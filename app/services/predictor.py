"""Predictor: memuat artifact model sekali dan menjalankan inference.

Predictor hanya bekerja dengan RaceStateProvider — ia tidak tahu (dan tidak
peduli) apakah data datang dari replay historis atau live timing.

Model yang dimuat mengikuti models/selection.json (hasil pemilihan berbasis
val 2024): primary_model bisa "hybrid" atau "lstm", dengan suffix artefak
per seed. Bila selection.json tidak ada, fallback ke artef hybrid polos.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier, XGBRegressor

from app.config import get_settings
from app.ml import dataset as ds
from app.ml.features import COMPOUNDS, WINDOW_LAP, bangun_sequences
from app.ml.nn import muat_model
from app.ml.pipeline import Preprocessor
from app.schemas.prediksi import CompoundBerikutnya, PrediksiPembalap
from app.services.providers import RaceState, RaceStateProvider

logger = logging.getLogger(__name__)


class ModelBelumSiapError(RuntimeError):
    """Artifact model belum dilatih / tidak lengkap."""


class Predictor:
    """Wadah model final (LSTM atau Hybrid XGBoost-LSTM) untuk inference."""

    def __init__(self, models_dir: Path | None = None) -> None:
        md = models_dir or get_settings().models_dir

        # Tentukan model utama + suffix artefak dari METADATA di models/
        # (selection.json -> metadata.json); tipe model TIDAK di-hardcode.
        self.primary = None
        suffix = ""
        sel_path = md / "selection.json"
        meta_path = md / "metadata.json"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text())
            self.primary = sel.get("primary_model")
            suffix = sel.get("artifacts", {}).get(self.primary or "", "")
        if self.primary is None and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.primary = meta.get("primary_model")
            suffix = meta.get("artifacts", {}).get(self.primary or "", "")
        if self.primary is None:
            raise ModelBelumSiapError(
                "Tipe model final tidak ditemukan di models/ "
                "(butuh selection.json atau metadata.json dengan 'primary_model').")
        if self.primary not in ("lstm", "hybrid"):
            raise ModelBelumSiapError(f"primary_model tidak dikenal: {self.primary}")
        self.suffix = suffix

        wajib = [md / f"preprocessor{suffix}.pkl",
                 md / f"{self.primary}{suffix}_model.pt",
                 md / f"{self.primary}{suffix}_config.json",
                 md / "metadata.json"]
        if self.primary == "hybrid":
            wajib += [md / f"xgb_laps{suffix}.json", md / f"xgb_compound{suffix}.json"]
        hilang = [str(p) for p in wajib if not p.exists()]
        if hilang:
            raise ModelBelumSiapError(
                "Artifact model belum lengkap: " + ", ".join(hilang)
                + ". Jalankan `python -m app.ml.train` dulu.")

        self.models_dir = md
        self.pre = Preprocessor.muat(md / f"preprocessor{suffix}.pkl")
        if self.primary == "hybrid":
            self.reg = XGBRegressor(); self.reg.load_model(md / f"xgb_laps{suffix}.json")
            self.clf = XGBClassifier(); self.clf.load_model(md / f"xgb_compound{suffix}.json")
        else:
            self.reg = self.clf = None
        self.config = json.loads((md / f"{self.primary}{suffix}_config.json").read_text())
        self.nn = muat_model(md / f"{self.primary}{suffix}_model.pt",
                             self.config["arsitektur"])
        self.window = int(self.config.get("window", WINDOW_LAP))
        self.metadata = json.loads((md / "metadata.json").read_text())
        logger.info("Model %s%s v%s dimuat dari %s", self.primary, suffix,
                    self.metadata.get("model_version", "?"), md)

    # ------------------------------------------------------------------
    def info(self) -> dict:
        """Informasi model untuk endpoint /model/info."""
        metrik = None
        p = self.models_dir / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            run2 = m.get("run2")
            if run2:
                # Format Run 2: {run2: {val/test: {model_terpilih_detail, mean_std}}}
                metrik = {}
                for split in ("val", "test"):
                    blok = run2.get(split, {})
                    det = blok.get("model_terpilih_detail", {}).get(self.primary)
                    if not det:
                        continue
                    metrik[split] = {
                        "laps_remaining": det["laps_remaining"],
                        "compound": {k: v for k, v in det["compound"].items()
                                     if k != "confusion_matrix"},
                        "mean_std_semua_seed": blok.get("mean_std", {}).get(self.primary),
                    }
            else:
                # Format lama: {val/test: {model: {...}}}
                metrik = {}
                for split in ("val", "test"):
                    if split in m and self.primary in m[split]:
                        v = m[split][self.primary]
                        metrik[split] = {
                            "laps_remaining": v["laps_remaining"],
                            "compound": {k: val for k, val in v["compound"].items()
                                         if k != "confusion_matrix"},
                        }
        return {
            "model_version": self.metadata.get("model_version", "?"),
            "trained_at": self.metadata.get("trained_at"),
            "primary_model": self.primary,
            "artifact_suffix": self.suffix,
            "metrics": metrik,
            "window_lap": self.window,
            "compounds": COMPOUNDS,
            "split": self.metadata.get("split", {}),
        }

    # ------------------------------------------------------------------
    def prediksi_pembalap(self, provider: RaceStateProvider, state: RaceState,
                          driver_code: str) -> PrediksiPembalap | None:
        """Prediksi untuk satu pembalap pada lap `state.lap`.

        Riwayat lap 1..N (hanya info sampai lap N) -> fitur -> sliding window
        -> LSTM (dengan estimasi awal XGBoost per timestep bila hybrid).
        """
        hist = provider.get_driver_history(state.year, state.round,
                                           driver_code, state.lap)
        if hist.empty:
            logger.warning("Tidak ada riwayat lap untuk %s di %d/%d lap<=%d",
                           driver_code, state.year, state.round, state.lap)
            return None

        baris_terakhir = hist.iloc[-1]
        if self.primary == "hybrid":
            X_flat = ds.fitur_xgb(hist, self.pre)
            laps_xgb = self.reg.predict(X_flat)
            proba_xgb = ds.proba_compound_full(self.clf, X_flat)
            M = ds.fitur_hybrid(hist, self.pre, laps_xgb, proba_xgb)
        else:
            M = ds.fitur_lstm(hist, self.pre)

        kolom = [f"f{i}" for i in range(M.shape[1])]
        sub = pd.DataFrame(M[-self.window:], columns=kolom)
        seq = bangun_sequences(sub, kolom, window=self.window)[-1:]
        with torch.no_grad():
            pred_laps, logits = self.nn(torch.from_numpy(seq))
            proba = torch.softmax(logits, dim=1).numpy()[0]
        sisa_lap = int(round(max(0.0, float(pred_laps.numpy()[0]))))

        kelas_terbaik = int(np.argmax(proba))
        return PrediksiPembalap(
            code=driver_code,
            driver_id=str(baris_terakhir.get("driver_id", driver_code)),
            current_compound=(str(baris_terakhir["current_compound"])
                              if pd.notna(baris_terakhir.get("current_compound")) else None),
            tyre_life=(int(baris_terakhir["tyre_life"])
                       if pd.notna(baris_terakhir.get("tyre_life")) else None),
            predicted_laps_to_pit=sisa_lap,
            predicted_pit_lap=state.lap + sisa_lap,
            next_compound=CompoundBerikutnya(
                label=COMPOUNDS[kelas_terbaik],
                probabilities={c: round(float(p), 4)
                               for c, p in zip(COMPOUNDS, proba)}),
        )

    def prediksi_tim(self, provider: RaceStateProvider, state: RaceState) -> list[PrediksiPembalap]:
        """Prediksi untuk semua pembalap pada state (biasanya 2 pembalap satu tim)."""
        hasil = []
        for d in sorted(state.drivers, key=lambda x: x.driver_code):
            p = self.prediksi_pembalap(provider, state, d.driver_code)
            if p is not None:
                hasil.append(p)
        return hasil
