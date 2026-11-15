"""Predictor: memuat artifact model sekali dan menjalankan inference.

Predictor hanya bekerja dengan RaceStateProvider — ia tidak tahu (dan tidak
peduli) apakah data datang dari replay historis atau live timing.
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
    """Wadah model hybrid (XGBoost + LSTM) untuk inference."""

    def __init__(self, models_dir: Path | None = None) -> None:
        md = models_dir or get_settings().models_dir
        wajib = [md / "preprocessor.pkl", md / "xgb_laps.json",
                 md / "xgb_compound.json", md / "hybrid_model.pt",
                 md / "hybrid_config.json", md / "metadata.json"]
        hilang = [str(p) for p in wajib if not p.exists()]
        if hilang:
            raise ModelBelumSiapError(
                "Artifact model belum lengkap: " + ", ".join(hilang)
                + ". Jalankan `python -m app.ml.train` dulu.")

        self.models_dir = md
        self.pre = Preprocessor.muat(md / "preprocessor.pkl")
        self.reg = XGBRegressor(); self.reg.load_model(md / "xgb_laps.json")
        self.clf = XGBClassifier(); self.clf.load_model(md / "xgb_compound.json")
        self.config = json.loads((md / "hybrid_config.json").read_text())
        self.nn = muat_model(md / "hybrid_model.pt", self.config["arsitektur"])
        self.window = int(self.config.get("window", WINDOW_LAP))
        self.metadata = json.loads((md / "metadata.json").read_text())
        logger.info("Model hybrid v%s dimuat dari %s",
                    self.metadata.get("model_version", "?"), md)

    # ------------------------------------------------------------------
    def info(self) -> dict:
        """Informasi model untuk endpoint /model/info."""
        metrik = None
        p = self.models_dir / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            metrik = {split: {model: {
                "laps_remaining": v["laps_remaining"],
                "compound": {k: val for k, val in v["compound"].items()
                             if k != "confusion_matrix"},
            } for model, v in d.items() if model != "n_rows"}
                for split, d in m.items() if split in ("val", "test")}
        return {
            "model_version": self.metadata.get("model_version", "?"),
            "trained_at": self.metadata.get("trained_at"),
            "metrics": metrik,
            "window_lap": self.window,
            "compounds": COMPOUNDS,
            "split": self.metadata.get("split", {}),
        }

    # ------------------------------------------------------------------
    def prediksi_pembalap(self, provider: RaceStateProvider, state: RaceState,
                          driver_code: str) -> PrediksiPembalap | None:
        """Prediksi untuk satu pembalap pada lap `state.lap`.

        Langkah: ambil riwayat lap 1..N pembalap (hanya info sampai lap N),
        hitung fitur flat -> estimasi awal XGBoost per baris -> rakit fitur
        hybrid -> sliding window -> LSTM -> dua head keluaran.
        """
        hist = provider.get_driver_history(state.year, state.round,
                                           driver_code, state.lap)
        if hist.empty:
            logger.warning("Tidak ada riwayat lap untuk %s di %d/%d lap<=%d",
                           driver_code, state.year, state.round, state.lap)
            return None

        baris_terakhir = hist.iloc[-1]
        X_flat = ds.fitur_xgb(hist, self.pre)
        laps_xgb = self.reg.predict(X_flat)
        proba_xgb = ds.proba_compound_full(self.clf, X_flat)
        M = ds.fitur_hybrid(hist, self.pre, laps_xgb, proba_xgb)

        # Ambil maksimum `window` baris terakhir; bangun_sequences melakukan
        # edge-padding bila riwayat lebih pendek dari window (awal race).
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
