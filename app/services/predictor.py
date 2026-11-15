"""Predictor: memuat artifact model sekali dan menjalankan inference.

Predictor hanya bekerja dengan RaceStateProvider — ia tidak tahu (dan tidak
peduli) apakah data datang dari replay historis atau live timing.

Model yang dimuat mengikuti models/metadata.json (+ selection.json):
  - "xgb_separate" (Run 3, FINAL): 2 XGB terpisah (K03-reg, K04-clf) + binary
    head "masih pit?" + pit window split-conformal. Prediksi per-baris, tanpa
    sequence/NN. Artefak di models/run3/, path dari metadata["artifacts"].
  - "lstm" / "hybrid" (Run 1/2, arsip): jalur sequence-based lama, dipertahankan
    untuk kompatibilitas test dan rollback, bukan model aktif.

Tipe model TIDAK di-hardcode — selalu dibaca dari metadata di models/.
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
from app.schemas.prediksi import (CompoundBerikutnya, JendelaPit,
                                  PeringatanPrediksi, PrediksiPembalap)
from app.services.providers import RaceState, RaceStateProvider

logger = logging.getLogger(__name__)

_BUCKET_KEYS = {"0-5": "mae_jarak_0_5", "6-15": "mae_jarak_6_15",
                ">15": "mae_jarak_gt15"}


class ModelBelumSiapError(RuntimeError):
    """Artifact model belum dilatih / tidak lengkap."""


class Predictor:
    """Wadah model final (LSTM atau Hybrid XGBoost-LSTM) untuk inference."""

    def __init__(self, models_dir: Path | None = None) -> None:
        md = models_dir or get_settings().models_dir
        self.models_dir = md

        # Tentukan model utama dari METADATA di models/ (TIDAK di-hardcode).
        # selection.json diperiksa dulu, fallback ke metadata.json.
        self.primary = None
        sel_path = md / "selection.json"
        meta_path = md / "metadata.json"
        meta_raw = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if sel_path.exists():
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
            self.primary = sel.get("primary_model")
        if self.primary is None:
            self.primary = meta_raw.get("primary_model")
        if self.primary is None:
            raise ModelBelumSiapError(
                "Tipe model final tidak ditemukan di models/ "
                "(butuh selection.json atau metadata.json dengan 'primary_model').")

        self.metadata = meta_raw
        self.suffix = ""
        self.nn = None
        self.config: dict = {}
        self.window = 1
        self.reg = self.clf = self.binary = None
        self.pre_binary = None
        self.conformal: dict = {}
        self.caveats_meta: dict = {}
        self.test_metrics: dict = {}

        if self.primary == "xgb_separate":
            # ----- Run 3 (FINAL): 2 XGB terpisah + binary head + conformal -----
            art = (meta_raw.get("artifacts", {}) or {}).get("xgb_separate")
            if not isinstance(art, dict):
                raise ModelBelumSiapError(
                    "metadata.json artifacts['xgb_separate'] harus berupa dict path relatif.")
            wajib = ["reg", "clf", "preprocessor", "conformal"]
            hilang = [k for k in wajib if k not in art or not (md / art[k]).exists()]
            if hilang:
                raise ModelBelumSiapError(
                    "Artefak Run 3 belum lengkap: " + ", ".join(hilang)
                    + ". Jalankan scripts/06_run3_final.py + 07_binary_head.py + 08_conformal.py.")
            self.reg = XGBRegressor()
            self.reg.load_model(md / art["reg"])
            self.clf = XGBClassifier()
            self.clf.load_model(md / art["clf"])
            if "binary" in art and (md / art["binary"]).exists():
                self.binary = XGBClassifier()
                self.binary.load_model(md / art["binary"])
            self.pre = Preprocessor.muat(md / art["preprocessor"])
            if "preprocessor_binary" in art and (md / art["preprocessor_binary"]).exists():
                self.pre_binary = Preprocessor.muat(md / art["preprocessor_binary"])
            self.conformal = json.loads((md / art["conformal"]).read_text(encoding="utf-8"))
            self.caveats_meta = meta_raw.get("caveats", {})
            self.test_metrics = meta_raw.get("test_metrics", {})
            self.window = int(meta_raw.get("window_lap", 1))
            logger.info("Model xgb_separate (Run 3) v%s dimuat dari %s",
                        meta_raw.get("model_version", "?"), md)
            return

        # ----- Run 1/2 (arsip): jalur sequence-based lstm/hybrid -----
        if self.primary not in ("lstm", "hybrid"):
            raise ModelBelumSiapError(f"primary_model tidak dikenal: {self.primary}")
        suffix = ""
        art_lama = meta_raw.get("artifacts", {}) or {}
        if isinstance(art_lama.get(self.primary), str):
            suffix = art_lama[self.primary]
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

        self.pre = Preprocessor.muat(md / f"preprocessor{suffix}.pkl")
        if self.primary == "hybrid":
            self.reg = XGBRegressor(); self.reg.load_model(md / f"xgb_laps{suffix}.json")
            self.clf = XGBClassifier(); self.clf.load_model(md / f"xgb_compound{suffix}.json")
        self.config = json.loads((md / f"{self.primary}{suffix}_config.json").read_text())
        self.nn = muat_model(md / f"{self.primary}{suffix}_model.pt",
                             self.config["arsitektur"])
        self.window = int(self.config.get("window", WINDOW_LAP))
        logger.info("Model %s%s v%s dimuat dari %s", self.primary, suffix,
                    self.metadata.get("model_version", "?"), md)

    # ------------------------------------------------------------------
    def info(self) -> dict:
        """Informasi model untuk endpoint /model/info."""
        base = {
            "model_version": self.metadata.get("model_version", "?"),
            "trained_at": self.metadata.get("trained_at"),
            "primary_model": self.primary,
            "window_lap": self.window,
            "compounds": COMPOUNDS,
            "split": self.metadata.get("split", {}),
        }

        if self.primary == "xgb_separate":
            # Run 3: metrik + conformal + binary + caveats langsung dari metadata
            bin_met = None
            mb_path = self.models_dir / "run3" / "metrics_binary.json"
            if mb_path.exists():
                mb = json.loads(mb_path.read_text(encoding="utf-8"))
                bin_met = {
                    "test": mb.get("final", {}).get("test"),
                    "val": mb.get("final", {}).get("val"),
                    "cv": mb.get("cv"),
                    "data_stats": mb.get("data_stats"),
                }
            base.update({
                "artifact_suffix": "",
                "metrics": {
                    "val": self.metadata.get("val_metrics"),
                    "test": self.metadata.get("test_metrics"),
                },
                "conformal": self.conformal or None,
                "binary_head": bin_met,
                "caveats": self.caveats_meta or None,
                "cv_selection": self.metadata.get("cv_selection"),
            })
            return base

        # ----- Run 1/2 (arsip): format lama dari metrics.json -----
        metrik = None
        p = self.models_dir / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            run2 = m.get("run2")
            if run2:
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
                metrik = {}
                for split in ("val", "test"):
                    if split in m and self.primary in m[split]:
                        v = m[split][self.primary]
                        metrik[split] = {
                            "laps_remaining": v["laps_remaining"],
                            "compound": {k: val for k, val in v["compound"].items()
                                         if k != "confusion_matrix"},
                        }
        base.update({"artifact_suffix": self.suffix, "metrics": metrik})
        return base

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

        if self.primary == "xgb_separate":
            return self._prediksi_xgb_separate(hist, state, driver_code)

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

    # ------------------------------------------------------------------
    # Run 3: inference XGB terpisah (reg + clf + binary) tanpa sequence/NN
    # ------------------------------------------------------------------
    def _prediksi_xgb_separate(self, hist: pd.DataFrame, state: RaceState,
                               driver_code: str) -> PrediksiPembalap:
        baris = hist.iloc[[-1]].copy()
        row0 = baris.iloc[0]

        X_rc = ds.fitur_xgb(baris, self.pre)
        sisa_lap = int(round(max(0.0, float(self.reg.predict(X_rc)[0]))))
        proba = ds.proba_compound_full(self.clf, X_rc)[0]
        kelas_terbaik = int(np.argmax(proba))
        compound_label = COMPOUNDS[kelas_terbaik]

        p_masih_pit = None
        if self.binary is not None and self.pre_binary is not None:
            X_bin = ds.fitur_xgb(baris, self.pre_binary)
            p_masih_pit = float(self.binary.predict_proba(X_bin)[0, 1])

        return PrediksiPembalap(
            code=driver_code,
            driver_id=str(row0.get("driver_id", driver_code)),
            current_compound=(str(row0["current_compound"])
                              if pd.notna(row0.get("current_compound")) else None),
            tyre_life=(int(row0["tyre_life"])
                       if pd.notna(row0.get("tyre_life")) else None),
            predicted_laps_to_pit=sisa_lap,
            predicted_pit_lap=state.lap + sisa_lap,
            next_compound=CompoundBerikutnya(
                label=compound_label,
                probabilities={c: round(float(p), 4)
                               for c, p in zip(COMPOUNDS, proba)}),
            probabilitas_masih_pit=(round(p_masih_pit, 4)
                                    if p_masih_pit is not None else None),
            jendela_pit=self._buat_jendela(sisa_lap, state.lap),
            peringatan=self._buat_peringatan(sisa_lap, compound_label),
        )

    def _buat_jendela(self, sisa_lap: int, current_lap: int) -> JendelaPit | None:
        """Pit window split-conformal: [pred-q, pred+q] di ruang laps_remaining."""
        if not self.conformal:
            return None
        q = float(self.conformal.get("kuantil_abs_resid", 0))
        if q <= 0:
            return None
        qi = int(round(q))
        lr_bawah = max(0, sisa_lap - qi)
        lr_atas = sisa_lap + qi
        return JendelaPit(
            lap_bawah=current_lap + lr_bawah,
            lap_atas=current_lap + lr_atas,
            tingkat_keyakinan=float(self.conformal.get("tingkat_keyakinan", 0.8)),
            coverage_kalibrasi=float(self.conformal.get("coverage_kalibrasi", 0.0)),
            setengah_lebar=round(q, 4),
        )

    def _buat_peringatan(self, sisa_lap: int,
                         compound_label: str) -> PeringatanPrediksi:
        """Caveat kontekstual: keandalan compound basah + akurasi per bucket jarak."""
        if sisa_lap <= 5:
            bucket = "0-5"
        elif sisa_lap <= 15:
            bucket = "6-15"
        else:
            bucket = ">15"
        mae_val = self.test_metrics.get("laps_remaining", {}).get(_BUCKET_KEYS.get(bucket, ""))
        wet = compound_label in ("INTERMEDIATE", "WET")
        catatan: list[str] = []
        if wet:
            catatan.append(self.caveats_meta.get(
                "wet_unreliable",
                "INTERMEDIATE/WET sangat tidak andal (data sangat sedikit di train/test)."))
        if bucket == ">15":
            catatan.append(
                "Prediksi pit >15 lap punya MAE tertinggi (~9.8 lap di test 2025); "
                "gunakan jendela_pit sebagai rentang, bukan titik prediksi tunggal.")
        return PeringatanPrediksi(
            compound_basah_tidak_andal=wet,
            bucket_jarak=bucket,
            mae_bucket_test=(round(float(mae_val), 4) if mae_val is not None else None),
            catatan=catatan,
        )

    def prediksi_tim(self, provider: RaceStateProvider, state: RaceState) -> list[PrediksiPembalap]:
        """Prediksi untuk semua pembalap pada state (biasanya 2 pembalap satu tim)."""
        hasil = []
        for d in sorted(state.drivers, key=lambda x: x.driver_code):
            p = self.prediksi_pembalap(provider, state, d.driver_code)
            if p is not None:
                hasil.append(p)
        return hasil
