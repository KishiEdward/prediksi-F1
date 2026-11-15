"""Evaluasi model: MAE/RMSE untuk laps remaining, accuracy/precision/recall/F1
untuk tyre compound, sesuai metrik pada laporan (subbab 2.2.9).

Jalankan dari root proyek:
    python -m app.ml.evaluate

Hasil disimpan ke models/metrics.json dan dicetak sebagai ringkasan.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)
from xgboost import XGBClassifier, XGBRegressor

from app.config import get_settings
from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.features import COMPOUNDS, WINDOW_LAP, muat_dataset
from app.ml.nn import muat_model
from app.ml.pipeline import Preprocessor

logger = logging.getLogger("evaluate")


def _metrik_regresi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, dan akurasi toleransi ±1 lap."""
    y_pred_clip = np.clip(y_pred, 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred_clip)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clip))),
        "dalam_1_lap": float(np.mean(np.abs(y_true - y_pred_clip) <= 1.0)),
    }


def _metrik_klasifikasi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1 (macro + per kelas) dan confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_kelas": {c: float(v) for c, v in zip(
            COMPOUNDS, f1_score(y_true, y_pred, average=None,
                                labels=list(range(len(COMPOUNDS))), zero_division=0))},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(COMPOUNDS)))).tolist(),
    }


def _prediksi_lstm(path_model: Path, path_config: Path, df: pd.DataFrame,
                   M: np.ndarray, pre: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Jalankan inference satu model NN (lstm/hybrid) pada df (sudah terurut)."""
    config = json.loads(path_config.read_text())
    model = muat_model(path_model, config["arsitektur"])
    y_cls = ds.target_dari_df(df, pre)
    X, _, _, _ = ds.bangun_dataset_sequence(df, M, y_cls, WINDOW_LAP)
    with torch.no_grad():
        pred_laps, logits = model(torch.from_numpy(X))
        proba = torch.softmax(logits, dim=1).numpy()
    return pred_laps.numpy(), proba


def main() -> None:
    setup_logging()
    s = get_settings()
    md = s.models_dir

    pre = Preprocessor.muat(md / "preprocessor.pkl")
    df = ds.urutkan_per_lap(muat_dataset())

    hasil: dict = {"evaluated_at": datetime.now(timezone.utc).isoformat()}

    for split in ("val", "test"):
        d = df[df["split"] == split].reset_index(drop=True)
        if d.empty:
            logger.warning("Split %s kosong, dilewati", split)
            continue
        y_reg = d["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls = ds.target_dari_df(d, pre)
        hasil[split] = {"n_rows": int(len(d))}

        X_xgb = ds.fitur_xgb(d, pre)

        # --- XGBoost (baseline) ---
        if (md / "xgb_laps.json").exists():
            reg = XGBRegressor(); reg.load_model(md / "xgb_laps.json")
            clf = XGBClassifier(); clf.load_model(md / "xgb_compound.json")
            hasil[split]["xgboost"] = {
                "laps_remaining": _metrik_regresi(y_reg, reg.predict(X_xgb)),
                "compound": _metrik_klasifikasi(y_cls, clf.predict(X_xgb)),
            }

        # --- LSTM tunggal (baseline) ---
        if (md / "lstm_model.pt").exists():
            pl, pr = _prediksi_lstm(md / "lstm_model.pt", md / "lstm_config.json",
                                    d, ds.fitur_lstm(d, pre), pre)
            hasil[split]["lstm"] = {
                "laps_remaining": _metrik_regresi(y_reg, pl),
                "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
            }

        # --- Hybrid XGBoost-LSTM ---
        if (md / "hybrid_model.pt").exists() and (md / "xgb_laps.json").exists():
            reg = XGBRegressor(); reg.load_model(md / "xgb_laps.json")
            clf = XGBClassifier(); clf.load_model(md / "xgb_compound.json")
            laps_xgb = reg.predict(X_xgb)
            proba_xgb = clf.predict_proba(X_xgb)
            M = ds.fitur_hybrid(d, pre, laps_xgb, proba_xgb)
            pl, pr = _prediksi_lstm(md / "hybrid_model.pt", md / "hybrid_config.json",
                                    d, M, pre)
            hasil[split]["hybrid"] = {
                "laps_remaining": _metrik_regresi(y_reg, pl),
                "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
            }

    (md / "metrics.json").write_text(json.dumps(hasil, indent=2))

    # Ringkasan terbaca
    for split in ("val", "test"):
        if split not in hasil:
            continue
        print(f"\n=== {split.upper()} ({hasil[split]['n_rows']} baris) ===")
        for nama_model in ("xgboost", "lstm", "hybrid"):
            if nama_model not in hasil[split]:
                continue
            m = hasil[split][nama_model]
            r, c = m["laps_remaining"], m["compound"]
            print(f"{nama_model:9s} | MAE {r['mae']:.2f} lap | RMSE {r['rmse']:.2f} lap | "
                  f"±1 lap {r['dalam_1_lap']:.1%} | acc {c['accuracy']:.1%} | "
                  f"F1-macro {c['f1_macro']:.3f}")
    print(f"\nMetrik lengkap: {md / 'metrics.json'}")


if __name__ == "__main__":
    main()
