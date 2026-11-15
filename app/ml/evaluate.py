"""Evaluasi model sesuai metrik laporan (subbab 2.2.9) + permintaan pengguna.

Jalankan SETELAH training (dan idealnya HANYA SEKALI untuk split test):
    python -m app.ml.evaluate

Isi laporan:
- Baseline (mayoritas & aturan tyre_life), XGBoost, LSTM, Hybrid.
- Regresi laps remaining: MAE, RMSE, dalam-±1-lap.
- Klasifikasi compound: accuracy, precision/recall/F1 MACRO + PER KELAS,
  confusion matrix.
- Analisis ablation (tanpa 2021 di train) HANYA pada val — split test tetap
  dievaluasi sekali untuk model utama agar tidak terjadi "test peeking".
- Semua hasil disimpan ke models/metrics.json.
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
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from xgboost import XGBClassifier, XGBRegressor

from app.config import get_settings
from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.baselines import muat_baselines
from app.ml.features import COMPOUNDS, WINDOW_LAP, muat_dataset
from app.ml.nn import muat_model
from app.ml.pipeline import Preprocessor

logger = logging.getLogger("evaluate")


def _metrik_regresi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred_clip)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred_clip))), 4),
        "dalam_1_lap": round(float(np.mean(np.abs(y_true - y_pred_clip) <= 1.0)), 4),
    }


def _metrik_klasifikasi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(COMPOUNDS)))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro",
                                                       zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro",
                                                 zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "per_kelas": {
            c: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
            for c, p, r, f, s in zip(COMPOUNDS, prec, rec, f1, supp)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _cetak_confusion(cm: list[list[int]], judul: str) -> None:
    print(f"  Confusion matrix ({judul}) — baris=aktual, kolom=prediksi:")
    head = "        " + "".join(f"{c[:5]:>7}" for c in COMPOUNDS)
    print(head)
    for nama, baris in zip(COMPOUNDS, cm):
        print(f"{nama[:6]:<8}" + "".join(f"{v:>7}" for v in baris))


def _prediksi_nn(mode: str, suffix: str, md: Path, d: pd.DataFrame,
                 pre: Preprocessor, xgb_reg=None, xgb_clf=None):
    """Inference model NN (lstm/hybrid) -> (pred_laps, proba)."""
    if mode == "lstm":
        M = ds.fitur_lstm(d, pre)
    else:
        X_flat = ds.fitur_xgb(d, pre)
        M = ds.fitur_hybrid(d, pre, xgb_reg.predict(X_flat),
                            ds.proba_compound_full(xgb_clf, X_flat))
    config = json.loads((md / f"{mode}{suffix}_config.json").read_text())
    model = muat_model(md / f"{mode}{suffix}_model.pt", config["arsitektur"])
    y_cls = ds.target_dari_df(d, pre)
    X, _, _, _ = ds.bangun_dataset_sequence(d, M, y_cls, WINDOW_LAP)
    with torch.no_grad():
        pred_laps, logits = model(torch.from_numpy(X))
        proba = torch.softmax(logits, dim=1).numpy()
    return pred_laps.numpy(), proba


def evaluasi_satu_split(d: pd.DataFrame, md: Path, suffix: str = "",
                        dengan_baseline: bool = True) -> dict:
    """Evaluasi semua model pada satu split dataframe."""
    pre = Preprocessor.muat(md / f"preprocessor{suffix}.pkl")
    y_reg = d["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(d, pre)
    hasil: dict = {"n_rows": int(len(d))}

    X_xgb = ds.fitur_xgb(d, pre)
    reg_path = md / f"xgb_laps{suffix}.json"
    clf_path = md / f"xgb_compound{suffix}.json"
    reg = clf = None
    if reg_path.exists():
        reg = XGBRegressor(); reg.load_model(reg_path)
        clf = XGBClassifier(); clf.load_model(clf_path)

    if dengan_baseline:
        mayor, tyre = muat_baselines(md / "baselines.json")
        hasil["baseline_mayoritas"] = {
            "laps_remaining": _metrik_regresi(y_reg, mayor.predict_laps(d)),
            "compound": _metrik_klasifikasi(y_cls, mayor.predict_compound(d)),
        }
        hasil["baseline_tyre_life"] = {
            "laps_remaining": _metrik_regresi(y_reg, tyre.predict_laps(d)),
            "compound": _metrik_klasifikasi(y_cls, tyre.predict_compound(d)),
        }

    if reg is not None:
        hasil["xgboost"] = {
            "laps_remaining": _metrik_regresi(y_reg, reg.predict(X_xgb)),
            "compound": _metrik_klasifikasi(y_cls, clf.predict(X_xgb)),
        }
    if (md / f"lstm{suffix}_model.pt").exists():
        pl, pr = _prediksi_nn("lstm", suffix, md, d, pre)
        hasil["lstm"] = {
            "laps_remaining": _metrik_regresi(y_reg, pl),
            "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
        }
    if (md / f"hybrid{suffix}_model.pt").exists() and reg is not None:
        pl, pr = _prediksi_nn("hybrid", suffix, md, d, pre, reg, clf)
        hasil["hybrid"] = {
            "laps_remaining": _metrik_regresi(y_reg, pl),
            "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
        }
    return hasil


def cetak_ringkasan(nama_split: str, hasil: dict) -> None:
    print(f"\n{'='*78}\n=== {nama_split} ({hasil['n_rows']} baris) ===")
    print(f"{'model':<22}{'MAE':>7}{'RMSE':>8}{'+-1lap':>8}{'acc':>8}"
          f"{'P-makro':>9}{'R-makro':>9}{'F1-makro':>10}")
    for model, m in hasil.items():
        if model == "n_rows":
            continue
        r, c = m["laps_remaining"], m["compound"]
        print(f"{model:<22}{r['mae']:>7.2f}{r['rmse']:>8.2f}{r['dalam_1_lap']:>8.1%}"
              f"{c['accuracy']:>8.1%}{c['precision_macro']:>9.3f}"
              f"{c['recall_macro']:>9.3f}{c['f1_macro']:>10.3f}")
    for model, m in hasil.items():
        if model == "n_rows":
            continue
        per_kelas = m["compound"]["per_kelas"]
        ringkas = " | ".join(f"{c[:4]}: F1={v['f1']:.2f} (n={v['support']})"
                             for c, v in per_kelas.items())
        print(f"  [{model}] per kelas -> {ringkas}")
        _cetak_confusion(m["compound"]["confusion_matrix"], model)


def main() -> None:
    setup_logging()
    s = get_settings()
    md = s.models_dir
    df = ds.urutkan_per_lap(muat_dataset())

    hasil: dict = {"evaluated_at": datetime.now(timezone.utc).isoformat()}

    # ---- Model utama: val + test (test HANYA di sini, sekali) ----
    d_val = df[df["split"] == "val"].reset_index(drop=True)
    d_test = df[df["split"] == "test"].reset_index(drop=True)
    hasil["val"] = evaluasi_satu_split(d_val, md)
    hasil["test"] = evaluasi_satu_split(d_test, md)
    cetak_ringkasan("VAL 2024", hasil["val"])
    cetak_ringkasan("TEST 2025 (sekali, akhir)", hasil["test"])

    # ---- Analisis ablation tanpa 2021: HANYA val (test tidak disentuh lagi) ----
    if (md / "preprocessor_no2021.pkl").exists():
        d_train21 = df[(df["split"] == "train") & (df["season"] == 2021)]
        hasil["ablation_no2021_val"] = evaluasi_satu_split(
            d_val, md, suffix="_no2021", dengan_baseline=False)
        cetak_ringkasan("ABLATION tanpa-2021 @ VAL 2024", hasil["ablation_no2021_val"])
        hasil["ablation_catatan"] = {
            "baris_2021_dibuang": int(len(d_train21)),
            "keterangan": ("Ablation hanya dievaluasi pada val 2024 agar split "
                           "test tetap sekali pakai untuk model utama."),
        }

    (md / "metrics.json").write_text(json.dumps(hasil, indent=2))
    print(f"\nMetrik lengkap tersimpan: {md / 'metrics.json'}")


if __name__ == "__main__":
    main()
