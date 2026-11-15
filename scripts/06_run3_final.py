# -*- coding: utf-8 -*-
"""Run 3 — latih konfigurasi terpilih di full train 2021-2023, evaluasi val 2024 + test 2025 SEKALI.

Pemenang CV (dari models/selection_criteria_v3.json, seleksi HANYA dari CV 2021-2024):
  - Regresi : K03_xgb_optuna_reg  (CV MAE 4.8119)
  - Klasifikasi: K04_xgb_optuna_cls (CV F1-makro 0.3259)

Test 2025 SUDAH PERNAH DILIHAT pada Run 1 dan Run 2 untuk pelaporan;
TIDAK dipakai untuk seleksi apa pun di Run 3. Tidak ada pemilihan ulang setelah ini.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.ml import dataset as ds  # noqa: E402
from app.ml import run3  # noqa: E402
from app.ml.evaluate import _metrik_klasifikasi, _metrik_regresi  # noqa: E402
from app.ml.features import muat_dataset  # noqa: E402
from app.ml.pipeline import Preprocessor  # noqa: E402

logger = logging.getLogger("run3.final")
MODELS = PROJECT_ROOT / "models"
OUT_DIR = MODELS / "run3"


def _muat_params(nama_file: str) -> dict:
    d = json.load(open(MODELS / nama_file, encoding="utf-8"))
    return d["best_params"]


def main() -> None:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = muat_dataset()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    train = df[df["season"] <= 2023].reset_index(drop=True)
    val = df[df["season"] == 2024].reset_index(drop=True)
    test = df[df["season"] == 2025].reset_index(drop=True)
    logger.info("train=%d val=%d test=%d", len(train), len(val), len(test))

    pre = Preprocessor().fit(train)
    X_tr = ds.fitur_xgb(train, pre)
    y_reg_tr = train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls_tr = run3.encode_compound(train)
    w_cls_tr = run3.bobot_kelas(y_cls_tr, mode="sqrt10")[y_cls_tr]

    params_reg = _muat_params("run3_optuna_reg.json")
    params_clf = _muat_params("run3_optuna_cls.json")

    logger.info("Latih XGBRegressor (K03 params)...")
    reg = XGBRegressor(**params_reg)
    reg.fit(X_tr, y_reg_tr)
    reg.save_model(str(OUT_DIR / "xgb_reg.json"))

    logger.info("Latih XGBClassifier (K04 params + bobot sqrt10)...")
    clf = XGBClassifier(**params_clf)
    clf.fit(X_tr, y_cls_tr, sample_weight=w_cls_tr)
    clf.save_model(str(OUT_DIR / "xgb_clf.json"))

    import pickle
    with open(OUT_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)

    hasil = {}
    for nama, split in [("val", val), ("test", test)]:
        if len(split) == 0:
            logger.warning("Split %s kosong, dilewati", nama)
            continue
        X = ds.fitur_xgb(split, pre)
        y_reg = split["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls = run3.encode_compound(split)
        pred_reg = np.clip(reg.predict(X), 0, None)
        proba = ds.proba_compound_full(clf, X)
        pred_cls = proba.argmax(axis=1)
        m_reg = _metrik_regresi(y_reg, pred_reg)
        m_cls = _metrik_klasifikasi(y_cls, pred_cls)
        hasil[nama] = {"laps_remaining": m_reg, "compound": m_cls,
                       "proba": proba.tolist()}
        logger.info("[%s] MAE=%.4f RMSE=%.4f | acc=%.4f F1=%.4f",
                    nama, m_reg["mae"], m_reg["rmse"],
                    m_cls["accuracy"], m_cls["f1_macro"])

    # Baseline untuk perbandingan
    bl_path = MODELS / "baselines.json"
    if bl_path.exists():
        bl = json.load(open(bl_path, encoding="utf-8"))
        hasil["baselines"] = bl

    meta = {
        "run": 3,
        "model_reg": "K03_xgb_optuna_reg",
        "model_cls": "K04_xgb_optuna_cls",
        "train_seasons": [2021, 2022, 2023],
        "val_season": 2024,
        "test_season": 2025,
        "params_reg": params_reg,
        "params_clf": params_clf,
        "bobot_kelas": "sqrt-inverse-freq cap 10",
        "target_reg": "laps_remaining asli (bukan log1p)",
        "pernyataan_test": ("Test 2025 sudah pernah dilihat pada Run 1 dan Run 2 "
                            "untuk pelaporan; tidak dipakai untuk seleksi Run 3. "
                            "Evaluasi ini SATU-SATUNYA evaluasi test Run 3, "
                            "tidak ada pemilihan ulang sesudahnya."),
        "cv_selection": {
            "reg_winner": {"config": "K03_xgb_optuna_reg", "cv_mae": 4.8119},
            "cls_winner": {"config": "K04_xgb_optuna_cls", "cv_f1_macro": 0.3259},
        },
    }
    out = {"meta": meta, "hasil": hasil}
    with open(OUT_DIR / "metrics_run3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    logger.info("SELESAI. Artefak di %s", OUT_DIR)


if __name__ == "__main__":
    main()
