# -*- coding: utf-8 -*-
"""Kalibrasi split-conformal untuk pit window Run 3.

Menggunakan residual dari fold cv2 (val 2024) — fold BERSIH yang tidak
terkontaminasi pool tuning Optuna (2021-2023). Interval simetris berbasis
kuantil |residual|, BUKAN P10-P90 mentah dari quantile regression.

Output: models/run3/conformal.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.ml.features import muat_dataset  # noqa: E402

logger = logging.getLogger("conformal")
OUT = PROJECT_ROOT / "models" / "run3" / "conformal.json"
PRED_CV2 = PROJECT_ROOT / "data" / "processed" / "run3_preds" / "K03_xgb_optuna_reg__cv2.npz"
ALPHA = 0.2  # interval 80%


def main() -> None:
    setup_logging()
    z = np.load(PRED_CV2)
    pred = z["pred_laps"].astype(float)

    df = muat_dataset()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    val = df[df["season"] == 2024].reset_index(drop=True)
    y = val["target_laps_remaining"].to_numpy(dtype=float)

    assert len(pred) == len(y), (
        f"Misalign: pred cv2 {len(pred)} baris vs val 2024 {len(y)} baris. "
        "Dataset mungkin dibangun ulang sejak CV — jalankan ulang stage xgb/optuna.")

    resid = y - pred
    abs_resid = np.abs(resid)
    n = len(abs_resid)
    # split conformal: level kuantile disesuaikan hingga (n+1)(1-alpha)/n
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = float(np.quantile(abs_resid, level, method="higher"))
    lo = pred - q
    hi = pred + q
    coverage = float(np.mean((y >= lo) & (y <= hi)))

    out = {
        "metode": "split-conformal simetris berbasis |residual| fold cv2 (val 2024)",
        "alpha": ALPHA,
        "tingkat_keyakinan": round(1 - ALPHA, 3),
        "kuantil_abs_resid": round(q, 4),
        "n_kalibrasi": n,
        "coverage_kalibrasi": round(coverage, 4),
        "level_kuantile_dipakai": round(float(level), 6),
        "mae_kalibrasi": round(float(np.mean(abs_resid)), 4),
        "catatan": ("Interval = [pred - q, pred + q], di-clip >= lap saat inference. "
                    "BUKAN P10-P90 mentah quantile regression (coverage hanya 52.8%). "
                    "cv2 dipilih sebagai set kalibrasi karena bebas dari overlap "
                    "pool tuning Optuna 2021-2023."),
        "sumber_config": "K03_xgb_optuna_reg",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    logger.info("Conformal: q=%.4f coverage=%.4f n=%d -> %s", q, coverage, n, OUT)


if __name__ == "__main__":
    main()
