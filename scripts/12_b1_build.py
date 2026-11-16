# -*- coding: utf-8 -*-
"""B1 — Bangun dataset eksperimen ke data/processed/b1/ (produksi TIDAK tertimpa).

Perubahan vs produksi (lihat models/selection_criteria_b1.json):
  - keep_inaccurate=True : pertahankan lap SC/VSC (IsAccurate=False), lap_time di-NaN
  - status_all_codes=True: parse SEMUA kode TrackStatus per lap (bukan digit pertama)
Hyperparameter model TIDAK diubah di sini (hanya dataset).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.logging_setup import setup_logging  # noqa: E402
from app.ml.features import PROCESSED, bangun_dataset  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("b1.build")


def main() -> None:
    setup_logging()
    out = PROCESSED / "b1"
    logger.info("Membangun dataset B1 ke %s ...", out)
    train_df = bangun_dataset(out_dir=out, keep_inaccurate=True, status_all_codes=True)

    inf = pd.read_parquet(out / "lap_inference.parquet")
    print("\n===== DIAGNOSTIK DATASET B1 =====")
    print(f"inference rows : {len(inf)}  (produksi lama: 106260)")
    print(f"training rows  : {len(train_df)}  (produksi lama: 57390)")
    print("\ntrack_status (inference):")
    print(inf["track_status"].value_counts().to_string())
    print("\nFitur status (sum / nunique) — HARUS hidup sekarang:")
    for c in ["is_green", "is_yellow", "is_sc", "is_vsc", "is_red", "laps_since_sc_vsc"]:
        col = pd.to_numeric(inf[c], errors="coerce")
        print(f"  {c:18s} sum={col.sum():8.0f}  nunique={col.nunique():3d}  "
              f"mean={col.mean():.4f}")
    print("\ntarget_laps_remaining (training):")
    t = train_df["target_laps_remaining"].astype(float)
    print(f"  min={t.min():.0f} max={t.max():.0f} mean={t.mean():.3f} "
          f"count==0={int((t==0).sum())} (harus 0)")
    print(f"  per musim: {train_df.groupby('season').size().to_dict()}")
    print("\nSELESAI. Dataset B1 di", out)


if __name__ == "__main__":
    main()
