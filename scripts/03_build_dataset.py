"""TAHAP (b): Bangun dataset siap latih dari data mentah.

Jalankan (setelah scripts 01 & 02 selesai):
    python scripts/03_build_dataset.py

Output:
- data/processed/lap_dataset.parquet   : baris training (punya target lengkap)
- data/processed/lap_inference.parquet : SEMUA lap (dipakai API replay)
- data/processed/dataset_summary.json  : ringkasan statistik untuk laporan

Anti-leakage dijamin oleh app/ml/features.py (lihat README bagian 6).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.ml.features import (FITUR_INRACE_NUMERIK, FITUR_PRA_NUMERIK,  # noqa: E402
                             bangun_dataset, muat_dataset_inference)

logger = logging.getLogger("build_dataset")


def ringkasan(df_inf: pd.DataFrame) -> dict:
    """Statistik dataset untuk laporan: baris/race per musim, NaN, target."""
    train = df_inf[df_inf["has_target"]]

    per_musim = {}
    for musim, g in df_inf.groupby("season"):
        per_musim[int(musim)] = {
            "jumlah_lap": int(len(g)),
            "jumlah_race": int(g["round"].nunique()),
            "jumlah_baris_training": int(g["has_target"].sum()),
        }

    # Distribusi target compound
    dist = train["target_next_compound"].astype(str).value_counts().to_dict()
    dist = {k: int(v) for k, v in dist.items()}

    # Persentase NaN per fitur (pada dataset training)
    semua_fitur = FITUR_PRA_NUMERIK + FITUR_INRACE_NUMERIK
    nan_pct = {}
    for k in semua_fitur:
        if k in train.columns:
            nan_pct[k] = round(float(train[k].isna().mean() * 100), 2)

    # Statistik target regresi
    tr = train["target_laps_remaining"]
    statistik_target = {
        "min": float(tr.min()), "maks": float(tr.max()),
        "rata2": round(float(tr.mean()), 2), "median": float(tr.median()),
    }

    # Split
    split = {k: int(v) for k, v in train["split"].value_counts().items()}
    race_per_split = {}
    for s in ("train", "val", "test"):
        g = train[train["split"] == s]
        race_per_split[s] = {
            "musim": sorted(int(x) for x in g["season"].unique()),
            "jumlah_race": int(len(g.groupby(["season", "round"]))),
            "jumlah_baris": int(len(g)),
        }

    return {
        "total_baris_inference": int(len(df_inf)),
        "total_baris_training": int(len(train)),
        "persentase_punya_target": round(float(df_inf["has_target"].mean() * 100), 2),
        "per_musim": per_musim,
        "distribusi_target_compound": dist,
        "statistik_laps_remaining": statistik_target,
        "nan_persen_per_fitur": nan_pct,
        "split": split,
        "race_per_split": race_per_split,
    }


def main() -> None:
    setup_logging()
    s = get_settings()

    logger.info("=== Membangun dataset (fitur + target + anti-leakage) ===")
    bangun_dataset()

    logger.info("=== Menghitung ringkasan ===")
    df_inf = muat_dataset_inference()
    ring = ringkasan(df_inf)

    out = s.processed_dir / "dataset_summary.json"
    out.write_text(json.dumps(ring, indent=2))
    logger.info("Ringkasan tersimpan: %s", out)

    print("\n--- RINGKASAN DATASET ---")
    print(json.dumps(ring, indent=2))


if __name__ == "__main__":
    main()
