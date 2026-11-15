"""Pembangunan matriks fitur & sequence untuk training/evaluasi/inference.

Modul ini dipakai bersama oleh app.ml.train, app.ml.evaluate, dan
app.services.predictor agar tidak ada perbedaan perlakuan data antara
proses training dan proses prediksi di API (train/serve consistency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.features import (COMPOUNDS, FITUR_INRACE_NUMERIK, FITUR_PRA_KATEGORIK,
                             FITUR_PRA_NUMERIK, bangun_sequences)
from app.ml.pipeline import Preprocessor

# Nama kolom kunci untuk mengelompokkan lap per pembalap per race
GRUP_KEYS = ["season", "round", "driver_code"]


def urutkan_per_lap(df: pd.DataFrame) -> pd.DataFrame:
    """Urutkan baris agar lap setiap pembalap-race berurutan naik."""
    return df.sort_values(GRUP_KEYS + ["lap_number"]).reset_index(drop=True)


def fitur_xgb(df: pd.DataFrame, pre: Preprocessor) -> np.ndarray:
    """Matriks fitur XGBoost = fitur pra-balapan + fitur in-race lap berjalan.

    XGBoost tidak melihat sejarah lap (model tabular), sesuai pembagian peran
    pada laporan: XGBoost untuk kondisi awal + kondisi lap saat ini.
    """
    return np.hstack([pre.transform_pra(df), pre.transform_inrace(df)])


def fitur_lstm(df: pd.DataFrame, pre: Preprocessor) -> np.ndarray:
    """Matriks fitur LSTM tunggal = hanya fitur in-race (tanpa pra-balapan)."""
    return pre.transform_inrace(df)


def fitur_hybrid(df: pd.DataFrame, pre: Preprocessor,
                 xgb_pred_laps: np.ndarray, xgb_proba: np.ndarray) -> np.ndarray:
    """Matriks fitur hybrid = in-race + pra-balapan + estimasi awal XGBoost.

    Mekanisme integrasi (disetujui pada rencana): estimasi XGBoost ditempel
    sebagai fitur tambahan pada SETIAP timestep LSTM.
    """
    laps_scaled = pre.transform_xgb_laps(xgb_pred_laps)
    return np.hstack([
        pre.transform_pra(df),
        pre.transform_inrace(df),
        laps_scaled,
        xgb_proba.astype(np.float32),
    ])


def bangun_dataset_sequence(df: pd.DataFrame, matriks_fitur: np.ndarray,
                            y_cls: np.ndarray, window: int) -> tuple:
    """Bentuk sequence per grup pembalap-race.

    df HARUS sudah terurut (pakai urutkan_per_lap) dan matriks_fitur serta
    y_cls selaras barisnya dengan df.

    Return: (X, y_reg, y_cls, meta) dengan X berbentuk (N, window, F).
    """
    df = df.reset_index(drop=True)
    y_reg = df["target_laps_remaining"].to_numpy(dtype=np.float32)

    # Batas grup: baris consecutive dengan kunci (season, round, driver_code) sama
    kunci = df[GRUP_KEYS].astype(str).agg("|".join, axis=1)
    batas = np.flatnonzero(np.r_[True, kunci[1:].to_numpy() != kunci[:-1].to_numpy()])
    batas_akhir = np.r_[batas[1:], len(df)]

    daftar_x: list[np.ndarray] = []
    for a, b in zip(batas, batas_akhir):
        sub = pd.DataFrame(matriks_fitur[a:b],
                           columns=[f"f{i}" for i in range(matriks_fitur.shape[1])])
        seq = bangun_sequences(sub, list(sub.columns), window=window)
        daftar_x.append(seq)

    X = np.concatenate(daftar_x, axis=0) if daftar_x else \
        np.zeros((0, window, matriks_fitur.shape[1]), dtype=np.float32)
    return X, y_reg, y_cls, df


def target_dari_df(df: pd.DataFrame, pre: Preprocessor) -> np.ndarray:
    """Indeks kelas compound (0..4) dari kolom target."""
    return pre.encode_target_compound(df)


def nama_fitur_xgb() -> list[str]:
    """Daftar nama kolom fitur XGBoost (untuk dokumentasi/metadata)."""
    return FITUR_PRA_NUMERIK + FITUR_PRA_KATEGORIK + FITUR_INRACE_NUMERIK


def nama_fitur_hybrid(n_xgb_proba: int = len(COMPOUNDS)) -> list[str]:
    """Daftar nama kolom fitur hybrid (per timestep LSTM)."""
    return (FITUR_PRA_NUMERIK + FITUR_PRA_KATEGORIK + FITUR_INRACE_NUMERIK
            + ["xgb_pred_laps"] + [f"xgb_proba_{c.lower()}" for c in COMPOUNDS])
