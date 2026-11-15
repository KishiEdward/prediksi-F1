"""Praproses data untuk model: scaler, encoder, pengisi nilai hilang.

ATURAN ANTI-LEAKAGE:
- Preprocessor.fit() HANYA boleh dipanggil dengan data split TRAIN.
  Median, StandardScaler, dan vocabulary kategori semuanya berasal dari train.
- Data val/test (dan data saat inference API) hanya melewati transform().
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.ml.features import COMPOUNDS, FITUR_INRACE_NUMERIK, FITUR_PRA_KATEGORIK, FITUR_PRA_NUMERIK


class Preprocessor:
    """Menyimpan semua hasil fit dari data train agar bisa dipakai ulang."""

    def __init__(self) -> None:
        self.median_pra: dict[str, float] = {}
        self.median_inrace: dict[str, float] = {}
        self.scaler_pra = StandardScaler()
        self.scaler_inrace = StandardScaler()
        self.vocab_kategorik: dict[str, dict[str, int]] = {}
        self.compounds: list[str] = list(COMPOUNDS)
        # scaler untuk estimasi XGBoost yang ditempel ke input hybrid
        self.scaler_xgb_laps = StandardScaler()

    # ---------------- fit (HANYA data train) ----------------
    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        for kolom in FITUR_PRA_NUMERIK:
            m = float(df_train[kolom].median())
            # kolom yang seluruhnya NaN di train (mis. data practice tak tersedia)
            # dinetralkan agar tidak meracuni imputasi
            self.median_pra[kolom] = 0.0 if np.isnan(m) else m
        for kolom in FITUR_INRACE_NUMERIK:
            m = float(df_train[kolom].median())
            self.median_inrace[kolom] = 0.0 if np.isnan(m) else m

        # fit dengan NaN dibiarkan: StandardScaler mengabaikan NaN saat fit
        pra = self._siapkan(df_train, FITUR_PRA_NUMERIK, self.median_pra, impute=False)
        inrace = self._siapkan(df_train, FITUR_INRACE_NUMERIK, self.median_inrace,
                               impute=False)
        self.scaler_pra.fit(pra)
        self.scaler_inrace.fit(inrace)
        for sc in (self.scaler_pra, self.scaler_inrace):
            sc.mean_ = np.nan_to_num(sc.mean_, nan=0.0)
            sc.scale_ = np.nan_to_num(sc.scale_, nan=1.0)
            sc.var_ = np.nan_to_num(sc.var_, nan=1.0)

        for kolom in FITUR_PRA_KATEGORIK:
            unik = sorted(df_train[kolom].astype(str).unique())
            # kode 0 dicadangkan untuk kategori tak dikenal (unseen)
            self.vocab_kategorik[kolom] = {v: i + 1 for i, v in enumerate(unik)}
        return self

    @staticmethod
    def _siapkan(df: pd.DataFrame, kolom: list[str], median: dict[str, float],
                 impute: bool) -> np.ndarray:
        """Siapkan matriks numerik; isi NaN dengan median train HANYA bila impute.

        XGBoost mendukung NaN secara native (missing value dipelajari arahnya),
        sehingga jalur XGBoost memakai impute=False. Jalur LSTM (numerik padat)
        memakai impute=True. StandardScaler mengabaikan NaN saat fit dan
        meneruskan NaN saat transform.
        """
        x = df[kolom].astype(float).copy()
        if impute:
            for k in kolom:
                x[k] = x[k].fillna(median[k])
        return x.to_numpy(dtype=np.float32)

    # ---------------- transform ----------------
    def transform_pra(self, df: pd.DataFrame, impute: bool = True) -> np.ndarray:
        """Fitur pra-balapan: numerik ter-scale + kode kategori."""
        x_num = self.scaler_pra.transform(
            self._siapkan(df, FITUR_PRA_NUMERIK, self.median_pra, impute))
        x_kat = np.column_stack([
            df[k].astype(str).map(self.vocab_kategorik[k]).fillna(0).to_numpy()
            for k in FITUR_PRA_KATEGORIK
        ]).astype(np.float32)
        return np.hstack([x_num, x_kat]).astype(np.float32)

    def transform_inrace(self, df: pd.DataFrame, impute: bool = True) -> np.ndarray:
        """Fitur in-race numerik ter-scale (dipakai sebagai timestep LSTM)."""
        return self.scaler_inrace.transform(
            self._siapkan(df, FITUR_INRACE_NUMERIK, self.median_inrace, impute)
        ).astype(np.float32)

    def transform_xgb_laps(self, pred: np.ndarray) -> np.ndarray:
        """Scale estimasi laps-remaining dari XGBoost (untuk fitur hybrid)."""
        return self.scaler_xgb_laps.transform(pred.reshape(-1, 1)).astype(np.float32)

    def fit_xgb_laps(self, pred: np.ndarray) -> None:
        """Fit scaler estimasi XGBoost memakai prediksi out-of-fold data TRAIN."""
        self.scaler_xgb_laps.fit(pred.reshape(-1, 1))

    def encode_target_compound(self, df: pd.DataFrame) -> np.ndarray:
        """Ubah label compound menjadi indeks kelas 0..4 (urutan COMPOUNDS)."""
        peta = {c: i for i, c in enumerate(self.compounds)}
        return df["target_next_compound"].astype(str).map(peta).fillna(-1).to_numpy().astype(int)

    # ---------------- simpan / muat ----------------
    def simpan(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def muat(path: str | Path) -> "Preprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 - artifact dibuat sendiri
