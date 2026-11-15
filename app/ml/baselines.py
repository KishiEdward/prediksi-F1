"""Model baseline sederhana sebagai pembanding (bukan model utama).

Dua baseline sesuai permintaan:
1. BaselineMayoritas  : selalu memprediksi nilai konstan (median laps remaining
                        train + compound tersering di train).
2. BaselineTyreLife   : aturan sederhana berbasis umur ban:
                        - sisa lap = median panjang stint compound saat ini - tyre_life
                        - compound berikutnya = transisi tersering dari compound
                          saat ini (dipelajari dari data train).

Keduanya HANYA belajar dari data train (parameter disimpan ke JSON).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import COMPOUNDS


class BaselineMayoritas:
    """Prediksi konstan: median laps remaining + compound mayoritas train."""

    def __init__(self) -> None:
        self.median_laps: float = 0.0
        self.compound_mayoritas: str = "HARD"

    def fit(self, df_train: pd.DataFrame) -> "BaselineMayoritas":
        self.median_laps = float(df_train["target_laps_remaining"].median())
        self.compound_mayoritas = str(df_train["target_next_compound"].mode().iloc[0])
        return self

    def predict_laps(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.median_laps, dtype=np.float32)

    def predict_compound(self, df: pd.DataFrame) -> np.ndarray:
        idx = COMPOUNDS.index(self.compound_mayoritas)
        return np.full(len(df), idx, dtype=int)

    def to_dict(self) -> dict:
        return {"jenis": "mayoritas", "median_laps": self.median_laps,
                "compound_mayoritas": self.compound_mayoritas}

    @staticmethod
    def from_dict(d: dict) -> "BaselineMayoritas":
        b = BaselineMayoritas()
        b.median_laps = d["median_laps"]
        b.compound_mayoritas = d["compound_mayoritas"]
        return b


class BaselineTyreLife:
    """Aturan sederhana: pit ketika tyre_life mendekati panjang stint tipikal."""

    def __init__(self) -> None:
        self.stint_median: dict[str, float] = {}
        self.transisi: dict[str, str] = {}      # compound sekarang -> berikutnya
        self.fallback_laps: float = 15.0
        self.fallback_compound: str = "HARD"

    def fit(self, df_train: pd.DataFrame) -> "BaselineTyreLife":
        d = df_train.copy()
        # Perkiraan panjang stint: pada lap SEBELUM pit (target==1), umur ban di
        # lap pit berikutnya ~ tyre_life + 1.
        pre_pit = d[d["target_laps_remaining"] == 1]
        if len(pre_pit):
            stint = (pre_pit["tyre_life"].astype(float) + 1).groupby(
                pre_pit["current_compound"].astype(str)).median()
            self.stint_median = {k: float(v) for k, v in stint.items() if not np.isnan(v)}
        # Transisi compound tersering
        for comp, g in d.groupby(d["current_compound"].astype(str)):
            mode = g["target_next_compound"].astype(str).mode()
            if len(mode):
                self.transisi[comp] = str(mode.iloc[0])
        self.fallback_laps = float(d["target_laps_remaining"].median())
        self.fallback_compound = str(d["target_next_compound"].mode().iloc[0])
        return self

    def predict_laps(self, df: pd.DataFrame) -> np.ndarray:
        comp = df["current_compound"].astype(str)
        life = pd.to_numeric(df["tyre_life"], errors="coerce")
        stint = comp.map(self.stint_median).fillna(self.fallback_laps)
        sisa = stint - life.fillna(0.0)
        return np.clip(sisa.to_numpy(dtype=float), 0, None).astype(np.float32)

    def predict_compound(self, df: pd.DataFrame) -> np.ndarray:
        comp = df["current_compound"].astype(str)
        label = comp.map(self.transisi).fillna(self.fallback_compound)
        peta = {c: i for i, c in enumerate(COMPOUNDS)}
        return label.map(peta).fillna(COMPOUNDS.index(self.fallback_compound)) \
            .to_numpy().astype(int)

    def to_dict(self) -> dict:
        return {"jenis": "tyre_life", "stint_median": self.stint_median,
                "transisi": self.transisi, "fallback_laps": self.fallback_laps,
                "fallback_compound": self.fallback_compound}

    @staticmethod
    def from_dict(d: dict) -> "BaselineTyreLife":
        b = BaselineTyreLife()
        b.stint_median = d["stint_median"]
        b.transisi = d["transisi"]
        b.fallback_laps = d["fallback_laps"]
        b.fallback_compound = d["fallback_compound"]
        return b


def simpan_baselines(path: Path, mayor: BaselineMayoritas,
                     tyre: BaselineTyreLife) -> None:
    path.write_text(json.dumps({"mayoritas": mayor.to_dict(),
                                "tyre_life": tyre.to_dict()}, indent=2))


def muat_baselines(path: Path) -> tuple[BaselineMayoritas, BaselineTyreLife]:
    d = json.loads(path.read_text())
    return BaselineMayoritas.from_dict(d["mayoritas"]), BaselineTyreLife.from_dict(d["tyre_life"])
