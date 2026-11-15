"""Test leakage pada DATASET ASLI (bukan sintetis/mock).

Semua test di sini di-skip otomatis bila dataset belum dibangun
(jalankan: python scripts/01..03 lebih dulu).

Test kuat (truncated rebuild): fitur pada lap N dihitung ulang dari data mentah
FastF1 yang DIPOTONG hanya sampai lap N. Bila hasil hitung ulang identik dengan
nilai tersimpan di dataset, terbukti fitur tidak memakai informasi masa depan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import get_settings
from app.ml import features

RAW = get_settings().raw_dir
PROCESSED = get_settings().processed_dir

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "lap_inference.parquet").exists(),
    reason="Dataset asli belum dibangun (jalankan scripts 01-03)",
)


@pytest.fixture(scope="module")
def df_inf() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "lap_inference.parquet")


# ---------------------------------------------------------------------------
# 1) Konsistensi delta & personal best pada data asli
# ---------------------------------------------------------------------------
def test_delta_prev_lap_konsisten_data_asli(df_inf):
    d = df_inf.sort_values(["season", "round", "driver_code", "lap_number"])
    g = d.groupby(["season", "round", "driver_code"], sort=False)
    recompute = d["lap_time_sec"] - g["lap_time_sec"].shift(1)
    mask = d["delta_prev_lap"].notna() & recompute.notna()
    np.testing.assert_allclose(d.loc[mask, "delta_prev_lap"].astype(float),
                               recompute[mask].astype(float), atol=1e-4)


def test_delta_pb_tidak_pernah_negatif_data_asli(df_inf):
    # pb_so_far = cumulative min sampai lap berjalan -> delta tak pernah negatif.
    # Bila fitur memakai min masa depan, akan muncul nilai negatif.
    d = df_inf["delta_pb_so_far"].dropna().astype(float)
    assert (d >= -1e-6).all(), "delta_pb_so_far negatif terdeteksi (indikasi leakage)"


def test_target_valid_data_asli(df_inf):
    d = df_inf[df_inf["has_target"]]
    assert (d["target_laps_remaining"] >= 0).all()
    assert d["target_next_compound"].isin(features.COMPOUNDS).all()
    # sisa lap menuju pit tidak mungkin melebihi sisa lap balapan
    assert (d["target_laps_remaining"] <= d["total_laps"]).all()


# ---------------------------------------------------------------------------
# 2) Truncated rebuild: bukti kuat tidak ada informasi masa depan
# ---------------------------------------------------------------------------
SAMPLES = [(2021, 1, "HAM", 20), (2022, 10, "VER", 25), (2023, 5, "ALO", 30)]


@pytest.mark.parametrize("season,rnd,driver,lap", SAMPLES)
def test_fitur_lap_N_identik_dengan_hitung_ulang_terpotong(df_inf, season, rnd,
                                                           driver, lap):
    """Hitung ulang fitur dari data mentah yang dipotong sampai lap N."""
    path_laps = RAW / "fastf1" / f"{season}_{rnd:02d}_laps.parquet"
    if not path_laps.exists():
        pytest.skip(f"raw fastf1 {season}/{rnd} tidak ada")

    asli = df_inf[(df_inf["season"] == season) & (df_inf["round"] == rnd)
                  & (df_inf["driver_code"] == driver) & (df_inf["lap_number"] == lap)]
    if asli.empty:
        pytest.skip(f"baris {season}/{rnd} {driver} lap {lap} tidak ada di dataset")
    asli = asli.iloc[0]

    # Potong data mentah: hanya lap <= N
    laps = pd.read_parquet(path_laps)
    laps = laps[laps["LapNumber"] <= lap]

    # Cuaca juga dipotong sampai waktu lap N (asof backward)
    time_batas = None
    baris_terakhir = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)]
    if not baris_terakhir.empty and "Time" in baris_terakhir.columns:
        t = pd.to_numeric(baris_terakhir["Time"].iloc[0], errors="coerce")
        time_batas = float(t) if pd.notna(t) else None
    weather_path = RAW / "fastf1" / f"{season}_{rnd:02d}_weather.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else \
        pd.DataFrame(columns=["season", "round", "Time"])
    if time_batas is not None and len(weather):
        weather = weather[pd.to_numeric(weather["Time"], errors="coerce") <= time_batas]

    pra = features._bangun_fitur_prabalapan()
    dibangun = features._bangun_fitur_inrace(laps, weather, pra)
    row = dibangun[(dibangun["driver_code"] == driver) & (dibangun["lap_number"] == lap)]
    assert not row.empty, "gagal merekonstruksi baris"
    row = row.iloc[0]

    kolom_uji = ["lap_time_sec", "delta_prev_lap", "delta_mean3_lap", "delta_pb_so_far",
                 "tyre_life", "position", "gap_to_leader", "race_progress",
                 "is_green", "is_sc", "is_vsc", "laps_since_sc_vsc", "track_status"]
    for k in kolom_uji:
        a, b = asli.get(k), row.get(k)
        if isinstance(a, str) or isinstance(b, str):
            assert a == b, f"{k}: {a} != {b}"
        else:
            if pd.isna(a) and pd.isna(b):
                continue
            np.testing.assert_allclose(float(a), float(b), atol=1e-4,
                                       err_msg=f"kolom {k} berbeda (indikasi leakage)")
