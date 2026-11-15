"""Unit test PENCEGAHAN DATA LEAKAGE (temporal).

Memverifikasi:
1. Sliding window sequence hanya memuat lap <= lap prediksi (tidak ada masa depan).
2. Fitur delta/lap in-race hanya memakai lap sebelumnya (shift, bukan pusat/masa depan).
3. Target laps_remaining dibentuk benar per stint dan stint terakhir tanpa target.
4. Preprocessor (scaler/encoder) hanya belajar dari data train.
5. Split kronologis: seluruh baris train lebih awal daripada val, val < test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import features
from app.ml.features import (FITUR_INRACE_NUMERIK, FITUR_PRA_NUMERIK, WINDOW_LAP,
                             bangun_sequences)
from app.ml.pipeline import Preprocessor


# ---------------------------------------------------------------------------
# 1) Sequence tidak boleh melihat lap masa depan
# ---------------------------------------------------------------------------
def test_sequence_tanpa_informasi_masa_depan():
    rng = np.random.default_rng(0)
    n, w, f = 12, WINDOW_LAP, 4
    M = rng.normal(size=(n, f)).astype(np.float32)
    df = pd.DataFrame(M, columns=[f"f{i}" for i in range(f)])

    seq = bangun_sequences(df, list(df.columns), window=w)
    assert seq.shape == (n, w, f)

    for i in range(n):
        # baris terakhir window HARUS = data lap i itu sendiri
        np.testing.assert_allclose(seq[i, -1], M[i])
        # seluruh isi window hanya berasal dari baris <= i (atau padding awal)
        for j in range(w):
            sumber = i - (w - 1 - j)
            if sumber >= 0:
                np.testing.assert_allclose(seq[i, j], M[sumber])
            else:
                # edge padding = mengulang baris pertama (tetap masa lalu)
                np.testing.assert_allclose(seq[i, j], M[0])


# ---------------------------------------------------------------------------
# 2) Fitur in-race hanya memakai informasi lap <= sekarang
# ---------------------------------------------------------------------------
def _laps_sintetis() -> pd.DataFrame:
    """Dua pembalap, 6 lap, waktu lap mudah ditebak."""
    baris = []
    for kode, basis in [("AAA", 90.0), ("BBB", 91.0)]:
        for lap in range(1, 7):
            baris.append({
                "season": 2024, "round": 1, "Driver": kode, "LapNumber": lap,
                "Stint": 1, "Time": lap * basis + (0.0 if kode == "AAA" else 0.5),
                "LapTime": basis + lap * 0.1,  # degradasi linear +0.1/lap
                "Compound": "MEDIUM", "TyreLife": float(lap), "FreshTyre": lap == 1,
                "Position": 1 if kode == "AAA" else 2, "PitIn": False, "PitOut": False,
                "TrackStatus": "1", "IsAccurate": True, "Deleted": False,
            })
    return pd.DataFrame(baris)


def test_fitur_inrace_hanya_masa_lalu():
    laps = _laps_sintetis()
    pra = pd.DataFrame([{"season": 2024, "round": 1, "total_laps": 6}])
    weather = pd.DataFrame(columns=["season", "round", "Time"])

    df = features._bangun_fitur_inrace(laps, weather, pra)
    aaa = df[df["driver_code"] == "AAA"].sort_values("lap_number").reset_index(drop=True)

    # delta_prev_lap pada lap t = LapTime(t) - LapTime(t-1) = +0.1 (bukan masa depan)
    np.testing.assert_allclose(aaa["delta_prev_lap"].iloc[2], 0.1, atol=1e-4)
    assert pd.isna(aaa["delta_prev_lap"].iloc[0])  # lap 1 tidak punya masa lalu

    # delta_pb_so_far memakai cumulative-min sampai lap berjalan saja
    np.testing.assert_allclose(aaa["delta_pb_so_far"].iloc[0], 0.0)
    assert (aaa["delta_pb_so_far"] >= -1e-9).all()  # tak pernah negatif (pb = min masa lalu)

    # race_progress hanya butuh total_laps (diketahui sebelum start) — bukan leakage
    np.testing.assert_allclose(aaa["race_progress"].iloc[5], 1.0)


# ---------------------------------------------------------------------------
# 3) Pembentukan target per stint
# ---------------------------------------------------------------------------
def test_target_laps_remaining_per_stint(tmp_path, monkeypatch):
    # Pitstop sintetis: driver AAA pit di lap 4 (dari total 6 lap)
    monkeypatch.setattr(features, "RAW", tmp_path)
    pd.DataFrame([
        {"season": 2024, "round": 1, "driverId": "aaa", "lap": "4"},
    ]).to_parquet(tmp_path / "jolpica_pitstops.parquet")

    baris = []
    for lap in range(1, 7):
        baris.append({
            "season": 2024, "round": 1, "driver_code": "AAA", "driver_id": "aaa",
            "lap_number": lap, "total_laps": 6.0,
            "current_compound": "MEDIUM" if lap <= 4 else "HARD",
        })
    df = pd.DataFrame(baris)

    hasil = features._bentuk_target(df)
    hasil = hasil.set_index("lap_number")

    # stint 1 (lap 1-4): target = 4 - t
    assert hasil.loc[1, "target_laps_remaining"] == 3
    assert hasil.loc[3, "target_laps_remaining"] == 1
    assert hasil.loc[4, "target_laps_remaining"] == 0
    # compound berikutnya = HARD (dipakai setelah pit)
    assert hasil.loc[1, "target_next_compound"] == "HARD"
    # stint terakhir (lap 5-6): tidak ada pit berikutnya -> tanpa target
    assert not hasil.loc[5, "has_target"]
    assert not hasil.loc[6, "has_target"]


def test_pit_di_lap_terakhir_dianggap_bukan_strategi(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "RAW", tmp_path)
    pd.DataFrame([
        {"season": 2024, "round": 1, "driverId": "aaa", "lap": "6"},  # = total lap
    ]).to_parquet(tmp_path / "jolpica_pitstops.parquet")

    baris = [{"season": 2024, "round": 1, "driver_code": "AAA", "driver_id": "aaa",
              "lap_number": lap, "total_laps": 6.0, "current_compound": "MEDIUM"}
             for lap in range(1, 7)]
    hasil = features._bentuk_target(pd.DataFrame(baris))
    assert (~hasil["has_target"]).all()


# ---------------------------------------------------------------------------
# 4) Preprocessor hanya belajar dari train
# ---------------------------------------------------------------------------
def _df_sintetis(n: int, offset: float) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {c: rng.normal(offset, 1.0, n) for c in FITUR_INRACE_NUMERIK + FITUR_PRA_NUMERIK}
    for c in FITUR_INRACE_NUMERIK + FITUR_PRA_NUMERIK:
        data[c] = np.abs(data[c]) + 0.1  # hindari NaN median
    data["driver_code"] = ["AAA"] * n
    data["team_id"] = ["t1"] * n
    data["circuit_id"] = ["c1"] * n
    return pd.DataFrame(data)


def test_preprocessor_fit_hanya_pada_train():
    train = _df_sintetis(50, offset=0.0)
    val = _df_sintetis(50, offset=1000.0)  # distribusi sangat berbeda

    pre = Preprocessor().fit(train)

    # Statistik scaler HARUS dari train saja
    np.testing.assert_allclose(pre.scaler_inrace.mean_,
                               train[FITUR_INRACE_NUMERIK].astype(float).mean().to_numpy(),
                               rtol=1e-4)
    # BUKAN dari gabungan train+val
    gabungan = pd.concat([train, val])
    assert not np.allclose(pre.scaler_inrace.mean_,
                           gabungan[FITUR_INRACE_NUMERIK].astype(float).mean().to_numpy())

    # Vocabulary kategori juga hanya dari train
    assert pre.vocab_kategorik["driver_code"] == {"AAA": 1}

    # Transform data tak dikenal tidak error, kode unseen = 0
    val["driver_code"] = "ZZZ"
    x = pre.transform_pra(val)
    assert x.shape[0] == 50


# ---------------------------------------------------------------------------
# 5) Split kronologis pada dataset nyata (dilewati bila belum dibuild)
# ---------------------------------------------------------------------------
def test_split_kronologis_dataset():
    from app.config import get_settings
    path = get_settings().processed_dir / "lap_dataset.parquet"
    if not path.exists():
        pytest.skip("Dataset belum dibangun; jalankan ekstraksi + python -m app.ml.features")
    df = pd.read_parquet(path, columns=["season", "split"])
    maks_train = df[df["split"] == "train"]["season"].max()
    min_val = df[df["split"] == "val"]["season"].min()
    maks_val = df[df["split"] == "val"]["season"].max()
    min_test = df[df["split"] == "test"]["season"].min()
    assert maks_train < min_val <= maks_val < min_test, \
        "Split harus kronologis: semua train lebih awal daripada val, val lebih awal dari test"
