"""TAHAP (a-3): Konversi dataset Kaggle "F1 Safety Car Research 2022-2025"
(data lap FastF1 yang sudah diekstrak pihak ketiga) ke format raw pipeline kita.

Latar belakang: server F1 memblokir sementara permintaan anonim saat ekstraksi
langsung baru mencapai 44/114 race (2021-2022). Dataset ini menutupi 2023-2025
(68 dari 70 race; Miami 2023 & Spain 2025 tidak ada -> menyusul via FastF1).

Dua mode:
  python scripts/02b_convert_kaggle.py            # konversi 2023-2025
  python scripts/02b_convert_kaggle.py --validate # validasi silang 2022 saja:
       membandingkan file Kaggle dengan hasil ekstraksi FastF1 kita sendiri.

Catatan provenance untuk skripsi: data pada dasarnya bersumber dari telemetri
FastF1; musim 2021-2022 diekstrak langsung oleh peneliti, musim 2023-2025
memakai arsip publik berbasis FastF1 yang SUDAH divalidasi silang terhadap
ekstraksi mandiri peneliti pada musim 2022 (lihat laporan validasi).
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("convert.kaggle")

KAGGLE_BASE = Path(r"C:\A_kuliah\Skripsi\skripsi-F1\Multi-Season Formula 1 Lap Dataset "
                   r"with Race-Contr (1)\Multi-Season Formula 1 Lap Dataset with Race-Contr"
                   r"\F1_SafetyCar_Research_2022_2025\F1_SafetyCar_Research_2022_2025\datasets")
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"


def norm_nama(nama: str) -> str:
    """Normalisasi nama event agar bisa dicocokkan dengan raceName Jolpica."""
    n = str(nama).lower().replace("_", " ")
    n = re.sub(r"[^a-z ]+", " ", n)
    n = n.replace("grand prix", "")
    return re.sub(r"\s+", " ", n).strip()


def peta_round_per_musim() -> dict[int, dict[str, int]]:
    """Buat peta {season: {nama_event_normalisasi: round}} dari jadwal Jolpica."""
    races = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "jolpica_races.parquet")
    races["season"] = races["season"].astype(int)
    races["round"] = races["round"].astype(int)
    peta: dict[int, dict[str, int]] = {}
    for _, r in races.iterrows():
        peta.setdefault(int(r["season"]), {})[norm_nama(r["raceName"])] = int(r["round"])
    return peta


def file_race_kaggle(year: int) -> list[tuple[str, Path]]:
    """Daftar (nama_event, path) file race Kaggle untuk satu musim (tanpa test)."""
    hasil = []
    for f in sorted(glob.glob(str(KAGGLE_BASE / f"year={year}" / "*.parquet"))):
        nama = os.path.basename(f).split("event=")[1].replace("_session=R.parquet", "")
        if "test" in nama or "track_session" in nama:
            continue
        hasil.append((nama, Path(f)))
    return hasil


def konversi_satu_race(year: int, round_: int, src: Path) -> None:
    """Konversi satu file race Kaggle -> parquet format pipeline kita."""
    df = pd.read_parquet(src)

    laps = pd.DataFrame({
        "season": year,
        "round": round_,
        "Driver": df["Driver"],
        "DriverNumber": np.nan,
        "LapNumber": pd.to_numeric(df["LapNumber"], errors="coerce"),
        "Stint": pd.to_numeric(df["Stint"], errors="coerce"),
        "Time": pd.to_numeric(df["Time"], errors="coerce"),
        "LapTime": pd.to_numeric(df["LapTime"], errors="coerce"),
        "Compound": df["Compound"].astype("string").str.upper(),
        "TyreLife": pd.to_numeric(df["TyreLife"], errors="coerce"),
        "FreshTyre": df["FreshTyre"],
        "Position": pd.to_numeric(df["Position"], errors="coerce"),
        # PitIn/PitOut diturunkan dari timestamp pit (True bila ada nilainya)
        "PitIn": df["PitInTime"].notna() if "PitInTime" in df else False,
        "PitOut": df["PitOutTime"].notna() if "PitOutTime" in df else False,
        "TrackStatus": df["TrackStatus"].astype("string"),
        "IsAccurate": df["IsAccurate"].astype(bool),
        "Deleted": None,  # tidak tersedia di dataset Kaggle
        "Sector1Time": pd.to_numeric(df.get("Sector1Time"), errors="coerce"),
        "Sector2Time": pd.to_numeric(df.get("Sector2Time"), errors="coerce"),
        "Sector3Time": pd.to_numeric(df.get("Sector3Time"), errors="coerce"),
    })
    prefix = OUT_DIR / f"{year}_{round_:02d}"
    laps.to_parquet(str(prefix) + "_laps.parquet")

    # Cuaca per lap -> file weather terpisah (format sama dengan ekstraksi FastF1)
    if "AirTemp" in df.columns:
        w = pd.DataFrame({
            "season": year, "round": round_,
            "Time": pd.to_numeric(df["Time"], errors="coerce"),
            "AirTemp": pd.to_numeric(df["AirTemp"], errors="coerce"),
            "Humidity": pd.to_numeric(df["Humidity"], errors="coerce"),
            "Pressure": pd.to_numeric(df.get("Pressure"), errors="coerce"),
            "Rainfall": pd.to_numeric(df["Rainfall"].astype(bool), errors="coerce"),
            "TrackTemp": pd.to_numeric(df["TrackTemp"], errors="coerce"),
            "WindDirection": pd.to_numeric(df.get("WindDirection"), errors="coerce"),
            "WindSpeed": pd.to_numeric(df.get("WindSpeed"), errors="coerce"),
        }).dropna(subset=["Time"]).drop_duplicates(subset=["Time"])
        w.to_parquet(str(prefix) + "_weather.parquet")

    # Track status per lap -> file trackstatus (dedup per Time)
    if "TrackStatus" in df.columns:
        ts = pd.DataFrame({
            "season": year, "round": round_,
            "Time": pd.to_numeric(df["Time"], errors="coerce"),
            "Status": df["TrackStatus"].astype("string"),
        }).dropna(subset=["Time"]).drop_duplicates(subset=["Time"])
        ts.to_parquet(str(prefix) + "_trackstatus.parquet")

    logger.info("Konversi %d round %d: %d baris lap", year, round_, len(laps))


def konversi(years: list[int]) -> None:
    """Konversi semua race Kaggle untuk daftar musim (lewati yang sudah ada)."""
    peta = peta_round_per_musim()
    dikonversi, dilewati, gagal = 0, 0, []
    for year in years:
        for nama, path in file_race_kaggle(year):
            kunci = norm_nama(nama)
            round_ = peta.get(year, {}).get(kunci)
            if round_ is None:
                gagal.append(f"{year} {nama} (round tidak ditemukan di jadwal Jolpica)")
                continue
            out = OUT_DIR / f"{year}_{round_:02d}_laps.parquet"
            if out.exists():
                dilewati += 1
                continue
            konversi_satu_race(year, round_, path)
            dikonversi += 1
    logger.info("SELESAI konversi: %d baru, %d dilewati (sudah ada), %d gagal: %s",
                dikonversi, dilewati, len(gagal), gagal or "-")


def validasi_silang(year: int = 2022) -> None:
    """Bandingkan file Kaggle dengan ekstraksi FastF1 mandiri untuk musim `year`.

    Memverifikasi: jumlah lap per race, kecocokan compound & TyreLife per
    (driver, lap), dan kecocokan LapTime. Hasil dicetak sebagai laporan.
    """
    peta = peta_round_per_musim()
    ringkasan = []
    for nama, path in file_race_kaggle(year):
        round_ = peta.get(year, {}).get(norm_nama(nama))
        if round_ is None:
            continue
        ours_path = OUT_DIR / f"{year}_{round_:02d}_laps.parquet"
        if not ours_path.exists():
            continue
        a = pd.read_parquet(ours_path)   # ekstraksi mandiri
        b = pd.read_parquet(path)        # kaggle
        a["LapNumber"] = a["LapNumber"].astype(float)
        b["LapNumber"] = pd.to_numeric(b["LapNumber"], errors="coerce")
        ka = a[["Driver", "LapNumber", "Compound", "TyreLife", "LapTime"]].rename(
            columns={"Compound": "comp_a", "TyreLife": "life_a", "LapTime": "lt_a"})
        kb = b[["Driver", "LapNumber", "Compound", "TyreLife", "LapTime"]].rename(
            columns={"Compound": "comp_b", "TyreLife": "life_b", "LapTime": "lt_b"})
        kb["LapNumber"] = kb["LapNumber"].astype(float)
        m = ka.merge(kb, on=["Driver", "LapNumber"], how="inner")
        if m.empty:
            ringkasan.append((year, round_, len(a), len(b), None, None, None))
            continue
        comp_match = float((m["comp_a"].astype("string").str.upper()
                            == m["comp_b"].astype("string").str.upper()).mean())
        life_match = float((m["life_a"].fillna(-1) == m["life_b"].fillna(-1)).mean())
        lt_close = float(np.isclose(m["lt_a"].astype(float), m["lt_b"].astype(float),
                                    atol=0.01, equal_nan=True).mean())
        ringkasan.append((year, round_, len(a), len(b), comp_match, life_match, lt_close))

    print(f"\n=== VALIDASI SILANG {year}: ekstraksi mandiri vs dataset Kaggle ===")
    print(f"{'round':>5} {'lap(kita)':>9} {'lap(kaggle)':>11} {'compound%':>9} "
          f"{'tyrelife%':>9} {'laptim%':>8}")
    ok = 0
    for row in ringkasan:
        year_, round_, na, nb, cm, lm, lt = row
        if cm is None:
            print(f"{round_:>5} {na:>9} {nb:>11}  (tidak ada baris cocok)")
            continue
        print(f"{round_:>5} {na:>9} {nb:>11} {cm*100:>8.1f}% {lm*100:>8.1f}% {lt*100:>7.1f}%")
        if cm >= 0.98 and lm >= 0.98 and lt >= 0.98:
            ok += 1
    print(f"\nRace dengan kecocokan >=98% pada semua aspek: {ok}/{len(ringkasan)}")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="hanya validasi silang musim 2022 (tanpa konversi)")
    ap.add_argument("--years", default="2023,2024,2025",
                    help="musim yang dikonversi (default: 2023,2024,2025)")
    args = ap.parse_args()

    if args.validate:
        validasi_silang(2022)
    else:
        years = [int(y) for y in args.years.split(",")]
        konversi(years)


if __name__ == "__main__":
    main()
