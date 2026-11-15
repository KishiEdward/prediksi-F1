"""TAHAP (a-2): Ekstraksi data in-race dari FastF1 -> Parquet per race.

Untuk tiap balapan utama musim 2021-2025 (daftar race dibaca dari hasil
script 01), simpan:
- laps        : LapTime, Compound, TireLife, Stint, Position, PitIn/PitOut, dll
- weather     : suhu udara/aspal, hujan, angin per waktu
- trackstatus : status lintasan (hijau/SC/VSC/dll) per waktu
- pitstops    : catatan pit stop versi FastF1 (bila tersedia)

Jalankan:  python scripts/02_extract_fastf1.py
Idempotent: file per-race yang sudah ada akan dilewati; cache FastF1
mencegah download ulang data mentah dari server F1.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.services.fastf1_svc import muat_sesi_race  # noqa: E402

logger = logging.getLogger("extract.fastf1")

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1"

# Kolom laps yang disimpan (hanya yang ada di versi FastF1 terpasang)
KOLOM_LAPS = [
    "Driver", "DriverNumber", "LapNumber", "Stint", "Time", "LapTime", "Compound",
    "TireLife", "FreshTyre", "Position", "PitIn", "PitOut", "TrackStatus",
    "IsAccurate", "Deleted", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
]


def _timedelta_ke_detik(df: pd.DataFrame, kolom: str) -> None:
    """Ubah kolom timedelta menjadi detik (float) agar mudah dipakai ML."""
    if kolom in df.columns:
        df[kolom] = df[kolom].dt.total_seconds().astype("Float64")


def ekstrak_satu_race(tahun: int, ronde: int) -> bool:
    """Ekstrak satu race. Return True bila file baru dibuat, False bila dilewati."""
    prefix = OUT_DIR / f"{tahun}_{ronde:02d}"
    if Path(str(prefix) + "_laps.parquet").exists():
        logger.info("Lewati %d round %d (sudah ada)", tahun, ronde)
        return False

    try:
        sesi = muat_sesi_race(tahun, ronde)
    except Exception as exc:  # race batal / data tidak tersedia
        logger.warning("Gagal memuat %d round %d: %s", tahun, ronde, exc)
        return False

    # --- Laps ---
    laps = sesi.laps.copy()
    kolom = [k for k in KOLOM_LAPS if k in laps.columns]
    laps = laps[kolom].reset_index(drop=True)
    for k in ("Time", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        _timedelta_ke_detik(laps, k)
    laps.insert(0, "season", tahun)
    laps.insert(1, "round", ronde)
    laps.to_parquet(str(prefix) + "_laps.parquet")

    # --- Weather ---
    w = sesi.weather_data.copy()
    _timedelta_ke_detik(w, "Time")
    w.insert(0, "season", tahun)
    w.insert(1, "round", ronde)
    w.to_parquet(str(prefix) + "_weather.parquet")

    # --- Track status ---
    ts = sesi.track_status.copy()
    _timedelta_ke_detik(ts, "Time")
    ts.insert(0, "season", tahun)
    ts.insert(1, "round", ronde)
    ts.to_parquet(str(prefix) + "_trackstatus.parquet")

    # --- Pit stops FastF1 (opsional, tergantung versi) ---
    pit = getattr(sesi, "pit_stops", None)
    if pit is not None and len(pit) > 0:
        p = pit.reset_index(drop=True).copy()
        _timedelta_ke_detik(p, "duration")
        p.insert(0, "season", tahun)
        p.insert(1, "round", ronde)
        p.to_parquet(str(prefix) + "_pitstops.parquet")

    logger.info("Selesai %d round %d: %d lap", tahun, ronde, len(laps))
    return True


def main() -> None:
    setup_logging()
    s = get_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jadwal = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "jolpica_races.parquet")
    jadwal = jadwal[jadwal["season"].astype(int).isin(s.seasons)]

    baru, gagal = 0, 0
    total = len(jadwal)
    for i, row in jadwal.iterrows():
        tahun, ronde = int(row["season"]), int(row["round"])
        logger.info("[%d/%d] %s %d round %d", len(jadwal[jadwal.index <= i]), total,
                    row.get("raceName", ""), tahun, ronde)
        try:
            if ekstrak_satu_race(tahun, ronde):
                baru += 1
        except Exception as exc:
            gagal += 1
            logger.error("Error %d round %d: %s", tahun, ronde, exc, exc_info=True)

    logger.info("SELESAI. %d race baru diekstrak, %d gagal. Output: %s", baru, gagal, OUT_DIR)


if __name__ == "__main__":
    main()
