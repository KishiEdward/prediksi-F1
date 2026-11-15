"""TAHAP (a-1): Ekstraksi data historis dari Jolpica F1 API -> Parquet.

Ambil musim 2021-2025 (bisa diubah lewat .env):
- races        : jadwal + info sirkuit per round
- results      : hasil balapan utama per pembalap
- qualifying   : hasil kualifikasi
- sprint       : hasil sprint race
- pitstops     : catatan pit stop (lap, durasi, waktu)
- drivers      : daftar pembalap per musim
- constructors : daftar konstruktor/tim per musim
- circuits     : karakteristik sirkuit

Jalankan:  python scripts/01_extract_jolpica.py
Idempotent: respons di-cache ke disk; file parquet ditimpa ulang tiap run.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.services.jolpica import JolpicaClient  # noqa: E402

logger = logging.getLogger("extract.jolpica")

RAW = PROJECT_ROOT / "data" / "raw"


def _races_per_musim(client: JolpicaClient, musim: int, endpoint: str,
                     record_key: str, per_round: bool = False) -> pd.DataFrame:
    """Ambil data bersarang per-race (results/qualifying/sprint/pitstops).

    Bila per_round=True, endpoint dipanggil per round (`{musim}/{round}/...`)
    karena Ergast/Jolpica tidak mendukung `{musim}/pitstops.json` langsung.
    """
    if per_round:
        jadwal = client.get_semua_halaman(f"{musim}.json", ["RaceTable", "Races"])
        races = []
        for r in jadwal:
            sub = client.get_semua_halaman(
                f"{musim}/{r['round']}/{endpoint}.json", ["RaceTable", "Races"])
            races.extend(sub)
    else:
        races = client.get_semua_halaman(f"{musim}/{endpoint}.json", ["RaceTable", "Races"])
    baris: list[dict] = []
    for r in races:
        for item in r.get(record_key, []):
            baris.append({
                "season": int(r["season"]),
                "round": int(r["round"]),
                "raceName": r.get("raceName"),
                **item,
            })
    df = pd.json_normalize(baris) if baris else pd.DataFrame()
    logger.info("  %s %d: %d baris", endpoint, musim, len(df))
    return df


def main() -> None:
    setup_logging()
    s = get_settings()
    client = JolpicaClient()
    RAW.mkdir(parents=True, exist_ok=True)

    # 1) Jadwal race (termasuk sirkuit & tanggal)
    semua = []
    for musim in s.seasons:
        races = client.get_semua_halaman(f"{musim}.json", ["RaceTable", "Races"])
        df = pd.json_normalize(races)
        logger.info("races %d: %d seri", musim, len(df))
        semua.append(df)
    pd.concat(semua, ignore_index=True).to_parquet(RAW / "jolpica_races.parquet")

    # 2) Data bersarang per race
    for endpoint, key, per_round in [("results", "Results", False),
                                     ("qualifying", "QualifyingResults", False),
                                     ("sprint", "SprintResults", False),
                                     ("pitstops", "PitStops", True)]:
        semua = [_races_per_musim(client, m, endpoint, key, per_round) for m in s.seasons]
        semua = [d for d in semua if not d.empty]
        if semua:
            pd.concat(semua, ignore_index=True).to_parquet(RAW / f"jolpica_{endpoint}.parquet")
        else:
            logger.warning("Tidak ada data untuk %s", endpoint)

    # 3) Referensi drivers & constructors per musim
    for endpoint, kunci in [("drivers", "DriverTable"), ("constructors", "ConstructorTable")]:
        semua = []
        for musim in s.seasons:
            rows = client.get_semua_halaman(f"{musim}/{endpoint}.json", [kunci])
            df = pd.json_normalize(rows)
            if not df.empty:
                df["season"] = musim
            semua.append(df)
        pd.concat(semua, ignore_index=True).to_parquet(RAW / f"jolpica_{endpoint}.parquet")
        logger.info("%s: total %d baris", endpoint, sum(len(d) for d in semua))

    # 4) Latihan bebas (FP1-FP3): TIDAK DIEKSTRAK.
    # Jolpica menolak endpoint practice ("Endpoint does not support final
    # filter"), sehingga hasil FP tidak tersedia dari sumber data penelitian.
    # "Hasil sesi sebelumnya" tetap tercakup oleh kualifikasi + sprint.
    # Dicatat sebagai keterbatasan data (lihat README).

    # 5) Referensi circuits (statis lintas musim)
    circuits = client.get_semua_halaman("circuits.json", ["CircuitTable", "Circuits"])
    pd.json_normalize(circuits).to_parquet(RAW / "jolpica_circuits.parquet")
    logger.info("circuits: %d sirkuit", len(circuits))

    logger.info("SELESAI. Parquet tersimpan di %s", RAW)


if __name__ == "__main__":
    main()
