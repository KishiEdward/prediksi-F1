"""Setup logging terstruktur untuk seluruh aplikasi.

Format: waktu | level | nama modul | pesan. Cukup jelas untuk debugging
di laptop maupun di container (Docker/Render/Railway membaca stdout).
"""
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Konfigurasi logging global ke stdout dengan format terstruktur."""
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # FastF1 sangat cerewet di level INFO; cukup tampilkan peringatan.
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
