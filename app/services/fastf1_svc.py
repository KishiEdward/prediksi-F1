"""Wrapper tipis untuk pustaka FastF1.

Tanggung jawab:
- Mengaktifkan cache disk agar data sesi tidak diunduh dua kali.
- Menyediakan akses ke sesi Race (balapan utama) per tahun/round.
"""
import logging

import fastf1
from fastf1.core import Session

from app.config import get_settings

logger = logging.getLogger(__name__)

_cache_aktif = False


def aktifkan_cache() -> None:
    """Aktifkan cache FastF1 sekali saja (idempotent)."""
    global _cache_aktif
    if not _cache_aktif:
        s = get_settings()
        s.fastf1_cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(s.fastf1_cache_path))
        _cache_aktif = True
        logger.info("Cache FastF1 aktif di %s", s.fastf1_cache_path)


def muat_sesi_race(tahun: int, ronde: int) -> Session:
    """Muat sesi balapan utama (R) untuk tahun & round tertentu.

    Data laps + weather diunduh dari server F1 pada pemanggilan pertama,
    selanjutnya dibaca dari cache disk.
    """
    aktifkan_cache()
    sesi = fastf1.get_session(tahun, ronde, "R")
    sesi.load(laps=True, weather=True, telemetry=False, messages=False)
    return sesi
