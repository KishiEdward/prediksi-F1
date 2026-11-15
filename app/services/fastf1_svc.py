"""Wrapper tipis untuk pustaka FastF1.

Tanggung jawab:
- Mengaktifkan cache disk agar data sesi tidak diunduh dua kali.
- Memuat kredensial akun F1 dari .env (F1_USERNAME / F1_PASSWORD) bila ada,
  agar request terautentikasi (limit lebih longgar, jarang diblokir).
  JANGAN PERNAH menaruh kredensial di kode atau commit .env ke git.
- Menyediakan akses ke sesi Race (balapan utama) per tahun/round.
"""
import logging
import os

import fastf1
from fastf1.core import Session

from app.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

_cache_aktif = False


def _muat_kredensial_f1_dari_env_file() -> None:
    """Baca F1_USERNAME/F1_PASSWORD dari file .env ke environment proses.

    fastf1 membaca kredensial dari environment variable, bukan dari .env,
    jadi kita salin manual (tanpa menimpa variable yang sudah diset).
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for baris in env_file.read_text(encoding="utf-8").splitlines():
        baris = baris.strip()
        if not baris or baris.startswith("#") or "=" not in baris:
            continue
        key, _, value = baris.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key in ("F1_USERNAME", "F1_PASSWORD") and value:
            os.environ.setdefault(key, value)
    if os.environ.get("F1_USERNAME"):
        logger.info("Kredensial akun F1 terdeteksi dari .env (autentikasi aktif)")


def aktifkan_cache() -> None:
    """Aktifkan cache FastF1 + kredensial sekali saja (idempotent)."""
    global _cache_aktif
    if not _cache_aktif:
        s = get_settings()
        s.fastf1_cache_path.mkdir(parents=True, exist_ok=True)
        _muat_kredensial_f1_dari_env_file()
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
