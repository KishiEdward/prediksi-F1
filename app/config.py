"""Konfigurasi terpusat aplikasi.

Semua pengaturan dibaca dari environment / file .env lewat pydantic-settings,
sehingga tidak ada path atau secret yang di-hardcode di kode.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Root proyek = folder yang memuat package `app`
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Pengaturan aplikasi yang bisa diubah lewat file .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="F1_",
        extra="ignore",
    )

    # Domain yang diizinkan mengakses API (CORS)
    api_cors_origins: list[str] = ["*"]
    api_port: int = 8000

    # Jolpica F1 API
    jolpica_base_url: str = "https://api.jolpi.ca/ergast/f1/"
    jolpica_min_interval: float = 0.5

    # Rentang musim data penelitian
    season_start: int = 2021
    season_end: int = 2025

    # Cache FastF1 (relatif terhadap root proyek)
    fastf1_cache_dir: str = "data/raw/fastf1_cache"

    # ----- Path turunan (tidak perlu diset manual) -----
    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def fastf1_cache_path(self) -> Path:
        p = Path(self.fastf1_cache_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def jolpica_cache_dir(self) -> Path:
        return self.raw_dir / "jolpica_cache"

    @property
    def seasons(self) -> list[int]:
        """Daftar musim dari season_start sampai season_end."""
        return list(range(self.season_start, self.season_end + 1))


@lru_cache
def get_settings() -> Settings:
    """Ambil satu instance Settings (di-cache agar .env hanya dibaca sekali)."""
    return Settings()
