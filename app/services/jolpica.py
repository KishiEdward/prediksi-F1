"""Client untuk Jolpica F1 API (pengganti Ergast).

Fitur:
- Retry dengan exponential backoff saat gagal (jaringan / 429 / 5xx).
- Jeda minimum antar request agar sopan terhadap rate limit publik.
- Cache respons JSON ke disk sehingga pipeline idempotent:
  menjalankan ulang tidak mengunduh data yang sama dua kali.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAKS_PERCOBAAN = 5
_BACKOFF_AWAL = 2.0  # detik; dilipatgandakan tiap gagal


class JolpicaClient:
    """Client HTTP kecil dengan retry + cache disk untuk Jolpica F1 API."""

    def __init__(self, base_url: str | None = None, cache_dir: Path | None = None,
                 min_interval: float | None = None) -> None:
        s = get_settings()
        self.base_url = (base_url or s.jolpica_base_url).rstrip("/")
        self.cache_dir = cache_dir or s.jolpica_cache_dir
        self.min_interval = min_interval if min_interval is not None else s.jolpica_min_interval
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._waktu_request_terakhir: float = 0.0

    def _path_cache(self, url: str) -> Path:
        """Nama file cache = hash dari URL lengkap (aman untuk semua karakter)."""
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")

    def _tunggu_jeda(self) -> None:
        """Pastikan ada jeda minimum antar request (hormati rate limit)."""
        selisih = time.monotonic() - self._waktu_request_terakhir
        if selisih < self.min_interval:
            time.sleep(self.min_interval - selisih)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ambil satu endpoint JSON. path contoh: '2024/results.json'.

        Mengembalikan dict hasil parse JSON. Respons di-cache ke disk;
        jika cache ada, tidak ada request jaringan yang dikirim.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            # params di-encode ke query string agar key cache konsisten
            url = str(httpx.URL(url, params=params))
        else:
            url = str(httpx.URL(url))

        file_cache = self._path_cache(url)
        if file_cache.exists():
            return json.loads(file_cache.read_text(encoding="utf-8"))

        backoff = _BACKOFF_AWAL
        for percobaan in range(1, _MAKS_PERCOBAAN + 1):
            self._tunggu_jeda()
            try:
                self._waktu_request_terakhir = time.monotonic()
                resp = httpx.get(url, timeout=30.0)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"status {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                data = resp.json()
                file_cache.write_text(json.dumps(data), encoding="utf-8")
                return data
            except httpx.HTTPStatusError as exc:
                # 4xx (selain 429) = kesalahan permanen, tidak perlu di-retry
                if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    logger.error("Permintaan ditolak permanen (%d): %s",
                                 exc.response.status_code, url)
                    raise
                if percobaan == _MAKS_PERCOBAAN:
                    raise
                logger.warning("Percobaan %d/%d gagal untuk %s (%s). Tunggu %.1f detik...",
                               percobaan, _MAKS_PERCOBAAN, url, exc, backoff)
                time.sleep(backoff)
                backoff *= 2
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                if percobaan == _MAKS_PERCOBAAN:
                    logger.error("Gagal mengambil %s setelah %d percobaan: %s",
                                 url, _MAKS_PERCOBAAN, exc)
                    raise
                logger.warning("Percobaan %d/%d gagal untuk %s (%s). Tunggu %.1f detik...",
                               percobaan, _MAKS_PERCOBAAN, url, exc, backoff)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("tidak seharusnya sampai sini")  # penenang type checker

    def get_semua_halaman(self, path: str, kunci_list: list[str],
                          limit: int = 100) -> list[dict[str, Any]]:
        """Ambil semua halaman (pagination offset/limit) dari satu endpoint.

        `kunci_list` adalah jalur key menuju array hasil,
        contoh ["RaceTable", "Races"].
        """
        semua: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self.get_json(path, params={"limit": limit, "offset": offset})
            node: Any = data.get("MRData", data)
            total = int(node.get("total", 0))
            for k in kunci_list:
                node = node.get(k, {})
            if isinstance(node, dict):
                # beberapa endpoint memakai key jamak, mis. "Races"/"Results"
                nilai = next((v for v in node.values() if isinstance(v, list)), [])
            else:
                nilai = node or []
            semua.extend(nilai)
            offset += limit
            if offset >= total or not nilai:
                break
        return semua
