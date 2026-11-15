"""Skema Pydantic v2 untuk request/response API.

Catatan framing: seluruh keluaran adalah PREDIKSI/PERKIRAAN keputusan pit
observasional (pola keputusan tim pada data historis), BUKAN rekomendasi
strategi optimal.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompoundBerikutnya(BaseModel):
    """Prediksi compound pada pit stop berikutnya + probabilitas tiap kelas."""
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Compound dengan probabilitas tertinggi")
    probabilities: dict[str, float] = Field(
        description="Probabilitas tiap compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET)")


class PrediksiPembalap(BaseModel):
    """Prediksi untuk satu pembalap pada lap tertentu."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Kode 3 huruf pembalap, contoh VER")
    driver_id: str
    current_compound: str | None = Field(description="Compound yang sedang dipakai")
    tyre_life: int | None = Field(description="Umur ban saat ini (lap)")
    predicted_laps_to_pit: int = Field(
        ge=0, description="Perkiraan sisa lap menuju pit stop berikutnya")
    predicted_pit_lap: int | None = Field(
        description="Perkiraan nomor lap pit stop berikutnya")
    next_compound: CompoundBerikutnya


class ResponsReplay(BaseModel):
    """Response endpoint /predict/replay."""
    model_config = ConfigDict(extra="forbid")

    year: int
    round: int
    lap: int
    team: str
    race_name: str
    track_status: str = Field(description="Status lintasan pada lap tersebut (GREEN/SC/VSC/RED)")
    drivers: list[PrediksiPembalap]
    note: str = ("Prediksi observasional berdasarkan pola keputusan tim "
                 "pada data 2021-2025, bukan strategi optimal.")
    model_version: str


class RaceMeta(BaseModel):
    """Metadata satu seri balapan."""
    model_config = ConfigDict(extra="forbid")

    year: int
    round: int
    race_name: str
    circuit_id: str
    circuit_name: str
    date: str
    total_laps: int | None = None


class DriverMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    driver_id: str
    name: str


class TeamMeta(BaseModel):
    """Tim beserta dua pembalapnya pada musim tersebut."""
    model_config = ConfigDict(extra="forbid")

    team_id: str
    team_name: str
    drivers: list[DriverMeta] = Field(min_length=0, max_length=2)


class ModelInfo(BaseModel):
    """Informasi model yang sedang dimuat API."""
    model_config = ConfigDict(extra="forbid")

    model_version: str
    trained_at: str | None = None
    metrics: dict | None = Field(default=None, description="Ringkasan metrik evaluasi")
    window_lap: int
    compounds: list[str]
    split: dict


class ResponsError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str


class ResponsLiveBelumTersedia(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str = ("Prediksi live belum tersedia (Tahap 2). Saat ini API hanya "
                   "mendukung mode replay balapan historis 2021-2025.")
