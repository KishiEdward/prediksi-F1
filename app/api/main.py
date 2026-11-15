"""Aplikasi FastAPI: route health, meta, prediksi replay, SSE, dan info model.

Jalankan dari root proyek:
    python -m app.main          (atau: uvicorn app.api.main:app --reload)
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.logging_setup import setup_logging
from app.schemas.prediksi import (ModelInfo, RaceMeta, ResponsError, ResponsLiveBelumTersedia,
                                  ResponsReplay, TeamMeta)
from app.services.predictor import ModelBelumSiapError, Predictor
from app.services.providers import RaceStateProvider, ReplayProvider

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Startup: muat provider & predictor SEKALI
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    app.state.provider = ReplayProvider()
    try:
        app.state.predictor = Predictor()
    except ModelBelumSiapError as exc:
        app.state.predictor = None
        logger.warning("Model belum siap (%s). Endpoint meta tetap aktif; "
                       "endpoint prediksi akan menjawab 503.", exc)
    yield


app = FastAPI(
    title="F1 Pit Strategy Prediction API",
    description=("API prediksi (perkiraan) keputusan pit stop dan pemilihan ban "
                 "Formula One berbasis arsitektur hybrid XGBoost-LSTM. Keluaran "
                 "bersifat observasional — pola keputusan tim pada data 2021-2025 — "
                 "BUKAN rekomendasi strategi optimal."),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_provider(request: Request) -> RaceStateProvider:
    return request.app.state.provider


def get_predictor(request: Request) -> Predictor:
    pred = request.app.state.predictor
    if pred is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum dilatih / artifact belum lengkap. Jalankan "
                   "`python -m app.ml.features` lalu `python -m app.ml.train`.")
    return pred


def _normalisasi_nama_team(team: str) -> str:
    return team.strip().lower().replace("_", " ")


def _cari_team_id(provider: RaceStateProvider, year: int, team: str) -> tuple[str, str]:
    """Cocokkan input pengguna (nama/id tim) -> (team_id, team_name)."""
    q = _normalisasi_nama_team(team)
    for t in provider.list_teams(year):
        if _normalisasi_nama_team(t["team_name"]) == q or t["team_id"].lower() == team.strip().lower():
            return t["team_id"], t["team_name"]
    raise HTTPException(
        status_code=404,
        detail=f"Tim '{team}' tidak ditemukan pada musim {year}. "
               f"Lihat /meta/teams?year={year} untuk daftar tim.")


def _race_atau_404(provider: RaceStateProvider, year: int, round_: int) -> pd.Series:
    try:
        races = provider.list_races(year)
    except Exception:
        races = []
    for r in races:
        if int(r["round"]) == round_:
            return r
    raise HTTPException(status_code=404,
                        detail=f"Race tidak ditemukan: tahun {year} round {round_}.")


# ---------------------------------------------------------------------------
# Endpoint dasar
# ---------------------------------------------------------------------------
@app.get("/health", tags=["sistem"])
def health(request: Request) -> dict:
    """Cek API hidup + apakah model sudah dimuat."""
    return {"status": "ok",
            "model_loaded": request.app.state.predictor is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["sistem"])
def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfo:
    return ModelInfo(**predictor.info())


# ---------------------------------------------------------------------------
# Endpoint metadata
# ---------------------------------------------------------------------------
@app.get("/meta/seasons", tags=["meta"])
def meta_seasons(provider: RaceStateProvider = Depends(get_provider)) -> dict:
    return {"seasons": provider.list_seasons()}


@app.get("/meta/races", response_model=list[RaceMeta], tags=["meta"])
def meta_races(year: int = Query(..., ge=2021, le=2025),
               provider: RaceStateProvider = Depends(get_provider)) -> list[RaceMeta]:
    hasil = []
    for r in provider.list_races(year):
        sirkuit = r.get("Circuit", {}) if isinstance(r.get("Circuit"), dict) else {}
        hasil.append(RaceMeta(
            year=int(r["season"]), round=int(r["round"]),
            race_name=str(r.get("raceName", "")),
            circuit_id=str(sirkuit.get("circuitId", "")),
            circuit_name=str(sirkuit.get("circuitName", "")),
            date=str(r.get("date", ""))[:10],
            total_laps=None,
        ))
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Tidak ada race untuk tahun {year}.")
    return hasil


@app.get("/meta/teams", response_model=list[TeamMeta], tags=["meta"])
def meta_teams(year: int = Query(..., ge=2021, le=2025),
               provider: RaceStateProvider = Depends(get_provider)) -> list[TeamMeta]:
    return [TeamMeta(**t) for t in provider.list_teams(year)]


# ---------------------------------------------------------------------------
# Endpoint prediksi
# ---------------------------------------------------------------------------
@app.get("/predict/replay", response_model=ResponsReplay,
         responses={503: {"model": ResponsError}}, tags=["prediksi"])
def predict_replay(
    year: int = Query(..., ge=2021, le=2025, description="Musim"),
    round: int = Query(..., ge=1, le=25, description="Nomor round balapan"),
    team: str = Query(..., min_length=2, description="Nama/id tim, contoh: Red Bull"),
    lap: int = Query(..., ge=1, le=100, description="Nomor lap saat prediksi dibuat"),
    provider: RaceStateProvider = Depends(get_provider),
    predictor: Predictor = Depends(get_predictor),
) -> ResponsReplay:
    """Prediksi untuk 2 pembalap satu tim pada lap tertentu (mode replay)."""
    _race_atau_404(provider, year, round)
    team_id, team_name = _cari_team_id(provider, year, team)
    try:
        state = provider.get_race_state(year, round, lap, team_id=team_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'"))
    if not state.drivers:
        raise HTTPException(status_code=404,
                            detail=f"Tidak ada data pembalap {team_name} pada lap {lap}.")

    prediksi = predictor.prediksi_tim(provider, state)
    if not prediksi:
        raise HTTPException(status_code=404,
                            detail="Riwayat lap pembalap tidak tersedia pada titik ini.")
    return ResponsReplay(
        year=year, round=round, lap=lap, team=team_name,
        race_name=state.race_name, track_status=state.track_status,
        drivers=prediksi, model_version=predictor.metadata.get("model_version", "0.1.0"),
    )


@app.get("/predict/replay/stream", tags=["prediksi"])
async def predict_replay_stream(
    request: Request,
    year: int = Query(..., ge=2021, le=2025),
    round: int = Query(..., ge=1, le=25),
    team: str = Query(..., min_length=2),
    provider: RaceStateProvider = Depends(get_provider),
    predictor: Predictor = Depends(get_predictor),
) -> EventSourceResponse:
    """SSE: kirim satu event prediksi per lap (untuk animasi replay di website)."""
    _race_atau_404(provider, year, round)
    team_id, team_name = _cari_team_id(provider, year, team)

    # Total lap race ini diambil dari data
    race_df = provider.laps[(provider.laps["season"] == year)
                            & (provider.laps["round"] == round)] if hasattr(provider, "laps") \
        else None
    total_laps = int(race_df["lap_number"].max()) if race_df is not None and len(race_df) else 60

    async def generator() -> AsyncIterator[dict]:
        for lap in range(1, total_laps + 1):
            if await request.is_disconnected():
                break
            try:
                state = provider.get_race_state(year, round, lap, team_id=team_id)
            except KeyError:
                continue  # lap tanpa data (mis. red flag) dilewati
            preds = predictor.prediksi_tim(provider, state)
            if not preds:
                continue
            payload = ResponsReplay(
                year=year, round=round, lap=lap, team=team_name,
                race_name=state.race_name, track_status=state.track_status,
                drivers=preds,
                model_version=predictor.metadata.get("model_version", "0.1.0"),
            )
            yield {"event": "lap", "data": payload.model_dump_json()}
        yield {"event": "end", "data": json.dumps({"total_laps": total_laps})}

    return EventSourceResponse(generator())


@app.get("/predict/live", status_code=501,
         responses={501: {"model": ResponsLiveBelumTersedia}}, tags=["prediksi"])
def predict_live(team: str = Query(..., min_length=2)) -> JSONResponse:
    """TAHAP 2 (belum diimplementasikan): prediksi dari live timing."""
    return JSONResponse(
        status_code=501,
        content=ResponsLiveBelumTersedia().model_dump(),
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})
