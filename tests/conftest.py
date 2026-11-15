"""Fixture bersama untuk test: provider & predictor palsu (tanpa data/API nyata)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.ml.features import FITUR_INRACE_NUMERIK, FITUR_PRA_NUMERIK
from app.schemas.prediksi import CompoundBerikutnya, PrediksiPembalap
from app.services.providers import DriverLapState, RaceState, RaceStateProvider

KOLOM_FITUR = FITUR_PRA_NUMERIK + FITUR_INRACE_NUMERIK


class FakeProvider(RaceStateProvider):
    """Provider palsu: satu race, satu tim (2 pembalap), beberapa lap sintetis."""

    def __init__(self) -> None:
        self.year, self.round = 2024, 5
        self.team_id, self.team_name = "red_bull", "Red Bull"
        self.total_laps = 57
        self.drivers = [
            DriverLapState("VER", "max_verstappen", self.team_id, 18, "MEDIUM", 14),
            DriverLapState("PER", "sergio_perez", self.team_id, 18, "MEDIUM", 14),
        ]
        # Riwayat lap sintetis 1..total untuk 2 pembalap
        baris = []
        for lap in range(1, self.total_laps + 1):
            for d in self.drivers:
                row = {c: 0.0 for c in KOLOM_FITUR}
                row.update({
                    "season": self.year, "round": self.round,
                    "driver_code": d.driver_code, "driver_id": d.driver_id,
                    "team_id": self.team_id, "lap_number": lap,
                    "current_compound": "MEDIUM", "tyre_life": float(lap),
                    "total_laps": self.total_laps, "target_laps_remaining": float(lap % 7),
                    "target_next_compound": "HARD",
                })
                baris.append(row)
        self.laps = pd.DataFrame(baris)

    def list_seasons(self) -> list[int]:
        return [2024]

    def list_races(self, year: int) -> list[pd.Series]:
        if year != self.year:
            return []
        return [pd.Series({
            "season": self.year, "round": self.round, "raceName": "Chinese Grand Prix",
            "date": "2024-04-21",
            "Circuit": {"circuitId": "shanghai", "circuitName": "Shanghai International Circuit"},
        })]

    def list_teams(self, year: int) -> list[dict]:
        return [{
            "team_id": self.team_id, "team_name": self.team_name,
            "drivers": [
                {"code": "VER", "driver_id": "max_verstappen", "name": "Max Verstappen"},
                {"code": "PER", "driver_id": "sergio_perez", "name": "Sergio Perez"},
            ],
        }]

    def get_race_state(self, year: int, round_: int, lap: int,
                       team_id: str | None = None) -> RaceState:
        if (year, round_) != (self.year, self.round) or not 1 <= lap <= self.total_laps:
            raise KeyError(f"Race/lap tidak ditemukan: {year}/{round_} lap {lap}")
        drivers = [DriverLapState(d.driver_code, d.driver_id, d.team_id, lap,
                                  "MEDIUM", lap) for d in self.drivers]
        if team_id is not None and team_id != self.team_id:
            drivers = []
        return RaceState(year, round_, "Chinese Grand Prix", "shanghai",
                         self.total_laps, lap, "GREEN", drivers)

    def get_driver_history(self, year: int, round_: int, driver_code: str,
                           until_lap: int) -> pd.DataFrame:
        return self.laps[(self.laps["driver_code"] == driver_code)
                         & (self.laps["lap_number"] <= until_lap)].copy()


class FakePredictor:
    """Predictor palsu: mengembalikan angka tetap tanpa model."""

    def __init__(self) -> None:
        self.metadata = {"model_version": "0.0.0-test"}

    def info(self) -> dict:
        return {"model_version": "0.0.0-test", "trained_at": None, "metrics": None,
                "window_lap": 5, "compounds": ["SOFT", "MEDIUM", "HARD",
                                               "INTERMEDIATE", "WET"],
                "split": {"train": "2021-2023", "val": "2024", "test": "2025"}}

    def prediksi_pembalap(self, provider, state, driver_code):
        return PrediksiPembalap(
            code=driver_code, driver_id=driver_code.lower(),
            current_compound="MEDIUM", tyre_life=state.lap,
            predicted_laps_to_pit=6, predicted_pit_lap=state.lap + 6,
            next_compound=CompoundBerikutnya(
                label="HARD",
                probabilities={"SOFT": 0.05, "MEDIUM": 0.2, "HARD": 0.7,
                               "INTERMEDIATE": 0.03, "WET": 0.02}))

    def prediksi_tim(self, provider, state):
        return [self.prediksi_pembalap(provider, state, d.driver_code)
                for d in sorted(state.drivers, key=lambda x: x.driver_code)]


@pytest.fixture
def client():
    """TestClient dengan provider & predictor palsu (tanpa menjalankan lifespan)."""
    from fastapi.testclient import TestClient

    from app.api.main import app
    app.state.provider = FakeProvider()
    app.state.predictor = FakePredictor()
    return TestClient(app)
