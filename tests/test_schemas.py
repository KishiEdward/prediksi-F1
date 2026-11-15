"""Test validasi skema Pydantic v2."""
import pytest
from pydantic import ValidationError

from app.schemas.prediksi import (CompoundBerikutnya, PrediksiPembalap,
                                  ResponsLiveBelumTersedia, ResponsReplay)


def _driver(**kw) -> PrediksiPembalap:
    base = dict(
        code="VER", driver_id="max_verstappen", current_compound="MEDIUM",
        tyre_life=14, predicted_laps_to_pit=6, predicted_pit_lap=24,
        next_compound=CompoundBerikutnya(
            label="HARD",
            probabilities={"SOFT": 0.05, "MEDIUM": 0.2, "HARD": 0.7,
                           "INTERMEDIATE": 0.03, "WET": 0.02}))
    base.update(kw)
    return PrediksiPembalap(**base)


def test_prediksi_pembalap_valid():
    d = _driver()
    assert d.code == "VER"
    assert d.predicted_pit_lap == 24


def test_prediksi_pembalap_tolak_field_tambahan():
    with pytest.raises(ValidationError):
        _driver_str = dict(
            code="VER", driver_id="x", current_compound="MEDIUM", tyre_life=1,
            predicted_laps_to_pit=1, predicted_pit_lap=2,
            next_compound=CompoundBerikutnya(
                label="HARD", probabilities={"HARD": 1.0}),
            field_liar="tidak boleh ada")
        PrediksiPembalap(**_driver_str)


def test_laps_to_pit_tidak_boleh_negatif():
    with pytest.raises(ValidationError):
        _driver(predicted_laps_to_pit=-1)


def test_respons_replay_lengkap():
    r = ResponsReplay(year=2024, round=5, lap=18, team="Red Bull",
                      race_name="Chinese Grand Prix", track_status="GREEN",
                      drivers=[_driver(), _driver(code="PER")],
                      model_version="0.1.0")
    assert len(r.drivers) == 2
    assert "observasional" in r.note


def test_live_501_pesan_jelas():
    m = ResponsLiveBelumTersedia()
    assert "Tahap 2" in m.detail
    assert "replay" in m.detail
