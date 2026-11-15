"""Test pemilihan 2 pembalap per tim pada ReplayProvider (tanpa file data nyata)."""
import pandas as pd
import pytest

from app.services.providers import ReplayProvider


def _bikin_provider() -> ReplayProvider:
    """ReplayProvider tanpa membaca disk: isi DataFrame kecil secara manual."""
    p = object.__new__(ReplayProvider)
    p.races = pd.DataFrame([{
        "season": 2024, "round": 5, "raceName": "Chinese Grand Prix",
        "date": "2024-04-21",
        "Circuit": {"circuitId": "shanghai", "circuitName": "Shanghai"},
    }])
    # Tim red_bull: VER 5 start, PER 5 start, ZZZ cuma 1 start (pembalap pengganti)
    baris = []
    for rnd in range(1, 6):
        baris += [
            {"season": 2024, "round": rnd, "Constructor.constructorId": "red_bull",
             "Driver.driverId": "max_verstappen", "Driver.code": "VER",
             "Driver.givenName": "Max", "Driver.familyName": "Verstappen"},
            {"season": 2024, "round": rnd, "Constructor.constructorId": "red_bull",
             "Driver.driverId": "sergio_perez", "Driver.code": "PER",
             "Driver.givenName": "Sergio", "Driver.familyName": "Perez"},
        ]
    baris.append({"season": 2024, "round": 6, "Constructor.constructorId": "red_bull",
                  "Driver.driverId": "pengganti", "Driver.code": "ZZZ",
                  "Driver.givenName": "Pembalap", "Driver.familyName": "Ketiga"})
    p.results = pd.DataFrame(baris)
    p.constructors = pd.DataFrame([{"constructorId": "red_bull", "name": "Red Bull"}])

    kolom_laps = ["season", "round", "driver_code", "driver_id", "team_id",
                  "lap_number", "total_laps", "track_status", "current_compound",
                  "tyre_life"]
    baris_laps = []
    for lap in (17, 18):
        for kode, did in [("VER", "max_verstappen"), ("PER", "sergio_perez")]:
            baris_laps.append({
                "season": 2024, "round": 5, "driver_code": kode, "driver_id": did,
                "team_id": "red_bull", "lap_number": lap, "total_laps": 57,
                "track_status": "GREEN", "current_compound": "MEDIUM", "tyre_life": lap,
            })
    p.laps = pd.DataFrame(baris_laps, columns=kolom_laps)
    return p


def test_list_teams_memilih_dua_pembalap_teratas():
    p = _bikin_provider()
    teams = p.list_teams(2024)
    assert len(teams) == 1
    t = teams[0]
    assert t["team_id"] == "red_bull" and t["team_name"] == "Red Bull"
    # tepat 2 pembalap, yaitu yang paling sering tampil (bukan si pengganti 1 start)
    assert len(t["drivers"]) == 2
    assert {d["driver_id"] for d in t["drivers"]} == {"max_verstappen", "sergio_perez"}


def test_get_race_state_filter_tim():
    p = _bikin_provider()
    state = p.get_race_state(2024, 5, 18, team_id="red_bull")
    assert len(state.drivers) == 2
    assert state.track_status == "GREEN"
    assert state.total_laps == 57
    # tim lain -> tidak ada pembalap
    state2 = p.get_race_state(2024, 5, 18, team_id="ferrari")
    assert state2.drivers == []


def test_get_race_state_race_tidak_ada():
    p = _bikin_provider()
    with pytest.raises(KeyError):
        p.get_race_state(2024, 99, 10)


def test_get_driver_history_hanya_sampai_lap_diminta():
    p = _bikin_provider()
    h = p.get_driver_history(2024, 5, "VER", until_lap=17)
    assert list(h["lap_number"]) == [17]  # lap 18 TIDAK boleh ikut (anti-leakage)
    h2 = p.get_driver_history(2024, 5, "VER", until_lap=18)
    assert list(h2["lap_number"]) == [17, 18]
