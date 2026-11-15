"""Test endpoint API dengan provider & predictor palsu (tanpa data nyata)."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_meta_seasons(client):
    r = client.get("/meta/seasons")
    assert r.status_code == 200
    assert r.json() == {"seasons": [2024]}


def test_meta_races(client):
    r = client.get("/meta/races", params={"year": 2024})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["race_name"] == "Chinese Grand Prix"
    assert data[0]["circuit_id"] == "shanghai"


def test_meta_races_tahun_kosong_404(client):
    r = client.get("/meta/races", params={"year": 2021})
    assert r.status_code == 404


def test_meta_teams_dua_pembalap(client):
    r = client.get("/meta/teams", params={"year": 2024})
    assert r.status_code == 200
    teams = r.json()
    assert len(teams) == 1
    assert teams[0]["team_id"] == "red_bull"
    # aturan bisnis: tepat 2 pembalap per tim
    assert len(teams[0]["drivers"]) == 2
    assert {d["code"] for d in teams[0]["drivers"]} == {"VER", "PER"}


def test_predict_replay_sukses(client):
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 5, "team": "Red Bull", "lap": 18})
    assert r.status_code == 200
    data = r.json()
    assert data["year"] == 2024 and data["round"] == 5 and data["lap"] == 18
    assert data["team"] == "Red Bull"
    assert data["track_status"] == "GREEN"
    assert len(data["drivers"]) == 2
    ver = next(d for d in data["drivers"] if d["code"] == "VER")
    assert ver["current_compound"] == "MEDIUM"
    assert ver["predicted_laps_to_pit"] == 6
    assert ver["predicted_pit_lap"] == 24
    assert ver["next_compound"]["label"] == "HARD"
    assert abs(sum(ver["next_compound"]["probabilities"].values()) - 1.0) < 1e-6
    # framing: harus menyebut prediksi observasional (bukan klaim optimal)
    assert "observasional" in data["note"].lower()


def test_predict_replay_tim_nama_case_insensitive(client):
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 5, "team": "red bull", "lap": 10})
    assert r.status_code == 200


def test_predict_replay_tim_dipakai_juga_id(client):
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 5, "team": "red_bull", "lap": 10})
    assert r.status_code == 200


def test_predict_replay_tim_tidak_dikenal_404(client):
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 5, "team": "Ferrari", "lap": 10})
    assert r.status_code == 404
    assert "tidak ditemukan" in r.json()["detail"].lower()


def test_predict_replay_race_tidak_ada_404(client):
    # round 25 valid secara parameter tapi tidak ada di provider
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 25, "team": "Red Bull", "lap": 10})
    assert r.status_code == 404


def test_predict_replay_lap_lebih_dari_race_404(client):
    # lap 99 lolos validasi Pydantic (<=100) tapi melebihi total lap race (57)
    r = client.get("/predict/replay",
                   params={"year": 2024, "round": 5, "team": "Red Bull", "lap": 99})
    assert r.status_code == 404


def test_predict_replay_validasi_param(client):
    # year di luar rentang -> 422 dari Pydantic
    r = client.get("/predict/replay",
                   params={"year": 1999, "round": 5, "team": "Red Bull", "lap": 10})
    assert r.status_code == 422


def test_predict_live_501(client):
    r = client.get("/predict/live", params={"team": "Red Bull"})
    assert r.status_code == 501
    assert "tahap 2" in r.json()["detail"].lower()


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"] == "0.0.0-test"
    assert data["window_lap"] == 5
    assert data["compounds"] == ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def test_predictor_belum_dimuat_503():
    """Bila artifact model belum ada, endpoint prediksi menjawab 503 (bukan crash)."""
    from fastapi.testclient import TestClient

    from app.api.main import app
    from tests.conftest import FakeProvider
    app.state.provider = FakeProvider()
    app.state.predictor = None
    c = TestClient(app)
    r = c.get("/predict/replay",
              params={"year": 2024, "round": 5, "team": "Red Bull", "lap": 18})
    assert r.status_code == 503
    assert "python -m app.ml.train" in r.json()["detail"]
    r2 = c.get("/model/info")
    assert r2.status_code == 503
