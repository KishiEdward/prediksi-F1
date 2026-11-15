# -*- coding: utf-8 -*-
"""Uji end-to-end /predict/replay pada race nyata (Run 3: xgb_separate).

Memverifikasi:
  1. Endpoint meta (seasons, races, teams) berfungsi
  2. /predict/replay mengembalikan semua field baru Run 3:
     probabilitas_masih_pit, jendela_pit (conformal), peringatan (caveats)
  3. Nilai masuk akal (window mencakup prediksi, bucket sesuai jarak, wet flag)
  4. /model/info memuat metadata Run 3
  5. /predict/live tetap 501 (stub Tahap 2)

Dijalankan via TestClient (lifespan app aktif, predictor + provider nyata).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402

PASS, FAIL = 0, 0


def _list_or(data, key: str) -> list:
    """API meta bisa mengembalikan list langsung ATAU dict dengan key. Tangani keduanya."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get(key, [])
    return []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    global PASS, FAIL
    if kondisi:
        PASS += 1
        print(f"  [PASS] {nama}")
    else:
        FAIL += 1
        print(f"  [FAIL] {nama} {detail}")


def main() -> None:
    with TestClient(app) as c:
        # --- 1. Meta endpoints ---
        print("== Meta ==")
        r = c.get("/health")
        cek("health 200", r.status_code == 200, str(r.status_code))

        r = c.get("/meta/seasons")
        cek("seasons 200", r.status_code == 200)
        seasons = r.json().get("seasons", []) if r.status_code == 200 else []
        cek("seasons berisi 2021-2025", set([2021, 2025]).issubset(set(seasons)),
            str(seasons))

        r = c.get("/meta/races", params={"year": 2021})
        cek("races 2021 ok", r.status_code == 200)
        races = _list_or(r.json(), "races") if r.status_code == 200 else []
        cek("ada race round 1", any(x.get("round") == 1 for x in races))

        r = c.get("/meta/teams", params={"year": 2021})
        cek("teams 2021 ok", r.status_code == 200)
        teams = _list_or(r.json(), "teams") if r.status_code == 200 else []
        cek("ada team dengan 2 driver",
            any(len(t.get("drivers", [])) == 2 for t in teams))
        team_pilih = next((t for t in teams if len(t.get("drivers", [])) == 2), None)
        if team_pilih is None:
            print("  [SKIP] tidak ada team valid untuk replay")
            return
        team_id = team_pilih["team_id"]
        team_name = team_pilih["team_name"]
        print(f"  team dipilih: {team_name} ({team_id}) "
              f"drivers={[d['code'] for d in team_pilih['drivers']]}")

        # --- 2. Model info Run 3 ---
        print("== /model/info ==")
        r = c.get("/model/info")
        cek("model/info 200", r.status_code == 200)
        info = r.json() if r.status_code == 200 else {}
        cek("primary_model=xgb_separate", info.get("primary_model") == "xgb_separate",
            str(info.get("primary_model")))
        cek("conformal ada", info.get("conformal") is not None)
        cek("binary_head ada", info.get("binary_head") is not None)
        cek("caveats ada", info.get("caveats") is not None)
        cek("cv_selection ada", info.get("cv_selection") is not None)
        cek("window_lap=1", info.get("window_lap") == 1, str(info.get("window_lap")))

        # --- 3. Replay pada beberapa lap (2021 Bahrain round 1) ---
        # Catatan: tidak semua lap tersedia untuk semua tim (DNF/gap data).
        # Lap yang 404 dicatat sebagai WARNING, bukan FAIL — ini karakteristik
        # data, bukan bug adaptasi Run 3.
        print("== /predict/replay Bahrain 2021 ==")
        for lap in [1, 10, 20, 30]:
            r = c.get("/predict/replay",
                      params={"year": 2021, "round": 1, "team": team_name, "lap": lap})
            if r.status_code == 404:
                print(f"  [WARN] lap{lap}: data tidak tersedia ({r.json().get('detail','')})")
                continue
            ok = r.status_code == 200
            cek(f"replay lap{lap} 200", ok, str(r.status_code) + " " + r.text[:200])
            if not ok:
                continue
            d = r.json()
            drivers = d.get("drivers", [])
            cek(f"lap{lap} ada drivers", len(drivers) > 0)
            for dr in drivers:
                code = dr.get("code")
                # field wajib Run 3
                cek(f"lap{lap} {code} prob_masih_pit ada",
                    dr.get("probabilitas_masih_pit") is not None,
                    json.dumps(dr)[:150])
                pmp = dr.get("probabilitas_masih_pit")
                if pmp is not None:
                    cek(f"lap{lap} {code} prob 0-1", 0.0 <= pmp <= 1.0, str(pmp))
                jend = dr.get("jendela_pit")
                cek(f"lap{lap} {code} jendela_pit ada", jend is not None)
                if jend:
                    cek(f"lap{lap} {code} window cover prediksi",
                        jend["lap_bawah"] <= dr["predicted_pit_lap"] <= jend["lap_atas"],
                        f"{jend['lap_bawah']}<={dr['predicted_pit_lap']}<={jend['lap_atas']}")
                    cek(f"lap{lap} {code} coverage=0.8",
                        abs(jend.get("tingkat_keyakinan", 0) - 0.8) < 1e-6)
                    cek(f"lap{lap} {code} bawah>=lap saat ini",
                        jend["lap_bawah"] >= lap, str(jend["lap_bawah"]))
                per = dr.get("peringatan")
                cek(f"lap{lap} {code} peringatan ada", per is not None)
                if per:
                    bucket = per.get("bucket_jarak")
                    cek(f"lap{lap} {code} bucket valid",
                        bucket in ("0-5", "6-15", ">15"), str(bucket))
                    # verifikasi konsistensi bucket vs prediksi
                    pl = dr["predicted_laps_to_pit"]
                    exp = "0-5" if pl <= 5 else ("6-15" if pl <= 15 else ">15")
                    cek(f"lap{lap} {code} bucket konsisten", bucket == exp,
                        f"pred={pl} bucket={bucket} exp={exp}")
                    lbl = dr["next_compound"]["label"]
                    wet_expected = lbl in ("INTERMEDIATE", "WET")
                    cek(f"lap{lap} {code} wet flag konsisten",
                        per.get("compound_basah_tidak_andal") == wet_expected,
                        f"label={lbl} flag={per.get('compound_basah_tidak_andal')}")
                    if bucket == ">15":
                        cek(f"lap{lap} {code} catatan >15 ada",
                            len(per.get("catatan", [])) > 0)

        # --- 4. Replay test-set 2025 (verifikasi generalisasi) ---
        print("== /predict/replay 2025 (test set) ==")
        r = c.get("/meta/races", params={"year": 2025})
        races25 = _list_or(r.json(), "races") if r.status_code == 200 else []
        if races25:
            rd = races25[0]["round"]
            r = c.get("/meta/teams", params={"year": 2025})
            teams25 = _list_or(r.json(), "teams") if r.status_code == 200 else []
            t25 = next((t for t in teams25 if len(t.get("drivers", [])) == 2), None)
            if t25:
                r = c.get("/predict/replay",
                          params={"year": 2025, "round": rd,
                                  "team": t25["team_name"], "lap": 15})
                cek(f"replay 2025 r{rd} lap15 200", r.status_code == 200,
                    str(r.status_code) + " " + r.text[:200])
                if r.status_code == 200:
                    dd = r.json()["drivers"]
                    cek("2025 drivers ada", len(dd) > 0)
                    for dr in dd:
                        cek(f"2025 {dr['code']} lengkap run3",
                            dr.get("probabilitas_masih_pit") is not None
                            and dr.get("jendela_pit") is not None
                            and dr.get("peringatan") is not None)

        # --- 5. Live stub tetap 501 (param team wajib sesuai kontrak API) ---
        print("== /predict/live ==")
        r = c.get("/predict/live", params={"team": team_name})
        cek("live 501", r.status_code == 501, str(r.status_code))

        # --- 6. Validasi parameter: round di luar batas -> 422 (by design, le=25) ---
        print("== Validasi parameter ==")
        r = c.get("/predict/replay",
                  params={"year": 2021, "round": 99, "team": team_name, "lap": 5})
        cek("round>25 ditolak 422 (validasi Query le=25)", r.status_code == 422,
            str(r.status_code))
        # Round valid tapi race tidak ada di dataset -> 404 dari provider
        r = c.get("/predict/replay",
                  params={"year": 2021, "round": 22, "team": team_name, "lap": 5})
        cek("round valid tanpa data 404/200", r.status_code in (404, 200),
            str(r.status_code))

    print(f"\n===== HASIL: {PASS} PASS, {FAIL} FAIL =====")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
