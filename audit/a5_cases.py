# -*- coding: utf-8 -*-
"""AUDIT 5 — Analisis kasus: 5 peristiwa pit acak (test) + 3 galat terbesar.

READ-ONLY inference. Test 2025 HANYA untuk deskripsi kasus (bukan dasar
rekomendasi). Seed acak dicantumkan. Per kasus: prediksi per lap menjelang pit,
jendela conformal, compound prediksi vs aktual, selisih, penjelasan teknis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import muat_dataset, muat_inference, muat_model_run3, prediksi_run3  # noqa: E402
from app.ml.features import COMPOUNDS  # noqa: E402

SEED = 20261115
pd.set_option("display.width", 220)


def q_window(cur_lap: int, pred_laps: float, q: float):
    qi = int(round(q))
    lo = cur_lap + max(0, int(round(pred_laps)) - qi)
    hi = cur_lap + int(round(pred_laps)) + qi
    return lo, hi


def cetak_kasus(judul, inf_race, reg, clf, pre, binary, pre_bin, q,
                s, r, d, p_lap):
    print(f"\n{'='*74}\n{judul}\n{'='*74}")
    g = inf_race[(inf_race.season == s) & (inf_race["round"] == r) &
                 (inf_race.driver_code == d)].sort_values("lap_number")
    if g.empty:
        print("  (data tidak ditemukan)"); return
    race_name = g.iloc[0].get("race_name", "?")
    print(f"  {race_name} {s} round {r} | driver {d} | pit aktual di lap {p_lap}")
    # prediksi pada lap mendekati pit
    lo_lap = max(1, p_lap - 6)
    sub = g[(g.lap_number >= lo_lap) & (g.lap_number <= p_lap)]
    if sub.empty:
        print("  (tidak ada lap mendekati pit di dataset inference)"); return
    pl, proba, pc, pb = prediksi_run3(sub.reset_index(drop=True), reg, clf, pre, binary, pre_bin)
    print(f"  {'lap':>4} {'cur_comp':>9} {'tyre':>5} {'pred_laps':>9} {'pred_pit':>8} "
          f"{'window':>12} {'pred_comp':>12} {'P(masih)':>8} {'trackstat':>9}")
    for i, (_, row) in enumerate(sub.iterrows()):
        cur = int(row.lap_number)
        w = q_window(cur, pl[i], q)
        cc = str(row.get("current_compound"))[:9]
        tl = row.get("tyre_life")
        tls = f"{int(tl)}" if pd.notna(tl) else "-"
        pm = f"{pb[i]:.3f}" if pb is not None else "-"
        print(f"  {cur:>4} {cc:>9} {tls:>5} {pl[i]:>9.2f} {cur+round(pl[i]):>8} "
              f"[{w[0]},{w[1]}]".ljust(62) +
              f" {COMPOUNDS[pc[i]]:>12} {pm:>8} {str(row.get('track_status')):>9}")
    # ringkasan galat pada lap terdekat sebelum pit
    near = sub[sub.lap_number == p_lap - 1]
    if not near.empty:
        idx = list(sub.lap_number).index(p_lap - 1)
        pred_pit = (p_lap - 1) + round(pl[idx])
        print(f"  -> Pada lap {p_lap-1}: pred pit lap={pred_pit} (aktual {p_lap}), "
              f"galat={pred_pit-p_lap:+d} lap; pred compound={COMPOUNDS[pc[idx]]}")


def main() -> None:
    print("=" * 78)
    print(f"AUDIT 5 — ANALISIS KASUS (test 2025, seed acak={SEED})")
    print("=" * 78)
    dsf = muat_dataset()
    inf = muat_inference()
    reg, clf, binary, pre, pre_bin, conf = muat_model_run3()
    q = float(conf.get("kuantil_abs_resid", 0))
    print(f"q conformal = {q} lap (window = pred +/- {round(q)} lap)")

    test = dsf[dsf["season"] == 2025].reset_index(drop=True)
    # bangun daftar peristiwa pit (driver, race, pit lap) dari target
    test = test.copy()
    test["pit_lap"] = test["lap_number"] + test["target_laps_remaining"]
    events = (test.groupby(["season", "round", "driver_code", "pit_lap"])
              .size().reset_index(name="n"))
    inf2 = inf.copy()
    inf2["pred_err_proxy"] = 0  # placeholder

    # prediksi semua baris test sekali (untuk ranking galat)
    pl_all, proba_all, pc_all, pb_all = prediksi_run3(test, reg, clf, pre, binary, pre_bin)
    test["pred_pit_lap"] = test["lap_number"] + np.round(np.clip(pl_all, 0, None))
    test["abs_err"] = np.abs(test["pred_pit_lap"] - test["pit_lap"])

    # --- 5 peristiwa acak ---
    rng = np.random.default_rng(SEED)
    ev = events[events["n"] >= 3]  # peristiwa dgn cukup lap mendekati
    idx = rng.choice(len(ev), size=min(5, len(ev)), replace=False)
    pilihan = ev.iloc[idx]
    print("\n##### 5 PERISTIWA PIT ACAK (seed=%d) #####" % SEED)
    for _, e in pilihan.iterrows():
        cetak_kasus(f"KASUS ACAK: {int(e['season'])} R{int(e['round'])} "
                    f"{e['driver_code']} pit lap {int(e['pit_lap'])}",
                    inf, reg, clf, pre, binary, pre_bin, q,
                    int(e["season"]), int(e["round"]), e["driver_code"], int(e["pit_lap"]))

    # --- 3 galat terbesar (per peristiwa, pakai median galat stint) ---
    print("\n\n##### 3 PERISTIWA GALAT TERBESAR (terpisah dari sampel acak) #####")
    ev_err = (test.groupby(["season", "round", "driver_code", "pit_lap"])
              .agg(med_err=("abs_err", "median"), max_err=("abs_err", "max"),
                   n=("abs_err", "size")).reset_index())
    ev_err = ev_err[ev_err["n"] >= 3].sort_values("med_err", ascending=False)
    for _, e in ev_err.head(3).iterrows():
        cetak_kasus(f"GALAT BESAR: {int(e['season'])} R{int(e['round'])} "
                    f"{e['driver_code']} pit lap {int(e['pit_lap'])} "
                    f"(median galat {e['med_err']:.1f} lap)",
                    inf, reg, clf, pre, binary, pre_bin, q,
                    int(e["season"]), int(e["round"]), e["driver_code"], int(e["pit_lap"]))

    # konteks: distribusi galat peristiwa di test
    print("\n\n##### KONTEKS GALAT PERISTIWA (test 2025) #####")
    print(f"  median |galat pit lap| per peristiwa: {ev_err['med_err'].median():.2f}")
    print(f"  p90: {ev_err['med_err'].quantile(0.9):.2f}  max: {ev_err['med_err'].max():.2f}")
    print(f"  fraksi peristiwa median galat <=3: {(ev_err['med_err']<=3).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
