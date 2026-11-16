# -*- coding: utf-8 -*-
"""AUDIT 1 — Ground truth: kualitas label & batas atas akurasi.

READ-ONLY. Menjawab pemeriksaan #1: jumlah pit akibat red flag/SC/VSC/pit
berturut-turut; baris berlabel tidak pasti; estimasi batas atas akurasi akibat
label bising. Semua dari data mentah + dataset olahan yang sudah ada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (muat_dataset, muat_inference, muat_pitstops,  # noqa: E402
                     status_lintasan_per_lap)

pd.set_option("display.width", 200)


def main() -> None:
    print("=" * 78)
    print("AUDIT 1 — GROUND TRUTH & LABEL NOISE")
    print("=" * 78)

    pit = muat_pitstops()
    st = status_lintasan_per_lap()
    inf = muat_inference()
    dsf = muat_dataset()

    # --- 1. Total pit stop & kondisi lintasan saat pit ---
    pit["lap"] = pit["lap"].astype(int)
    key = pit[["season", "round", "lap"]].rename(columns={"lap": "lap"})
    merged = key.merge(st, on=["season", "round", "lap"], how="left")
    for c in ("sc", "vsc", "red", "yellow"):
        merged[c] = merged[c].fillna(False).astype(bool)
    n_pit = len(pit)
    print(f"\n[1] TOTAL PIT STOP (jolpica, semua musim): {n_pit}")
    print(f"    Pit saat SC   : {int(merged['sc'].sum()):5d}  ({merged['sc'].mean()*100:.1f}%)")
    print(f"    Pit saat VSC  : {int(merged['vsc'].sum()):5d}  ({merged['vsc'].mean()*100:.1f}%)")
    print(f"    Pit saat RED  : {int(merged['red'].sum()):5d}  ({merged['red'].mean()*100:.1f}%)")
    print(f"    Pit saat YELLOW: {int(merged['yellow'].sum()):5d}  ({merged['yellow'].mean()*100:.1f}%)")
    scvsc = (merged['sc'] | merged['vsc'])
    print(f"    Pit saat SC/VSC (gabungan): {int(scvsc.sum()):5d}  ({scvsc.mean()*100:.1f}%)")
    print(f"    -> Pit SC/VSC/RED TIDAK TERLIHAT model (is_sc/is_vsc/is_red=0 semua).")

    # --- 2. Red flag: seharusnya dibuang dari target, tapi track_status tak pernah RED ---
    print("\n[2] RED-FLAG PIT EXCLUSION (features.py:416 cek track_status=='RED')")
    print(f"    Pit di lap ber-status RED (sumber mentah): {int(merged['red'].sum())}")
    n_red_in_dataset = int((inf['track_status'] == 'RED').sum())
    print(f"    Baris inference dgn track_status=='RED': {n_red_in_dataset}  (harusnya >0 jika fitur hidup)")
    print(f"    -> Karena track_status tak pernah 'RED', n_red_flag SELALU 0;")
    print(f"       pit red-flag (jika ada) TIDAK dibuang => label bising.")

    # --- 3. Pit berturut-turut (double-stack / within 2 laps) ---
    pit_sorted = pit.sort_values(["season", "round", "driverId", "lap"])
    pit_sorted["prev_lap"] = pit_sorted.groupby(["season", "round", "driverId"])["lap"].shift(1)
    gap = pit_sorted["lap"] - pit_sorted["prev_lap"]
    consec = int(((gap == 1)).sum())
    close = int(((gap <= 2)).sum())
    print(f"\n[3] PIT BERTURUT-TURUT (driver sama):")
    print(f"    Pit dgn jarak tepat 1 lap dari pit sebelumnya: {consec}")
    print(f"    Pit dgn jarak <= 2 lap: {close}")
    print(f"    -> pit sangat berdekatan = kemungkinan double-stack/kesalahan data/SC.")

    # --- 4. Baris berlabel tidak pasti ---
    print("\n[4] BARIS BERLABEL TIDAK PASTI")
    n_inf = len(inf)
    n_has = int(inf['has_target'].sum())
    print(f"    lap_inference total          : {n_inf}")
    print(f"    has_target=True (dipakai latih): {n_has}  ({n_has/n_inf*100:.1f}%)")
    print(f"    has_target=False (stint akhir/DNF/tanpa pit berikut): {n_inf-n_has}  ({(n_inf-n_has)/n_inf*100:.1f}%)")
    # next_compound tak dikenal di antara has_target
    tgt_comp_null = int(dsf['target_next_compound'].isna().sum())
    print(f"    dataset: target_next_compound NaN: {tgt_comp_null}")
    comp_vals = dsf['target_next_compound'].dropna().unique()
    print(f"    nilai target_next_compound unik: {sorted(map(str,comp_vals))}")

    # --- 5. Distribusi target laps_remaining ---
    print("\n[5] DISTRIBUSI TARGET laps_remaining (dataset has_target)")
    t = dsf['target_laps_remaining'].astype(float)
    print(f"    n={len(t)} min={t.min():.0f} max={t.max():.0f} mean={t.mean():.2f} median={t.median():.0f} std={t.std():.2f}")
    print(f"    count==0: {int((t==0).sum())} | ==1: {int((t==1).sum())} | <=5: {int((t<=5).sum())} ({(t<=5).mean()*100:.1f}%)")
    q = np.percentile(t, [10,25,50,75,90,95,99])
    print(f"    persentil 10/25/50/75/90/95/99: {[round(float(x),1) for x in q]}")

    # --- 6. Estimasi batas atas akurasi akibat label bising ---
    print("\n[6] ESTIMASI BATAS ATAS AKURASI (label noise)")
    p_scvsc = float(scvsc.mean())
    p_red = float(merged['red'].mean())
    print(f"    Fraksi pit dipicu SC/VSC (tak terlihat model): {p_scvsc*100:.1f}%")
    print(f"    Fraksi pit saat RED: {p_red*100:.1f}%")
    print(f"    HIPOTESIS: pit SC/VSC bersifat reaktif (keputusan mendadak saat SC),")
    print(f"      sulit diprediksi tanpa fitur SC. Jika {p_scvsc*100:.0f}% pit tak terprediksi")
    print(f"      sistematis, batas atas hit-rate peristiwa pit ~ {(1-p_scvsc)*100:.0f}% untuk timing tepat.")
    print(f"    (Angka ini batas kasar; verifikasi kausal butuh eksperimen Fase B.)")

    # --- 7. Konsistensi pitstop mentah vs target dataset (rujuk audit lama, cek baru) ---
    print("\n[7] CROSS-CHECK: target vs pitstop mentah (sampel)")
    # ambil beberapa driver-race, bandingkan pit berikutnya dari dataset vs mentah
    st_set = set(zip(st['season'], st['round'], st['lap']))
    sample = dsf.sample(min(2000, len(dsf)), random_state=7)
    mismatch = 0
    checked = 0
    pit_lookup = {}
    for (s, r, d), g in pit.groupby(["season", "round", "driverId"]):
        pit_lookup[(s, r, d)] = sorted(g['lap'].astype(int).tolist())
    for _, row in sample.iterrows():
        s, r = int(row['season']), int(row['round'])
        d = row['driver_id']; t_lap = int(row['lap_number'])
        if pd.isna(row['target_laps_remaining']):
            continue
        pits = [p for p in pit_lookup.get((s, r, d), []) if p < row.get('total_laps', 999)]
        nxt = [p for p in pits if p >= t_lap]
        if not nxt:
            continue
        expected = nxt[0] - t_lap
        checked += 1
        if abs(expected - float(row['target_laps_remaining'])) > 0.5:
            mismatch += 1
    print(f"    sampel dicek={checked} mismatch={mismatch} ({mismatch/max(checked,1)*100:.2f}%)")
    print(f"    (red-flag TIDAK dibuang di cek ini karena track_status mentah; ")
    print(f"     dataset juga tidak membuang => konsisten, tapi keduanya salah bila ada red-flag pit)")


if __name__ == "__main__":
    main()
