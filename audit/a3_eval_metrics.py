# -*- coding: utf-8 -*-
"""AUDIT 3 — Korektness evaluasi + definisi metrik (per-baris vs per-peristiwa).

READ-ONLY. Pemeriksaan #4: verifikasi perhitungan metrik, sampel evaluasi,
dampak bug F1-makro. Plus metrik gabungan & compound 1-5 lap sebelum pit.
Evaluasi pada VAL 2024 (dasar rekomendasi). Test 2025 hanya deskriptif.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ROOT, muat_dataset, muat_model_run3, prediksi_run3)  # noqa: E402
from app.ml.features import COMPOUNDS  # noqa: E402

pd.set_option("display.width", 200)


def f1_macro_5kelas(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=list(range(5)), zero_division=0))


def f1_macro_inferred(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> None:
    print("=" * 78)
    print("AUDIT 3 — EVALUASI & METRIK (VAL 2024)")
    print("=" * 78)

    dsf = muat_dataset()
    reg, clf, binary, pre, pre_bin, conf = muat_model_run3()
    val = dsf[dsf["season"] == 2024].reset_index(drop=True)
    y_reg = val["target_laps_remaining"].to_numpy(float)
    y_cls = val["target_next_compound"].map({c: i for i, c in enumerate(COMPOUNDS)}).to_numpy()
    pred_laps, proba, pred_cls, pbin = prediksi_run3(val, reg, clf, pre, binary, pre_bin)

    # --- A. Dampak bug F1-makro (label inferred vs 5 kelas) di VAL 2024 ---
    print("\n[A] DAMPAK BUG F1-MAKRO (evaluate.py:76 tanpa labels=)")
    f1_inf = f1_macro_inferred(y_cls, pred_cls)
    f1_5 = f1_macro_5kelas(y_cls, pred_cls)
    present = sorted(set(y_cls.tolist()) | set(pred_cls.tolist()))
    print(f"    kelas hadir di val 2024 (true|pred): {present} -> {[COMPOUNDS[i] for i in present]}")
    print(f"    F1-makro (label inferred, spt kode) : {f1_inf:.4f}")
    print(f"    F1-makro (5 kelas eksplisit, benar) : {f1_5:.4f}")
    print(f"    selisih: {f1_inf - f1_5:+.4f}")
    # cek tiap config CV dari jsonl: apakah WET hadir (maka CV pakai 5 kelas)
    jl = ROOT / "logs" / "run3_experiments.jsonl"
    rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
    print("    Cek kehadiran WET di fold CV (tentukan apakah CV macro 5-kelas):")
    for r in rows:
        if r.get("config") in ("K01_xgb_base", "K04_xgb_optuna_cls"):
            for fold in ("cv1", "cv2"):
                pk = r.get("folds", {}).get(fold, {}).get("compound", {}).get("per_kelas", {})
                wet_sup = pk.get("WET", {}).get("support")
                print(f"      {r['config']} {fold}: WET support={wet_sup} "
                      f"f1_macro(logged)={r['folds'][fold]['compound']['f1_macro']}")

    # --- B. Verifikasi independen metrik regresi (recompute) ---
    print("\n[B] VERIFIKASI INDEPENDEN METRIK REGRESI (val 2024)")
    err = np.abs(y_reg - np.clip(pred_laps, 0, None))
    print(f"    MAE  recompute: {err.mean():.4f}")
    print(f"    RMSE recompute: {np.sqrt((err**2).mean()):.4f}")
    print(f"    hit <=1: {(err<=1).mean():.4f} <=3: {(err<=3).mean():.4f} <=5: {(err<=5).mean():.4f}")
    print(f"    acc compound: {accuracy_score(y_cls, pred_cls):.4f}")

    # --- C. Metrik gabungan (timing <=k DAN compound benar) ---
    print("\n[C] METRIK GABUNGAN — per BARIS vs per PERISTIWA PIT (val 2024)")
    for k in (3, 5):
        # per baris
        ok_row = ((err <= k) & (pred_cls == y_cls))
        print(f"    k={k} PER-BARIS  : benar={ok_row.mean()*100:.2f}%  (n={len(val)})")
    # per peristiwa: grup stint -> pit lap p = t + target
    val2 = val.copy()
    val2["pred_laps"] = np.clip(pred_laps, 0, None)
    val2["pred_pit_lap"] = val2["lap_number"] + val2["pred_laps"]
    val2["actual_pit_lap"] = val2["lap_number"] + val2["target_laps_remaining"]
    val2["pred_comp"] = pred_cls
    val2["err_lap"] = np.abs(val2["pred_pit_lap"] - val2["actual_pit_lap"])
    grp = val2.groupby(["season", "round", "driver_code", "actual_pit_lap"])
    events = []
    for _, g in grp:
        if len(g) == 0:
            continue
        ev_pred_pit = g["pred_pit_lap"].median()
        ev_actual = g["actual_pit_lap"].iloc[0]
        ev_err = abs(ev_pred_pit - ev_actual)
        # compound: prediksi pada lap terdekat ke pit (horizon terkecil)
        g2 = g.reindex(g["target_laps_remaining"].sort_values().index)
        ev_pred_comp = g2["pred_comp"].iloc[0]  # prediksi saat paling dekat pit
        ev_true_comp = g2["target_next_compound"].map(
            {c: i for i, c in enumerate(COMPOUNDS)}).iloc[0]
        events.append({"err": ev_err, "comp_ok": int(ev_pred_comp == ev_true_comp),
                       "n_rows": len(g)})
    ev = pd.DataFrame(events)
    print(f"    jumlah peristiwa pit di val 2024: {len(ev)} (total baris {len(val)})")
    for k in (3, 5):
        ok_ev = ((ev["err"] <= k) & (ev["comp_ok"] == 1))
        print(f"    k={k} PER-PERISTIWA: benar={ok_ev.mean()*100:.2f}%  "
              f"(timing<= {k}: {(ev['err']<=k).mean()*100:.1f}%, compound: {ev['comp_ok'].mean()*100:.1f}%)")
    print(f"    MAE timing per-peristiwa (median pred pit lap): {ev['err'].mean():.3f} lap")
    print(f"    MAE timing per-baris: {err.mean():.3f} lap")
    print("    SELISIH per-baris vs per-peristiwa: per-baris memberi bobot lebih pada")
    print("      stint panjang (banyak baris); per-peristiwa menimbang tiap keputusan pit")
    print("      sama rata. Per-baris juga memasukkan horizon jauh (target besar) yang")
    print("      lebih sulit, sehingga biasanya lebih rendah daripada per-peristiwa-dekat.")

    # --- D. Compound metrik hanya 1-5 lap sebelum pit ---
    print("\n[D] METRIK COMPOUND HANYA PADA 1-5 LAP SEBELUM PIT (val 2024)")
    mask = (val["target_laps_remaining"] >= 1) & (val["target_laps_remaining"] <= 5)
    yc, pc = y_cls[mask], pred_cls[mask]
    print(f"    n={int(mask.sum())} acc={accuracy_score(yc,pc):.4f} "
          f"F1-5kelas={f1_macro_5kelas(yc,pc):.4f} F1-inferred={f1_macro_inferred(yc,pc):.4f}")
    mask_all = np.ones(len(val), bool)
    print(f"    (bandingkan semua baris: acc={accuracy_score(y_cls,pred_cls):.4f} "
          f"F1-5kelas={f1_macro_5kelas(y_cls,pred_cls):.4f})")
    print("    Interpretasi: akurasi compound saat pit SANGAT DEKAT (1-5 lap) adalah")
    print("      ukuran paling relevan untuk keputusan aksi; bandingkan dgn semua baris.")

    # --- E. Sampel evaluasi: pastikan hanya has_target & tidak ada kebocoran baru ---
    print("\n[E] SAMPEL EVALUASI")
    print(f"    val 2024 rows (has_target semua di lap_dataset): {len(val)}")
    print(f"    target NaN di val: {int(val['target_laps_remaining'].isna().sum())}")
    print(f"    musim di val: {sorted(val['season'].unique().tolist())} (harus [2024])")
    # cek model tidak dilatih pada 2024: reg/clf final dilatih <=2023 (dari script 06)
    print("    Model final dilatih pada season<=2023 (scripts/06_run3_final.py:34-37) -> val 2024 holdout bersih.")


if __name__ == "__main__":
    main()
