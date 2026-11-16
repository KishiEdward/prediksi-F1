# -*- coding: utf-8 -*-
"""AUDIT 2 — Plafon informasi, kurva belajar, kualitas fitur.

READ-ONLY (inference + agregasi; TIDAK ada training baru).
Pemeriksaan #2 (plafon informasi), #3 (kurva belajar dari hasil CV yang ada),
#5 (kualitas fitur: importance, redundan/noise, fitur mati).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ROOT, muat_dataset, muat_inference, muat_model_run3,  # noqa: E402
                     prediksi_run3)
from app.ml.dataset import nama_fitur_xgb  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)


def variance_explained_by_groups(y: np.ndarray, groups: pd.DataFrame) -> float:
    """R^2 non-parametrik: 1 - SS_within/SS_total. Tanpa training (hanya mean grup)."""
    y = np.asarray(y, float)
    ss_total = float(((y - y.mean()) ** 2).sum())
    if ss_total == 0:
        return 0.0
    key = groups.astype(str).agg("|".join, axis=1)
    ss_within = 0.0
    for _, idx in key.groupby(key).groups.items():
        yg = y[np.asarray(idx)]
        if len(yg):
            ss_within += float(((yg - yg.mean()) ** 2).sum())
    return 1.0 - ss_within / ss_total


def main() -> None:
    print("=" * 78)
    print("AUDIT 2 — PLAFON INFORMASI, KURVA BELAJAR, KUALITAS FITUR")
    print("=" * 78)

    dsf = muat_dataset()
    inf = muat_inference()
    reg, clf, binary, pre, pre_bin, conf = muat_model_run3()
    fitur = nama_fitur_xgb()

    # --- A. Fitur mati / degenerat pada matriks fitur XGB yang SEBENARNYA dipakai ---
    print("\n[A] FITUR MATI/DEGENERAT (diperiksa di matriks fitur model, train 2021-2023)")
    train = dsf[dsf["season"] <= 2023]
    Xtr = __import__("app.ml.dataset", fromlist=["fitur_xgb"]).fitur_xgb(train, pre)
    print(f"    shape X_train={Xtr.shape} (baris, fitur); nama fitur={len(fitur)}")
    dead = []
    for j, name in enumerate(fitur):
        col = Xtr[:, j]
        col_num = pd.to_numeric(pd.Series(col), errors="coerce")
        nunq = col_num.nunique(dropna=False)
        std = float(np.nanstd(col_num.to_numpy(dtype=float))) if nunq > 0 else 0.0
        nan_frac = float(col_num.isna().mean())
        if nunq <= 1 or std == 0.0:
            dead.append((name, nunq, nan_frac))
    if dead:
        print("    Fitur KONSTAN/nol-varians (tidak memberi informasi):")
        for name, nunq, nanf in dead:
            print(f"      - {name}: nunique={nunq} nan_frac={nanf:.3f}")
    else:
        print("    Tidak ada fitur konstan.")
    # cek khusus fitur status lintasan
    print("    Cek khusus fitur status lintasan (train):")
    for name in ["is_green", "is_yellow", "is_sc", "is_vsc", "is_red", "laps_since_sc_vsc"]:
        if name in fitur:
            j = fitur.index(name)
            col = pd.to_numeric(pd.Series(Xtr[:, j]), errors="coerce")
            print(f"      {name}: nunique={col.nunique()} mean={col.mean():.4f} "
                  f"min={col.min()} max={col.max()}")

    # --- B. Feature importance dari model Run 3 (gain) ---
    print("\n[B] FEATURE IMPORTANCE (model Run 3, importance gain)")
    for label, model in [("REG (K03)", reg), ("CLF (K04)", clf), ("BINARY", binary)]:
        if model is None:
            continue
        try:
            imp = model.feature_importances_
        except Exception as e:
            print(f"    {label}: tidak bisa ambil importance ({e})")
            continue
        order = np.argsort(imp)[::-1]
        print(f"  --- {label}: top 12 ---")
        for r, j in enumerate(order[:12], 1):
            print(f"    {r:2d}. {fitur[j]:28s} {imp[j]:.4f}")
        zero = int((imp == 0).sum())
        print(f"    fitur importance==0: {zero}/{len(imp)}")

    # --- C. Plafon informasi: varians laps_to_pit dijelaskan tyre_life+compound ---
    print("\n[C] PLAFON INFORMASI (val 2024) — varians target dijelaskan fitur sederhana")
    val = dsf[dsf["season"] == 2024].reset_index(drop=True)
    y_val = val["target_laps_remaining"].to_numpy(float)
    # R2 non-parametrik oleh (tyre_life, current_compound) saja
    g1 = val[["tyre_life", "current_compound"]]
    r2_tl_comp = variance_explained_by_groups(y_val, g1)
    g2 = val[["tyre_life"]]
    r2_tl = variance_explained_by_groups(y_val, g2)
    g3 = val[["stint_number", "current_compound"]]
    r2_stint = variance_explained_by_groups(y_val, g3)
    print(f"    R2 oleh tyre_life saja                 : {r2_tl:.4f}")
    print(f"    R2 oleh (tyre_life, current_compound)  : {r2_tl_comp:.4f}")
    print(f"    R2 oleh (stint_number, current_compound): {r2_stint:.4f}")
    # R2 model penuh
    pred_laps, proba, pred_cls, pbin = prediksi_run3(val, reg, clf, pre, binary, pre_bin)
    ss_res = float(((y_val - pred_laps) ** 2).sum())
    ss_tot = float(((y_val - y_val.mean()) ** 2).sum())
    r2_model = 1 - ss_res / ss_tot
    mae_model = float(np.abs(y_val - pred_laps).mean())
    print(f"    R2 model Run 3 (reg)                   : {r2_model:.4f}  (MAE {mae_model:.3f})")
    print(f"    -> Selisih R2 model vs (tyre_life+compound): {r2_model - r2_tl_comp:+.4f}")
    print(f"    Interpretasi: jika R2 model ~ R2 tyre_life+compound, model tak banyak")
    print(f"      menambah info di luar umur ban & compound.")

    # --- D. Konsistensi dua pembalap satu tim (situasi serupa) ---
    print("\n[D] KONSISTENSI TEAMMATE (proxy plafon informasi strategis)")
    # Untuk tiap (season,round, stint_number), bandingkan laps_to_pit aktual 2 rekan setim
    # pada stint yang sama. Selisih kecil = keputusan tim sinkron (bisa diprediksi dari
    # kondisi bersama); selisih besar = keputusan individual/sulit diprediksi.
    d = dsf.copy()
    diffs = []
    for (s, r, st_num), g in d.groupby(["season", "round", "stint_number"]):
        # ambil satu baris per driver (laps_to_pit pada awal stint ~ panjang stint)
        per_drv = g.groupby("driver_code")["target_laps_remaining"].max()
        if len(per_drv) == 2:
            v = per_drv.to_numpy()
            diffs.append(abs(v[0] - v[1]))
    if diffs:
        diffs = np.array(diffs, float)
        print(f"    n pasangan teammate-stint: {len(diffs)}")
        print(f"    |selisih laps_to_pit| antar teammate: mean={diffs.mean():.2f} "
              f"median={np.median(diffs):.2f} p90={np.percentile(diffs,90):.2f}")
        print(f"    fraksi selisih<=2 lap: {(diffs<=2).mean()*100:.1f}%  <=5 lap: {(diffs<=5).mean()*100:.1f}%")
        print(f"    Interpretasi: selisih besar => banyak keputusan pit bersifat")
        print(f"      individual/reaktif, bukan fungsi kondisi bersama => plafon akurasi.")

    # --- E. Kurva belajar: dari hasil CV/ablation yang SUDAH ADA (tanpa training) ---
    print("\n[E] KURVA BELAJAR (dari log CV/ablation yang sudah ada, tanpa training baru)")
    jl = ROOT / "logs" / "run3_experiments.jsonl"
    if jl.exists():
        rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
        for r in rows:
            c = r.get("config", "")
            if c in ("K01_xgb_base", "K03_xgb_optuna_reg"):
                f = r.get("folds", {})
                cv1 = f.get("cv1", {}).get("laps_remaining", {}).get("mae")
                cv2 = f.get("cv2", {}).get("laps_remaining", {}).get("mae")
                print(f"    {c}: cv1(train21-22->val23) MAE={cv1} | cv2(train21-23->val24) MAE={cv2}")
        print("    Catatan: cv1 vs cv2 BEDA tahun val, jadi bukan kurva belajar bersih.")
    # ablation Run 2 (train dgn vs tanpa 2021, val 2024 sama) = perbandingan bersih
    m = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    ab = m.get("ablation_no2021") or m.get("run2", {}).get("ablation")
    print("    Ablation Run 2 (val 2024 sama, train +/- 2021) — lihat metrics.json:")
    # cari ringkasan ablation
    val_full = m.get("run2", {}).get("val", {}).get("mean_std", {})
    for mt in ("lstm", "xgboost", "hybrid"):
        d2 = val_full.get(mt, {})
        if d2:
            print(f"      {mt} val2024 (train 2021-23): MAE={d2.get('laps_remaining.mae_mean')} "
                  f"F1={d2.get('compound.f1_macro_mean')}")


if __name__ == "__main__":
    main()
