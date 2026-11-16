# -*- coding: utf-8 -*-
"""AUDIT 4 — Segmentasi error + pelanggaran aturan dua compound kering.

READ-ONLY. Pemeriksaan #6 (rule violation, hanya diukur) & #7 (segmentasi dgn
jumlah sampel). Evaluasi pada VAL 2024 + CV (cache pred) untuk dasar rekomendasi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ROOT, muat_dataset, muat_model_run3, prediksi_run3)  # noqa: E402
from app.ml.features import COMPOUNDS  # noqa: E402

pd.set_option("display.width", 220)
DRY = {"SOFT", "MEDIUM", "HARD"}


def seg_table(df: pd.DataFrame, err: np.ndarray, comp_ok: np.ndarray, key: str,
              label: str, min_n: int = 30) -> None:
    print(f"\n  --- Segmentasi oleh {label} ---")
    d = df.copy()
    d["_err"] = err
    d["_ok"] = comp_ok.astype(int)
    g = d.groupby(key).agg(n=("_err", "size"), mae=("_err", "mean"),
                           comp_acc=("_ok", "mean")).reset_index()
    g = g[g["n"] >= min_n].sort_values("mae", ascending=False)
    for _, r in g.iterrows():
        print(f"    {str(r[key])[:22]:22s} n={int(r['n']):5d}  MAE={r['mae']:6.3f}  "
              f"acc_comp={r['comp_acc']:.3f}")
    small = d.groupby(key).size()
    small = small[small < min_n]
    if len(small):
        print(f"    (segmen dgn n<{min_n} disembunyikan: {len(small)} segmen, "
              f"total {int(small.sum())} baris)")


def main() -> None:
    print("=" * 78)
    print("AUDIT 4 — SEGMENTASI & PELANGGARAN ATURAN (VAL 2024)")
    print("=" * 78)

    dsf = muat_dataset()
    reg, clf, binary, pre, pre_bin, conf = muat_model_run3()
    val = dsf[dsf["season"] == 2024].reset_index(drop=True)
    y_reg = val["target_laps_remaining"].to_numpy(float)
    cmap = {c: i for i, c in enumerate(COMPOUNDS)}
    y_cls = val["target_next_compound"].map(cmap).to_numpy()
    pred_laps, proba, pred_cls, pbin = prediksi_run3(val, reg, clf, pre, binary, pre_bin)
    err = np.abs(y_reg - np.clip(pred_laps, 0, None))
    comp_ok = (pred_cls == y_cls)

    # --- A. Segmentasi ---
    print("\n[A] SEGMENTASI ERROR (val 2024)")
    # jarak ke pit (bucket target)
    val["_bucket"] = pd.cut(val["target_laps_remaining"], bins=[0, 5, 15, 999],
                            labels=["0-5", "6-15", ">15"])
    seg_table(val, err, comp_ok, "_bucket", "jarak ke pit (bucket target)", min_n=1)
    # tyre age bucket
    val["_tyreage"] = pd.cut(val["tyre_life"], bins=[-1, 5, 15, 25, 999],
                             labels=["0-5", "6-15", "16-25", ">25"])
    seg_table(val, err, comp_ok, "_tyreage", "umur ban (tyre_life)", min_n=1)
    # stint number
    seg_table(val, err, comp_ok, "stint_number", "nomor stint", min_n=30)
    # circuit
    seg_table(val, err, comp_ok, "circuit_id", "sirkuit", min_n=50)
    # team
    seg_table(val, err, comp_ok, "team_id", "tim", min_n=50)
    # current compound
    seg_table(val, err, comp_ok, "current_compound", "compound saat ini", min_n=30)
    # track status (degenerat)
    print("\n  --- Segmentasi oleh track_status (PERIKSA degenerasi) ---")
    print("    value_counts track_status val 2024:", dict(val["track_status"].value_counts()))
    print("    is_sc sum:", int(val["is_sc"].sum()), "is_vsc sum:", int(val["is_vsc"].sum()),
          "is_red sum:", int(val["is_red"].sum()))
    print("    -> Tidak bisa segmentasi SC/VSC/RED: fitur mati (semua 0). Ini itself temuan.")

    # --- B. Pelanggaran aturan dua compound kering ---
    print("\n[B] PELANGGARAN ATURAN DUA COMPOUND KERING (hanya diukur)")
    print("    Definisi: untuk tiap peristiwa pit, hitung himpunan compound KERING")
    print("    (SOFT/MEDIUM/HARD) yang dipakai driver SEBELUM pit ini (dari riwayat")
    print("    current_compound). 'Pelanggaran potensial' = compound prediksi berikutnya")
    print("    adalah kering DAN union(dry_so_far, prediksi) berukuran 1 (artinya belum")
    print("    ada 2 spec kering berbeda). Upper bound: stop berikutnya bisa memperkenalkan")
    print("    compound baru. Dibandingkan dengan laju pada data AKTUAL (baseline).")
    val2 = val.copy()
    val2["pred_comp_name"] = [COMPOUNDS[i] for i in pred_cls]
    val2["true_comp_name"] = val2["target_next_compound"]
    # riwayat dry compound per driver-race hingga lap tertentu
    val2 = val2.sort_values(["season", "round", "driver_code", "lap_number"])
    dry_hist = {}
    viol_pred = viol_true = n_event = 0
    for (s, r, d), g in val2.groupby(["season", "round", "driver_code"]):
        used_dry = set()
        for _, row in g.iterrows():
            cc = row["current_compound"]
            if cc in DRY:
                used_dry.add(cc)
        # evaluasi per peristiwa pit: ambil baris terdekat sebelum tiap pit
        for p, gp in g.groupby(g["lap_number"] + g["target_laps_remaining"]):
            if gp.empty:
                continue
            row = gp.iloc[gp["target_laps_remaining"].argmin()]  # paling dekat pit
            # dry dipakai sebelum pit = dry dari riwayat s.d. lap row
            sub = g[g["lap_number"] <= row["lap_number"]]
            dry_before = {c for c in sub["current_compound"].unique() if c in DRY}
            n_event += 1
            pc = row["pred_comp_name"]
            tc = row["true_comp_name"]
            if pc in DRY and len(dry_before | {pc}) <= 1:
                viol_pred += 1
            if tc in DRY and len(dry_before | {tc}) <= 1:
                viol_true += 1
    print(f"    jumlah peristiwa pit dinilai: {n_event}")
    print(f"    pelanggaran potensial oleh PREDIKSI : {viol_pred} ({viol_pred/max(n_event,1)*100:.2f}%)")
    print(f"    pelanggaran potensial oleh AKTUAL   : {viol_true} ({viol_true/max(n_event,1)*100:.2f}%)")
    print("    Interpretasi: jika prediksi >> aktual, model menghasilkan pilihan compound")
    print("      yang secara sporting rule kurang masuk akal (mis. menyarakan compound sama).")

    # --- C. Segmentasi lintas musim pakai cache pred CV (cv1=2023, cv2=2024) ---
    print("\n[C] ERROR PER MUSIM (dari cache prediksi CV K03)")
    pred_dir = ROOT / "data" / "processed" / "run3_preds"
    for fold, yr in [("cv1", 2023), ("cv2", 2024)]:
        fp = pred_dir / f"K03_xgb_optuna_reg__{fold}.npz"
        if not fp.exists():
            continue
        z = np.load(fp)
        pl = np.clip(z["pred_laps"], 0, None)
        d = dsf[dsf["season"] == yr].reset_index(drop=True)
        if len(d) != len(pl):
            print(f"    {fold} ({yr}): ukuran tak cocok ({len(d)} vs {len(pl)}), lewati")
            continue
        yt = d["target_laps_remaining"].to_numpy(float)
        e = np.abs(yt - pl)
        print(f"    {fold} (val {yr}): n={len(d)} MAE={e.mean():.4f} "
              f"hit<=3={(e<=3).mean():.4f} hit<=5={(e<=5).mean():.4f}")


if __name__ == "__main__":
    main()
