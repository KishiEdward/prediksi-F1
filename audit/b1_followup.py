# -*- coding: utf-8 -*-
"""B1 follow-up: (a) truncated-rebuild causality test untuk fitur status BARU,
(b) kasus Monaco 2025 (aturan wajib dua pit?), (c) evaluasi test 2025 SEKALI
untuk B1 sebagai pandangan keempat. READ-ONLY thd produksi; tidak mengubah model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.logging_setup import setup_logging  # noqa: E402
from app.ml import dataset as ds, run3  # noqa: E402
from app.ml.evaluate import _metrik_klasifikasi, _metrik_regresi  # noqa: E402
from app.ml.features import COMPOUNDS, PROCESSED, TRACK_STATUS_MAP  # noqa: E402
from app.ml.pipeline import Preprocessor  # noqa: E402
import glob  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("b1.followup")
MODELS = Path(__file__).resolve().parent.parent / "models"
KEY = ["season", "round", "driver_code", "lap_number"]


# ---------------------------------------------------------------------------
# (a) Causality test: fitur status lap N hanya bergantung data <= lap N
# ---------------------------------------------------------------------------
def status_features_from_trackstatus(ts_series: pd.Series) -> pd.DataFrame:
    """Replikasi logika status all-code (features.py status_all_codes=True)."""
    ts = ts_series.astype("string").fillna("")
    out = pd.DataFrame(index=ts.index)
    out["is_green"] = ts.str.contains("1").astype(float)
    out["is_yellow"] = ts.str.contains("2").astype(float)
    out["is_sc"] = ts.str.contains("4").astype(float)
    out["is_red"] = ts.str.contains("5").astype(float)
    out["is_vsc"] = ts.str.contains("6").astype(float)
    out["track_status"] = np.select(
        [out["is_red"] == 1, out["is_vsc"] == 1, out["is_sc"] == 1,
         out["is_yellow"] == 1, out["is_green"] == 1],
        ["RED", "VSC", "SC", "YELLOW", "GREEN"], default="OTHER")
    return out


def laps_since_scvsc(scvsc: np.ndarray) -> np.ndarray:
    hitung = np.zeros(len(scvsc)); sejak = 99
    for i, v in enumerate(scvsc):
        sejak = 0 if v == 1 else sejak + 1
        hitung[i] = min(sejak, 10)
    return hitung


def causal_test():
    print("=" * 74)
    print("(a) TRUNCATED-REBUILD CAUSALITY TEST — fitur status B1")
    print("=" * 74)
    files = sorted(glob.glob(str(PROCESSED.parent / "raw" / "fastf1" / "*_laps.parquet")))[:6]
    total_checked = 0; mismatches = 0
    for f in files:
        L = pd.read_parquet(f)
        if "TrackStatus" not in L.columns or "LapNumber" not in L.columns:
            continue
        L = L.dropna(subset=["LapNumber"]).copy()
        L["LapNumber"] = pd.to_numeric(L["LapNumber"], errors="coerce")
        L = L.dropna(subset=["LapNumber"])
        L["LapNumber"] = L["LapNumber"].astype(int)
        for drv, g in L.groupby("Driver"):
            g = g.sort_values("LapNumber").reset_index(drop=True)
            if len(g) < 8:
                continue
            # FULL
            full = status_features_from_trackstatus(g["TrackStatus"])
            scvsc_full = ((full["is_sc"] + full["is_vsc"]).clip(0, 1)).to_numpy()
            full["lssv"] = laps_since_scvsc(scvsc_full)
            # TRUNCATED at lap N (buang semua lap > N), rebuild
            N = int(g["LapNumber"].quantile(0.6))
            gt = g[g["LapNumber"] <= N].reset_index(drop=True)
            if len(gt) < 3:
                continue
            tr = status_features_from_trackstatus(gt["TrackStatus"])
            scvsc_tr = ((tr["is_sc"] + tr["is_vsc"]).clip(0, 1)).to_numpy()
            tr["lssv"] = laps_since_scvsc(scvsc_tr)
            # bandingkan baris lap <= N
            fl = full[full.index < len(gt)]
            for col in ["is_sc", "is_vsc", "is_red", "is_yellow", "is_green", "lssv", "track_status"]:
                a = fl[col].to_numpy(); b = tr[col].to_numpy()
                total_checked += len(a)
                if not np.array_equal(a.astype(str), b.astype(str)):
                    mismatches += int((a.astype(str) != b.astype(str)).sum())
    print(f"  file dicek: {len(files)} | sel dibandingkan: {total_checked} | mismatch: {mismatches}")
    print(f"  HASIL: {'LULUS (kausal)' if mismatches==0 else 'GAGAL (ada kebocoran!)'}")
    print("  Alasan: is_* hanya dari TrackStatus lap itu sendiri; laps_since_sc_vsc")
    print("          dihitung mundur dalam driver-race. Tidak ada lap masa depan dipakai.")
    return mismatches == 0


# ---------------------------------------------------------------------------
# (b) Kasus Monaco: aturan wajib dua pit 2025?
# ---------------------------------------------------------------------------
def monaco_case():
    print("\n" + "=" * 74)
    print("(b) KASUS MONACO — aturan wajib dua pit 2025?")
    print("=" * 74)
    pit = pd.read_parquet(PROCESSED.parent / "raw" / "jolpica_pitstops.parquet")
    pit["season"] = pd.to_numeric(pit["season"], errors="coerce").astype(int)
    pit["round"] = pd.to_numeric(pit["round"], errors="coerce").astype(int)
    pit["lap"] = pd.to_numeric(pit["lap"], errors="coerce")
    pit = pit.dropna(subset=["lap"])
    # round Monaco bervariasi per tahun; cari via raceName
    mon = pit[pit["raceName"].astype(str).str.contains("Monaco", case=False, na=False)]
    print("  Jumlah pit stop per pembalap di Monaco, per musim:")
    for season in sorted(mon["season"].unique()):
        g = mon[mon["season"] == season]
        per_drv = g.groupby("driverId").size()
        n_drivers = len(per_drv)
        stops = per_drv.to_numpy()
        frac_2plus = float((stops >= 2).mean()) if len(stops) else 0
        print(f"    {season}: drivers={n_drivers:2d} mean_stops={stops.mean():.2f} "
              f"max={stops.max()} fraksi>=2 stop={frac_2plus*100:.0f}%")
    # detail 2025: 3 driver galat terbesar
    print("\n  Detail Monaco 2025 (COL, ALO, STR) — lap pit aktual:")
    m25 = mon[mon["season"] == 2025]
    for d in ["colapinto", "albon", "stroll"]:
        sub = m25[m25["driverId"].astype(str).str.contains(d.split('o')[0][:3], case=False, na=False)]
    # pakai kode dari dataset
    inf = pd.read_parquet(PROCESSED / "b1" / "lap_inference.parquet")
    m25r = inf[(inf.season == 2025) & (inf.race_name.astype(str).str.contains("Monaco"))]
    rnd = int(m25r["round"].iloc[0]) if len(m25r) else 8
    mp = pit[(pit.season == 2025) & (pit["round"] == rnd)]
    for code in ["COL", "ALO", "STR"]:
        # map code->driverId via dataset
        did = m25r[m25r.driver_code == code]["driver_id"]
        did = did.iloc[0] if len(did) else None
        if did is None:
            continue
        stops = sorted(mp[mp.driverId == did]["lap"].astype(int).tolist())
        print(f"    {code} ({did}): pit laps = {stops}  (jumlah={len(stops)})")
    print("\n  Interpretasi: jika 2025 mean_stops & fraksi>=2 melonjak vs 2021-2024,")
    print("  itu konsisten dengan aturan wajib dua pit Monaco 2025 (regulasi baru),")
    print("  yang TIDAK diketahui model (dilatih 2021-2023 pola satu-stop).")


# ---------------------------------------------------------------------------
# (c) Test 2025 SEKALI untuk B1 (pandangan keempat)
# ---------------------------------------------------------------------------
def b1_test_once():
    print("\n" + "=" * 74)
    print("(c) TEST 2025 SEKALI — B1 (pandangan keempat; B1 DITOLAK oleh CV)")
    print("=" * 74)
    reg_p = json.loads((MODELS / "run3_optuna_reg.json").read_text())["best_params"]
    clf_p = json.loads((MODELS / "run3_optuna_cls.json").read_text())["best_params"]
    b1 = pd.read_parquet(PROCESSED / "b1" / "lap_dataset.parquet")
    base = pd.read_parquet(PROCESSED / "lap_dataset.parquet")
    for d in (b1, base):
        for c in ("season", "round", "lap_number"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(int)
    b1_tr = b1[b1.season <= 2023]; b1_te = b1[b1.season == 2025].reset_index(drop=True)
    pre = Preprocessor().fit(b1_tr)
    Xtr = ds.fitur_xgb(b1_tr, pre); Xte = ds.fitur_xgb(b1_te, pre)
    y_reg_tr = b1_tr["target_laps_remaining"].to_numpy(np.float32)
    y_cls_tr = run3.encode_compound(b1_tr)
    w = run3.bobot_kelas(y_cls_tr, "sqrt10")[y_cls_tr]
    reg = XGBRegressor(objective="reg:squarederror", **reg_p); reg.fit(Xtr, y_reg_tr, verbose=False)
    clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **clf_p)
    clf.fit(Xtr, y_cls_tr, sample_weight=w, verbose=False)
    pred_laps = np.clip(reg.predict(Xte), 0, None)
    pred_cls = ds.proba_compound_full(clf, Xte).argmax(1)
    y_reg = b1_te["target_laps_remaining"].to_numpy(float)
    y_cls = run3.encode_compound(b1_te)
    mr = _metrik_regresi(y_reg, pred_laps); mc = _metrik_klasifikasi(y_cls, pred_cls)
    print(f"  B1 test 2025 (SELURUH baris B1, n={len(b1_te)}): MAE={mr['mae']:.4f} "
          f"RMSE={mr['rmse']:.4f} hit<=3={mr['dalam_3_lap']:.4f} F1={mc['f1_macro']:.4f} acc={mc['accuracy']:.4f}")
    print(f"    bucket: 0-5={mr['mae_jarak_0_5']} 6-15={mr['mae_jarak_6_15']} >15={mr['mae_jarak_gt15']}")
    # common rows dengan test Run3 (base)
    base_te = base[base.season == 2025].reset_index(drop=True)
    b1p = b1_te[KEY].copy(); b1p["pl"] = pred_laps; b1p["pc"] = pred_cls
    b1p["yr"] = y_reg; b1p["yc"] = y_cls
    # prediksi BASE model pada baris base (sudah ada di metrics_run3); recompute cepat:
    pre_b = Preprocessor().fit(base[base.season <= 2023])
    reg_b = XGBRegressor(objective="reg:squarederror", **reg_p)
    reg_b.fit(ds.fitur_xgb(base[base.season <= 2023], pre_b),
              base[base.season <= 2023]["target_laps_remaining"].to_numpy(np.float32), verbose=False)
    clf_b = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **clf_p)
    ycb = run3.encode_compound(base[base.season <= 2023])
    clf_b.fit(ds.fitur_xgb(base[base.season <= 2023], pre_b), ycb,
              sample_weight=run3.bobot_kelas(ycb, "sqrt10")[ycb], verbose=False)
    base_te2 = base_te.copy()
    Xb = ds.fitur_xgb(base_te2, pre_b)
    base_te2["pl"] = np.clip(reg_b.predict(Xb), 0, None)
    base_te2["pc"] = ds.proba_compound_full(clf_b, Xb).argmax(1)
    m = base_te2[KEY + ["pl", "pc", "target_laps_remaining", "target_next_compound"]].merge(
        b1p[KEY + ["pl", "pc"]], on=KEY, how="inner", suffixes=("_base", "_b1"))
    yr = m["target_laps_remaining"].to_numpy(float)
    yc = m["target_next_compound"].map({c: i for i, c in enumerate(COMPOUNDS)}).to_numpy(int)
    mae_base = float(np.abs(yr - m["pl_base"].to_numpy(float)).mean())
    mae_b1 = float(np.abs(yr - m["pl_b1"].to_numpy(float)).mean())
    f1_base = _metrik_klasifikasi(yc, m["pc_base"].to_numpy(int))["f1_macro"]
    f1_b1 = _metrik_klasifikasi(yc, m["pc_b1"].to_numpy(int))["f1_macro"]
    print(f"  BARIS SAMA test 2025 (n={len(m)}): MAE base={mae_base:.4f} b1={mae_b1:.4f} "
          f"({mae_base-mae_b1:+.4f}) | F1 base={f1_base:.4f} b1={f1_b1:.4f}")
    print("  Catatan: B1 DITOLAK oleh CV; test ini hanya pandangan keempat transparan,")
    print("           TIDAK dipakai untuk seleksi/promosi.")
    out = {"b1_test_all_rows": {"mae": mr["mae"], "f1_macro": mc["f1_macro"],
                               "accuracy": mc["accuracy"], "n": len(b1_te),
                               "buckets": {"0_5": mr["mae_jarak_0_5"], "6_15": mr["mae_jarak_6_15"],
                                           "gt15": mr["mae_jarak_gt15"]}},
           "b1_test_same_rows": {"mae_base": round(mae_base, 4), "mae_b1": round(mae_b1, 4),
                                 "f1_base": f1_base, "f1_b1": f1_b1, "n": len(m)}}
    (MODELS / "b1_test_once.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    setup_logging()
    causal_test()
    monaco_case()
    b1_test_once()


if __name__ == "__main__":
    main()
