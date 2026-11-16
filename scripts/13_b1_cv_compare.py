# -*- coding: utf-8 -*-
"""B1 — Perbandingan CV: BASE (Run 3, fitur status mati) vs B1 (fitur status hidup).

Hyperparameter K03 (reg) & K04 (clf) PERSIS SAMA, tanpa tuning ulang.
Seleksi HANYA dari CV 2021-2024. Test 2025 TIDAK disentuh di sini.

Dua basis perbandingan (sesuai pra-registrasi models/selection_criteria_b1.json):
  1. SELURUH BARIS: tiap model pada val fold-nya sendiri (populasi berbeda).
  2. BARIS SAMA: irisan baris val (key season/round/driver/lap), kedua model
     dinilai terhadap target B1 yang terkoreksi (red-flag exclusion aktif).
Keputusan utama dari BASIS BARIS SAMA.
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
from app.ml import dataset as ds  # noqa: E402
from app.ml import run3  # noqa: E402
from app.ml.evaluate import _metrik_klasifikasi, _metrik_regresi  # noqa: E402
from app.ml.features import COMPOUNDS, PROCESSED  # noqa: E402
from app.ml.pipeline import Preprocessor  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("b1.cv")
MODELS = Path(__file__).resolve().parent.parent / "models"
FOLDS = [("cv1", [2021, 2022], [2023]), ("cv2", [2021, 2022, 2023], [2024])]
KEY = ["season", "round", "driver_code", "lap_number"]


def load_params():
    reg_p = json.loads((MODELS / "run3_optuna_reg.json").read_text())["best_params"]
    clf_p = json.loads((MODELS / "run3_optuna_cls.json").read_text())["best_params"]
    return reg_p, clf_p


def train_predict(d_tr, d_va, reg_p, clf_p):
    """Latih K03-reg + K04-clf pada d_tr, prediksi d_va. Return pred_laps, pred_cls, y_reg, y_cls, key."""
    pre = Preprocessor().fit(d_tr)
    X_tr, X_va = ds.fitur_xgb(d_tr, pre), ds.fitur_xgb(d_va, pre)
    y_reg_tr = d_tr["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls_tr = run3.encode_compound(d_tr)
    w = run3.bobot_kelas(y_cls_tr, "sqrt10")[y_cls_tr]

    reg = XGBRegressor(objective="reg:squarederror", **reg_p)
    reg.fit(X_tr, y_reg_tr, verbose=False)
    pred_laps = np.clip(reg.predict(X_va), 0, None)

    clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **clf_p)
    clf.fit(X_tr, y_cls_tr, sample_weight=w, verbose=False)
    proba = ds.proba_compound_full(clf, X_va)
    pred_cls = proba.argmax(1)

    y_reg_va = d_va["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls_va = run3.encode_compound(d_va)
    keydf = d_va[KEY].reset_index(drop=True)
    return pred_laps, pred_cls, y_reg_va, y_cls_va, keydf


def fold_slice(df, train_years, val_years):
    tr = df[df["season"].isin(train_years)].reset_index(drop=True)
    va = df[df["season"].isin(val_years)].reset_index(drop=True)
    return tr, va


def main() -> None:
    setup_logging()
    reg_p, clf_p = load_params()
    base = pd.read_parquet(PROCESSED / "lap_dataset.parquet")
    b1 = pd.read_parquet(PROCESSED / "b1" / "lap_dataset.parquet")
    for d in (base, b1):
        for c in ("season", "round", "lap_number"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(int)

    hasil = {"folds": {}, "params_reg": reg_p, "params_clf": clf_p}
    agg = {k: [] for k in ["base_all_mae", "b1_all_mae", "same_mae_base",
                           "same_mae_b1", "base_all_f1", "b1_all_f1",
                           "same_f1_base", "same_f1_b1"]}

    for fold, tr_years, va_years in FOLDS:
        b_tr, b_va = fold_slice(base, tr_years, va_years)
        n_tr, n_va = fold_slice(b1, tr_years, va_years)
        logger.info("[%s] BASE train=%d val=%d | B1 train=%d val=%d",
                    fold, len(b_tr), len(b_va), len(n_tr), len(n_va))

        # --- BASE ---
        bp_laps, bp_cls, by_reg, by_cls, bkey = train_predict(b_tr, b_va, reg_p, clf_p)
        m_base_all = _metrik_regresi(by_reg, bp_laps)
        c_base_all = _metrik_klasifikasi(by_cls, bp_cls)

        # --- B1 ---
        np_laps, np_cls, ny_reg, ny_cls, nkey = train_predict(n_tr, n_va, reg_p, clf_p)
        m_b1_all = _metrik_regresi(ny_reg, np_laps)
        c_b1_all = _metrik_klasifikasi(ny_cls, np_cls)

        # --- BARIS SAMA: irisan key, nilai terhadap target B1 (terkoreksi) ---
        bdf = bkey.copy(); bdf["pred_laps_base"] = bp_laps; bdf["pred_cls_base"] = bp_cls
        ndf = nkey.copy(); ndf["pred_laps_b1"] = np_laps; ndf["pred_cls_b1"] = np_cls
        ndf["y_reg_b1"] = ny_reg; ndf["y_cls_b1"] = ny_cls
        merged = bdf.merge(ndf, on=KEY, how="inner")
        # juga bawa target BASE untuk hitung berapa yang targetnya berubah (red-flag)
        bdf2 = bkey.copy(); bdf2["y_reg_base"] = by_reg
        merged = merged.merge(bdf2, on=KEY, how="left")
        n_same = len(merged)
        tgt_changed = int((merged["y_reg_base"] != merged["y_reg_b1"]).sum())

        yr = merged["y_reg_b1"].to_numpy(float)
        m_same_base = _metrik_regresi(yr, merged["pred_laps_base"].to_numpy(float))
        m_same_b1 = _metrik_regresi(yr, merged["pred_laps_b1"].to_numpy(float))
        yc = merged["y_cls_b1"].to_numpy(int)
        c_same_base = _metrik_klasifikasi(yc, merged["pred_cls_base"].to_numpy(int))
        c_same_b1 = _metrik_klasifikasi(yc, merged["pred_cls_b1"].to_numpy(int))

        hasil["folds"][fold] = {
            "n_val_base": len(b_va), "n_val_b1": len(n_va), "n_same": n_same,
            "target_berubah_redflag": tgt_changed,
            "all_rows": {"base_mae": m_base_all["mae"], "b1_mae": m_b1_all["mae"],
                         "base_f1": c_base_all["f1_macro"], "b1_f1": c_b1_all["f1_macro"],
                         "base_hit3": m_base_all["dalam_3_lap"], "b1_hit3": m_b1_all["dalam_3_lap"],
                         "base_b_gt15": m_base_all["mae_jarak_gt15"], "b1_b_gt15": m_b1_all["mae_jarak_gt15"],
                         "base_b_0_5": m_base_all["mae_jarak_0_5"], "b1_b_0_5": m_b1_all["mae_jarak_0_5"],
                         "base_b_6_15": m_base_all["mae_jarak_6_15"], "b1_b_6_15": m_b1_all["mae_jarak_6_15"]},
            "same_rows": {"base_mae": m_same_base["mae"], "b1_mae": m_same_b1["mae"],
                          "base_f1": c_same_base["f1_macro"], "b1_f1": c_same_b1["f1_macro"],
                          "base_hit3": m_same_base["dalam_3_lap"], "b1_hit3": m_same_b1["dalam_3_lap"],
                          "base_b_gt15": m_same_base["mae_jarak_gt15"], "b1_b_gt15": m_same_b1["mae_jarak_gt15"],
                          "base_b_0_5": m_same_base["mae_jarak_0_5"], "b1_b_0_5": m_same_b1["mae_jarak_0_5"],
                          "base_b_6_15": m_same_base["mae_jarak_6_15"], "b1_b_6_15": m_same_b1["mae_jarak_6_15"]},
        }
        agg["base_all_mae"].append(m_base_all["mae"]); agg["b1_all_mae"].append(m_b1_all["mae"])
        agg["same_mae_base"].append(m_same_base["mae"]); agg["same_mae_b1"].append(m_same_b1["mae"])
        agg["base_all_f1"].append(c_base_all["f1_macro"]); agg["b1_all_f1"].append(c_b1_all["f1_macro"])
        agg["same_f1_base"].append(c_same_base["f1_macro"]); agg["same_f1_b1"].append(c_same_b1["f1_macro"])
        logger.info("[%s] ALL  MAE base=%.4f b1=%.4f | SAME MAE base=%.4f b1=%.4f (n=%d, tgt berubah=%d)",
                    fold, m_base_all["mae"], m_b1_all["mae"],
                    m_same_base["mae"], m_same_b1["mae"], n_same, tgt_changed)

    # --- Agregasi CV ---
    mean = lambda k: round(float(np.mean(agg[k])), 4)
    cv = {k: mean(k) for k in agg}
    cv["same_mae_improve"] = round(cv["same_mae_base"] - cv["same_mae_b1"], 4)
    cv["all_mae_improve"] = round(cv["base_all_mae"] - cv["b1_all_mae"], 4)
    cv["same_f1_improve"] = round(cv["same_f1_b1"] - cv["same_f1_base"], 4)
    hasil["cv"] = cv

    # --- Keputusan (pra-registrasi) ---
    imp = cv["same_mae_improve"]
    f1_drop = cv["same_f1_base"] - cv["same_f1_b1"]
    bucket_worsen = max(
        hasil["folds"]["cv1"]["same_rows"]["b1_b_gt15"] - hasil["folds"]["cv1"]["same_rows"]["base_b_gt15"],
        hasil["folds"]["cv2"]["same_rows"]["b1_b_gt15"] - hasil["folds"]["cv2"]["same_rows"]["base_b_gt15"],
        hasil["folds"]["cv1"]["same_rows"]["b1_b_0_5"] - hasil["folds"]["cv1"]["same_rows"]["base_b_0_5"],
        hasil["folds"]["cv2"]["same_rows"]["b1_b_0_5"] - hasil["folds"]["cv2"]["same_rows"]["base_b_0_5"],
        hasil["folds"]["cv1"]["same_rows"]["b1_b_6_15"] - hasil["folds"]["cv1"]["same_rows"]["base_b_6_15"],
        hasil["folds"]["cv2"]["same_rows"]["b1_b_6_15"] - hasil["folds"]["cv2"]["same_rows"]["base_b_6_15"],
    )
    if imp > 0.20:
        putus = "TERIMA B1 (perbaikan MAE baris-sama > 0.20 lap)"
    elif imp >= 0.10:
        ok = (f1_drop <= 0.005) and (bucket_worsen <= 0.30)
        putus = ("TERIMA B1 (band 0.10-0.20, syarat terpenuhi)" if ok else
                 "TOLAK B1 (band 0.10-0.20 tapi F1 turun >0.005 atau bucket memburuk >0.30)")
    else:
        putus = "TOLAK B1 (perbaikan MAE baris-sama < 0.10 lap)"
    hasil["keputusan"] = {
        "same_mae_improve": imp, "f1_drop_same": round(f1_drop, 4),
        "bucket_worsen_max": round(float(bucket_worsen), 4),
        "catatan": "Keputusan final juga mensyaratkan truncated-rebuild test LULUS (lihat b1_causal).",
        "keputusan_metric": putus,
    }

    (MODELS / "b1_cv_results.json").write_text(
        json.dumps(hasil, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 76)
    print("RINGKASAN CV B1 (mean cv1+cv2)")
    print("=" * 76)
    print(f"  SELURUH BARIS : MAE base={cv['base_all_mae']:.4f}  b1={cv['b1_all_mae']:.4f}  "
          f"(perbaikan {cv['all_mae_improve']:+.4f})")
    print(f"                  F1  base={cv['base_all_f1']:.4f}  b1={cv['b1_all_f1']:.4f}")
    print(f"  BARIS SAMA    : MAE base={cv['same_mae_base']:.4f}  b1={cv['same_mae_b1']:.4f}  "
          f"(perbaikan {cv['same_mae_improve']:+.4f})  <-- BASIS KEPUTUSAN")
    print(f"                  F1  base={cv['same_f1_base']:.4f}  b1={cv['same_f1_b1']:.4f}  "
          f"(perbaikan {cv['same_f1_improve']:+.4f})")
    print(f"  bucket >15 memburuk maks: {bucket_worsen:+.4f} | F1 drop same: {f1_drop:+.4f}")
    print(f"\n  KEPUTUSAN (metrik): {putus}")
    print("  Tersimpan: models/b1_cv_results.json")


if __name__ == "__main__":
    main()
