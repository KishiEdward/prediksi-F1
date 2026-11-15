# -*- coding: utf-8 -*-
"""Binary head "masih pit?" — model terpisah sesuai persetujuan opsi (a).

Target per baris (driver-lap):
  1 = pembalap MASIH akan pit lagi sebelum finis (ada pit stop di lap > current)
  0 = pembalap TIDAK pit lagi dan FINIS (classified)
  NaN = DNF tanpa pit berikutnya -> SENSOR (di-flag is_dnf, TIDAK dilabel 'no pit')

Fitur: sama persis dengan model utama (FITUR_XGB Bab III), tanpa fitur baru.
CV kronologis: cv1 (2021-22->2023), cv2 (2021-23->2024), grup per race.
Test 2025 dievaluasi SEKALI. Artefak: models/run3/xgb_binary.json.
"""
from __future__ import annotations

import bisect
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.ml import dataset as ds  # noqa: E402
from app.ml.features import muat_dataset_inference  # noqa: E402
from app.ml.pipeline import Preprocessor  # noqa: E402

logger = logging.getLogger("binary_head")
MODELS = PROJECT_ROOT / "models"
OUT_DIR = MODELS / "run3"
RAW = PROJECT_ROOT / "data" / "raw"
JSONL_LOG = PROJECT_ROOT / "logs" / "run3_experiments.jsonl"


def _is_classified(status: str) -> bool:
    """True jika pembalap diklasifikasikan finis (bukan DNF/retired)."""
    s = str(status).strip().lower()
    return s == "finished" or s.startswith("+")


def bangun_target_biner() -> pd.DataFrame:
    """Gabung lap_inference + pitstops + results -> kolom is_dnf, has_future_pit, binary_label."""
    inf = muat_dataset_inference()
    inf["season"] = pd.to_numeric(inf["season"], errors="coerce").astype(int)
    inf["round"] = pd.to_numeric(inf["round"], errors="coerce").astype(int)
    inf["lap_number"] = pd.to_numeric(inf["lap_number"], errors="coerce").astype(int)

    pits = pd.read_parquet(RAW / "jolpica_pitstops.parquet")
    pits["season"] = pd.to_numeric(pits["season"], errors="coerce").astype(int)
    pits["round"] = pd.to_numeric(pits["round"], errors="coerce").astype(int)
    pits["lap"] = pd.to_numeric(pits["lap"], errors="coerce").astype(int)
    pit_set = set(zip(pits["season"], pits["round"], pits["driverId"], pits["lap"]))

    res = pd.read_parquet(RAW / "jolpica_results.parquet")
    res["season"] = pd.to_numeric(res["season"], errors="coerce").astype(int)
    res["round"] = pd.to_numeric(res["round"], errors="coerce").astype(int)
    res["is_classified"] = res["status"].apply(_is_classified)
    rs = res["season"].to_numpy()
    rr = res["round"].to_numpy()
    rd = res["Driver.driverId"].to_numpy()
    rc = res["is_classified"].to_numpy()
    status_map = {(rs[i], rr[i], rd[i]): bool(rc[i]) for i in range(len(res))}

    # max pit lap per driver-race untuk cek future pit efisien
    pit_by_dr = {}
    for s, r, d, lp in pit_set:
        pit_by_dr.setdefault((s, r, d), []).append(lp)
    for k in pit_by_dr:
        pit_by_dr[k].sort()

    is_dnf = np.zeros(len(inf), dtype=bool)
    has_future_pit = np.zeros(len(inf), dtype=bool)
    seasons_arr = inf["season"].to_numpy()
    rounds_arr = inf["round"].to_numpy()
    drivers_arr = inf["driver_id"].to_numpy()
    laps_arr = inf["lap_number"].to_numpy()
    for i in range(len(inf)):
        key = (seasons_arr[i], rounds_arr[i], drivers_arr[i])
        classified = status_map.get(key, True)
        is_dnf[i] = not classified
        pits_list = pit_by_dr.get(key, [])
        idx = bisect.bisect_right(pits_list, laps_arr[i])
        has_future_pit[i] = idx < len(pits_list)

    inf["is_dnf"] = is_dnf
    inf["has_future_pit"] = has_future_pit
    # label: 1 jika future pit; 0 jika tidak & classified; NaN jika tidak & DNF (sensor)
    label = np.where(has_future_pit, 1.0,
                     np.where(~is_dnf, 0.0, np.nan))
    inf["binary_label"] = label
    return inf


def metrik_biner(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support_0": int((y_true == 0).sum()),
        "support_1": int((y_true == 1).sum()),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ValueError:
            pass
    return m


def cv_folds(df: pd.DataFrame):
    for name, train_seasons, val_season in [
        ("cv1", [2021, 2022], 2023),
        ("cv2", [2021, 2022, 2023], 2024),
    ]:
        tr = df[df["season"].isin(train_seasons)]
        va = df[df["season"] == val_season]
        yield name, tr, va


PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_lambda": 1.0, "tree_method": "hist", "n_jobs": -1,
    "random_state": 42, "eval_metric": "logloss",
}


def main() -> None:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Membangun target biner...")
    df = bangun_target_biner()
    n_total = len(df)
    n_sensor = int(df["binary_label"].isna().sum())
    n_dnf = int(df["is_dnf"].sum())
    labeled = df.dropna(subset=["binary_label"]).copy()
    labeled["binary_label"] = labeled["binary_label"].astype(int)
    pos_rate = float(labeled["binary_label"].mean())
    logger.info("total=%d sensor(DNF-no-pit)=%d is_dnf_rows=%d labeled=%d pos_rate=%.4f",
                n_total, n_sensor, n_dnf, len(labeled), pos_rate)

    hasil_fold = {}
    for fold, tr, va in cv_folds(labeled):
        va_l = va.dropna(subset=["binary_label"])
        if len(va_l) == 0:
            logger.warning("fold %s val kosong", fold)
            continue
        pre = Preprocessor().fit(tr)
        X_tr, X_va = ds.fitur_xgb(tr, pre), ds.fitur_xgb(va_l, pre)
        y_tr = tr["binary_label"].to_numpy(dtype=int)
        y_va = va_l["binary_label"].to_numpy(dtype=int)
        # bobot kelas inverse-freq sederhana (cap 10)
        cnt = np.bincount(y_tr, minlength=2).astype(float)
        w = np.where(cnt > 0, len(y_tr) / (2 * np.maximum(cnt, 1)), 1.0)
        w = np.minimum(w, 10.0); w = w / w.mean()
        sw = w[y_tr]
        clf = XGBClassifier(**PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=sw)
        prob = clf.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = metrik_biner(y_va, pred, prob)
        hasil_fold[fold] = m
        logger.info("[%s] acc=%.4f f1=%.4f auc=%.4f sup0=%d sup1=%d",
                    fold, m["accuracy"], m["f1"], m.get("roc_auc", -1),
                    m["support_0"], m["support_1"])

    cv_agg = {}
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = [f[k] for f in hasil_fold.values() if k in f]
        if vals:
            cv_agg[k] = round(float(np.mean(vals)), 4)
    logger.info("CV binary: %s", json.dumps(cv_agg))

    # Train final di 2021-2023, test SEKALI di 2025
    train_full = labeled[labeled["season"] <= 2023]
    test = labeled[labeled["season"] == 2025]
    pre_full = Preprocessor().fit(train_full)
    X_tr = ds.fitur_xgb(train_full, pre_full)
    y_tr = train_full["binary_label"].to_numpy(dtype=int)
    cnt = np.bincount(y_tr, minlength=2).astype(float)
    w = np.where(cnt > 0, len(y_tr) / (2 * np.maximum(cnt, 1)), 1.0)
    w = np.minimum(w, 10.0); w = w / w.mean()
    clf_final = XGBClassifier(**PARAMS)
    clf_final.fit(X_tr, y_tr, sample_weight=w[y_tr])
    clf_final.save_model(str(OUT_DIR / "xgb_binary.json"))

    val24 = labeled[labeled["season"] == 2024]
    hasil_split = {}
    for nama, split in [("val", val24), ("test", test)]:
        if len(split) == 0:
            continue
        X = ds.fitur_xgb(split, pre_full)
        y = split["binary_label"].to_numpy(dtype=int)
        prob = clf_final.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = metrik_biner(y, pred, prob)
        hasil_split[nama] = m
        logger.info("[binary %s] acc=%.4f f1=%.4f auc=%.4f sup0=%d sup1=%d",
                    nama, m["accuracy"], m["f1"], m.get("roc_auc", -1),
                    m["support_0"], m["support_1"])

    import pickle
    with open(OUT_DIR / "preprocessor_binary.pkl", "wb") as f:
        pickle.dump(pre_full, f)

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": "binary_head",
        "config": "B01_xgb_binary",
        "cv": cv_agg,
        "folds": hasil_fold,
        "final": hasil_split,
        "data_stats": {"total_rows": n_total, "censored_dnf_no_pit": n_sensor,
                       "is_dnf_rows": n_dnf, "labeled_rows": len(labeled),
                       "pos_rate_labeled": round(pos_rate, 4)},
        "catatan": ("Binary head 'masih pit?'. DNF tanpa pit berikutnya = sensor "
                    "(label NaN, di-flag is_dnf, TIDAK dilabel 0). Fitur sama "
                    "dengan model utama (Bab III), tanpa fitur baru."),
    }
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    with open(OUT_DIR / "metrics_binary.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    logger.info("SELESAI. Artefak binary di %s", OUT_DIR)


if __name__ == "__main__":
    main()
