# -*- coding: utf-8 -*-
"""Helper bersama untuk skrip audit (READ-ONLY: tidak melatih/mengubah apa pun).

Menyediakan: pemuatan dataset, model Run 3 (untuk inference), dan pemetaan
status lintasan dari file mentah (untuk analisis ground truth).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml import dataset as ds  # noqa: E402
from app.ml.features import COMPOUNDS  # noqa: E402
from app.ml.pipeline import Preprocessor  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
RUN3 = MODELS / "run3"


def muat_inference() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "processed" / "lap_inference.parquet")
    for c in ("season", "round", "lap_number"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    return df


def muat_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "processed" / "lap_dataset.parquet")
    for c in ("season", "round", "lap_number"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    return df


def muat_model_run3():
    """Muat reg/clf/binary + preprocessor Run 3 untuk INFERENCE (bukan training)."""
    reg = XGBRegressor(); reg.load_model(RUN3 / "xgb_reg.json")
    clf = XGBClassifier(); clf.load_model(RUN3 / "xgb_clf.json")
    pre = Preprocessor.muat(RUN3 / "preprocessor.pkl")
    binary = None
    if (RUN3 / "xgb_binary.json").exists():
        binary = XGBClassifier(); binary.load_model(RUN3 / "xgb_binary.json")
    pre_bin = None
    if (RUN3 / "preprocessor_binary.pkl").exists():
        import pickle
        with open(RUN3 / "preprocessor_binary.pkl", "rb") as f:
            pre_bin = pickle.load(f)
    conformal = json.loads((RUN3 / "conformal.json").read_text(encoding="utf-8"))
    return reg, clf, binary, pre, pre_bin, conformal


def prediksi_run3(df: pd.DataFrame, reg, clf, pre, binary=None, pre_bin=None):
    """Inference per-baris. Return (pred_laps, proba[5], pred_cls, p_biner|None)."""
    X = ds.fitur_xgb(df, pre)
    pred_laps = np.clip(reg.predict(X), 0, None)
    proba = ds.proba_compound_full(clf, X)
    pred_cls = proba.argmax(1)
    p_bin = None
    if binary is not None and pre_bin is not None:
        Xb = ds.fitur_xgb(df, pre_bin)
        p_bin = binary.predict_proba(Xb)[:, 1]
    return pred_laps, proba, pred_cls, p_bin


def status_lintasan_per_lap() -> pd.DataFrame:
    """Status lintasan per (season,round,lap) dari file laps MENTAH (tanpa filter).

    Mengembalikan kolom: sc, vsc, red, yellow, green (bool) berdasarkan APAKAH
    kode status muncul DI MANA PUN dalam string TrackStatus multi-kode.
    Ini sumber kebenaran untuk analisis ground truth (bukan fitur model).
    """
    rows = []
    for f in sorted(glob.glob(str(DATA / "raw" / "fastf1" / "*_laps.parquet"))):
        name = Path(f).stem  # e.g. 2021_01_laps
        try:
            s, r = name.split("_")[0], name.split("_")[1]
            L = pd.read_parquet(f, columns=["LapNumber", "TrackStatus"])
        except Exception:
            continue
        L = L.dropna(subset=["LapNumber"])
        ln = pd.to_numeric(L["LapNumber"], errors="coerce")
        ts = L["TrackStatus"].astype("string").fillna("")
        g = pd.DataFrame({"season": int(s), "round": int(r),
                          "lap": ln.astype("Int64"), "ts": ts}).dropna(subset=["lap"])
        g["lap"] = g["lap"].astype(int)
        agg = g.groupby(["season", "round", "lap"])["ts"].agg(
            sc=lambda x: x.str.contains("4").any(),
            vsc=lambda x: x.str.contains("6").any(),
            red=lambda x: x.str.contains("5").any(),
            yellow=lambda x: x.str.contains("2").any(),
        ).reset_index()
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "round", "lap", "sc", "vsc", "red", "yellow"])


def muat_pitstops() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "raw" / "jolpica_pitstops.parquet")
    p["season"] = pd.to_numeric(p["season"], errors="coerce").astype(int)
    p["round"] = pd.to_numeric(p["round"], errors="coerce").astype(int)
    p["lap"] = pd.to_numeric(p["lap"], errors="coerce")
    return p.dropna(subset=["lap"]).copy()
