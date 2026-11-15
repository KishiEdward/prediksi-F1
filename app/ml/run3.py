"""Run 3: framework eksperimen dengan CV kronologis (protokol selection_criteria_v3.json).

Prinsip:
- Seleksi HANYA dari CV fold: cv1 = train 2021-22 -> val 2023; cv2 = train 2021-23 -> val 2024.
- Scaler/encoder/bobot kelas fit HANYA pada train tiap fold.
- Test 2025 tidak disentuh di modul ini.
- Setiap konfigurasi dicatat ke logs/run3_experiments.jsonl (termasuk yang gagal).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

from app.config import PROJECT_ROOT
from app.ml import dataset as ds
from app.ml.evaluate import _metrik_klasifikasi, _metrik_regresi
from app.ml.features import COMPOUNDS, WINDOW_LAP, muat_dataset
from app.ml.pipeline import Preprocessor

logger = logging.getLogger("run3")

PRED_DIR = PROJECT_ROOT / "data" / "processed" / "run3_preds"
JSONL_LOG = PROJECT_ROOT / "logs" / "run3_experiments.jsonl"

FOLDS = [("cv1", [2021, 2022], [2023]), ("cv2", [2021, 2022, 2023], [2024])]

SEEDS = [42, 43, 44]
EPOCHS_LSTM = 12       # epoch tetap (tanpa early stop di fold val yang dinilai)
HIDDEN = 64
DROPOUT = 0.3
LR = 3e-4
BATCH = 256
CLIP_FITUR = 4.0       # clip fitur ter-scale untuk jalur NN


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def muat_data() -> pd.DataFrame:
    return ds.urutkan_per_lap(muat_dataset())


def bobot_kelas(y: np.ndarray, mode: str) -> np.ndarray:
    """mode: 'none' | 'inv20' (Run 2) | 'sqrt10' (lebih halus, Run 3)."""
    n, k = len(y), len(COMPOUNDS)
    counts = np.bincount(y, minlength=k).astype(float)
    if mode == "none":
        w = np.ones(k)
    elif mode == "inv20":
        w = np.where(counts > 0, n / (k * np.maximum(counts, 1)), 0.0)
        w = np.minimum(w, 20.0)
    elif mode == "sqrt10":
        w = np.where(counts > 0, np.sqrt(n / (k * np.maximum(counts, 1))), 0.0)
        w = np.minimum(w, 10.0)
    else:
        raise ValueError(mode)
    pos = w[w > 0]
    if len(pos):
        w = w / pos.mean()
    return w.astype(np.float32)


def _log_config(config: str, stage: str, hasil_fold: dict, cv: dict,
                catatan: str = "") -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "stage": stage,
           "config": config, "folds": hasil_fold, "cv": cv, "catatan": catatan}
    JSONL_LOG.parent.mkdir(exist_ok=True)
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info("[%s] CV: %s", config, json.dumps(cv))


def _ringkas_cv(hasil_fold: dict) -> dict:
    """Rata-ratakan metrik antar fold."""
    keys_reg = ["mae", "rmse", "dalam_1_lap", "dalam_3_lap", "dalam_5_lap",
                "mae_jarak_0_5", "mae_jarak_6_15", "mae_jarak_gt15"]
    keys_cls = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv = {}
    for k in keys_reg + keys_cls:
        vals = []
        for f in hasil_fold.values():
            blok = f.get("laps_remaining", {}) if k in keys_reg else f.get("compound", {})
            v = blok.get(k)
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            cv[k] = round(float(np.mean(vals)), 4)
    return cv


def _simpan_pred(nama: str, fold: str, pred_laps: np.ndarray, proba: np.ndarray,
                 ekstra: dict | None = None) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{nama}__{fold}.npz",
                        pred_laps=pred_laps, proba=proba, **(ekstra or {}))


def _muat_pred(nama: str, fold: str) -> dict:
    z = np.load(PRED_DIR / f"{nama}__{fold}.npz")
    return {k: z[k] for k in z.files}


def _y_folds(df: pd.DataFrame):
    """yield (nama_fold, df_train, df_val) sesuai protokol."""
    for nama, train_years, val_years in FOLDS:
        d_tr = df[df["season"].isin(train_years)].reset_index(drop=True)
        d_va = df[df["season"].isin(val_years)].reset_index(drop=True)
        yield nama, d_tr, d_va


def evaluasi_pred(y_reg: np.ndarray, y_cls: np.ndarray,
                  pred_laps: np.ndarray, proba: np.ndarray) -> dict:
    return {"laps_remaining": _metrik_regresi(y_reg, pred_laps),
            "compound": _metrik_klasifikasi(y_cls, proba.argmax(1))}


# ---------------------------------------------------------------------------
# Konfigurasi XGBoost (model regresi & klasifikasi TERPISAH)
# ---------------------------------------------------------------------------
PARAM_XGB_DASAR = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                       n_jobs=-1)


def run_xgb(nama: str, df: pd.DataFrame, params_reg: dict, params_clf: dict,
            target_log1p: bool, weight_mode: str, stage: str = "xgb",
            catatan: str = "") -> dict:
    hasil_fold = {}
    for fold, d_tr, d_va in _y_folds(df):
        pre = Preprocessor().fit(d_tr)
        X_tr, X_va = ds.fitur_xgb(d_tr, pre), ds.fitur_xgb(d_va, pre)
        y_reg = d_tr["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls = ds.target_dari_df(d_tr, pre)
        y_reg_va = d_va["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls_va = ds.target_dari_df(d_va, pre)

        t_reg = np.log1p(y_reg) if target_log1p else y_reg
        reg = XGBRegressor(objective="reg:squarederror", **params_reg)
        reg.fit(X_tr, t_reg, verbose=False)
        p = reg.predict(X_va)
        pred_laps = np.expm1(p) if target_log1p else p

        w = bobot_kelas(y_cls, weight_mode)[y_cls]
        clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS),
                            **params_clf)
        clf.fit(X_tr, y_cls, sample_weight=w, verbose=False)
        proba = ds.proba_compound_full(clf, X_va)

        _simpan_pred(nama, fold, pred_laps, proba)
        hasil_fold[fold] = evaluasi_pred(y_reg_va, y_cls_va, pred_laps, proba)
        logger.info("[%s] fold %s: MAE %.3f | F1 %.4f", nama, fold,
                    hasil_fold[fold]["laps_remaining"]["mae"],
                    hasil_fold[fold]["compound"]["f1_macro"])
    cv = _ringkas_cv(hasil_fold)
    _log_config(nama, stage, hasil_fold, cv, catatan)
    return cv


# ---------------------------------------------------------------------------
# Konfigurasi LSTM terpisah (regresi Huber-log1p, klasifikasi CE berbobot)
# ---------------------------------------------------------------------------
class _LSTMReg(nn.Module):
    def __init__(self, f: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(f, hidden, 1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _LSTMCls(nn.Module):
    def __init__(self, f: int, hidden: int = HIDDEN, dropout: float = DROPOUT,
                 k: int = len(COMPOUNDS)):
        super().__init__()
        self.lstm = nn.LSTM(f, hidden, 1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ReLU(), nn.Linear(32, k))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _latih_nn(model: nn.Module, X: np.ndarray, y: np.ndarray, loss_fn,
              epochs: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + ep))
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model


def run_lstm(nama: str, df: pd.DataFrame, seed: int = 42, stage: str = "lstm",
             catatan: str = "") -> dict:
    """LSTM reg & cls terpisah; target log1p terstandardisasi; loss Huber;
    fitur di-clip ±CLIP_FITUR; epoch tetap (tanpa early stop di fold val)."""
    hasil_fold = {}
    for fold, d_tr, d_va in _y_folds(df):
        pre = Preprocessor().fit(d_tr)
        y_cls_tr = ds.target_dari_df(d_tr, pre)
        y_cls_va = ds.target_dari_df(d_va, pre)
        y_reg_tr = d_tr["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_reg_va = d_va["target_laps_remaining"].to_numpy(dtype=np.float32)

        M_tr = np.clip(ds.fitur_lstm(d_tr, pre), -CLIP_FITUR, CLIP_FITUR)
        M_va = np.clip(ds.fitur_lstm(d_va, pre), -CLIP_FITUR, CLIP_FITUR)
        X_tr, _, _, _ = ds.bangun_dataset_sequence(d_tr, M_tr, y_cls_tr, WINDOW_LAP)
        X_va, _, _, _ = ds.bangun_dataset_sequence(d_va, M_va, y_cls_va, WINDOW_LAP)

        # --- regresi: Huber pada log1p terstandardisasi (statistik dari train fold)
        t_tr = np.log1p(y_reg_tr)
        mu, sd = float(t_tr.mean()), float(t_tr.std() + 1e-6)
        tz_tr = ((t_tr - mu) / sd).astype(np.float32)
        model_r = _LSTMReg(X_tr.shape[2])
        _latih_nn(model_r, X_tr, tz_tr, nn.HuberLoss(delta=1.0), EPOCHS_LSTM, seed)
        with torch.no_grad():
            pz = model_r(torch.from_numpy(X_va)).numpy()
        pred_laps = np.clip(np.expm1(pz * sd + mu), 0, None)

        # --- klasifikasi: CE berbobot sqrt10
        w = torch.from_numpy(bobot_kelas(y_cls_tr, "sqrt10"))
        model_c = _LSTMCls(X_tr.shape[2])
        _latih_nn(model_c, X_tr, y_cls_tr, nn.CrossEntropyLoss(weight=w),
                  EPOCHS_LSTM, seed)
        with torch.no_grad():
            proba = torch.softmax(model_c(torch.from_numpy(X_va)), dim=1).numpy()

        _simpan_pred(nama, fold, pred_laps, proba)
        hasil_fold[fold] = evaluasi_pred(y_reg_va, y_cls_va, pred_laps, proba)
        logger.info("[%s] fold %s: MAE %.3f | F1 %.4f", nama, fold,
                    hasil_fold[fold]["laps_remaining"]["mae"],
                    hasil_fold[fold]["compound"]["f1_macro"])
    cv = _ringkas_cv(hasil_fold)
    _log_config(nama, stage, hasil_fold, cv, catatan)
    return cv


# ---------------------------------------------------------------------------
# Kuantil XGBoost (P10/P50/P90)
# ---------------------------------------------------------------------------
def run_quantile(nama: str, df: pd.DataFrame, params: dict,
                 stage: str = "quantile", catatan: str = "") -> dict:
    hasil_fold = {}
    for fold, d_tr, d_va in _y_folds(df):
        pre = Preprocessor().fit(d_tr)
        X_tr, X_va = ds.fitur_xgb(d_tr, pre), ds.fitur_xgb(d_va, pre)
        y_tr = np.log1p(d_tr["target_laps_remaining"].to_numpy(dtype=np.float32))
        y_va = d_va["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls_va = ds.target_dari_df(d_va, pre)

        D_tr = xgb.DMatrix(X_tr, label=y_tr)
        D_va = xgb.DMatrix(X_va)
        preds = {}
        for q in (0.1, 0.5, 0.9):
            p = {**params, "objective": "reg:quantileerror", "quantile_alpha": q,
                 "tree_method": "hist", "verbosity": 0}
            bst = xgb.train(p, D_tr, num_boost_round=params.get("n_estimators", 400))
            preds[q] = np.clip(np.expm1(bst.predict(D_va)), 0, None)

        pred_laps = preds[0.5]
        coverage = float(np.mean((y_va >= preds[0.1]) & (y_va <= preds[0.9])))
        # proba dummy (kuantil hanya regresi) -> pakai classifier tersimpan bila ada
        proba = np.zeros((len(d_va), len(COMPOUNDS)), dtype=np.float32)
        _simpan_pred(nama, fold, pred_laps, proba,
                     ekstra={"p10": preds[0.1], "p90": preds[0.9]})
        m = evaluasi_pred(y_va, y_cls_va, pred_laps,
                          np.eye(len(COMPOUNDS))[np.zeros(len(d_va), int)])  # cls placeholder
        m["coverage_p10_p90"] = coverage
        m.pop("compound")
        hasil_fold[fold] = m
        logger.info("[%s] fold %s: MAE(P50) %.3f | coverage P10-P90 %.1f%%",
                    nama, fold, m["laps_remaining"]["mae"], coverage * 100)
    cv = _ringkas_cv(hasil_fold)
    cov = [f.get("coverage_p10_p90") for f in hasil_fold.values()]
    cv["coverage_p10_p90"] = round(float(np.mean(cov)), 4)
    _log_config(nama, stage, hasil_fold, cv, catatan)
    return cv


# ---------------------------------------------------------------------------
# Ensemble (rata-rata prediksi tersimpan)
# ---------------------------------------------------------------------------
def run_ensemble(nama: str, df: pd.DataFrame, anggota: list[str],
                 stage: str = "ensemble", catatan: str = "") -> dict:
    """Rata-rata prediksi regresi & probabilitas klasifikasi dari konfigurasi anggota."""
    hasil_fold = {}
    for fold, _, d_va in _y_folds(df):
        y_reg_va = d_va["target_laps_remaining"].to_numpy(dtype=np.float32)
        y_cls_va = encode_compound(d_va)
        preds, probas, nama_proba = [], [], []
        for a in anggota:
            z = _muat_pred(a, fold)
            preds.append(z["pred_laps"])
            if z["proba"].shape[1] == len(COMPOUNDS) and z["proba"].sum() > 0:
                probas.append(z["proba"])
                nama_proba.append(a)
        pred_laps = np.mean(preds, axis=0)
        proba = np.mean(probas, axis=0) if probas else \
            np.zeros((len(d_va), len(COMPOUNDS)), dtype=np.float32)
        _simpan_pred(nama, fold, pred_laps, proba)
        hasil_fold[fold] = {"laps_remaining": _metrik_regresi(y_reg_va, pred_laps)}
        if probas:
            hasil_fold[fold]["compound"] = _metrik_klasifikasi(y_cls_va, proba.argmax(1))
        logger.info("[%s] fold %s: MAE %.3f%s", nama, fold,
                    hasil_fold[fold]["laps_remaining"]["mae"],
                    f" | F1 {hasil_fold[fold]['compound']['f1_macro']:.4f}" if probas else "")
    cv = _ringkas_cv(hasil_fold)
    _log_config(nama, stage, hasil_fold, cv,
                catatan + f" | anggota={anggota} | probaDari={nama_proba if probas else '-'}")
    return cv


def encode_compound(d_va: pd.DataFrame) -> np.ndarray:
    """Encoder compound bersifat TETAP (urutan COMPOUNDS) — tidak perlu fit."""
    peta = {c: i for i, c in enumerate(COMPOUNDS)}
    return d_va["target_next_compound"].astype(str).map(peta).fillna(-1) \
        .to_numpy().astype(int)
