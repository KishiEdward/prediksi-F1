"""Pelatihan model: XGBoost (baseline), LSTM (baseline), Hybrid XGBoost-LSTM.

Jalankan dari root proyek:
    python -m app.ml.train                # latih semua model
    python -m app.ml.train --epochs 10    # epoch lebih sedikit (uji coba cepat)
    python -m app.ml.train --models xgb   # hanya XGBoost

Sesuai laporan:
- Split kronologis per musim: train 2021-2023, val 2024, test 2025.
- Preprocessor di-fit HANYA pada data train (anti-leakage).
- Hybrid: estimasi XGBoost dipakai sebagai fitur tambahan tiap timestep LSTM.
  Untuk data train, fitur XGBoost dihasilkan secara out-of-fold (GroupKFold per
  race) agar hybrid tidak melihat prediksi in-sample XGBoost yang overfit.
- Dua target: regresi laps remaining (MSE) + klasifikasi compound (CrossEntropy).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

from app.config import get_settings
from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.features import COMPOUNDS, FITUR_INRACE_NUMERIK, WINDOW_LAP, muat_dataset
from app.ml.nn import ModelLSTM
from app.ml.pipeline import Preprocessor

logger = logging.getLogger("train")

MODEL_VERSION = "0.1.0"
SEED = 42

PARAM_XGB = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
    random_state=SEED, n_jobs=-1,
)


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _grup_race(df: pd.DataFrame) -> np.ndarray:
    """ID unik per race untuk GroupKFold (satu race tidak boleh terbelah)."""
    return (df["season"].astype(int) * 100 + df["round"].astype(int)).to_numpy()


def latih_xgb(df_train: pd.DataFrame, pre: Preprocessor,
              models_dir: Path) -> tuple[XGBRegressor, XGBClassifier]:
    """Latih XGBoost regresi + klasifikasi pada seluruh data train."""
    X = ds.fitur_xgb(df_train, pre)
    y_reg = df_train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(df_train, pre)

    logger.info("Melatih XGBRegressor (%d baris)...", len(X))
    reg = XGBRegressor(objective="reg:squarederror", **PARAM_XGB)
    reg.fit(X, y_reg, verbose=False)

    logger.info("Melatih XGBClassifier (%d kelas)...", len(COMPOUNDS))
    clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **PARAM_XGB)
    clf.fit(X, y_cls, verbose=False)

    reg.save_model(models_dir / "xgb_laps.json")
    clf.save_model(models_dir / "xgb_compound.json")
    return reg, clf


def prediksi_xgb_oof(df_train: pd.DataFrame, pre: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Prediksi out-of-fold XGBoost pada data train (untuk fitur hybrid).

    GroupKFold per race: model fold tidak pernah melihat race yang diprediksi,
    sehingga fitur hybrid pada train tidak bocor.
    """
    X = ds.fitur_xgb(df_train, pre)
    y_reg = df_train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(df_train, pre)
    groups = _grup_race(df_train)

    oof_laps = np.zeros(len(X), dtype=np.float32)
    oof_proba = np.zeros((len(X), len(COMPOUNDS)), dtype=np.float32)

    gkf = GroupKFold(n_splits=5)
    for i, (tr, va) in enumerate(gkf.split(X, groups=groups), start=1):
        logger.info("  XGB OOF fold %d/5 (train=%d, holdout=%d)", i, len(tr), len(va))
        reg = XGBRegressor(objective="reg:squarederror", **PARAM_XGB)
        reg.fit(X[tr], y_reg[tr], verbose=False)
        clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **PARAM_XGB)
        clf.fit(X[tr], y_cls[tr], verbose=False)
        oof_laps[va] = reg.predict(X[va])
        oof_proba[va] = clf.predict_proba(X[va])
    return oof_laps, oof_proba


def _latih_lstm(model: ModelLSTM, X_tr: np.ndarray, yreg_tr: np.ndarray, ycls_tr: np.ndarray,
                X_va: np.ndarray, yreg_va: np.ndarray, ycls_va: np.ndarray,
                epochs: int, batch_size: int, lr: float, patience: int,
                nama: str) -> dict:
    """Loop training PyTorch (CPU) dengan early stopping pada loss validasi."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_reg_fn = nn.MSELoss()
    loss_cls_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr)
    yrt = torch.from_numpy(yreg_tr)
    yct = torch.from_numpy(ycls_tr)
    Xv = torch.from_numpy(X_va)
    yrv = torch.from_numpy(yreg_va)
    ycv = torch.from_numpy(ycls_va)

    n = len(Xt)
    terbaik = float("inf")
    state_terbaik = None
    tanpa_maju = 0
    riwayat = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            pred_laps, logits = model(Xt[idx])
            loss = loss_reg_fn(pred_laps, yrt[idx]) + loss_cls_fn(logits, yct[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss) * len(idx)
        loss_train = total_loss / n

        model.eval()
        with torch.no_grad():
            vl, lg = model(Xv)
            loss_val = float(loss_reg_fn(vl, yrv) + loss_cls_fn(lg, ycv))
        riwayat.append({"epoch": epoch, "loss_train": loss_train, "loss_val": loss_val})
        logger.info("[%s] epoch %d/%d | loss train %.4f | loss val %.4f | %.1f detik",
                    nama, epoch, epochs, loss_train, loss_val, time.time() - t0)

        if loss_val < terbaik - 1e-4:
            terbaik = loss_val
            state_terbaik = {k: v.clone() for k, v in model.state_dict().items()}
            tanpa_maju = 0
        else:
            tanpa_maju += 1
            if tanpa_maju >= patience:
                logger.info("[%s] early stop di epoch %d (val terbaik %.4f)",
                            nama, epoch, terbaik)
                break

    if state_terbaik is not None:
        model.load_state_dict(state_terbaik)
    return {"loss_val_terbaik": terbaik, "riwayat": riwayat}


def latih_nn(df_train: pd.DataFrame, df_val: pd.DataFrame, pre: Preprocessor,
             mode: str, models_dir: Path, epochs: int, hidden: int,
             batch_size: int, lr: float,
             xgb_info: dict | None = None) -> dict:
    """Latih LSTM (mode='lstm') atau Hybrid (mode='hybrid').

    Untuk hybrid, `xgb_info` berisi model XGB + cara membuat fitur XGB per df.
    """
    y_cls_tr = ds.target_dari_df(df_train, pre)
    y_cls_va = ds.target_dari_df(df_val, pre)

    if mode == "lstm":
        M_tr = ds.fitur_lstm(df_train, pre)
        M_va = ds.fitur_lstm(df_val, pre)
    elif mode == "hybrid":
        assert xgb_info is not None
        reg, clf, oof_laps, oof_proba = (xgb_info["reg"], xgb_info["clf"],
                                         xgb_info["oof_laps"], xgb_info["oof_proba"])
        M_tr = ds.fitur_hybrid(df_train, pre, oof_laps, oof_proba)
        laps_va = reg.predict(ds.fitur_xgb(df_val, pre))
        proba_va = clf.predict_proba(ds.fitur_xgb(df_val, pre))
        M_va = ds.fitur_hybrid(df_val, pre, laps_va, proba_va)
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")

    X_tr, yreg_tr, _, _ = ds.bangun_dataset_sequence(df_train, M_tr, y_cls_tr, WINDOW_LAP)
    X_va, yreg_va, _, _ = ds.bangun_dataset_sequence(df_val, M_va, y_cls_va, WINDOW_LAP)
    logger.info("[%s] sequence train %s | val %s", mode, X_tr.shape, X_va.shape)

    model = ModelLSTM(input_size=X_tr.shape[2], hidden_size=hidden, n_classes=len(COMPOUNDS))
    hasil = _latih_lstm(model, X_tr, yreg_tr, y_cls_tr, X_va, yreg_va, y_cls_va,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=5, nama=mode)

    torch.save(model.state_dict(), models_dir / f"{mode}_model.pt")
    (models_dir / f"{mode}_config.json").write_text(
        json.dumps({"arsitektur": model.config(), "window": WINDOW_LAP,
                    "training": hasil}, indent=2))
    return hasil


def main() -> None:
    setup_logging()
    _set_seed()
    s = get_settings()
    models_dir = s.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Latih model prediksi pit stop F1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--models", default="all", choices=["all", "xgb", "lstm", "hybrid"])
    args = ap.parse_args()

    logger.info("Memuat dataset...")
    df = muat_dataset()
    df = ds.urutkan_per_lap(df)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    logger.info("train=%d baris | val=%d baris | test=%d baris",
                len(df_train), len(df_val), (df["split"] == "test").sum())

    # Preprocessor di-fit HANYA pada train (anti-leakage)
    pre = Preprocessor().fit(df_train)

    xgb_info = None
    if args.models in ("all", "xgb", "hybrid"):
        reg, clf = latih_xgb(df_train, pre, models_dir)
        if args.models in ("all", "hybrid"):
            logger.info("Menghitung prediksi out-of-fold XGBoost untuk fitur hybrid...")
            oof_laps, oof_proba = prediksi_xgb_oof(df_train, pre)
            pre.fit_xgb_laps(oof_laps)
            xgb_info = {"reg": reg, "clf": clf, "oof_laps": oof_laps, "oof_proba": oof_proba}

    if args.models in ("all", "lstm"):
        latih_nn(df_train, df_val, pre, "lstm", models_dir,
                 args.epochs, args.hidden, args.batch, args.lr)

    if args.models in ("all", "hybrid"):
        if xgb_info is None:
            raise RuntimeError("Hybrid butuh XGBoost: jalankan dengan --models all/hybrid")
        latih_nn(df_train, df_val, pre, "hybrid", models_dir,
                 args.epochs, args.hidden, args.batch, args.lr, xgb_info)

    pre.simpan(models_dir / "preprocessor.pkl")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "window_lap": WINDOW_LAP,
        "compounds": COMPOUNDS,
        "train_rows": int(len(df_train)), "val_rows": int(len(df_val)),
        "test_rows": int((df["split"] == "test").sum()),
        "split": {"train": "2021-2023", "val": "2024", "test": "2025"},
        "fitur_xgb": ds.nama_fitur_xgb(),
        "fitur_hybrid": ds.nama_fitur_hybrid(),
        "fitur_lstm": FITUR_INRACE_NUMERIK,
        "hyperparameter": {"epochs": args.epochs, "hidden": args.hidden,
                           "batch": args.batch, "lr": args.lr, "xgb": PARAM_XGB},
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("SELESAI. Artifact tersimpan di %s", models_dir)


if __name__ == "__main__":
    main()
