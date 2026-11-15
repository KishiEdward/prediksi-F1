"""Pelatihan model: baseline, XGBoost, LSTM, dan Hybrid XGBoost-LSTM.

Jalankan dari root proyek:
    python -m app.ml.train                      # latih semua (main run)
    python -m app.ml.train --ablation-no2021    # analisis: train tanpa musim 2021
    python -m app.ml.train --resume             # lanjut dari checkpoint terakhir
    python -m app.ml.train --epochs 5           # uji coba cepat

Aturan main (disetujui pengguna):
- HANYA train 2021-2023 untuk belajar; val 2024 untuk early stopping/tuning;
  test 2025 TIDAK disentuh di sini (dievaluasi sekali di app/ml/evaluate.py).
- Preprocessor di-fit HANYA pada data train (anti-leakage).
- Hybrid: estimasi XGBoost ditempel per timestep; untuk baris train dipakai
  prediksi out-of-fold (GroupKFold per race) agar tidak bocor.
- Seed, log training, dan checkpoint disimpan agar bisa direproduksi/dilanjutkan.
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

from app.config import PROJECT_ROOT, get_settings
from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.baselines import BaselineMayoritas, BaselineTyreLife, simpan_baselines
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


def _setup_file_log(nama: str) -> Path:
    """Tambahkan handler file agar seluruh log training tersimpan."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"train_{nama}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    return path


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def latih_xgb(df_train: pd.DataFrame, pre: Preprocessor, models_dir: Path,
              suffix: str = "") -> tuple[XGBRegressor, XGBClassifier]:
    """Latih XGBoost regresi + klasifikasi pada seluruh data train."""
    X = ds.fitur_xgb(df_train, pre)
    y_reg = df_train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(df_train, pre)

    logger.info("Melatih XGBRegressor%s (%d baris)...", suffix, len(X))
    reg = XGBRegressor(objective="reg:squarederror", **PARAM_XGB)
    reg.fit(X, y_reg, verbose=False)

    logger.info("Melatih XGBClassifier%s (%d kelas)...", suffix, len(COMPOUNDS))
    clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **PARAM_XGB)
    clf.fit(X, y_cls, verbose=False)

    reg.save_model(models_dir / f"xgb_laps{suffix}.json")
    clf.save_model(models_dir / f"xgb_compound{suffix}.json")
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


# ---------------------------------------------------------------------------
# Neural network (LSTM & Hybrid) dengan checkpoint/resume
# ---------------------------------------------------------------------------
def _latih_lstm(model: ModelLSTM, X_tr: np.ndarray, yreg_tr: np.ndarray, ycls_tr: np.ndarray,
                X_va: np.ndarray, yreg_va: np.ndarray, ycls_va: np.ndarray,
                epochs: int, batch_size: int, lr: float, patience: int,
                nama: str, models_dir: Path, resume: bool = False) -> dict:
    """Loop training PyTorch (CPU) dengan early stopping + checkpoint.

    Checkpoint (epoch, bobot, optimizer, state early-stop) disimpan tiap epoch
    ke models/{nama}_checkpoint.pt sehingga training bisa dilanjutkan (--resume).
    Bobot terbaik (val loss terendah) disimpan ke models/{nama}_model.pt.
    """
    ckpt_path = models_dir / f"{nama}_checkpoint.pt"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_reg_fn = nn.MSELoss()
    loss_cls_fn = nn.CrossEntropyLoss()

    epoch_mulai, terbaik, tanpa_maju, riwayat = 1, float("inf"), 0, []
    state_terbaik = None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        epoch_mulai = ck["epoch"] + 1
        terbaik, tanpa_maju, riwayat = ck["terbaik"], ck["tanpa_maju"], ck["riwayat"]
        state_terbaik = ck.get("state_terbaik")
        logger.info("[%s] resume dari checkpoint epoch %d (val terbaik %.4f)",
                    nama, ck["epoch"], terbaik)

    Xt = torch.from_numpy(X_tr)
    yrt = torch.from_numpy(yreg_tr)
    yct = torch.from_numpy(ycls_tr)
    Xv = torch.from_numpy(X_va)
    yrv = torch.from_numpy(yreg_va)
    ycv = torch.from_numpy(ycls_va)

    n = len(Xt)
    for epoch in range(epoch_mulai, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
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
            torch.save(state_terbaik, models_dir / f"{nama}_model.pt")
            tanpa_maju = 0
        else:
            tanpa_maju += 1

        # checkpoint lengkap untuk resume
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "terbaik": terbaik, "tanpa_maju": tanpa_maju, "riwayat": riwayat,
                    "state_terbaik": state_terbaik}, ckpt_path)

        if tanpa_maju >= patience:
            logger.info("[%s] early stop di epoch %d (val terbaik %.4f)", nama, epoch, terbaik)
            break

    if state_terbaik is not None:
        model.load_state_dict(state_terbaik)
        torch.save(state_terbaik, models_dir / f"{nama}_model.pt")
    (models_dir / f"{nama}_config.json").write_text(json.dumps(
        {"arsitektur": model.config(), "window": WINDOW_LAP,
         "loss_val_terbaik": terbaik, "riwayat": riwayat}, indent=2))
    return {"loss_val_terbaik": terbaik, "riwayat": riwayat}


def latih_nn(df_train: pd.DataFrame, df_val: pd.DataFrame, pre: Preprocessor,
             mode: str, models_dir: Path, epochs: int, hidden: int,
             batch_size: int, lr: float, resume: bool = False,
             xgb_info: dict | None = None, suffix: str = "") -> dict:
    """Latih LSTM (mode='lstm') atau Hybrid (mode='hybrid')."""
    y_cls_tr = ds.target_dari_df(df_train, pre)
    y_cls_va = ds.target_dari_df(df_val, pre)

    if mode == "lstm":
        M_tr = ds.fitur_lstm(df_train, pre)
        M_va = ds.fitur_lstm(df_val, pre)
    elif mode == "hybrid":
        assert xgb_info is not None
        reg, clf = xgb_info["reg"], xgb_info["clf"]
        M_tr = ds.fitur_hybrid(df_train, pre, xgb_info["oof_laps"], xgb_info["oof_proba"])
        X_va_flat = ds.fitur_xgb(df_val, pre)
        M_va = ds.fitur_hybrid(df_val, pre, reg.predict(X_va_flat),
                               clf.predict_proba(X_va_flat))
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")

    X_tr, yreg_tr, _, _ = ds.bangun_dataset_sequence(df_train, M_tr, y_cls_tr, WINDOW_LAP)
    X_va, yreg_va, _, _ = ds.bangun_dataset_sequence(df_val, M_va, y_cls_va, WINDOW_LAP)
    logger.info("[%s%s] sequence train %s | val %s", mode, suffix, X_tr.shape, X_va.shape)

    nama = f"{mode}{suffix}"
    model = ModelLSTM(input_size=X_tr.shape[2], hidden_size=hidden, n_classes=len(COMPOUNDS))
    return _latih_lstm(model, X_tr, yreg_tr, y_cls_tr, X_va, yreg_va, y_cls_va,
                       epochs=epochs, batch_size=batch_size, lr=lr, patience=5,
                       nama=nama, models_dir=models_dir, resume=resume)


# ---------------------------------------------------------------------------
# Pipeline satu run (main / ablation)
# ---------------------------------------------------------------------------
def jalankan(df_train: pd.DataFrame, df_val: pd.DataFrame, models_dir: Path,
             args: argparse.Namespace, suffix: str = "") -> dict:
    """Latih seluruh model untuk satu konfigurasi data train."""
    _set_seed()
    pre_path = models_dir / f"preprocessor{suffix}.pkl"
    pre = Preprocessor().fit(df_train)

    # Baseline (hanya untuk run utama)
    if suffix == "":
        mayor = BaselineMayoritas().fit(df_train)
        tyre = BaselineTyreLife().fit(df_train)
        simpan_baselines(models_dir / "baselines.json", mayor, tyre)
        logger.info("Baseline disimpan: mayoritas=%s stint_median=%s",
                    mayor.compound_mayoritas, tyre.stint_median)

    logger.info("Melatih XGBoost%s...", suffix)
    reg, clf = latih_xgb(df_train, pre, models_dir, suffix)

    logger.info("Menghitung prediksi out-of-fold XGBoost%s untuk fitur hybrid...", suffix)
    oof_laps, oof_proba = prediksi_xgb_oof(df_train, pre)
    pre.fit_xgb_laps(oof_laps)
    xgb_info = {"reg": reg, "clf": clf, "oof_laps": oof_laps, "oof_proba": oof_proba}

    logger.info("Melatih LSTM%s...", suffix)
    latih_nn(df_train, df_val, pre, "lstm", models_dir, args.epochs, args.hidden,
             args.batch, args.lr, resume=args.resume, suffix=suffix)

    logger.info("Melatih Hybrid%s...", suffix)
    latih_nn(df_train, df_val, pre, "hybrid", models_dir, args.epochs, args.hidden,
             args.batch, args.lr, resume=args.resume, xgb_info=xgb_info, suffix=suffix)

    pre.simpan(pre_path)
    return {"train_rows": int(len(df_train)), "val_rows": int(len(df_val))}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Latih model prediksi pit stop F1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true",
                    help="lanjutkan NN dari checkpoint terakhir")
    ap.add_argument("--ablation-no2021", action="store_true",
                    help="latih varian tanpa musim 2021 (analisis), artifact bersuffix _no2021")
    args = ap.parse_args()

    nama_run = "ablation_no2021" if args.ablation_no2021 else "main"
    log_path = _setup_file_log(nama_run)
    logger.info("Log training: %s", log_path)

    s = get_settings()
    models_dir = s.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Memuat dataset...")
    df = ds.urutkan_per_lap(muat_dataset())
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    suffix = ""
    if args.ablation_no2021:
        df_train = df_train[df_train["season"] != 2021].reset_index(drop=True)
        suffix = "_no2021"
        logger.info("ABLATION: musim 2021 dibuang dari train")

    logger.info("train=%d baris (%s) | val=%d baris",
                len(df_train),
                sorted(df_train['season'].unique().tolist()), len(df_val))

    info = jalankan(df_train, df_val, models_dir, args, suffix)

    # Metadata (jangan menimpa metadata utama saat ablation)
    meta_path = models_dir / f"metadata{suffix}.json"
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "run": nama_run,
        "log_file": str(log_path),
        "window_lap": WINDOW_LAP,
        "compounds": COMPOUNDS,
        "train_rows": info["train_rows"], "val_rows": info["val_rows"],
        "test_rows": int((df["split"] == "test").sum()),
        "split": {"train": "2022-2023" if suffix else "2021-2023",
                  "val": "2024", "test": "2025"},
        "fitur_xgb": ds.nama_fitur_xgb(),
        "fitur_hybrid": ds.nama_fitur_hybrid(),
        "fitur_lstm": FITUR_INRACE_NUMERIK,
        "hyperparameter": {"epochs": args.epochs, "hidden": args.hidden,
                           "batch": args.batch, "lr": args.lr, "xgb": PARAM_XGB},
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("SELESAI. Artifact tersimpan di %s (metadata: %s)", models_dir, meta_path.name)


if __name__ == "__main__":
    main()
