"""Pelatihan model: baseline, XGBoost, LSTM, dan Hybrid XGBoost-LSTM.

Jalankan dari root proyek:
    python -m app.ml.train                        # main run, seed 42,43,44
    python -m app.ml.train --ablation-no2021      # analisis tanpa musim 2021
    python -m app.ml.train --resume               # lanjut dari checkpoint
    python -m app.ml.train --seeds 42             # satu seed saja (debug)

Aturan main (disetujui pengguna, Run 2):
- HANYA train 2021-2023 untuk belajar; val 2024 untuk early stopping, class
  weight, dan pemilihan model; test 2025 TIDAK disentuh di sini.
- Class weight (inverse frequency, dibatasi maks 20) untuk CE loss NN dan
  sample_weight XGBClassifier — mengatasi ketidakseimbangan compound.
- Multi-seed (default 3): artifact per seed bersuffix _s{seed}; mean±std val
  dilaporkan; model final dipilih DENGAN KRITERIA YANG DITULIS SEBELUM RUN
  (models/selection_criteria.json).
- Hybrid: estimasi XGBoost ditempel per timestep; baris train memakai prediksi
  out-of-fold (GroupKFold per race) agar tidak bocor.
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

MODEL_VERSION = "0.2.0"
SEED_DEFAULT = [42, 43, 44]

PARAM_XGB = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
    n_jobs=-1,
)

# Kriteria pemilihan model final — DITULIS SEBELUM training dijalankan
KRITERIA_SELEKSI = {
    "ditulis_pada": None,  # diisi saat runtime
    "data": "val 2024 (test 2025 TIDAK dipakai untuk pemilihan apa pun)",
    "metrik_utama": "mean F1-makro compound antar-seed (tertinggi menang)",
    "tie_break_1": "selisih mean F1-makro < 0.010 -> MAE laps-remaining val terendah",
    "tie_break_2": "masih seri -> nomor seed terkecil",
    "artefak_per_tipe": "seed dengan F1-makro val tertinggi untuk tipe tersebut",
    "catatan": ("Class weight inverse-frequency dibatasi maksimum 20 agar kelas "
                "WET (84 baris) tidak mendominasi gradien."),
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _grup_race(df: pd.DataFrame) -> np.ndarray:
    return (df["season"].astype(int) * 100 + df["round"].astype(int)).to_numpy()


def bobot_kelas(y: np.ndarray, cap: float = 20.0) -> np.ndarray:
    """Class weight inverse-frequency, dibatasi `cap`, dinormalisasi rata-rata 1."""
    n = len(y)
    k = len(COMPOUNDS)
    counts = np.bincount(y, minlength=k).astype(float)
    w = np.where(counts > 0, n / (k * np.maximum(counts, 1.0)), 0.0)
    w = np.minimum(w, cap)
    positif = w[w > 0]
    if len(positif):
        w = w / positif.mean()
    return w.astype(np.float32)


def _setup_file_log(nama: str) -> Path:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"train_{nama}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    return path


# ---------------------------------------------------------------------------
# XGBoost (dengan sample_weight klasifikasi)
# ---------------------------------------------------------------------------
def latih_xgb(df_train: pd.DataFrame, pre: Preprocessor, models_dir: Path,
              suffix: str, seed: int) -> tuple[XGBRegressor, XGBClassifier]:
    X = ds.fitur_xgb(df_train, pre)
    y_reg = df_train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(df_train, pre)
    w_cls = bobot_kelas(y_cls)[y_cls]

    param = {**PARAM_XGB, "random_state": seed}
    logger.info("Melatih XGBRegressor%s (seed=%d, %d baris)...", suffix, seed, len(X))
    reg = XGBRegressor(objective="reg:squarederror", **param)
    reg.fit(X, y_reg, verbose=False)

    logger.info("Melatih XGBClassifier%s (class-weighted)...", suffix)
    clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **param)
    clf.fit(X, y_cls, sample_weight=w_cls, verbose=False)

    reg.save_model(models_dir / f"xgb_laps{suffix}.json")
    clf.save_model(models_dir / f"xgb_compound{suffix}.json")
    return reg, clf


def prediksi_xgb_oof(df_train: pd.DataFrame, pre: Preprocessor,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Prediksi out-of-fold XGBoost pada train (fitur hybrid, anti-leakage)."""
    X = ds.fitur_xgb(df_train, pre)
    y_reg = df_train["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(df_train, pre)
    w_cls = bobot_kelas(y_cls)
    groups = _grup_race(df_train)

    oof_laps = np.zeros(len(X), dtype=np.float32)
    oof_proba = np.zeros((len(X), len(COMPOUNDS)), dtype=np.float32)

    param = {**PARAM_XGB, "random_state": seed}
    gkf = GroupKFold(n_splits=5)
    for i, (tr, va) in enumerate(gkf.split(X, groups=groups), start=1):
        logger.info("  XGB OOF fold %d/5 (train=%d, holdout=%d)", i, len(tr), len(va))
        reg = XGBRegressor(objective="reg:squarederror", **param)
        reg.fit(X[tr], y_reg[tr], verbose=False)
        clf = XGBClassifier(objective="multi:softprob", num_class=len(COMPOUNDS), **param)
        clf.fit(X[tr], y_cls[tr], sample_weight=w_cls[y_cls[tr]], verbose=False)
        oof_laps[va] = reg.predict(X[va])
        oof_proba[va] = ds.proba_compound_full(clf, X[va])
    return oof_laps, oof_proba


# ---------------------------------------------------------------------------
# Neural network dengan checkpoint/resume + weighted CE
# ---------------------------------------------------------------------------
def _latih_lstm(model: ModelLSTM, X_tr: np.ndarray, yreg_tr: np.ndarray, ycls_tr: np.ndarray,
                X_va: np.ndarray, yreg_va: np.ndarray, ycls_va: np.ndarray,
                epochs: int, batch_size: int, lr: float, patience: int, seed: int,
                nama: str, models_dir: Path, resume: bool = False) -> dict:
    """Training PyTorch (CPU): MSE + CrossEntropy berbobot kelas, early stopping
    pada val loss, checkpoint tiap epoch (bisa --resume)."""
    ckpt_path = models_dir / f"{nama}_checkpoint.pt"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_reg_fn = nn.MSELoss()
    w_tensor = torch.from_numpy(bobot_kelas(ycls_tr))
    loss_cls_fn = nn.CrossEntropyLoss(weight=w_tensor)
    logger.info("[%s] class weight CE: %s", nama,
                {c: round(float(w), 2) for c, w in zip(COMPOUNDS, w_tensor)})

    epoch_mulai, terbaik, tanpa_maju, riwayat = 1, float("inf"), 0, []
    state_terbaik = None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        epoch_mulai = ck["epoch"] + 1
        terbaik, tanpa_maju, riwayat = ck["terbaik"], ck["tanpa_maju"], ck["riwayat"]
        state_terbaik = ck.get("state_terbaik")
        logger.info("[%s] resume dari checkpoint epoch %d", nama, ck["epoch"])

    Xt, yrt, yct = torch.from_numpy(X_tr), torch.from_numpy(yreg_tr), torch.from_numpy(ycls_tr)
    Xv, yrv, ycv = torch.from_numpy(X_va), torch.from_numpy(yreg_va), torch.from_numpy(ycls_va)

    n = len(Xt)
    for epoch in range(epoch_mulai, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch))
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
        logger.info("[%s] epoch %d/%d | loss train %.4f | loss val %.4f | %.1f dtk",
                    nama, epoch, epochs, loss_train, loss_val, time.time() - t0)

        if loss_val < terbaik - 1e-4:
            terbaik = loss_val
            state_terbaik = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(state_terbaik, models_dir / f"{nama}_model.pt")
            tanpa_maju = 0
        else:
            tanpa_maju += 1

        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "terbaik": terbaik, "tanpa_maju": tanpa_maju, "riwayat": riwayat,
                    "state_terbaik": state_terbaik}, ckpt_path)

        if tanpa_maju >= patience:
            logger.info("[%s] early stop epoch %d (val terbaik %.4f)", nama, epoch, terbaik)
            break

    if state_terbaik is not None:
        model.load_state_dict(state_terbaik)
        torch.save(state_terbaik, models_dir / f"{nama}_model.pt")
    (models_dir / f"{nama}_config.json").write_text(json.dumps(
        {"arsitektur": model.config(), "window": WINDOW_LAP, "seed": seed,
         "loss_val_terbaik": terbaik, "riwayat": riwayat}, indent=2))
    return {"loss_val_terbaik": terbaik, "riwayat": riwayat}


def latih_nn(df_train: pd.DataFrame, df_val: pd.DataFrame, pre: Preprocessor,
             mode: str, models_dir: Path, args: argparse.Namespace, seed: int,
             suffix: str, xgb_info: dict | None = None) -> dict:
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
                               ds.proba_compound_full(clf, X_va_flat))
    else:
        raise ValueError(f"mode tidak dikenal: {mode}")

    X_tr, yreg_tr, _, _ = ds.bangun_dataset_sequence(df_train, M_tr, y_cls_tr, WINDOW_LAP)
    X_va, yreg_va, _, _ = ds.bangun_dataset_sequence(df_val, M_va, y_cls_va, WINDOW_LAP)
    logger.info("[%s%s] sequence train %s | val %s", mode, suffix, X_tr.shape, X_va.shape)

    nama = f"{mode}{suffix}"
    model = ModelLSTM(input_size=X_tr.shape[2], hidden_size=args.hidden,
                      dropout=args.dropout, n_classes=len(COMPOUNDS))
    return _latih_lstm(model, X_tr, yreg_tr, y_cls_tr, X_va, yreg_va, y_cls_va,
                       epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                       patience=args.patience, seed=seed, nama=nama,
                       models_dir=models_dir, resume=args.resume)


# ---------------------------------------------------------------------------
# Satu seed penuh: XGB + LSTM + Hybrid + metrik val
# ---------------------------------------------------------------------------
def jalankan_seed(df_train: pd.DataFrame, df_val: pd.DataFrame, models_dir: Path,
                  args: argparse.Namespace, seed: int, suffix: str) -> dict:
    """Latih semua model untuk satu seed; kembalikan metrik val per tipe."""
    from app.ml.evaluate import evaluasi_satu_split  # lokal: hindari import cycle

    _set_seed(seed)
    pre = Preprocessor().fit(df_train)

    reg, clf = latih_xgb(df_train, pre, models_dir, suffix, seed)
    oof_laps, oof_proba = prediksi_xgb_oof(df_train, pre, seed)
    pre.fit_xgb_laps(oof_laps)
    xgb_info = {"reg": reg, "clf": clf, "oof_laps": oof_laps, "oof_proba": oof_proba}

    latih_nn(df_train, df_val, pre, "lstm", models_dir, args, seed, suffix)
    latih_nn(df_train, df_val, pre, "hybrid", models_dir, args, seed, suffix, xgb_info)

    pre.simpan(models_dir / f"preprocessor{suffix}.pkl")

    hasil_val = evaluasi_satu_split(df_val, models_dir, suffix_map={
        "xgboost": suffix, "lstm": suffix, "hybrid": suffix,
        "pre_lstm": suffix, "pre_hybrid": suffix, "pre_xgb": suffix,
    }, dengan_baseline=False)
    return {tipe: hasil_val[tipe] for tipe in ("xgboost", "lstm", "hybrid")
            if tipe in hasil_val}


def ringkas_val(hasil_per_seed: dict[int, dict]) -> dict:
    """mean ± std metrik val antar-seed untuk tiap tipe model."""
    ringkas = {}
    tipe_list = ["xgboost", "lstm", "hybrid"]
    for tipe in tipe_list:
        f1s, maes, accs = [], [], []
        for seed, h in hasil_per_seed.items():
            if tipe not in h:
                continue
            f1s.append(h[tipe]["compound"]["f1_macro"])
            maes.append(h[tipe]["laps_remaining"]["mae"])
            accs.append(h[tipe]["compound"]["accuracy"])
        if not f1s:
            continue
        ringkas[tipe] = {
            "f1_makro_mean": round(float(np.mean(f1s)), 4),
            "f1_makro_std": round(float(np.std(f1s)), 4),
            "mae_mean": round(float(np.mean(maes)), 4),
            "mae_std": round(float(np.std(maes)), 4),
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs)), 4),
            "per_seed": {str(s): {"f1_makro": h[tipe]["compound"]["f1_macro"],
                                  "mae": h[tipe]["laps_remaining"]["mae"]}
                         for s, h in hasil_per_seed.items() if tipe in h},
        }
    return ringkas


def pilih_model(ringkas: dict) -> tuple[str, dict]:
    """Terapkan KRITERIA_SELEKSI pada ringkasan val -> (tipe utama, artefak per tipe)."""
    tipe_urut = sorted(ringkas.items(),
                       key=lambda kv: (-kv[1]["f1_makro_mean"], kv[1]["mae_mean"]))
    terbaik = tipe_urut[0]
    if len(tipe_urut) > 1:
        runner = tipe_urut[1]
        if abs(terbaik[1]["f1_makro_mean"] - runner[1]["f1_makro_mean"]) < 0.010:
            # tie-break: MAE terendah
            kandidat = [terbaik, runner]
            terbaik = min(kandidat, key=lambda kv: kv[1]["mae_mean"])
    primary = terbaik[0]
    artifacts = {}
    for tipe, r in ringkas.items():
        seed_terbaik = min(r["per_seed"].items(),
                           key=lambda kv: (-kv[1]["f1_makro"], kv[1]["mae"], int(kv[0])))
        artifacts[tipe] = f"_s{seed_terbaik[0]}"
    return primary, artifacts


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Latih model prediksi pit stop F1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEED_DEFAULT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ablation-no2021", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    ablation = args.ablation_no2021
    nama_run = "ablation_no2021" if ablation else "main"
    log_path = _setup_file_log(nama_run)
    logger.info("Log training: %s", log_path)

    s = get_settings()
    models_dir = s.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Kriteria seleksi DITULIS SEBELUM training (hanya untuk main run)
    if not ablation:
        kriteria = {**KRITERIA_SELEKSI,
                    "ditulis_pada": datetime.now(timezone.utc).isoformat()}
        (models_dir / "selection_criteria.json").write_text(json.dumps(kriteria, indent=2))
        logger.info("Kriteria seleksi ditulis SEBELUM training: %s", json.dumps(kriteria))

    logger.info("Memuat dataset...")
    df = ds.urutkan_per_lap(muat_dataset())
    df_train_all = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    if ablation:
        df_train_all = df_train_all[df_train_all["season"] != 2021].reset_index(drop=True)
        logger.info("ABLATION: musim 2021 dibuang dari train")

    # Baseline deterministik (sekali, hanya main run)
    if not ablation:
        mayor = BaselineMayoritas().fit(df_train_all)
        tyre = BaselineTyreLife().fit(df_train_all)
        simpan_baselines(models_dir / "baselines.json", mayor, tyre)
        logger.info("Baseline disimpan: mayoritas=%s stint_median=%s",
                    mayor.compound_mayoritas, tyre.stint_median)

    suffix_base = "_no2021" if ablation else ""
    hasil_per_seed: dict[int, dict] = {}
    for seed in seeds:
        suffix = f"{suffix_base}_s{seed}"
        logger.info("===== SEED %d (suffix %s) | train=%d baris =====",
                    seed, suffix, len(df_train_all))
        hasil_per_seed[seed] = jalankan_seed(df_train_all, df_val, models_dir,
                                             args, seed, suffix)

    ringkas = ringkas_val(hasil_per_seed)
    out_name = "multiseed_val_no2021.json" if ablation else "multiseed_val.json"
    (models_dir / out_name).write_text(json.dumps(
        {"run": nama_run, "seeds": seeds, "val": ringkas,
         "hyperparameter": vars(args)}, indent=2))
    logger.info("Ringkasan val multi-seed: %s", json.dumps(ringkas, indent=2))

    if not ablation:
        primary, artifacts = pilih_model(ringkas)
        selection = {
            "primary_model": primary,
            "artifacts": artifacts,
            "kriteria": KRITERIA_SELEKSI,
            "val_ringkas": ringkas,
            "seeds": seeds,
            "dipilih_pada": datetime.now(timezone.utc).isoformat(),
        }
        (models_dir / "selection.json").write_text(json.dumps(selection, indent=2))
        logger.info("MODEL FINAL (berdasarkan val): %s | artefak: %s", primary, artifacts)

        metadata = {
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "run": nama_run, "seeds": seeds, "seed_default": seeds[0],
            "log_file": str(log_path),
            "primary_model": primary, "artifacts": artifacts,
            "window_lap": WINDOW_LAP, "compounds": COMPOUNDS,
            "train_rows": int(len(df_train_all)), "val_rows": int(len(df_val)),
            "test_rows": int((df["split"] == "test").sum()),
            "split": {"train": "2021-2023", "val": "2024", "test": "2025"},
            "fitur_xgb": ds.nama_fitur_xgb(),
            "fitur_hybrid": ds.nama_fitur_hybrid(),
            "fitur_lstm": FITUR_INRACE_NUMERIK,
            "hyperparameter": vars(args) | {"xgb": PARAM_XGB},
            "class_weight_cap": 20.0,
        }
        (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("SELESAI. Artifact di %s", models_dir)


if __name__ == "__main__":
    main()
