"""Evaluasi model — metrik sesuai laporan (subbab 2.2.9) + permintaan pengguna.

Jalankan SETELAH training multi-seed selesai:
    python -m app.ml.evaluate

Perilaku:
- Run 2 (model hasil training class-weighted, artefak dipilih via selection.json)
  dievaluasi pada val 2024 dan test 2025.
- Run 1 (backup models/run1, tanpa class weight) dievaluasi ulang dengan metrik
  yang sama agar perbandingan adil — tidak ada keputusan apa pun dari test.
- Metrik regresi: MAE, RMSE, hit rate ±1/±3/±5 lap, MAE per jarak ke pit
  (0-5, 6-15, >15 lap).
- Metrik klasifikasi: accuracy, precision/recall/F1 makro + per kelas,
  confusion matrix.
- Semua hasil disimpan: models/metrics.json (Run 2) dan
  models/metrics_run1_extended.json (Run 1).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from xgboost import XGBClassifier, XGBRegressor

from app.config import get_settings
from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.baselines import muat_baselines
from app.ml.features import COMPOUNDS, WINDOW_LAP, muat_dataset
from app.ml.nn import muat_model
from app.ml.pipeline import Preprocessor

logger = logging.getLogger("evaluate")


def _metrik_regresi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/RMSE + hit rate ±1/±3/±5 lap + MAE per jarak ke pit (bucket y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    err = np.abs(y_true - y_pred_clip)

    def mae(mask: np.ndarray) -> float | None:
        return round(float(err[mask].mean()), 4) if mask.any() else None

    return {
        "mae": round(float(err.mean()), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred_clip))), 4),
        "dalam_1_lap": round(float(np.mean(err <= 1.0)), 4),
        "dalam_3_lap": round(float(np.mean(err <= 3.0)), 4),
        "dalam_5_lap": round(float(np.mean(err <= 5.0)), 4),
        "mae_jarak_0_5": mae(y_true <= 5),
        "mae_jarak_6_15": mae((y_true > 5) & (y_true <= 15)),
        "mae_jarak_gt15": mae(y_true > 15),
    }


def _metrik_klasifikasi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(COMPOUNDS)))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro",
                                                       zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro",
                                                 zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "per_kelas": {
            c: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
            for c, p, r, f, s in zip(COMPOUNDS, prec, rec, f1, supp)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _cetak_confusion(cm: list[list[int]], judul: str) -> None:
    print(f"  Confusion matrix ({judul}) — baris=aktual, kolom=prediksi:")
    print("        " + "".join(f"{c[:5]:>7}" for c in COMPOUNDS))
    for nama, baris in zip(COMPOUNDS, cm):
        print(f"{nama[:6]:<8}" + "".join(f"{v:>7}" for v in baris))


def _prediksi_nn(mode: str, suffix: str, md: Path, d: pd.DataFrame,
                 pre: Preprocessor, xgb_reg=None, xgb_clf=None):
    if mode == "lstm":
        M = ds.fitur_lstm(d, pre)
    else:
        X_flat = ds.fitur_xgb(d, pre)
        M = ds.fitur_hybrid(d, pre, xgb_reg.predict(X_flat),
                            ds.proba_compound_full(xgb_clf, X_flat))
    config = json.loads((md / f"{mode}{suffix}_config.json").read_text())
    model = muat_model(md / f"{mode}{suffix}_model.pt", config["arsitektur"])
    y_cls = ds.target_dari_df(d, pre)
    X, _, _, _ = ds.bangun_dataset_sequence(d, M, y_cls, WINDOW_LAP)
    with torch.no_grad():
        pred_laps, logits = model(torch.from_numpy(X))
        proba = torch.softmax(logits, dim=1).numpy()
    return pred_laps.numpy(), proba


def evaluasi_satu_split(d: pd.DataFrame, md: Path, suffix: str = "",
                        suffix_map: dict | None = None,
                        dengan_baseline: bool = True) -> dict:
    """Evaluasi baseline + XGB + LSTM + Hybrid pada satu split.

    suffix_map memungkinkan tiap tipe model memakai artefak seed berbeda
    (mis. {"xgboost": "_s42", "lstm": "_s43", "hybrid": "_s42",
           "pre_xgb": "_s42", "pre_lstm": "_s43", "pre_hybrid": "_s42"}).
    """
    sm = suffix_map or {k: suffix for k in
                        ("xgboost", "lstm", "hybrid", "pre_xgb", "pre_lstm", "pre_hybrid")}
    hasil: dict = {"n_rows": int(len(d))}

    # Target memakai preprocessor mana pun hasil fit train yang sama —
    # encoder compound bersifat deterministik; pakai pre lstm.
    pre_any = Preprocessor.muat(md / f"preprocessor{sm['pre_lstm']}.pkl")
    y_reg = d["target_laps_remaining"].to_numpy(dtype=np.float32)
    y_cls = ds.target_dari_df(d, pre_any)

    if dengan_baseline and (md / "baselines.json").exists():
        mayor, tyre = muat_baselines(md / "baselines.json")
        hasil["baseline_mayoritas"] = {
            "laps_remaining": _metrik_regresi(y_reg, mayor.predict_laps(d)),
            "compound": _metrik_klasifikasi(y_cls, mayor.predict_compound(d)),
        }
        hasil["baseline_tyre_life"] = {
            "laps_remaining": _metrik_regresi(y_reg, tyre.predict_laps(d)),
            "compound": _metrik_klasifikasi(y_cls, tyre.predict_compound(d)),
        }

    # --- XGBoost ---
    if (md / f"xgb_laps{sm['xgboost']}.json").exists():
        pre_x = Preprocessor.muat(md / f"preprocessor{sm['pre_xgb']}.pkl")
        reg = XGBRegressor(); reg.load_model(md / f"xgb_laps{sm['xgboost']}.json")
        clf = XGBClassifier(); clf.load_model(md / f"xgb_compound{sm['xgboost']}.json")
        X_flat = ds.fitur_xgb(d, pre_x)
        hasil["xgboost"] = {
            "laps_remaining": _metrik_regresi(y_reg, reg.predict(X_flat)),
            "compound": _metrik_klasifikasi(y_cls, clf.predict(X_flat)),
        }

    # --- LSTM ---
    if (md / f"lstm{sm['lstm']}_model.pt").exists():
        pre_l = Preprocessor.muat(md / f"preprocessor{sm['pre_lstm']}.pkl")
        pl, pr = _prediksi_nn("lstm", sm["lstm"], md, d, pre_l)
        hasil["lstm"] = {
            "laps_remaining": _metrik_regresi(y_reg, pl),
            "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
        }

    # --- Hybrid (fitur XGB dari seed yang sama dengan hybrid) ---
    if (md / f"hybrid{sm['hybrid']}_model.pt").exists() \
            and (md / f"xgb_laps{sm['hybrid']}.json").exists():
        pre_h = Preprocessor.muat(md / f"preprocessor{sm['pre_hybrid']}.pkl")
        reg = XGBRegressor(); reg.load_model(md / f"xgb_laps{sm['hybrid']}.json")
        clf = XGBClassifier(); clf.load_model(md / f"xgb_compound{sm['hybrid']}.json")
        pl, pr = _prediksi_nn("hybrid", sm["hybrid"], md, d, pre_h, reg, clf)
        hasil["hybrid"] = {
            "laps_remaining": _metrik_regresi(y_reg, pl),
            "compound": _metrik_klasifikasi(y_cls, pr.argmax(1)),
        }
    return hasil


def cetak_ringkasan(nama_split: str, hasil: dict) -> None:
    print(f"\n{'='*94}\n=== {nama_split} ({hasil['n_rows']} baris) ===")
    print(f"{'model':<22}{'MAE':>6}{'RMSE':>7}{'±1':>6}{'±3':>6}{'±5':>6}"
          f"{'MAE0-5':>8}{'MAE6-15':>8}{'MAE>15':>8}{'acc':>7}{'F1-makro':>9}")
    for model, m in hasil.items():
        if model == "n_rows":
            continue
        r, c = m["laps_remaining"], m["compound"]
        f = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "  -"
        print(f"{model:<22}{f(r['mae']):>6}{f(r['rmse']):>7}{r['dalam_1_lap']:>6.1%}"
              f"{r['dalam_3_lap']:>6.1%}{r['dalam_5_lap']:>6.1%}"
              f"{f(r['mae_jarak_0_5']):>8}{f(r['mae_jarak_6_15']):>8}"
              f"{f(r['mae_jarak_gt15']):>8}{c['accuracy']:>7.1%}{c['f1_macro']:>9.3f}")
    for model, m in hasil.items():
        if model == "n_rows":
            continue
        per_kelas = m["compound"]["per_kelas"]
        ringkas = " | ".join(f"{c[:4]}: F1={v['f1']:.2f} (n={v['support']})"
                             for c, v in per_kelas.items())
        print(f"  [{model}] per kelas -> {ringkas}")
        _cetak_confusion(m["compound"]["confusion_matrix"], model)


def _agregat_per_seed(per_seed: dict) -> dict:
    """Hitung mean ± std antar-seed untuk tiap tipe model & metrik skalar."""
    tipe_list = [t for t in ("xgboost", "lstm", "hybrid")
                 if all(t in h for h in per_seed.values())]
    out = {}
    for tipe in tipe_list:
        agg = {}
        for kelompok in ("laps_remaining", "compound"):
            kunci = [k for k, v in per_seed[list(per_seed)[0]][tipe][kelompok].items()
                     if isinstance(v, (int, float))]
            for k in kunci:
                nilai = [h[tipe][kelompok][k] for h in per_seed.values()]
                agg[f"{kelompok}.{k}_mean"] = round(float(np.mean(nilai)), 4)
                agg[f"{kelompok}.{k}_std"] = round(float(np.std(nilai)), 4)
        # F1 per kelas (mean±std antar seed)
        kelas = COMPOUNDS
        for c in kelas:
            f1s = [h[tipe]["compound"]["per_kelas"][c]["f1"] for h in per_seed.values()]
            agg[f"f1_{c}_mean"] = round(float(np.mean(f1s)), 4)
            agg[f"f1_{c}_std"] = round(float(np.std(f1s)), 4)
        out[tipe] = agg
    return out


def _cetak_mean_std(nama: str, agg: dict, n_seed: int) -> None:
    print(f"\n=== {nama} — mean ± std antar {n_seed} seed ===")
    print(f"{'model':<12}{'MAE':>16}{'RMSE':>16}{'±1lap':>15}{'±3lap':>15}"
          f"{'±5lap':>15}{'acc':>15}{'F1-makro':>16}")
    for tipe, a in agg.items():
        def ms(k1: str, k2: str, pct: bool = False) -> str:
            m, s = a[f"{k1}.{k2}_mean"], a[f"{k1}.{k2}_std"]
            if pct:
                return f"{m:.1%}±{s:.1%}"
            return f"{m:.2f}±{s:.2f}"
        print(f"{tipe:<12}{ms('laps_remaining','mae'):>16}{ms('laps_remaining','rmse'):>16}"
              f"{ms('laps_remaining','dalam_1_lap',True):>15}"
              f"{ms('laps_remaining','dalam_3_lap',True):>15}"
              f"{ms('laps_remaining','dalam_5_lap',True):>15}"
              f"{ms('compound','accuracy',True):>15}{ms('compound','f1_macro'):>16}")


def main() -> None:
    setup_logging()
    s = get_settings()
    md = s.models_dir
    df = ds.urutkan_per_lap(muat_dataset())
    d_val = df[df["split"] == "val"].reset_index(drop=True)
    d_test = df[df["split"] == "test"].reset_index(drop=True)

    keluaran: dict = {"evaluated_at": datetime.now(timezone.utc).isoformat()}

    # ---------- RUN 2: SEMUA seed dievaluasi (val & test), seleksi dari VAL ----------
    # Aturan (ditetapkan sebelum run): pemilihan model/seed HANYA dari val
    # (selection.json hasil training). Test dievaluasi SEKALI untuk semua seed
    # dan dilaporkan mean±std; TIDAK ADA pemilihan ulang berdasarkan test.
    sel_path = md / "selection.json"
    seeds = [42, 43, 44]
    primary, artifacts = "lstm", {}
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        seeds = sel.get("seeds", seeds)
        primary = sel.get("primary_model", primary)
        artifacts = sel.get("artifacts", {})
        keluaran["run2_primary_model"] = primary
        keluaran["run2_artifacts"] = artifacts
        keluaran["run2_kriteria"] = sel.get("kriteria")
    seed_terpilih = int(artifacts.get(primary, "_s42").replace("_s", "")) if artifacts else seeds[0]
    keluaran["run2_seed_terpilih_dari_val"] = seed_terpilih

    run2: dict = {}
    for nama_split, d_split in (("val", d_val), ("test", d_test)):
        per_seed = {}
        for i, seed in enumerate(seeds):
            smap = {k: f"_s{seed}" for k in
                    ("xgboost", "lstm", "hybrid", "pre_xgb", "pre_lstm", "pre_hybrid")}
            per_seed[str(seed)] = evaluasi_satu_split(
                d_split, md, suffix_map=smap, dengan_baseline=(i == 0))
        baseline = {k: v for k, v in per_seed[str(seeds[0])].items()
                    if k.startswith("baseline")}
        agg = _agregat_per_seed({k: v for k, v in per_seed.items()})
        run2[nama_split] = {"per_seed": per_seed, "mean_std": agg,
                            "baselines": baseline,
                            "model_terpilih_detail": per_seed[str(seed_terpilih)]}
        if nama_split == "test":
            cetak_ringkasan(f"RUN 2 — TEST 2025 seed {seed_terpilih} "
                            f"(seed dipilih dari VAL; test sekali evaluasi)",
                            per_seed[str(seed_terpilih)])
            _cetak_mean_std("RUN 2 — TEST 2025 semua seed", agg, len(seeds))
        else:
            _cetak_mean_std("RUN 2 — VAL 2024 semua seed", agg, len(seeds))
    keluaran["run2"] = run2

    # ---------- RUN 1 (backup, tanpa class weight) — metrik sama utk perbandingan ----------
    md1 = md / "run1"
    if (md1 / "preprocessor.pkl").exists():
        keluaran["run1"] = {
            "val": evaluasi_satu_split(d_val, md1),
            "test": evaluasi_satu_split(d_test, md1),
        }
        cetak_ringkasan("RUN 1 (lama) — VAL 2024", keluaran["run1"]["val"])
        cetak_ringkasan("RUN 1 (lama) — TEST 2025", keluaran["run1"]["test"])
        (md / "metrics_run1_extended.json").write_text(
            json.dumps(keluaran["run1"], indent=2))

    (md / "metrics.json").write_text(json.dumps(keluaran, indent=2))
    print(f"\nMetrik lengkap: {md / 'metrics.json'}")


if __name__ == "__main__":
    main()
