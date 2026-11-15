"""Driver eksperimen Run 3 (protokol: models/selection_criteria_v3.json).

Tahapan (dijalankan berurutan, berhenti bila tidak ada perbaikan nyata di CV):
    python scripts/05_run3_experiments.py --stage xgb       # K01, K02
    python scripts/05_run3_experiments.py --stage optuna --target-log1p 0|1
    python scripts/05_run3_experiments.py --stage lstm      # K05 (+seed 43/44)
    python scripts/05_run3_experiments.py --stage ensemble --anggota "K02,K05"
    python scripts/05_run3_experiments.py --stage quantile  # K08

SEMUA hasil (termasuk yang gagal) dicatat di logs/run3_experiments.jsonl.
Test 2025 TIDAK disentuh di sini.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.ml import run3  # noqa: E402
from app.ml.run3 import PARAM_XGB_DASAR, run_lstm, run_quantile, run_xgb  # noqa: E402

logger = logging.getLogger("run3.driver")
MODELS = PROJECT_ROOT / "models"


def stage_xgb(df: pd.DataFrame) -> None:
    # K01: target asli, bobot sqrt10 (referensi arsitektur terpisah)
    run_xgb("K01_xgb_base", df,
            {**PARAM_XGB_DASAR, "random_state": 42},
            {**PARAM_XGB_DASAR, "random_state": 42},
            target_log1p=False, weight_mode="sqrt10",
            catatan="XGB terpisah reg+clf, target asli, bobot sqrt-inv cap10")
    # K02: target log1p
    run_xgb("K02_xgb_log1p", df,
            {**PARAM_XGB_DASAR, "random_state": 42},
            {**PARAM_XGB_DASAR, "random_state": 42},
            target_log1p=True, weight_mode="sqrt10",
            catatan="K01 + target regresi log1p (clip fitur tidak dipakai utk tree)")


def stage_optuna(df: pd.DataFrame, target_log1p: bool, n_trials: int = 25) -> None:
    """Tuning Optuna DI DALAM pool train 2021-2023 (GroupKFold per race).
    Fold val CV dan test tidak pernah terlihat oleh Optuna."""
    import optuna
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.metrics import f1_score
    from app.ml import dataset as ds
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pool = df[df["season"] <= 2023].reset_index(drop=True)
    groups = run3_ds_grup(pool)
    gkf = list(GroupKFold(n_splits=3).split(pool, groups=groups))

    def objective_reg(trial):
        params = _space(trial)
        maes = []
        for tr, va in gkf:
            pre = _pre_fold(pool.iloc[tr])
            X_tr, X_va = _fitur(pool.iloc[tr], pre), _fitur(pool.iloc[va], pre)
            y_tr = pool.iloc[tr]["target_laps_remaining"].to_numpy(dtype=np.float32)
            y_va = pool.iloc[va]["target_laps_remaining"].to_numpy(dtype=np.float32)
            if target_log1p:
                y_tr = np.log1p(y_tr)
            reg = XGBRegressor(objective="reg:squarederror", **params)
            reg.fit(X_tr, y_tr, verbose=False)
            p = reg.predict(X_va)
            if target_log1p:
                p = np.expm1(p)
            maes.append(float(np.mean(np.abs(y_va - np.clip(p, 0, None)))))
        return float(np.mean(maes))

    def objective_cls(trial):
        params = _space(trial)
        f1s = []
        for tr, va in gkf:
            pre = _pre_fold(pool.iloc[tr])
            X_tr, X_va = _fitur(pool.iloc[tr], pre), _fitur(pool.iloc[va], pre)
            y_tr = run3.encode_compound(pool.iloc[tr])
            y_va = run3.encode_compound(pool.iloc[va])
            w = run3.bobot_kelas(y_tr, "sqrt10")[y_tr]
            clf = XGBClassifier(objective="multi:softprob", num_class=5, **params)
            clf.fit(X_tr, y_tr, sample_weight=w, verbose=False)
            pred = clf.predict(X_va)
            f1s.append(float(f1_score(y_va, pred, average="macro", zero_division=0)))
        return float(np.mean(f1s))

    study_reg = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_reg.optimize(objective_reg, n_trials=n_trials, show_progress_bar=False)
    study_cls = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cls.optimize(objective_cls, n_trials=n_trials, show_progress_bar=False)

    best_reg = {**study_reg.best_params, "tree_method": "hist", "n_jobs": -1,
                "random_state": 42}
    best_cls = {**study_cls.best_params, "tree_method": "hist", "n_jobs": -1,
                "random_state": 42}
    (MODELS / "run3_optuna_reg.json").write_text(json.dumps(
        {"best_params": best_reg, "cv_mae": study_reg.best_value,
         "n_trials": n_trials, "target_log1p": target_log1p,
         "semua_trial": [{"params": t.params, "value": t.value} for t in study_reg.trials]},
        indent=2))
    (MODELS / "run3_optuna_cls.json").write_text(json.dumps(
        {"best_params": best_cls, "cv_f1": study_cls.best_value,
         "n_trials": n_trials,
         "semua_trial": [{"params": t.params, "value": t.value} for t in study_cls.trials]},
        indent=2))
    logger.info("Optuna reg terbaik: MAE %.4f | params %s", study_reg.best_value, best_reg)
    logger.info("Optuna cls terbaik: F1 %.4f | params %s", study_cls.best_value, best_cls)

    # K03: reg tuned (+ clf dasar) ; K04: cls tuned (+ reg dasar)
    run_xgb("K03_xgb_optuna_reg", df, best_reg,
            {**PARAM_XGB_DASAR, "random_state": 42},
            target_log1p=target_log1p, weight_mode="sqrt10",
            catatan=f"Optuna reg (in-pool GroupKFold3), log1p={target_log1p}")
    run_xgb("K04_xgb_optuna_cls", df,
            {**PARAM_XGB_DASAR, "random_state": 42}, best_cls,
            target_log1p=target_log1p, weight_mode="sqrt10",
            catatan="Optuna cls (in-pool GroupKFold3), bobot sqrt10")


def _space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }


def _pre_fold(d_tr):
    from app.ml.pipeline import Preprocessor
    return Preprocessor().fit(d_tr)


def _fitur(d, pre):
    from app.ml import dataset as ds
    return ds.fitur_xgb(d, pre)


def run3_ds_grup(df):
    from app.ml.train import _grup_race
    return _grup_race(df)


def stage_lstm(df: pd.DataFrame, seeds: list[int]) -> None:
    for i, seed in enumerate(seeds):
        nama = "K05_lstm" if i == 0 else f"K05_lstm_s{seed}"
        run_lstm(nama, df, seed=seed,
                 catatan="LSTM reg(Huber,log1p,std) & cls(CE sqrt10) TERPISAH, "
                         "clip ±4, 12 epoch tetap")


def stage_ensemble(df: pd.DataFrame, anggota: list[str]) -> None:
    run_ensemble_safe("K06_ens_xgb_lstm", df, anggota[:2] if len(anggota) >= 2 else anggota)
    if len(anggota) >= 3:
        run_ensemble_safe("K07_ens_3seed", df, anggota)


def run_ensemble_safe(nama, df, anggota):
    from app.ml.run3 import run_ensemble
    run_ensemble(nama, df, anggota, catatan="ensemble rata-rata (reg & proba)")


def stage_quantile(df: pd.DataFrame) -> None:
    run_quantile("K08_xgb_quantile", df,
                 {**PARAM_XGB_DASAR, "n_estimators": 400},
                 catatan="Quantile P10/P50/P90 (log1p), titik=P50, tanpa tuning "
                         "(hemat budget konfigurasi)")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["xgb", "optuna", "lstm", "ensemble", "quantile"])
    ap.add_argument("--target-log1p", type=int, default=1, choices=[0, 1])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--anggota", default="", help="config ensemble, pisah koma")
    args = ap.parse_args()

    df = run3.muat_data()
    if args.stage == "xgb":
        stage_xgb(df)
    elif args.stage == "optuna":
        stage_optuna(df, bool(args.target_log1p), args.trials)
    elif args.stage == "lstm":
        stage_lstm(df, [int(x) for x in args.seeds.split(",")])
    elif args.stage == "ensemble":
        anggota = [a for a in args.anggota.split(",") if a]
        if not anggota:
            raise SystemExit("--anggota wajib untuk stage ensemble")
        stage_ensemble(df, anggota)
    elif args.stage == "quantile":
        stage_quantile(df)


if __name__ == "__main__":
    main()
