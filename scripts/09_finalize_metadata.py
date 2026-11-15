# -*- coding: utf-8 -*-
"""Finalisasi metadata Run 3: update models/metadata.json + selection.json
agar predictor memuat 2 XGB terpisah (K03-reg, K04-clf) + binary head + conformal.

Run 3 diterima sebagai hasil FINAL. Run 2 diarsipkan di metadata["run2_archive"]
untuk provenance, bukan untuk loading aktif.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("finalize_meta")
MODELS = PROJECT_ROOT / "models"
RUN3 = MODELS / "run3"


def main() -> None:
    setup_logging()

    meta_lama = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    sel_lama = json.loads((MODELS / "selection.json").read_text(encoding="utf-8"))

    m_reg_cls = json.loads((RUN3 / "metrics_run3.json").read_text(encoding="utf-8"))
    m_bin = json.loads((RUN3 / "metrics_binary.json").read_text(encoding="utf-8"))
    conf = json.loads((RUN3 / "conformal.json").read_text(encoding="utf-8"))

    test_rc = m_reg_cls["hasil"]["test"]
    val_rc = m_reg_cls["hasil"]["val"]
    test_bin = m_bin["final"]["test"]

    artefak = {
        "reg": "run3/xgb_reg.json",
        "clf": "run3/xgb_clf.json",
        "binary": "run3/xgb_binary.json",
        "preprocessor": "run3/preprocessor.pkl",
        "preprocessor_binary": "run3/preprocessor_binary.pkl",
        "conformal": "run3/conformal.json",
        "metrics": "run3/metrics_run3.json",
        "metrics_binary": "run3/metrics_binary.json",
    }
    hilang = [v for v in artefak.values() if not (MODELS / v).exists()]
    if hilang:
        raise FileNotFoundError(f"Artefak Run 3 belum lengkap: {hilang}")

    cv_sel = m_reg_cls["meta"]["cv_selection"]
    cv_sel["reg_winner"]["cv2_only_mae"] = 4.7731
    cv_sel["cls_winner"]["cv2_only_f1"] = None  # cls cv2 fold tersedia di jsonl bila perlu
    cv_sel["bias_note"] = (
        "Optuna tuning memakai pool 2021-2023 yang overlap dengan val fold cv1 (2023), "
        "sehingga cv1 berpotensi bias optimis untuk K03/K04. Fold cv2 (val 2024) bersih. "
        "Perbandingan cv2-only: K03 MAE 4.7731 vs LSTM 5.5865 (selisih 0.813 lap), "
        "LEBIH BESAR dari selisih cv1 (0.350). Keunggulan K03 bukan artefak kontaminasi; "
        "jika ada, cv1 justru under-estimate. Seleksi tetap valid berdasarkan cv2.")

    caveats = {
        "wet_unreliable": (
            "INTERMEDIATE/WET sangat tidak andal: test 2025 hanya 94 baris INTER (F1=0.005) "
            "dan 0 baris WET. Prediksi compound basah harus diperlakukan sebagai "
            "ketidakpastian tinggi, bukan sinyal actionable."),
        "accuracy_by_distance": {
            "0-5": {"mae_test": test_rc["laps_remaining"]["mae_jarak_0_5"],
                    "interpretasi": "Prediksi pit dekat paling akurat"},
            "6-15": {"mae_test": test_rc["laps_remaining"]["mae_jarak_6_15"],
                     "interpretasi": "Akurasi sedang"},
            ">15": {"mae_test": test_rc["laps_remaining"]["mae_jarak_gt15"],
                    "interpretasi": "Prediksi pit jauh paling tidak akurat (MAE ~9.8 lap)"},
        },
        "conformal_window": (
            f"Jendela pit menggunakan split-conformal (bukan P10-P90 mentah quantile "
            f"regression yang coverage-nya hanya 52.8%). Setengah-lebar q="
            f"{conf['kuantil_abs_resid']} lap, coverage kalibrasi "
            f"{conf['coverage_kalibrasi']:.1%} di fold cv2."),
        "binary_head": (
            "Head biner 'masih pit?' dilatih terpisah dengan censoring DNF "
            "(9.853 baris DNF-tanpa-pit di-flag, TIDAK dilabel 0). "
            f"Test 2025: acc={test_bin['accuracy']}, F1={test_bin['f1']}, "
            f"AUC={test_bin.get('roc_auc')}."),
    }

    meta_baru = {
        "model_version": "0.3.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "run": 3,
        "primary_model": "xgb_separate",
        "artifacts": {"xgb_separate": artefak},
        "window_lap": 1,
        "compounds": meta_lama.get("compounds",
                                   ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
        "split": {"train": "2021-2023", "val": "2024", "test": "2025"},
        "fitur_xgb": meta_lama.get("fitur_xgb", []),
        "cv_selection": cv_sel,
        "conformal": conf,
        "caveats": caveats,
        "test_metrics": {
            "laps_remaining": {k: v for k, v in test_rc["laps_remaining"].items()
                               if k != "confusion_matrix"},
            "compound": {k: v for k, v in test_rc["compound"].items()
                         if k not in ("confusion_matrix", "per_kelas")},
            "binary": {k: v for k, v in test_bin.items()
                       if k != "confusion_matrix"},
        },
        "val_metrics": {
            "laps_remaining": {k: v for k, v in val_rc["laps_remaining"].items()
                               if k != "confusion_matrix"},
            "compound": {k: v for k, v in val_rc["compound"].items()
                         if k not in ("confusion_matrix", "per_kelas")},
        },
        "run2_archive": {
            "primary_model": sel_lama.get("primary_model"),
            "artifacts": sel_lama.get("artifacts"),
            "model_version": meta_lama.get("model_version"),
            "catatan": "Run 2 (LSTM s42) diarsipkan; Run 3 diterima sebagai final.",
        },
    }

    sel_baru = {
        "primary_model": "xgb_separate",
        "artifacts": {"xgb_separate": artefak},
        "kriteria": {
            "ditulis_pada": m_reg_cls["meta"].get("pernyataan_test"),
            "data": "CV kronologis 2021-2024 (cv1: 2021-22->2023; cv2: 2021-23->2024), grup per race",
            "metrik_utama_regresi": "CV MAE rata-rata (primer); tie-break <0.05 lap -> bucket 0-5/6-15/>15",
            "metrik_utama_klasifikasi": "CV F1-makro rata-rata (primer); tie-break <0.005 -> accuracy",
            "pemenang_terpisah": "Regresi dan klasifikasi dipilih TERPISAH per target dari CV",
            "stop_rule": "MAE membaik <0.15 DAN F1 membaik <0.010 -> berhenti",
            "test_protocol": "Test 2025 dievaluasi SEKALI, tidak untuk seleksi, tidak ada pemilihan ulang",
        },
        "cv_ringkas": cv_sel,
        "dipilih_pada": datetime.now(timezone.utc).isoformat(),
    }

    (MODELS / "metadata.json").write_text(
        json.dumps(meta_baru, indent=1, ensure_ascii=False), encoding="utf-8")
    (MODELS / "selection.json").write_text(
        json.dumps(sel_baru, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("metadata.json + selection.json diperbarui -> primary_model=xgb_separate (Run 3)")
    logger.info("Test 2025: MAE=%.4f F1=%.4f binary_acc=%.4f conformal_q=%.4f",
                test_rc["laps_remaining"]["mae"], test_rc["compound"]["f1_macro"],
                test_bin["accuracy"], conf["kuantil_abs_resid"])


if __name__ == "__main__":
    main()
