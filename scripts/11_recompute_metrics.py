# -*- coding: utf-8 -*-
"""P2b — Hitung ulang metrik klasifikasi Run 1/2/3 dengan KONVENSI BARU yang konsisten.

Konvensi: macro (precision/recall/F1) HANYA atas kelas dengan sampel nyata di
y_true (support>0). per_kelas melaporkan semua kelas. Dihitung ulang dari
CONFUSION MATRIX tersimpan (bukan re-inference) agar identik & bebas drift.

Output: models/metrics_corrected.json + tabel perbandingan lama vs baru.
Tidak mengubah metrics.json/metrics_run1_extended.json/metrics_run3.json asli.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.features import COMPOUNDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def metrics_from_cm(cm) -> dict:
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    prec = np.where(predicted > 0, tp / np.maximum(predicted, 1e-12), 0.0)
    rec = np.where(support > 0, tp / np.maximum(support, 1e-12), 0.0)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = [i for i in range(len(COMPOUNDS)) if support[i] > 0]

    def macro(a):
        return round(float(np.mean([a[i] for i in present])), 4) if present else 0.0

    total = cm.sum()
    return {
        "accuracy": round(float(tp.sum() / total), 4) if total else 0.0,
        "precision_macro": macro(prec),
        "recall_macro": macro(rec),
        "f1_macro": macro(f1),
        "macro_over_classes": [COMPOUNDS[i] for i in present],
        "per_kelas": {COMPOUNDS[i]: {"precision": round(float(prec[i]), 4),
                                     "recall": round(float(rec[i]), 4),
                                     "f1": round(float(f1[i]), 4),
                                     "support": int(support[i])}
                      for i in range(len(COMPOUNDS))},
    }


def get_cm(blok):
    """Ambil confusion_matrix dari sebuah blok metrik compound."""
    if isinstance(blok, dict):
        if "confusion_matrix" in blok:
            return blok["confusion_matrix"]
        if "compound" in blok and isinstance(blok["compound"], dict):
            return blok["compound"].get("confusion_matrix")
    return None


def old_f1(blok):
    if isinstance(blok, dict):
        if "f1_macro" in blok:
            return blok.get("f1_macro")
        if "compound" in blok and isinstance(blok["compound"], dict):
            return blok["compound"].get("f1_macro")
    return None


def main() -> None:
    out = {"konvensi": "macro atas kelas dengan support>0 di y_true; per_kelas lengkap",
           "run1": {}, "run2": {}, "run3": {}, "run3_cv": {}}
    rows_cmp = []

    # ---------- RUN 1 ----------
    r1 = json.loads((MODELS / "metrics_run1_extended.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        out["run1"][split] = {}
        for model, blok in r1.get(split, {}).items():
            cm = get_cm(blok)
            if cm is None:
                continue
            new = metrics_from_cm(cm)
            of1 = old_f1(blok)
            out["run1"][split][model] = new
            rows_cmp.append(("Run1", split, model, of1, new["f1_macro"],
                             new["macro_over_classes"]))

    # ---------- RUN 2 (per seed + mean/std) ----------
    m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
    r2 = m.get("run2", {})
    for split in ("val", "test"):
        out["run2"][split] = {}
        ps = r2.get(split, {}).get("per_seed", {})
        models = set()
        for s in ps:
            models |= set(ps[s].keys())
        for model in sorted(models):
            per_seed_new = {}
            for s in ("42", "43", "44"):
                blk = ps.get(s, {}).get(model)
                cm = get_cm(blk) if blk else None
                if cm is None:
                    continue
                per_seed_new[s] = metrics_from_cm(cm)
            if not per_seed_new:
                continue
            f1s = [v["f1_macro"] for v in per_seed_new.values()]
            accs = [v["accuracy"] for v in per_seed_new.values()]
            agg = {
                "f1_macro_mean": round(float(np.mean(f1s)), 4),
                "f1_macro_std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
                "per_seed": per_seed_new,
            }
            out["run2"][split][model] = agg
            # old logged mean
            old_ms = r2.get(split, {}).get("mean_std", {}).get(model, {})
            of1 = old_ms.get("compound.f1_macro_mean")
            rows_cmp.append(("Run2", split, model + "(mean)", of1, agg["f1_macro_mean"],
                             per_seed_new[list(per_seed_new)[0]]["macro_over_classes"]))

    # ---------- RUN 3 (val/test) ----------
    r3 = json.loads((MODELS / "run3" / "metrics_run3.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        blok = r3.get("hasil", {}).get(split, {})
        cm = get_cm(blok)
        if cm is None:
            continue
        new = metrics_from_cm(cm)
        of1 = old_f1(blok)
        out["run3"][split] = new
        rows_cmp.append(("Run3", split, "K03+K04", of1, new["f1_macro"],
                         new["macro_over_classes"]))

    # ---------- RUN 3 CV (per fold + mean) dari jsonl ----------
    jl = ROOT / "logs" / "run3_experiments.jsonl"
    if jl.exists():
        for line in open(jl, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            cfg = r.get("config", "")
            if not cfg.startswith("K0"):
                continue  # lewati B01 (binary head, CM 2x2 — konvensi berbeda)
            folds = r.get("folds", {})
            cv_f1 = []
            entry = {"folds": {}}
            for fold, fb in folds.items():
                cm = get_cm(fb)
                if cm is None:
                    continue
                new = metrics_from_cm(cm)
                entry["folds"][fold] = {"f1_macro": new["f1_macro"],
                                        "accuracy": new["accuracy"],
                                        "macro_over_classes": new["macro_over_classes"]}
                cv_f1.append(new["f1_macro"])
            if cv_f1:
                entry["cv_f1_macro_mean"] = round(float(np.mean(cv_f1)), 4)
                out["run3_cv"][cfg] = entry
                of1 = r.get("cv", {}).get("f1_macro")
                rows_cmp.append(("Run3CV", "cv", cfg, of1, entry["cv_f1_macro_mean"],
                                 entry["folds"][list(entry["folds"])[0]]["macro_over_classes"]))

    (MODELS / "metrics_corrected.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------- Tabel perbandingan ----------
    print("=" * 96)
    print("PERBANDINGAN F1-MAKRO: LAMA (label inferred) vs BARU (kelas hadir di y_true)")
    print("=" * 96)
    print(f"{'Run':7} {'split':5} {'model':16} {'F1 lama':>8} {'F1 baru':>8} {'delta':>8}  kelas-makro")
    for run, split, model, of1, nf1, classes in rows_cmp:
        d = (nf1 - of1) if (of1 is not None and nf1 is not None) else None
        ds = f"{d:+.4f}" if d is not None else "  n/a"
        ofs = f"{of1:.4f}" if of1 is not None else "  n/a"
        print(f"{run:7} {split:5} {model[:16]:16} {ofs:>8} {nf1:>8.4f} {ds:>8}  {classes}")
    print("\nTersimpan: models/metrics_corrected.json")


if __name__ == "__main__":
    main()
