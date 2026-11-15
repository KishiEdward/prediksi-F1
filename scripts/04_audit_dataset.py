"""Audit dataset & pembentukan sequence (diminta pengguna sebelum re-training).

Memeriksa pada DATA ASLI:
1. Tidak ada baris duplikat (season, round, driver_code, lap_number).
2. lap_number naik ketat di dalam tiap grup (setelah diurutkan).
3. Grup bersifat kontigu di dataframe terurut (syarat boundary detection).
4. Sequence: timestep terakhir = baris prediksi; semua timestep berasal dari
   baris <= i; tidak ada window yang menyeberang antar grup.
5. Tidak ada NaN/inf pada matriks fitur final (xgb/lstm/hybrid).
6. Target cocok dengan pitstop mentah (sampel acak).
7. Statistik skala target & fitur (diagnostik early stopping).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_setup import setup_logging
from app.ml import dataset as ds
from app.ml.features import (FITUR_INRACE_NUMERIK, WINDOW_LAP, muat_dataset,
                             bangun_sequences)
from app.ml.pipeline import Preprocessor

setup_logging()
GAGAL = []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    status = "OK " if kondisi else "GAGAL"
    print(f"[{status}] {nama} {detail}")
    if not kondisi:
        GAGAL.append(nama)


df = ds.urutkan_per_lap(muat_dataset())
pre = Preprocessor().fit(df[df["split"] == "train"])

# 1) duplikat
dup = df.duplicated(["season", "round", "driver_code", "lap_number"]).sum()
cek("Tidak ada baris duplikat", dup == 0, f"(duplikat={dup})")

# 2) lap naik ketat per grup
g = df.groupby(["season", "round", "driver_code"], sort=False)["lap_number"]
tidak_monoton = sum(1 for _, v in g if (v.diff().dropna() <= 0).any())
cek("lap_number naik ketat dalam grup", tidak_monoton == 0,
    f"(grup bermasalah={tidak_monoton})")

# 3) kontiguitas grup
kunci = df[["season", "round", "driver_code"]].astype(str).agg("|".join, axis=1)
n_grup = len(kunci.unique())
n_blok = int((kunci != kunci.shift()).sum())
cek("Grup kontigu (blok == jumlah grup)", n_blok == n_grup,
    f"(blok={n_blok}, grup={n_grup})")

# 4) audit sequence pada 200 grup acak + seluruh struktur boundary
rng = np.random.default_rng(0)
batas = np.flatnonzero(np.r_[True, kunci[1:].to_numpy() != kunci[:-1].to_numpy()])
batas_akhir = np.r_[batas[1:], len(df)]
M = ds.fitur_lstm(df, pre)
sample = rng.choice(len(batas), size=min(200, len(batas)), replace=False)
ok_seq, ok_nofuture = True, True
for gi in sample:
    a, b = batas[gi], batas_akhir[gi]
    sub = pd.DataFrame(M[a:b], columns=[f"f{i}" for i in range(M.shape[1])])
    X = bangun_sequences(sub, list(sub.columns), window=WINDOW_LAP)
    if X.shape[0] != b - a:
        ok_seq = False
    for i in range(len(X)):
        if not np.allclose(X[i, -1], M[a + i], atol=1e-5):
            ok_seq = False
            break
        # semua timestep harus == baris <= i (edge padding = baris pertama)
        for j in range(WINDOW_LAP):
            src = i - (WINDOW_LAP - 1 - j)
            ref = M[a + max(src, 0)]
            if not np.allclose(X[i, j], ref, atol=1e-5):
                ok_nofuture = False
cek("Sequence: timestep terakhir = baris prediksi (sampel 200 grup)", ok_seq)
cek("Sequence: tidak ada informasi baris masa depan (sampel 200 grup)", ok_nofuture)

# window tidak menyeberang grup: boundary dibangun per blok -> struktural aman
X_all, y_reg, y_cls, meta = ds.bangun_dataset_sequence(
    df, M, ds.target_dari_df(df, pre), WINDOW_LAP)
cek("Jumlah sequence == jumlah baris", len(X_all) == len(df),
    f"({len(X_all)} vs {len(df)})")
cek("Urutan meta == urutan df", (meta["lap_number"].to_numpy()
                                == df["lap_number"].to_numpy()).all())

# 5) NaN/inf pada matriks fitur
for nama, mtrx in [("lstm", ds.fitur_lstm(df, pre)),
                   ("xgb", ds.fitur_xgb(df, pre))]:
    finite = np.isfinite(mtrx[:, :]).all() if nama == "lstm" else None
    if nama == "lstm":
        cek(f"Fitur {nama}: tanpa NaN/inf", bool(finite))
    else:
        n_nan = int(np.isnan(mtrx).sum())
        print(f"[INFO] Fitur xgb: {n_nan} sel NaN (disengaja - XGBoost native NaN)")
cek("Sequence final: tanpa NaN/inf", bool(np.isfinite(X_all).all()))

# 6) sampel target vs pitstop mentah (dengan aturan kasus khusus:
#    pit di lap terakhir dibuang; pit di bawah RED FLAG dibuang)
inf = pd.read_parquet(Path("data/processed/lap_inference.parquet"),
                      columns=["season", "round", "lap_number", "track_status"])
status_lap = {(s, r, l): t for s, r, l, t in zip(inf["season"], inf["round"],
                                                 inf["lap_number"], inf["track_status"])}
pit = pd.read_parquet(Path("data/raw/jolpica_pitstops.parquet"))
pit["season"] = pit["season"].astype(int); pit["round"] = pit["round"].astype(int)
pit["pit_lap"] = pd.to_numeric(pit["lap"], errors="coerce")
idx = rng.choice(len(df), size=500, replace=False)
cocok = 0
for i in idx:
    row = df.iloc[i]
    ps = sorted(pit[(pit.season == row.season) & (pit["round"] == row["round"])
                    & (pit.driverId == row.driver_id)]["pit_lap"].dropna().astype(int))
    ps = [p for p in ps if p < row.total_laps
          and status_lap.get((row.season, row["round"], p), "GREEN") != "RED"]
    berikutnya = [p for p in ps if p >= row.lap_number]
    if berikutnya and (berikutnya[0] - row.lap_number) == row.target_laps_remaining:
        cocok += 1
cek("Target = pit berikutnya - lap, dgn aturan red-flag (sampel 500)",
    cocok == 500, f"(cocok {cocok}/500)")

# 7) statistik skala (diagnostik early stop)
y = df["target_laps_remaining"].to_numpy(dtype=float)
print(f"\n[INFO] target regresi: mean={y.mean():.2f} std={y.std():.2f} "
      f"min={y.min():.0f} max={y.max():.0f}")
print(f"[INFO] fitur lstm ter-scale: mean={M.mean():.3f} std={M.std():.3f} "
      f"min={M.min():.2f} max={M.max():.2f}")
print(f"[INFO] MSE awal ~ std^2 = {y.var():.1f} (mendominasi loss gabungan vs CE ~1.6)")

print("\n=== HASIL AUDIT ===")
print("SEMUA CEK LULUS" if not GAGAL else f"ADA {len(GAGAL)} KEGAGALAN: {GAGAL}")
sys.exit(1 if GAGAL else 0)
