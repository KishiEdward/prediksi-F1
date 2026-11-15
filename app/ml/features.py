"""Pembentukan fitur dan target dari data mentah (Tahap b).

Alur:
1. Baca Parquet dari data/raw (Jolpica + FastF1).
2. Bangun fitur pra-balapan (statis per driver-race) — semua statistik historis
   pembalap dihitung EXPANDING (hanya dari race SEBELUM race berjalan) agar
   tidak ada kebocoran informasi lintas waktu.
3. Bangun fitur in-race per lap — hanya memakai data sampai lap tersebut.
4. Bentuk target per stint:
   - target_laps_remaining (regresi) = lap pit berikutnya - lap saat ini
   - target_next_compound (klasifikasi 5 kelas) = compound sesudah pit berikutnya
5. Tandai split kronologis: train 2021-2023, val 2024, test 2025.
6. Simpan data/processed/lap_dataset.parquet.

Prinsip anti-leakage (lihat README):
- Fitur pada lap N hanya dihitung dari data <= lap N.
- Statistik pembalap memakai hasil race sebelumnya saja (shift sebelum expanding mean).
- Split berdasarkan musim (grup race), bukan acak per baris.
- Scaler/encoder HANYA di-fit pada data train (dilakukan di app/ml/train.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

RAW = PROJECT_ROOT / "data" / "raw"
FASTF1_DIR = RAW / "fastf1"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Panjang sliding window untuk sequence LSTM (jumlah lap yang dilihat model)
WINDOW_LAP = 5

# Kelas compound target (label baku sesuai laporan)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# Pemetaan kode TrackStatus FastF1 -> label kondisi lintasan
TRACK_STATUS_MAP = {"1": "GREEN", "2": "SC", "3": "VSC", "4": "SC", "5": "RED", "6": "OTHER"}

# Jumlah tikungan per sirkuit (data referensi publik; tidak tersedia di Jolpica)
JUMLAH_TIKUNGAN = {
    "albert_park": 14, "americas": 20, "bahrain": 15, "baku": 20,
    "catalunya": 16, "hungaroring": 14, "imola": 19, "interlagos": 15,
    "istanbul": 14, "jeddah": 27, "las_vegas": 17, "le_castellet": 10,
    "losail": 16, "marina_bay": 19, "miami": 19, "monaco": 19, "monza": 11,
    "portimao": 15, "red_bull_ring": 10, "rodriguez": 17, "shanghai": 16,
    "silverstone": 18, "sochi": 18, "spa": 19, "suzuka": 18,
    "villeneuve": 14, "yas_marina": 16, "zandvoort": 14,
}

# ---------------------------------------------------------------------------
# Daftar kolom fitur (dipakai konsisten oleh train, evaluate, dan predictor)
# ---------------------------------------------------------------------------
FITUR_PRA_NUMERIK = [
    "grid_position", "quali_position", "quali_best_sec_rel", "sprint_position",
    "driver_starts_prev", "driver_avg_finish_prev",
    "driver_avg_points_prev", "driver_avg_grid_prev", "driver_dnf_rate_prev",
    "team_avg_points_prev", "circuit_len_km", "n_corners", "total_laps",
]
FITUR_PRA_KATEGORIK = ["driver_code", "team_id", "circuit_id"]

FITUR_INRACE_NUMERIK = [
    "lap_time_sec", "delta_prev_lap", "delta_mean3_lap", "delta_pb_so_far",
    "tyre_life", "stint_number", "fresh_tyre", "position", "position_delta_prev",
    "gap_to_leader", "gap_to_car_ahead", "race_progress", "laps_to_race_end",
    "is_green", "is_sc", "is_vsc", "is_red", "laps_since_sc_vsc",
    "air_temp", "track_temp", "rainfall", "humidity",
] + [f"compound_{c.lower()}" for c in COMPOUNDS]

KOLOM_TARGET = ["target_laps_remaining", "target_next_compound"]
KOLOM_KUNCI = ["season", "round", "driver_code", "lap_number"]
KONTEKS_NON_FITUR = ["driver_id", "team_id", "circuit_id", "race_name",
                     "current_compound", "track_status", "split"]


def parse_waktu_telemetri(x: object) -> float:
    """Ubah string waktu F1 ('1:16.741' / '58.794' / '1:30.5') menjadi detik."""
    if not isinstance(x, str) or not x.strip():
        return np.nan
    x = x.strip()
    try:
        if ":" in x:
            menit, detik = x.split(":")
            return float(menit) * 60 + float(detik)
        return float(x)
    except ValueError:
        return np.nan


def _kode_negara_ke_id(country: str) -> int:
    """Encoder deterministik sederhana untuk negara (tanpa perlu fit)."""
    return abs(hash(country)) % 997 if isinstance(country, str) else -1


# ---------------------------------------------------------------------------
# Fitur pra-balapan
# ---------------------------------------------------------------------------
def _bangun_fitur_prabalapan() -> pd.DataFrame:
    """Bangun satu baris fitur statis per (season, round, driver).

    Semua statistik 'prev' dihitung dari race SEBELUMNYA saja (expanding,
    di-shift 1) sehingga tidak memakai hasil race yang sedang diprediksi.
    """
    results = pd.read_parquet(RAW / "jolpica_results.parquet")
    races = pd.read_parquet(RAW / "jolpica_races.parquet")
    circuits = pd.read_parquet(RAW / "jolpica_circuits.parquet")
    quali = pd.read_parquet(RAW / "jolpica_qualifying.parquet")

    df = results[[
        "season", "round", "raceName", "grid", "position", "positionText", "points",
        "laps", "status", "Driver.driverId", "Driver.code", "Constructor.constructorId",
    ]].rename(columns={
        "Driver.driverId": "driver_id", "Driver.code": "driver_code",
        "Constructor.constructorId": "team_id", "raceName": "race_name",
    })

    # Sirkuit per race datang dari tabel races (results tidak selalu memuatnya)
    sirkuit_per_race = races[["season", "round", "Circuit.circuitId", "Circuit.circuitName"]] \
        .rename(columns={"Circuit.circuitId": "circuit_id", "Circuit.circuitName": "circuit_name"})
    df = df.drop(columns=[c for c in df.columns if c.startswith("Circuit.")], errors="ignore")
    df = df.merge(sirkuit_per_race, on=["season", "round"], how="left")

    # Posisi finish numerik (NaN bila DNF/DNS/DSQ)
    df["finish_position"] = pd.to_numeric(df["positionText"], errors="coerce")
    df["is_dnf"] = (~df["status"].isin(["Finished", "+1 Lap", "+2 Laps"]) &
                    df["finish_position"].isna()).astype(float)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

    # Total lap race = maksimum lap yang diselesaikan pemenang
    total_laps = df.groupby(["season", "round"])["laps"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").max())
    df["total_laps"] = total_laps

    # ---- Statistik historis pembalap (expanding, HANYA race sebelumnya) ----
    df = df.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)
    g = df.groupby("driver_id", group_keys=False)
    df["driver_starts_prev"] = g.cumcount()
    for kolom in ["finish_position", "points", "grid", "is_dnf"]:
        df[f"tmp_{kolom}"] = df[kolom]
        df[f"driver_avg_{kolom}_prev"] = (
            g[f"tmp_{kolom}"].apply(lambda s: s.expanding().mean().shift(1)))
    df = df.rename(columns={
        "driver_avg_finish_position_prev": "driver_avg_finish_prev",
        "driver_avg_points_prev": "driver_avg_points_prev",
        "driver_avg_grid_prev": "driver_avg_grid_prev",
        "driver_avg_is_dnf_prev": "driver_dnf_rate_prev",
    }).drop(columns=[c for c in df.columns if c.startswith("tmp_")])

    # ---- Statistik historis tim (expanding, hanya race sebelumnya) ----
    df = df.sort_values(["team_id", "season", "round"]).reset_index(drop=True)
    gt = df.groupby("team_id", group_keys=False)
    df["team_avg_points_prev"] = gt["points"].apply(
        lambda s: s.expanding().mean().shift(1))

    # ---- Karakteristik sirkuit ----
    sirk = circuits.rename(columns={
        "circuitId": "circuit_id", "circuitName": "circuit_name_ref",
        "Location.country": "country", "Location.locality": "locality",
        "circuitLength": "circuit_len_km", "circuitType": "circuit_type",
    })[["circuit_id", "country", "locality", "circuit_len_km", "circuit_type"]]
    sirk["circuit_len_km"] = pd.to_numeric(sirk["circuit_len_km"], errors="coerce")
    df = df.merge(sirk, on="circuit_id", how="left")
    df["n_corners"] = df["circuit_id"].map(JUMLAH_TIKUNGAN).astype("Float64")

    # ---- Kualifikasi: posisi + waktu terbaik relatif ----
    q = quali.rename(columns={"Driver.driverId": "driver_id"})
    for k in ("Q1", "Q2", "Q3"):
        if k in q.columns:
            q[f"{k}_sec"] = q[k].map(parse_waktu_telemetri)
    qcols = [f"{k}_sec" for k in ("Q1", "Q2", "Q3") if f"{k}_sec" in q.columns]
    q["quali_best_sec"] = q[qcols].min(axis=1)
    # relatif terhadap waktu terbaik pada sesi yang sama (hapus efek panjang sirkuit)
    q["quali_best_sec_rel"] = q["quali_best_sec"] - q.groupby(
        ["season", "round"])["quali_best_sec"].transform("min")
    df = df.merge(q[["season", "round", "driver_id", "position", "quali_best_sec_rel"]]
                  .rename(columns={"position": "quali_position"}),
                  on=["season", "round", "driver_id"], how="left")
    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce")

    # ---- Sprint race ----
    sprint_path = RAW / "jolpica_sprint.parquet"
    if sprint_path.exists():
        sp = pd.read_parquet(sprint_path).rename(columns={"Driver.driverId": "driver_id"})
        sp["sprint_position"] = pd.to_numeric(sp.get("positionText", sp.get("position")),
                                              errors="coerce")
        df = df.merge(sp[["season", "round", "driver_id", "sprint_position"]],
                      on=["season", "round", "driver_id"], how="left")
    else:
        df["sprint_position"] = np.nan

    # ---- Latihan bebas (FP) ----
    # CATATAN KETERBATASAN: Jolpica tidak mendukung endpoint hasil practice
    # ("Endpoint does not support final filter"), sehingga fitur FP tidak dapat
    # dibentuk. "Hasil sesi sebelumnya" (Bab III) tetap tercakup oleh kualifikasi
    # dan sprint. Fitur fp_best_position dihapus dari daftar fitur.

    df["grid_position"] = df["grid"]
    df["country_id"] = df["country"].map(_kode_negara_ke_id)

    kolom = (["season", "round", "driver_id", "driver_code", "team_id", "race_name",
              "circuit_id", "circuit_name", "country", "total_laps"]
             + FITUR_PRA_NUMERIK + ["country_id", "finish_position"])
    df = df[[c for c in kolom if c in df.columns]].copy()
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Fitur in-race per lap
# ---------------------------------------------------------------------------
def _muat_laps_fastf1() -> pd.DataFrame:
    """Gabungkan semua file laps FastF1 per-race menjadi satu DataFrame."""
    files = sorted(FASTF1_DIR.glob("*_laps.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Belum ada data FastF1 di {FASTF1_DIR}. Jalankan scripts/02_extract_fastf1.py dulu.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info("Laps FastF1: %d baris dari %d file", len(df), len(files))
    return df


def _muat_weather_trackstatus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gabungkan data cuaca dan status lintasan per-race."""
    w_files = sorted(FASTF1_DIR.glob("*_weather.parquet"))
    t_files = sorted(FASTF1_DIR.glob("*_trackstatus.parquet"))
    weather = pd.concat([pd.read_parquet(f) for f in w_files], ignore_index=True) if w_files \
        else pd.DataFrame(columns=["season", "round", "Time"])
    tstatus = pd.concat([pd.read_parquet(f) for f in t_files], ignore_index=True) if t_files \
        else pd.DataFrame(columns=["season", "round", "Time", "Status"])
    return weather, tstatus


def _bangun_fitur_inrace(laps: pd.DataFrame, weather: pd.DataFrame,
                         pra: pd.DataFrame) -> pd.DataFrame:
    """Bangun fitur per lap. Hanya memakai informasi sampai lap berjalan."""
    df = laps.rename(columns={"Driver": "driver_code", "LapNumber": "lap_number"}).copy()
    df = df[df["lap_number"].notna()]
    df["lap_number"] = df["lap_number"].astype(int)
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)

    # Hanya lap yang datanya akurat & tidak dihapus (kualitas data)
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]  # noqa: E712
    if "Deleted" in df.columns:
        df = df[df["Deleted"] == False]  # noqa: E712

    df = df.sort_values(["season", "round", "driver_code", "lap_number"]).reset_index(drop=True)
    grp = df.groupby(["season", "round", "driver_code"], group_keys=False, sort=False)

    # --- Waktu lap & delta (semuanya hanya melihat lap <= sekarang) ---
    df["lap_time_sec"] = pd.to_numeric(df["LapTime"], errors="coerce")
    df["delta_prev_lap"] = grp["lap_time_sec"].diff(1)
    df["delta_mean3_lap"] = df["lap_time_sec"] - grp["lap_time_sec"].transform(
        lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    df["pb_so_far"] = grp["lap_time_sec"].cummin()
    df["delta_pb_so_far"] = df["lap_time_sec"] - df["pb_so_far"]

    # --- Ban ---
    df["tyre_life"] = pd.to_numeric(df.get("TireLife"), errors="coerce")
    df["stint_number"] = pd.to_numeric(df.get("Stint"), errors="coerce")
    df["fresh_tyre"] = (df.get("FreshTyre") == True).astype(float) \
        if "FreshTyre" in df.columns else np.where(df["tyre_life"] <= 1, 1.0, np.nan)
    df["current_compound"] = df["Compound"].astype("string").str.upper()
    for c in COMPOUNDS:
        df[f"compound_{c.lower()}"] = (df["current_compound"] == c).astype(float)

    # --- Posisi & gap ---
    df["position"] = pd.to_numeric(df.get("Position"), errors="coerce")
    df["position_delta_prev"] = grp["position"].diff(1) * -1  # positif = naik posisi
    df["time_elapsed"] = pd.to_numeric(df.get("Time"), errors="coerce")

    per_lap = df.groupby(["season", "round", "lap_number"], sort=False)
    df["gap_to_leader"] = df["time_elapsed"] - per_lap["time_elapsed"].transform("min")
    # gap ke mobil persis di depan: urutkan per posisi dalam lap yang sama
    df_sorted = df.sort_values(["season", "round", "lap_number", "position"])
    df_sorted["gap_to_car_ahead"] = (
        df_sorted.groupby(["season", "round", "lap_number"], sort=False)["time_elapsed"].diff(1))
    # diff antar pembalap berurutan hanya valid bila satu lap sama (sudah dijamin groupby)
    df = df_sorted.sort_values(["season", "round", "driver_code", "lap_number"]).reset_index(drop=True)
    grp = df.groupby(["season", "round", "driver_code"], group_keys=False, sort=False)

    # --- Progres balapan ---
    total = pra.set_index(["season", "round"])["total_laps"]
    df["total_laps"] = [total.get((s, r), np.nan)
                        for s, r in zip(df["season"], df["round"])]
    df["race_progress"] = df["lap_number"] / df["total_laps"]
    df["laps_to_race_end"] = df["total_laps"] - df["lap_number"]

    # --- Status lintasan ---
    kode = df["TrackStatus"].astype("string").str.extract(r"(\d)")[0].fillna("1")
    df["track_status"] = kode.map(TRACK_STATUS_MAP).fillna("OTHER")
    df["is_green"] = (df["track_status"] == "GREEN").astype(float)
    df["is_sc"] = (df["track_status"] == "SC").astype(float)
    df["is_vsc"] = (df["track_status"] == "VSC").astype(float)
    df["is_red"] = (df["track_status"] == "RED").astype(float)
    scvsc = (df["is_sc"] + df["is_vsc"]).clip(0, 1)
    # jumlah lap sejak SC/VSC terakhir (0 = sedang/lap ini; dibatasi 10)
    hitung = np.zeros(len(df))
    sejak = 99
    kunci_lama = None
    for i, (s, r, d, v) in enumerate(zip(df["season"], df["round"],
                                         df["driver_code"], scvsc)):
        kunci = (s, r, d)
        if kunci != kunci_lama:
            sejak = 99
            kunci_lama = kunci
        sejak = 0 if v == 1 else sejak + 1
        hitung[i] = min(sejak, 10)
    df["laps_since_sc_vsc"] = hitung

    # --- Cuaca: gabungkan sampel cuaca TERDEKAT SEBELUM waktu lap (asof, anti-leak) ---
    if len(weather) > 0 and "time_elapsed" in df.columns:
        w = weather.rename(columns={
            "Time": "time_elapsed", "AirTemp": "air_temp", "TrackTemp": "track_temp",
            "Rainfall": "rainfall", "Humidity": "humidity", "WindSpeed": "wind_speed",
        })
        kolom_w = [c for c in ["season", "round", "time_elapsed", "air_temp",
                               "track_temp", "rainfall", "humidity", "wind_speed"]
                   if c in w.columns]
        w = w[kolom_w].sort_values("time_elapsed")
        df = pd.merge_asof(
            df.sort_values("time_elapsed"), w.sort_values("time_elapsed"),
            on="time_elapsed", by=["season", "round"], direction="backward",
            tolerance=600.0,
        ).sort_values(["season", "round", "driver_code", "lap_number"]).reset_index(drop=True)
    else:
        for c in ["air_temp", "track_temp", "rainfall", "humidity"]:
            df[c] = np.nan

    # Lap tanpa catatan waktu (pit lap dsb.) tetap dipakai; nilai NaN akan
    # ditangani saat training (median train) — di sini dibiarkan NaN.
    df = df.drop(columns=["pb_so_far", "time_elapsed"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Target per stint
# ---------------------------------------------------------------------------
def _bentuk_target(df: pd.DataFrame) -> pd.DataFrame:
    """Bentuk target laps_remaining & next_compound berdasarkan stint.

    Aturan (sesuai laporan subbab 2.2.7 + 1.4):
    - Pit pada lap p mengakhiri stint; untuk lap t <= p target = p - t.
    - Setelah pit, target dihitung ulang untuk pit berikutnya.
    - Baris tanpa pit berikutnya (stint terakhir / DNF) tidak punya target dan
      tidak dipakai untuk training (tetap disimpan untuk inference).
    - Pit di lap terakhir race dianggap bukan keputusan strategi -> dibuang.
    - KASUS KHUSUS (subbab 1.4): pit yang terjadi di bawah RED FLAG dibuang
      dari daftar pit (ganti ban gratis saat red flag bukan keputusan strategi).
    - Compound berikutnya = compound VALID pertama setelah lap pit (out-lap
      sering berstatus tidak akurat sehingga tidak selalu ada di lap p+1).
    """
    pit = pd.read_parquet(RAW / "jolpica_pitstops.parquet")
    pit = pit.rename(columns={"driverId": "driver_id"})[
        ["season", "round", "driver_id", "lap"]].copy()
    pit["season"] = pit["season"].astype(int)
    pit["round"] = pit["round"].astype(int)
    pit["pit_lap"] = pd.to_numeric(pit["lap"], errors="coerce")
    pit = pit.dropna(subset=["pit_lap"])
    pit["pit_lap"] = pit["pit_lap"].astype(int)

    df = df.sort_values(["season", "round", "driver_code", "lap_number"]).reset_index(drop=True)

    # Status lintasan per (season, round, lap) untuk deteksi red flag
    status_lap = (df.drop_duplicates(["season", "round", "lap_number"])
                  .set_index(["season", "round", "lap_number"])["track_status"]
                  .to_dict()) if "track_status" in df.columns else {}

    pit_per_driver = {}
    n_red_flag = 0
    for (s, r, d), ls in pit.groupby(["season", "round", "driver_id"])["pit_lap"]:
        bersih = []
        for p in sorted(set(ls)):
            if status_lap.get((s, r, p), "GREEN") == "RED":
                n_red_flag += 1  # pit saat red flag -> bukan keputusan strategi
                continue
            bersih.append(p)
        pit_per_driver[(s, r, d)] = bersih
    if n_red_flag:
        logger.info("Pit stop di bawah red flag dibuang dari target: %d kejadian", n_red_flag)

    laps_rem = np.full(len(df), np.nan)
    for i in range(len(df)):
        s, r = df.at[i, "season"], df.at[i, "round"]
        d = df.at[i, "driver_id"]
        t = df.at[i, "lap_number"]
        total = df.at[i, "total_laps"]
        pits = pit_per_driver.get((s, r, d), [])
        pits = [p for p in pits if p < total]  # buang pit di lap terakhir
        berikutnya = [p for p in pits if p >= t]
        if not berikutnya:
            continue
        laps_rem[i] = berikutnya[0] - t
    df["target_laps_remaining"] = laps_rem

    # Compound berikutnya: compound VALID pertama pada lap > p per driver-race.
    # Dibangun dari daftar (lap, compound) yang sudah terurut per driver-race.
    next_comp = np.full(len(df), None, dtype=object)
    for (s, r, kode), g in df.groupby(["season", "round", "driver_code"], sort=False):
        lap_arr = g["lap_number"].to_numpy()
        comp_arr = g["current_compound"].to_numpy()
        urutan = np.argsort(lap_arr)
        lap_sorted, comp_sorted = lap_arr[urutan], comp_arr[urutan]
        for i in g.index:
            if pd.isna(df.at[i, "target_laps_remaining"]):
                continue
            p = int(df.at[i, "lap_number"] + df.at[i, "target_laps_remaining"])
            pos = np.searchsorted(lap_sorted, p, side="right")
            for j in range(pos, len(comp_sorted)):
                if comp_sorted[j] in COMPOUNDS:
                    next_comp[i] = comp_sorted[j]
                    break
    df["target_next_compound"] = next_comp

    # Tandai baris dengan ground truth lengkap (untuk training).
    # Baris tanpa target (stint terakhir / DNF / compound tak dikenal) TETAP
    # disimpan di file inference agar API bisa memprediksi lap mana pun.
    df["has_target"] = df["target_laps_remaining"].notna() & df["target_next_compound"].notna()
    return df


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------
def bangun_dataset() -> pd.DataFrame:
    """Bangun dataset final per lap dan simpan sebagai Parquet."""
    s = get_settings()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("Membangun fitur pra-balapan...")
    pra = _bangun_fitur_prabalapan()

    logger.info("Memuat data FastF1...")
    laps = _muat_laps_fastf1()
    weather, _ = _muat_weather_trackstatus()

    logger.info("Membangun fitur in-race...")
    df = _bangun_fitur_inrace(laps, weather, pra)

    # Gabungkan fitur statis (join per season-round-driver_code)
    pra_join = pra.drop(columns=["country", "circuit_name", "finish_position",
                                 "total_laps"], errors="ignore")
    df = df.merge(pra_join, on=["season", "round", "driver_code"], how="left")
    # merge_asof bisa mengacak tipe; pastikan kunci integer
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)

    logger.info("Membentuk target per stint...")
    df = _bentuk_target(df)

    # Split kronologis sesuai laporan: train <=2023, val 2024, test 2025
    df["split"] = np.where(df["season"] <= 2023, "train",
                           np.where(df["season"] == 2024, "val", "test"))
    df = df[df["season"].isin(s.seasons)]

    kolom_akhir = (KOLOM_KUNCI + KONTEKS_NON_FITUR + FITUR_PRA_NUMERIK
                   + FITUR_INRACE_NUMERIK + KOLOM_TARGET + ["has_target"])
    df = df[[c for c in kolom_akhir if c in df.columns]].copy()
    df = df.sort_values(["season", "round", "driver_code", "lap_number"]).reset_index(drop=True)

    # 1) File inference: SEMUA lap (dipakai API replay, termasuk stint terakhir)
    out_inf = PROCESSED / "lap_inference.parquet"
    df.to_parquet(out_inf)
    logger.info("Dataset inference tersimpan: %s (%d baris)", out_inf, len(df))

    # 2) File training: hanya baris dengan target lengkap
    train_df = df[df["has_target"]].drop(columns=["has_target"]).copy()
    train_df["target_next_compound"] = train_df["target_next_compound"].astype(str)
    out = PROCESSED / "lap_dataset.parquet"
    train_df.to_parquet(out)
    logger.info("Dataset training tersimpan: %s (%d baris, kolom=%d)",
                out, len(train_df), len(train_df.columns))
    logger.info("Distribusi split: %s", train_df["split"].value_counts().to_dict())
    logger.info("Distribusi target compound: %s",
                train_df["target_next_compound"].value_counts().to_dict())
    return train_df


def muat_dataset_inference() -> pd.DataFrame:
    """Muat dataset SEMUA lap (untuk inference API replay)."""
    return pd.read_parquet(PROCESSED / "lap_inference.parquet")


def muat_dataset() -> pd.DataFrame:
    """Muat dataset terproses (hasil bangun_dataset) untuk training/API."""
    return pd.read_parquet(PROCESSED / "lap_dataset.parquet")


def bangun_sequences(fitur_df: pd.DataFrame, kolom_fitur: list[str],
                     window: int = WINDOW_LAP) -> np.ndarray:
    """Bentuk array sequence (n, window, n_fitur) dari baris lap berurutan.

    `fitur_df` HARUS sudah:
    - terfilter satu pembalap dalam satu race (atau dipanggil per grup),
    - terurut naik berdasarkan lap_number,
    - kolom fitur numerik sudah di-scale & NaN sudah diisi.
    Awal race (lap < window) di-pad dengan mengulang nilai lap pertama
    (edge padding) supaya semua lap tetap punya sequence.
    """
    x = fitur_df[kolom_fitur].to_numpy(dtype=np.float32)
    n = len(x)
    if n == 0:
        return np.zeros((0, window, len(kolom_fitur)), dtype=np.float32)
    pad = np.repeat(x[:1], window - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    idx = np.arange(n)[:, None] + np.arange(window)[None, :]
    return xp[idx]  # bentuk (n, window, n_fitur)


if __name__ == "__main__":
    from app.logging_setup import setup_logging
    setup_logging()
    bangun_dataset()
