# F1 Pit Strategy Prediction API

Backend API (Python) yang **memprediksi kapan mobil F1 masuk pit dan ban apa yang
dipakai berikutnya**, dibangun berdasarkan arsitektur pada skripsi *"Pengembangan
Arsitektur Hibrida XGBoost-LSTM untuk Rekomendasi Strategi Pit dan Pemilihan Ban
Formula One"*.

> **Penting (framing):** keluaran API adalah **prediksi/perkiraan keputusan pit**
> yang dipelajari dari keputusan tim yang *benar-benar terjadi* (ground truth
> observasional, data 2021–2025). Ini **bukan** rekomendasi strategi optimal
> secara matematis.

> **Model aktif: Run 3 (`xgb_separate`, v0.3.0).** Setelah tiga putaran eksperimen
> dengan protokol seleksi pra-registrasi, model final adalah **dua XGBoost terpisah**
> (regresi laps-to-pit hasil tuning Optuna + klasifikasi compound berbobot sqrt-inv-freq)
> ditambah **head biner "masih pit?"** dan **pit window terkalibrasi split-conformal**.
> Arsitektur hybrid XGBoost-LSTM dan LSTM tunggal dari Run 1/2 diarsipkan — tetap
> dapat dimuat untuk perbandingan, tetapi bukan model aktif. Alasan dan bukti
> empiris ada di §10.

---

## 1. Cara Kerja (Ringkas untuk Pemula)

1. **Ambil data** — skrip mengunduh data historis (Jolpica F1 API) dan data
   balapan/telemetri (FastF1), lalu menyimpannya sebagai file Parquet di `data/raw`.
2. **Bentuk fitur & target** — `app/ml/features.py` menggabungkan semuanya menjadi
   tabel per *pembalap-race-lap*, lalu membuat dua target:
   - `laps remaining` (regresi): sisa lap menuju pit stop berikutnya.
   - `next compound` (klasifikasi): ban berikutnya — SOFT/MEDIUM/HARD/INTERMEDIATE/WET.
3. **Training & seleksi model** — tiga putaran eksperimen (Run 1→2→3) dengan
   protokol seleksi pra-registrasi. Semua kandidat dicatat, termasuk yang gagal:
   - **Run 1**: XGBoost, LSTM, Hybrid XGBoost-LSTM (loss gabungan MSE+CE).
   - **Run 2**: + class weight, lr 3e-4, dropout 0.3, 3 seed, kriteria val-only.
   - **Run 3 (FINAL)**: model regresi & klasifikasi **terpisah** (tanpa loss
     gabungan), tuning Optuna per target, ensemble/quantile diuji tapi tidak
     memperbaiki, head biner "masih pit?" dengan censoring DNF, pit window
     split-conformal. Pemenang dipilih **terpisah per target dari CV kronologis
     2021-2024**; test 2025 hanya dievaluasi sekali di akhir.
   Model aktif: **XGBoost regresi (K03) + XGBoost klasifikasi (K04) + XGBoost
   biner**, semuanya per-baris tabular tanpa sequence/NN.
4. **API** — FastAPI memuat model sekali saat startup, lalu melayani prediksi
   mode **replay** (per lap atau streaming SSE). Mode **live** adalah Tahap 2
   (saat ini menjawab `501`).

```
Website ── HTTP ──> FastAPI ──> Predictor ──> RaceStateProvider
                                   │              ├── ReplayProvider (Tahap 1, aktif)
                                   │              └── LiveProvider  (Tahap 2, stub)
                                   └── artifact di models/run3/ (XGB reg + clf + binary
                                       + conformal.json), dimuat via metadata.json
```

## 2. Struktur Folder

```
Model/
├── app/
│   ├── api/main.py           # semua route FastAPI
│   ├── schemas/prediksi.py   # skema Pydantic v2 (validasi ketat)
│   ├── services/
│   │   ├── jolpica.py        # client Jolpica: retry+backoff, rate limit, cache disk
│   │   ├── fastf1_svc.py     # wrapper FastF1 + cache
│   │   ├── providers.py      # RaceStateProvider / ReplayProvider / LiveProvider
│   │   └── predictor.py      # muat model, jalankan inference
│   ├── ml/
│   │   ├── features.py       # feature engineering + target + anti-leakage
│   │   ├── pipeline.py       # Preprocessor (scaler/encoder, fit HANYA di train)
│   │   ├── dataset.py        # matriks fitur & sliding window (train = serve)
│   │   ├── nn.py             # arsitektur LSTM 2 head (PyTorch)
│   │   ├── train.py          # python -m app.ml.train
│   │   └── evaluate.py       # python -m app.ml.evaluate
│   ├── config.py             # pengaturan via .env (pydantic-settings)
│   └── main.py               # entry point server
├── scripts/                  # 01 Jolpica, 02 FastF1, 02b Kaggle convert,
│                             # 03 build dataset, 04 audit, 05 Run3 CV experiments,
│                             # 06 Run3 final train, 07 binary head, 08 conformal,
│                             # 09 finalize metadata, 10 E2E replay test
├── tests/                    # pytest (schema, endpoint, 2 pembalap/tim, leakage)
├── data/raw/                 # Parquet mentah + cache API (tidak masuk git)
├── data/processed/           # dataset hasil feature engineering + run3_preds/
├── models/                   # metadata.json, selection.json, baselines.json
│   ├── run1/, run2 _s*       # arsip Run 1/2 (LSTM/hybrid, bukan model aktif)
│   └── run3/                 # MODEL AKTIF: xgb_reg/clf/binary.json,
│                             # preprocessor*.pkl, conformal.json, metrics_*.json
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## 3. Setup dari Nol (Windows/Linux/Mac)

Butuh **Python 3.11** (cek: `py -3.11 --version` atau `python3.11 --version`).

```bash
# 1. Masuk folder proyek
cd Model

# 2. Buat virtual environment
py -3.11 -m venv .venv                # Windows
.venv\Scripts\activate                # Windows (Git Bash: source .venv/Scripts/activate)
# python3.11 -m venv .venv && source .venv/bin/activate   # Linux/Mac

# 3. Install PyTorch versi CPU (lebih kecil, tanpa GPU)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install sisanya
pip install -r requirements.txt

# 5. Buat file konfigurasi
copy .env.example .env                # Windows
# cp .env.example .env                # Linux/Mac
```

## 4. Ambil Data (sekali saja, ±1–3 jam)

```bash
python scripts/01_extract_jolpica.py   # data historis 2021-2025 (API Jolpica)
python scripts/02_extract_fastf1.py    # data balapan per lap (FastF1, unduhan besar)
```

- Keduanya **idempotent**: aman dijalankan ulang. Respons Jolpica di-cache ke
  `data/raw/jolpica_cache`, data FastF1 ke `data/raw/fastf1_cache`, dan file
  per-race yang sudah ada akan dilewati.
- Kalau koneksi putus di tengah, cukup jalankan ulang skrip yang sama.

## 5. Bentuk Dataset, Training, dan Evaluasi

```bash
# Bentuk fitur + target (output: data/processed/lap_dataset.parquet & lap_inference.parquet)
python -m app.ml.features

# ===== Run 3 (MODEL AKTIF) =====
# Protokol CV ditulis SEBELUM eksperimen: models/selection_criteria_v3.json
python scripts/05_run3_experiments.py --stage xgb        # K01-K02 baseline CV
python scripts/05_run3_experiments.py --stage optuna --target-log1p 0 --trials 25  # K03-K04
python scripts/05_run3_experiments.py --stage lstm --seeds 42,43,44   # K05
python scripts/05_run3_experiments.py --stage ensemble --anggota "K03_xgb_optuna_reg,K05_lstm,K05_lstm_s43,K05_lstm_s44"  # K06-K07
python scripts/05_run3_experiments.py --stage quantile   # K08
python scripts/06_run3_final.py      # latih winner CV di train 2021-2023, test SEKALI
python scripts/07_binary_head.py     # head biner "masih pit?" (DNF di-flag sensor)
python scripts/08_conformal.py       # kalibrasi pit window dari residual cv2
python scripts/09_finalize_metadata.py  # update metadata.json + selection.json -> Run 3
python scripts/10_e2e_replay.py      # uji end-to-end /predict/replay (64 check)

# ===== Run 1/2 (ARSIP, bukan model aktif) =====
python -m app.ml.train                # LSTM/hybrid, default 40 epoch early stop
python -m app.ml.evaluate             # evaluasi Run 1+2 -> models/metrics.json
```

### Arti metrik evaluasi (bahasa sederhana)

**Target `laps remaining` (regresi):**
| Metrik | Artinya |
|---|---|
| **MAE** | Rata-rata meleset berapa lap. MAE 1.8 = prediksi pit biasanya meleset ±1.8 lap. |
| **RMSE** | Seperti MAE tapi kesalahan besar dihukum lebih berat. RMSE jauh di atas MAE = ada beberapa prediksi yang meleset parah. |
| **dalam_1_lap** | Persentase prediksi yang meleset maksimal 1 lap — cara paling intuitif menilai akurasi waktu pit. |

**Target `next compound` (klasifikasi):**
| Metrik | Artinya |
|---|---|
| **accuracy** | Persentase tebakan ban yang benar. |
| **precision** | Dari semua yang *ditebak* HARD, berapa persen yang benar HARD. |
| **recall** | Dari semua yang *sebenarnya* HARD, berapa persen yang berhasil ditebak. |
| **F1** | Rata-rata harmonik precision & recall (imbang keduanya). Macro = rata-rata antar kelas, penting karena kelas WET/INTERMEDIATE jarang muncul. |
| **confusion matrix** | Tabel 5×5: baris = ban aktual, kolom = tebakan. Diagonal = benar. |

## 6. Pencegahan Data Leakage

| Risiko | Pencegahan | Kode |
|---|---|---|
| Split acak per baris bocorkan race yang sama ke train & test | Split **kronologis per musim**: train 2021–2023, val 2024, test 2025 | `app/ml/features.py` (kolom `split`) |
| Fitur lap N memakai info masa depan | Delta lap time pakai `shift`, personal-best pakai `cummin`, cuaca pakai `merge_asof` **backward**, sequence hanya lap ≤ N | `app/ml/features.py`, `app/ml/dataset.py` |
| Statistik pembalap bocorkan hasil race itu sendiri | Expanding mean **di-shift 1** (hanya race-race sebelumnya) | `app/ml/features.py::_bangun_fitur_prabalapan` |
| Scaler/encoder belajar dari data test | `Preprocessor.fit()` hanya dipanggil dengan data train | `app/ml/pipeline.py`, dipanggil di `app/ml/train.py` |
| Fitur XGBoost untuk hybrid bocor di data train | Prediksi XGBoost dibuat **out-of-fold** (GroupKFold per race — satu race tidak pernah terbelah) | `app/ml/train.py::prediksi_xgb_oof` |
| LSTM melihat "masa depan" | LSTM **satu arah** (bukan BiLSTM), window 5 lap ke belakang | `app/ml/nn.py` |
| Verifikasi | Unit test khusus leakage (sequence, shift, target stint, scaler train-only, split kronologis) | `tests/test_leakage.py` |

## 6b. Perlakuan Kasus Khusus (sesuai Bab III)

| Kasus | Perlakuan | Dasar |
|---|---|---|
| Pit stop di bawah **red flag** | Dibuang dari target (ganti ban gratis saat red flag bukan keputusan strategi) | Subbab 1.4: kondisi Red Flag = kondisi khusus prapemrosesan |
| **Sprint race** | Tidak dimodelkan sebagai race; hanya balapan utama. Hasil sprint dipakai sebagai fitur pra-balapan (`sprint_position`) | Subbab 3.1.2: lingkup = balapan utama |
| Pit stop saat **SC/VSC** | Tetap dipakai (keputusan strategis nyata); SC/VSC menjadi fitur input | Subbab 1.4 & 3.2.3 |
| Pit di **lap terakhir** | Dibuang (bukan keputusan strategi) | Prapemrosesan data |
| Compound **UNKNOWN/kosong** (terutama 2021) | Baris tanpa compound target valid tidak dipakai training; compound berjalan tak dikenal → one-hot nol semua | Elisitasi #6 (menangani data tidak lengkap) |
| Lap **tidak akurat / dihapus** (in/out lap) | Dibuang dari fitur; sequence LSTM sedikit terkompresi (didokumentasikan) | Kualitas data |
| Stint terakhir / **DNF** tanpa pit berikutnya | Disimpan untuk inference tetapi tanpa target (tidak dipakai training) | Target = pit berikutnya (subbab 2.2.7) |

## 6c. Run 2 — Perubahan konfigurasi & protokol evaluasi

Perubahan konfigurasi di luar 7 batasan pengguna (semua ditetapkan dari
train/val saja; **test 2025 tidak dipakai untuk keputusan apa pun**; scaler/
encoder/imputasi hanya di-fit pada train). Versi mesin-baca:
`models/selection_criteria.json`.

1. **Class weight**: inverse-frequency, **dibatasi maks 20** lalu dinormalisasi
   rata-rata 1 — tanpa cap, bobot WET (84 baris) ≈ 137× dan mengganggu stabilitas.
2. **Loss gabungan**: MSE + CrossEntropy berbobot, keseimbangan 1:1 (tidak diubah).
3. **Gradient clipping** max-norm 5.0 (stabilitas, praktik standar).
4. **Target TIDAK dinormalisasi**; prediksi di-clip ≥ 0 hanya saat metrik/tampilan API.
5. **Fitur TIDAK di-clip**; StandardScaler + median imputasi fit HANYA di train;
   jalur XGBoost membiarkan NaN (native), jalur LSTM diimputasi.
6. **Epochs maks 40, patience 7** (early stop pada val loss gabungan).
7. **Arsitektur/optimizer**: LSTM 1 lapis hidden 64, dropout 0.3, Adam lr 3e-4, batch 256.
8. **Reproduksibilitas**: seed 42/43/44; shuffle per epoch pakai generator
   `seed*1000+epoch`.
9. **XGBoost**: 400 tree, depth 6, lr 0.05, subsample/colsample 0.9, `random_state=seed`.
10. **Padding sequence**: riwayat < window di-pad mengulang lap pertama (edge padding).
11. **Fitur hybrid di train**: estimasi XGBoost out-of-fold GroupKFold(5) per race.

**Protokol evaluasi Run 2** (ditetapkan sebelum run):
- Pemilihan tipe model & seed **hanya dari val 2024** (kriteria: mean F1-makro
  tertinggi antar-seed; seri <0,010 → MAE val terendah) → `models/selection.json`.
- **Test 2025 dievaluasi SEKALI untuk SEMUA seed** (42/43/44), dilaporkan
  mean±std + detail seed terpilih. Tidak ada pemilihan ulang sesudahnya.
- Run 1 (tanpa class weight, `models/run1/`) dievaluasi dengan metrik identik
  untuk perbandingan berdampingan.

## 6d. Run 3 (FINAL) — Protokol, perubahan konfigurasi, dan catatan bias CV

Versi mesin-baca: `models/selection_criteria_v3.json` (ditulis SEBELUM eksperimen).

**Protokol seleksi:**
- CV kronologis 2 fold, grup per race: cv1 (train 2021-22 → val 2023),
  cv2 (train 2021-23 → val 2024). Scaler/encoder/bobot HANYA fit per fold-train.
- Metrik primer: regresi = CV MAE rata-rata; klasifikasi = CV F1-makro rata-rata.
  Pemenang dipilih **TERPISAH per target** (boleh dari konfigurasi berbeda).
- Stop rule: berhenti bila MAE membaik <0,15 lap **DAN** F1 membaik <0,010.
- Budget maks 10 konfigurasi; terpakai 8 (K01-K08), semua dicatat di
  `logs/run3_experiments.jsonl` termasuk yang gagal/memburuk.
- Test 2025 dievaluasi **SEKALI** di akhir, tidak untuk seleksi, tidak ada
  pemilihan ulang. Test 2025 sudah pernah dilihat pada Run 1 & Run 2 untuk
  pelaporan — dinyatakan secara transparan, tidak dipakai untuk keputusan Run 3.

**Perubahan konfigurasi di luar batasan awal (semua diputuskan dari CV/train):**
1. Model regresi & klasifikasi **terpisah** (tanpa loss gabungan MSE+CE).
2. Target regresi diuji dengan log1p (K02) → **ditolak**, MAE CV lebih buruk.
3. LSTM Run 3: Huber loss pada target log1p terstandardisasi, clip fitur ±4
   (jalur NN saja), epoch tetap 12 tanpa early stop di fold val.
4. Class weight klasifikasi: sqrt-inverse-frequency cap 10 ("sqrt10"), lebih
   halus dari Run 2 (inv-freq cap 20).
5. Tuning Optuna 25 trial TPE (seed 42) per target, GroupKFold(3) di pool 2021-2023.
6. Pit window API menggunakan **split-conformal** dari residual cv2
   (q=7,38 lap, coverage kalibrasi 80,01%), BUKAN P10-P90 mentah quantile
   regression (coverage hanya 52,8% → terlalu sempit, tidak dipakai).
7. Head biner "masih pit?": DNF tanpa pit berikutnya = **sensor** (label NaN,
   di-flag `is_dnf`, TIDAK dilabel 0). 9.853 baris sensor dari 106.260.

**⚠️ Catatan bias CV (wajib diketahui pembaca hasil):**
Pool tuning Optuna adalah 2021-2023, yang **overlap dengan validation fold cv1
(val 2023)**. Artinya cv1 berpotensi bias optimis untuk K03/K04. Fold **cv2
(val 2024) bersih** karena 2024 tidak masuk pool tuning.

Perbandingan cv2-only (fold bersih):

| Config | cv1 MAE (⚠️ overlap) | cv2 MAE (bersih) |
|---|---|---|
| K03_xgb_optuna_reg | 4,851 | **4,773** |
| K05_lstm (3 seed mean) | 5,201±0,039 | **5,587±0,012** |
| Selisih K03 vs LSTM | 0,350 | **0,813** |

Keunggulan K03 justru **LEBIH BESAR di fold bersih** (0,813 vs 0,350 lap).
Jika kontaminasi cv1 menggelembungkan hasil K03, selisih cv2 seharusnya menyempit —
yang terjadi sebaliknya. Seleksi K03 tetap valid berdasarkan bukti cv2 yang bersih;
cv1 jika ada biasnya justru *under-estimate*. Angka CV rata-rata (gabungan cv1+cv2)
yang dilaporkan di §10 tetap mencantumkan kedua fold untuk transparansi, tetapi
pembaca harus menimbang cv2 sebagai bukti utama.

## 7. Menjalankan API

```bash
python -m app.main          # http://localhost:8000
# dokumentasi interaktif: http://localhost:8000/docs
```

### Endpoint

| Endpoint | Fungsi |
|---|---|
| `GET /health` | Cek API hidup & model termuat |
| `GET /meta/seasons` | Daftar musim tersedia |
| `GET /meta/races?year=2024` | Daftar race satu musim |
| `GET /meta/teams?year=2024` | Daftar tim + **2 pembalapnya** |
| `GET /predict/replay?year=2024&round=5&team=Red%20Bull&lap=18` | Prediksi kedua pembalap tim pada lap itu |
| `GET /predict/replay/stream?year=2024&round=5&team=Red%20Bull` | SSE per lap (untuk animasi replay) |
| `GET /predict/live?team=...` | **Tahap 2** — saat ini menjawab `501` |
| `GET /model/info` | Versi model, metrik, tanggal training |

### Contoh `curl`

```bash
curl "http://localhost:8000/predict/replay?year=2024&round=5&team=Red%20Bull&lap=18"
```

Contoh response (Run 3, disederhanakan):

```json
{
  "year": 2021, "round": 1, "lap": 10, "team": "Alfa Romeo",
  "race_name": "Bahrain Grand Prix", "track_status": "GREEN",
  "drivers": [
    {
      "code": "GIO", "driver_id": "antonio_giovinazzi",
      "current_compound": "MEDIUM", "tyre_life": 10,
      "predicted_laps_to_pit": 8, "predicted_pit_lap": 18,
      "next_compound": {"label": "HARD",
        "probabilities": {"SOFT": 0.08, "MEDIUM": 0.17, "HARD": 0.72,
                          "INTERMEDIATE": 0.03, "WET": 0.00}},
      "probabilitas_masih_pit": 0.9412,
      "jendela_pit": {
        "lap_bawah": 11, "lap_atas": 25,
        "tingkat_keyakinan": 0.8, "coverage_kalibrasi": 0.8001,
        "setengah_lebar": 7.3818
      },
      "peringatan": {
        "compound_basah_tidak_andal": false,
        "bucket_jarak": "6-15",
        "mae_bucket_test": 3.7804,
        "catatan": []
      }
    },
    {"code": "RAI", "...": "..."}
  ],
  "note": "Prediksi observasional berdasarkan pola keputusan tim pada data 2021-2025, bukan strategi optimal.",
  "model_version": "0.3.0"
}
```

> **Catatan field Run 3:**
> - `probabilitas_masih_pit`: keluaran head biner terpisah (DNF di-flag sensor,
>   bukan dilabel 0). Null jika model biner tidak dimuat.
> - `jendela_pit`: interval **split-conformal** dari residual cv2, BUKAN P10-P90
>   mentah quantile regression (yang coverage-nya hanya 52,8%). Selalu gunakan
>   rentang ini, bukan `predicted_pit_lap` tunggal, untuk keputusan tampilan.
> - `peringatan`: caveat kontekstual. `compound_basah_tidak_andal=true` saat
>   prediksi INTERMEDIATE/WET (data sangat sedikit, F1≈0). `bucket_jarak` +
>   `mae_bucket_test` memberi akurasi historis sesuai jarak prediksi; bucket
>   ">15" selalu disertai catatan agar tidak mengandalkan titik prediksi.

### Contoh `fetch` dari website

```js
// Prediksi satu lap
const res = await fetch(
  "http://localhost:8000/predict/replay?year=2024&round=5&team=Red Bull&lap=18"
);
const data = await res.json();
console.log(data.drivers[0].predicted_pit_lap, data.drivers[0].next_compound.label);

// Streaming replay per lap (SSE) — untuk animasi
const es = new EventSource(
  "http://localhost:8000/predict/replay/stream?year=2024&round=5&team=Red Bull"
);
es.addEventListener("lap", (e) => {
  const perLap = JSON.parse(e.data);
  // perbarui grafik/animasi dengan perLap.drivers[...]
});
es.addEventListener("end", () => es.close());
```

> Ubah `F1_API_CORS_ORIGINS` di `.env` ke domain website Anda saat produksi.

## 8. Docker & Deploy

### Docker lokal

```bash
docker compose up --build     # API di http://localhost:8000
```

Image sudah memuat `data/processed` + `models` (lihat `Dockerfile`), jadi
ekstraksi data & training cukup dilakukan sekali di laptop Anda.

### Render

1. Push repo ke GitHub.
2. **New → Web Service** → pilih repo → Runtime: **Docker**.
3. Pastikan `data/processed/` dan `models/` ikut ter-commit / tersedia saat build
   (atau pasang **Persistent Disk** dan jalankan ekstraksi+training di shell Render).
4. Deploy. Instance terkecil (512 MB–2 GB RAM) cukup untuk inference CPU.

### Railway

1. **New Project → Deploy from GitHub repo**.
2. Railway otomatis mendeteksi `Dockerfile`.
3. Deploy, lalu aktifkan **Public Networking** untuk mendapat URL.

Catatan deploy: platform serverless seperti Vercel **tidak cocok** (model +
cache besar, bukan fungsi pendek). Render/Railway dengan container lebih tepat.

## 9. Menjalankan Test

```bash
python -m pytest tests -v
```

Cakupan: validasi skema Pydantic, semua endpoint (Jolpica/FastF1 di-mock),
aturan 2 pembalap per tim, dan **unit test anti-leakage temporal**.

## 10. Ringkasan Hasil Evaluasi — Run 1 / Run 2 / Run 3 Berdampingan

Detail lengkap: `models/metrics.json` (Run 1+2), `models/run3/metrics_run3.json`,
`models/run3/metrics_binary.json`, `logs/run3_experiments.jsonl` (semua 8 config CV).

**Model final (Run 3, dipilih HANYA dari CV 2021-2024):** XGBoost regresi K03
(Optuna-tuned) + XGBoost klasifikasi K04 (Optuna-tuned, sqrt10 weights) +
head biner "masih pit?" + pit window split-conformal. Versi 0.3.0.

### 10.1 Test 2025 (dievaluasi SEKALI per run; Run 2 = mean±std 3 seed)

| Model | MAE | ±1 | ±3 | ±5 | Bucket 0-5 | 6-15 | >15 | F1-makro | Acc |
|---|---|---|---|---|---|---|---|---|---|
| Baseline mayoritas | 7,37 | 12,7% | 29,6% | 46,3% | 7,03 | **2,45** | 13,76 | 0,151 | 0,431 |
| Baseline tyre_life | 7,90 | 12,5% | 29,2% | 46,0% | **3,95** | 5,86 | 13,44 | 0,339 | **0,616** |
| R1 XGBoost | 6,75 | 12,0% | 33,4% | 50,6% | 4,53 | 5,10 | 10,48 | 0,340 | 0,551 |
| R1 LSTM | 6,05 | 12,0% | 35,5% | 54,4% | 5,82 | 3,97 | **8,81** | 0,374 | 0,584 |
| R1 Hybrid | 6,65 | 12,2% | 34,8% | 52,0% | 4,23 | 4,31 | 11,39 | 0,248 | 0,442 |
| R2 XGBoost | 6,69±0,05 | — | — | — | — | — | — | 0,268±0,00 | 0,539±0,00 |
| R2 LSTM | **5,99±0,06** | — | 35,3% | 54,1% | 5,65 | 3,74 | 9,05 | 0,337±0,01 | 0,448±0,02 |
| R2 Hybrid | 6,77±0,13 | — | — | — | — | — | — | 0,263±0,03 | 0,351±0,06 |
| **R3 K03+K04** | 6,08 | 12,4% | **36,6%** | **55,6%** | 4,97 | 3,78 | 9,80 | **0,375** | 0,555 |
| R3 binary head | — | — | — | — | — | — | — | acc 0,878 / F1 0,908 / AUC 0,954 | — |

### 10.2 Temuan jujur (tidak disamarkan)

- **(a) Regresi TIDAK membaik di test vs Run 2 LSTM** (6,08 vs 5,99; selisih
  +0,09 lap ≈ 1,6σ Run 2). K03 menang telak di CV (4,81 vs 5,39 LSTM; cv2-only
  4,77 vs 5,59) tetapi transfer ke test 2025 sedikit lebih lemah. Ini hasil
  negatif yang jujur untuk target regresi — CV dan test bisa divergen.
- **(b) Klasifikasi F1 TERBAIK** (0,375), unggul tipis dari R1 LSTM (0,374,
  selisih 0,0009 ≈ noise) dan jelas dari R2 LSTM (0,337, +0,038). Optuna +
  sqrt10 berhasil memulihkan F1 yang turun di Run 2 akibat class weight agresif.
- **(c) Accuracy R3 (0,555) < R1 LSTM (0,584)** tapi F1 lebih tinggi → R3 lebih
  seimbang antar kelas (recall minoritas lebih baik). Sesuai metrik primer
  F1-makro, bukan accuracy yang bias ke kelas mayoritas (HARD 46,8%).
- **(d) Ensemble (K06/K07) dan Quantile (K08) TIDAK memperbaiki** → tidak dipilih
  sesuai stop rule pra-registrasi. Rata-rata XGB+LSTM justru menarik MAE naik
  karena LSTM reg Run 3 (5,39 CV) jauh lebih lemah dari XGB tuned (4,81).
- **(e) Quantile mentah P10-P90 coverage hanya 52,8%** (nominal 80%) → interval
  terlalu sempit, tidak dipakai. Diganti **split-conformal** dari residual cv2:
  q=7,38 lap, coverage kalibrasi **80,01%** (tepat nominal by construction).
- **(f) INTERMEDIATE/WET tetap tidak terpelajari**: test 2025 hanya 94 baris INTER
  (F1=0,005) dan 0 baris WET. Tidak digabung ke kelas lain sesuai instruksi.
  API menandai `compound_basah_tidak_andal=true` saat prediksi jatuh ke kelas ini.
- **(g) Bucket jarak campur**: R3 terbaik di 0-5 (4,97 vs R2-LSTM 5,65) dan
  hit rate ±3/±5, tapi terburuk di >15 (9,80 vs 8,81 R1-LSTM). Prediksi pit
  jauh tetap sulit untuk semua model.
- **(h) Miami 2023 (round 5) & Spain 2025 (round 9) dilewati** sesuai direktif
  (112/114 race). Server FastF1 belum unblock; tidak mempengaruhi Run 3 secara
  material tetapi dicatat sebagai keterbatasan cakupan data.

### 10.3 Audit Hybrid vs Bab III (Eksperimen 5 Run 3)

**Kesimpulan: arsitektur hybrid SESUAI Bab III §2.2.5 secara konseptual.**
Ketidakstabilan berasal dari detail training yang TIDAK dispesifikasikan Bab III,
bukan pelanggaran desain.

| Aspek Bab III | Implementasi | Status |
|---|---|---|
| ŷ₀ = XGB(X_hist) estimasi awal | OOF XGB preds (laps scaled + 5 proba) ditempel di setiap timestep LSTM | ✅ Sesuai rumus H_t = LSTM(X_t, H_{t-1}, ŷ₀) |
| H_t sama, target terpisah | Shared LSTM body, 2 head (reg MSE + cls CE) | ✅ Sesuai |
| Unidirectional (bukan BiLSTM) | LSTM 1 arah | ✅ Sesuai |
| Loss weighting antar task | MSE + CE **tanpa bobot eksplisit** | ⚠️ Gap (Bab III tidak menyebut) |
| Konsistensi ŷ₀ train vs inference | Scaler fit pada **OOF**, inference pakai **full model** | ⚠️ Gap distribution shift |
| Feature clipping NN | Tidak ada di Run 2 hybrid (baru ditambah Run 3 LSTM terpisah) | ⚠️ Gap |

**Penyebab seed 42 hybrid acc 26,5% (Run 2 test):**
Terverifikasi dari `models/metrics.json`: s42=26,48%, s43=40,59%, s44=38,21%
(std 6,2% didorong entirely oleh s42).

1. **MSE mendominasi CE ~48×** (MSE ~77 vs CE ~1,6). `loss = loss_reg + loss_cls`
   polos tanpa λ → gradien regresi menenggelamkan klasifikasi. Head cls dapat
   sinyal lemah/noisy → sangat sensitif inisialisasi.
2. **OOF→full-model distribution shift.** Training lihat OOF preds (model exclude
   fold); inference lihat full-model preds. Scaler `fit_xgb_laps` juga di-fit pada
   OOF. Mapping yang dipelajari LSTM tidak transfer sempurna.
3. **Optimization landscape rugged.** Kombinasi (1)+(2) membuat basin attraksi cls
   head sempit. Seed 42 jatuh di basin buruk → cls head hampir tidak belajar →
   acc collapse. Seed 43/44 landing lebih baik.
4. **Tidak ada feature clipping di Run 2 hybrid.** Outlier ±50σ memperparah gradien.

**Rekomendasi (BELUM diimplementasikan, butuh persetujuan karena menyentuh
arsitektur training):** tambah λ_reg·MSE + λ_cls·CE dengan λ di-tune via CV,
atau normalisasi MSE per-batch agar se-magnitudo dengan CE; konsistenkan sumber
preds XGB (full-model OOF-fold atau calibration layer). Run 3 menghindari masalah
ini dengan **memisahkan model regresi dan klasifikasi sepenuhnya** — tidak ada
loss gabungan, tidak ada OOF fusion, tidak ada instability antar-seed.

### 10.4 Head biner "masih pit?" (Run 3)

Model terpisah, bukan bagian dari arsitektur hybrid. Target: 1 = masih akan pit
lagi sebelum finis; 0 = tidak pit lagi & classified; **NaN = DNF tanpa pit
berikutnya (sensor, 9.853 baris, di-flag `is_dnf`, TIDAK dilabel 0)**.

| Split | Acc | F1 | AUC | Support (0/1) |
|---|---|---|---|---|
| cv1 (val 2023) | 0,876 | 0,903 | 0,941 | 6192/11922 |
| cv2 (val 2024) | 0,878 | 0,908 | 0,953 | 7108/12452 |
| Test 2025 | 0,878 | 0,908 | 0,954 | 7103/12400 |

Stabil across CV/val/test (tidak ada overfitting). Confusion matrix test:
TN=5412 FP=1691 FN=685 TP=11715 → recall kelas-1 94,5%, FN rate rendah 5,5%.

## 11. Keterbatasan (jujur)

**Umum (semua run):**
- Model mempelajari **keputusan aktual tim**, bukan strategi terbaik; bila tim
  salah keputusan, model meniru pola tersebut.
- Data publik (FastF1/Jolpica/Kaggle) tidak memuat semua informasi internal tim
  (suhu ban detail, beban bahan bakar, dsb.) — prediksi pasti kalah informasi
  dibanding pit wall sungguhan.
- **Data latihan bebas (FP) tidak tersedia**: Jolpica menolak endpoint practice
  ("Endpoint does not support final filter"), sehingga fitur hasil FP tidak
  dibentuk. "Hasil sesi sebelumnya" (Bab III) tetap tercakup kualifikasi + sprint.
- Undercut/overcut adalah keputusan taktis manusia; deviasi 1–3 lap antara
  prediksi dan eksekusi aktual adalah hal wajar (lihat laporan subbab 3.2.3).
- Mode live (Tahap 2) belum diimplementasikan; `LiveProvider` masih stub.

**Spesifik Run 3 (model aktif):**
- **Regresi tidak membaik di test vs Run 2 LSTM** (MAE 6,08 vs 5,99) meski menang
  telak di CV. Divergensi CV→test nyata; dilaporkan apa adanya, tidak disamarkan
  dengan memilih ulang berbasis test.
- **Pit window conformal lebar (±7,38 lap)** — mencerminkan ketidakpastian model
  yang sesungguhnya (MAE test ~6 lap). Interval sempit akan under-cover; lebar
  adalah harga kejujuran kalibrasi 80%. Gunakan rentang, bukan titik prediksi.
- **Quantile regression P10/P50/P90 ditinggalkan**: coverage mentah hanya 52,8%
  (nominal 80%) → interval terlalu sempit dan menyesatkan. Split-conformal
  dipilih sebagai pengganti yang terkalibrasi by construction.
- **INTERMEDIATE/WET tidak andal secara permanen**: 94 baris INTER dan 0 baris WET
  di test 2025; F1≈0. API menandai prediksi basah dengan
  `compound_basah_tidak_andal=true` agar konsumen tidak salah tafsir. Kelas TIDAK
  digabung ke dry compounds sesuai instruksi.
- **Head biner tidak memprediksi DNF**: baris DNF-tanpa-pit disensor (9.853 baris),
  jadi model hanya menjawab "masih pit?" untuk skenario non-DNF. Prediksi DNF itu
  sendiri di luar scope dan memerlukan model survival/hazard terpisah (belum dibuat,
  butuh persetujuan).
- **Dua preprocessor terpisah** (reg/clf fit pada has_target rows; binary fit pada
  labeled inference rows) — benar secara metodologis (masing-masing kalibrasi ke
  populasi trainingnya) tetapi menambah kompleksitas serving. Predictor menangani
  otomatis; konsumen API tidak perlu aware.
- **Miami 2023 (round 5) & Spain 2025 (round 9) dilewati** sesuai direktif pengguna.
  112/114 race tersedia. Server FastF1 memblokir IP saat ekstraksi; tidak di-retry
  ulang setelah Run 3 dimulai agar tidak mengganggu protokol. Dicatat sebagai
  keterbatasan cakupan, bukan kesalahan pipeline.
- **Bias CV struktural**: pool tuning Optuna (2021-2023) overlap dengan val fold cv1
  (2023). Fold cv2 (2024) bersih dan mengonfirmasi keunggulan K03 justru LEBIH BESAR
  (selisih 0,813 vs 0,350 lap). Detail di §6d. Pembaca hasil harus menimbang cv2
  sebagai bukti utama, bukan CV rata-rata gabungan.
- **Bucket >15 lap tetap lemah** (MAE 9,80 di test) untuk semua model ML. Prediksi
  pit jarak jauh secara inheren sulit karena strategi bisa berubah drastis akibat
  SC/VSC/cuaca yang belum terjadi saat prediksi dibuat.
