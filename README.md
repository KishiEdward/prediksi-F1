# F1 Pit Strategy Prediction API

Backend API (Python) yang **memprediksi kapan mobil F1 masuk pit dan ban apa yang
dipakai berikutnya**, dibangun sesuai arsitektur **hybrid XGBoost-LSTM** pada
skripsi *"Pengembangan Arsitektur Hibrida XGBoost-LSTM untuk Rekomendasi Strategi
Pit dan Pemilihan Ban Formula One"*.

> **Penting (framing):** keluaran API adalah **prediksi/perkiraan keputusan pit**
> yang dipelajari dari keputusan tim yang *benar-benar terjadi* (ground truth
> observasional, data 2021–2025). Ini **bukan** rekomendasi strategi optimal
> secara matematis.

---

## 1. Cara Kerja (Ringkas untuk Pemula)

1. **Ambil data** — skrip mengunduh data historis (Jolpica F1 API) dan data
   balapan/telemetri (FastF1), lalu menyimpannya sebagai file Parquet di `data/raw`.
2. **Bentuk fitur & target** — `app/ml/features.py` menggabungkan semuanya menjadi
   tabel per *pembalap-race-lap*, lalu membuat dua target:
   - `laps remaining` (regresi): sisa lap menuju pit stop berikutnya.
   - `next compound` (klasifikasi): ban berikutnya — SOFT/MEDIUM/HARD/INTERMEDIATE/WET.
3. **Training** — tiga model dilatih dan dibandingkan (sesuai laporan):
   - **XGBoost** (baseline): fitur pra-balapan + kondisi lap saat ini (tabular).
   - **LSTM** (baseline): urutan 5 lap terakhir (temporal, satu arah).
   - **Hybrid XGBoost-LSTM** (model utama): estimasi awal XGBoost ditempel sebagai
     fitur tambahan di *setiap timestep* LSTM; LSTM punya dua head keluaran
     (regresi + klasifikasi) dari representasi temporal yang sama.
4. **API** — FastAPI memuat model sekali saat startup, lalu melayani prediksi
   mode **replay** (per lap atau streaming SSE). Mode **live** adalah Tahap 2
   (saat ini menjawab `501`).

```
Website ── HTTP ──> FastAPI ──> Predictor ──> RaceStateProvider
                                   │              ├── ReplayProvider (Tahap 1, aktif)
                                   │              └── LiveProvider  (Tahap 2, stub)
                                   └── artifact di models/ (XGBoost + LSTM hybrid)
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
├── scripts/                  # ekstraksi data (01 Jolpica, 02 FastF1)
├── tests/                    # pytest (schema, endpoint, 2 pembalap/tim, leakage)
├── data/raw/                 # Parquet mentah + cache API (tidak masuk git)
├── data/processed/           # dataset hasil feature engineering
├── models/                   # artifact model + metrik + metadata versi
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

# Latih ketiga model (CPU; XGBoost ±menit, LSTM/hybrid ±30–90 menit)
python -m app.ml.train                # default 20 epoch, early stopping
python -m app.ml.train --epochs 5     # uji coba cepat

# Evaluasi + simpan models/metrics.json
python -m app.ml.evaluate
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

Contoh response (disederhanakan):

```json
{
  "year": 2024, "round": 5, "lap": 18, "team": "Red Bull",
  "race_name": "Chinese Grand Prix", "track_status": "GREEN",
  "drivers": [
    {
      "code": "VER", "driver_id": "max_verstappen",
      "current_compound": "MEDIUM", "tyre_life": 14,
      "predicted_laps_to_pit": 6, "predicted_pit_lap": 24,
      "next_compound": {"label": "HARD",
        "probabilities": {"SOFT": 0.05, "MEDIUM": 0.20, "HARD": 0.70,
                          "INTERMEDIATE": 0.03, "WET": 0.02}}
    },
    {"code": "PER", "...": "..."}
  ],
  "note": "Prediksi observasional berdasarkan pola keputusan tim pada data 2021-2025, bukan strategi optimal.",
  "model_version": "0.1.0"
}
```

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

## 10. Keterbatasan (jujur)

- Model mempelajari **keputusan aktual tim**, bukan strategi terbaik; bila tim
  salah keputusan, model meniru pola tersebut.
- Data publik (FastF1) tidak memuat semua informasi internal tim (suhu ban
  detail, beban bahan bakar, dsb.) — prediksi pasti kalah informasi dibanding
  pit wall sungguhan.
- **Data latihan bebas (FP) tidak tersedia**: Jolpica menolak endpoint practice
  ("Endpoint does not support final filter"), sehingga fitur hasil FP tidak
  dibentuk. "Hasil sesi sebelumnya" (Bab III) tetap tercakup kualifikasi + sprint.
- Kelas INTERMEDIATE/WET sangat jarang sehingga akurasinya lebih rendah
  (lihat F1 per kelas di `models/metrics.json`).
- Undercut/overcut adalah keputusan taktis manusia; deviasi 1–3 lap antara
  prediksi dan eksekusi aktual adalah hal wajar (lihat laporan subbab 3.2.3).
- Mode live (Tahap 2) belum diimplementasikan; `LiveProvider` masih stub.
