# AUDIT TEKNIS: Mengapa Performa Prediksi Pit F1 Terbatas

**Sifat:** FASE A (read-only). Tidak ada kode/dataset/model/konfigurasi yang diubah; tidak ada training baru.
Semua analisis memakai **VAL 2024** dan **CV 2021-2024** sebagai dasar; **test 2025 hanya untuk deskripsi kasus**.
Skrip analisis baru: `audit/_common.py`, `audit/a1_ground_truth.py`, `audit/a2_ceiling_features.py`,
`audit/a3_eval_metrics.py`, `audit/a4_segments_rules.py`, `audit/a5_cases.py`.

**Label bukti:** FAKTA (ada bukti file:fungsi:baris / hasil skrip) · HIPOTESIS (cara uji + perkiraan perbaikan maks) · TIDAK DAPAT DIPASTIKAN.

---

## 0. Verifikasi Konteks (sesuai permintaan)

| Klaim konteks | Status | Bukti |
|---|---|---|
| Model final = 2 XGBoost terpisah (reg + clf) + head biner + conformal window | **FAKTA — sesuai** | `models/metadata.json`: `primary_model="xgb_separate"`, `run=3`, `version=0.3.0`; artifacts reg/clf/binary/preprocessor/preprocessor_binary/conformal |
| Hybrid XGBoost-LSTM & LSTM = kandidat tidak terpilih | **FAKTA — sesuai** | `models/selection.json` + README §10; Run 3 winner K03(reg)/K04(clf) dari CV |
| Mode replay lap-per-lap; `/predict/live` stub 501 | **FAKTA — sesuai** | `app/api/main.py:232-234` (`status_code=501`); E2E `scripts/10_e2e_replay.py` live→501 PASS |
| Framing "prediksi keputusan tim nyata", bukan optimal | **FAKTA — sesuai** | `app/schemas/prediksi.py:3-6`, `ResponsReplay.note` |
| Audit lama (rujuk, tak diulang): sequence audit, leakage test, audit hybrid README, verifikasi target vs pitstop | **FAKTA — ada** | `scripts/04_audit_dataset.py`, `tests/test_leakage_realdata.py`, README §10.3 |

**Catatan verifikasi tambahan (baru):** target-vs-pitstop diverifikasi ulang independen di `audit/a1` — **0/2000 mismatch** (FAKTA), mengonfirmasi audit lama.

---

## RINGKASAN EKSEKUTIF

Penyebab dominan terbatasnya performa **bukan arsitektur model**, melainkan **kehilangan informasi di pipeline fitur** ditambah **sifat reaktif keputusan pit**:

1. **SELURUH keluarga fitur status lintasan MATI** (6/44 fitur importance = 0). `is_sc/is_vsc/is_red/laps_since_sc_vsc` konstan; `is_green/is_yellow` nyaris konstan. Akar: filter `IsAccurate==True` menghapus semua lap SC/VSC sebelum flag dibuat. → **FAKTA**
2. **28,7% pit terjadi saat SC/VSC dan 8,5% saat RED** — kondisi pemicu yang **tidak terlihat model**. → **FAKTA**
3. **Pengecualian pit red-flag (Bab III) adalah dead code** — `track_status` tak pernah `"RED"`, jadi `n_red_flag` selalu 0. → **FAKTA**
4. **112 file trackstatus interval (kaya: kode 1/2/4/6/7 + Time) diekstrak tapi TIDAK PERNAH dipakai** feature engineering. → **FAKTA**
5. **Bug rumus F1-makro**: `evaluate.py:76` tanpa `labels=` → di test 2025 (WET absen) F1 dilaporkan 0,3752 padahal macro 5-kelas sejati 0,3001. **Seleksi CV TIDAK tercemar** (WET hadir di cv1/cv2), hanya pelaporan test yang menggelembung. → **FAKTA**
6. **Ketidakcocokan objective vs metrik**: regresi dilatih `reg:squarederror` (MSE) tapi diseleksi/dilaporkan pakai MAE. → **FAKTA** (dampak = HIPOTESIS, kecil)
7. **Akurasi gabungan (timing ±3 DAN compound benar) hanya ~20% (per-peristiwa) / 24% (per-baris)** di val 2024. → **FAKTA**
8. Kasus galat terbesar: **3/3 Monaco 2025** (salah +32 s.d. +38 lap) — model memprediksi stint HARD panjang (lap ~48-51), realitas pit lap 13-17 saat ada **VSC** (terkonfirmasi di trackstatus mentah Monaco). → **FAKTA**

**Kesimpulan inti:** sisa error terutama berasal dari **informasi data yang dibuang (status lintasan)** dan **ketidakpastian intrinsik pit reaktif (cuaca/SC/regulasi)**, bukan dari arsitektur atau target. Ada **2 cacat nyata yang bisa diperbaiki** (pipeline track-status; bug metrik F1) dan **1 penyelarasan rumus** (objective MAE).

---

## 1. GROUND TRUTH & LABEL NOISE (Pemeriksaan #1)

### 1.1 Komposisi pit stop terhadap kondisi lintasan
- **Komponen:** dataset target / kualitas label.
- **Kondisi saat ini:** Dari **4161 pit stop** (jolpica, semua musim), kondisi lintasan pada lap pit (sumber: `TrackStatus` laps mentah, cek digit di posisi mana pun):
  - SC: **1013 (24,3%)** · VSC: **264 (6,3%)** · **SC/VSC gabungan: 1195 (28,7%)** · RED: **354 (8,5%)** · YELLOW: 851 (20,5%).
- **Label:** **FAKTA** — `audit/a1_ground_truth.py` bagian [1]; sumber `data/raw/fastf1/*_laps.parquet`.
- **Dampak kuantitatif:** ~29% pit dipicu kondisi yang **fiturnya nol semua** di model. Jika pit SC/VSC bersifat reaktif dan tak terprediksi tanpa fitur SC, ini menekan plafon hit-rate peristiwa. Perkiraan kasar plafon timing-akurat ≈ 70% (lihat 1.6, HIPOTESIS).
- **Usaha perbaikan:** sedang (perbaiki pipeline track-status — lihat §5.1). **Prioritas: TERTINGGI.**

### 1.2 Pengecualian pit red-flag = dead code
- **Komponen:** `app/ml/features.py::_bentuk_target`.
- **Kondisi saat ini:** Baris 416: `if status_lap.get((s,r,p),"GREEN") == "RED": n_red_flag += 1; continue`. Karena `track_status` di dataset **tak pernah bernilai "RED"** (0 baris di `lap_inference`), kondisi ini **tak pernah benar**.
- **Label:** **FAKTA** — `audit/a1` [2]: "Baris inference dgn track_status=='RED': 0"; `features.py:416`.
- **Bukti kode:**
  ```python
  # features.py:416
  if status_lap.get((s, r, p), "GREEN") == "RED":
      n_red_flag += 1; continue   # tak pernah dieksekusi
  ```
- **Dampak:** 354 pit di lap ber-status RED (mentah) **tidak dibuang** → diperlakukan sebagai keputusan strategis → **label bising**. Bertentangan dengan Bab III subbab 1.4 dan klaim README §6b.
- **Usaha:** kecil-sedang (perbaiki sumber status). **Prioritas: TINGGI.**

### 1.3 Pit berturut-turut (indikasi artifact/double-stack)
- **Kondisi saat ini:** 264 pit terjadi **tepat 1 lap** setelah pit sebelumnya oleh driver sama; 308 dalam ≤2 lap.
- **Label:** **FAKTA** — `audit/a1` [3].
- **Dampak:** berkontribusi pada 3153 baris ber-`target==1` (5,5% dataset). Sebagian mungkin sah (pit beruntun saat SC), sebagian artifact. Menambah noise pada bucket jarak 0-5.
- **Usaha:** kecil (inspeksi manual sampel). **Prioritas: RENDAH.**

### 1.4 Baris berlabel tidak pasti
- **Kondisi saat ini:** `lap_inference` 106.260 baris: **has_target=True 57.390 (54,0%)**, False 48.870 (46,0%, = stint akhir/DNF/tanpa pit berikut). `target_next_compound` NaN = **0** di `lap_dataset` (semua baris target punya compound valid).
- **Label:** **FAKTA** — `audit/a1` [4]; `features.py:460`.
- **Dampak:** 46% baris inference tak berlabel target (benar secara desain — tak ada pit berikutnya untuk diprediksi). Bukan cacat, tapi membatasi jumlah sampel latih.

### 1.5 Distribusi target & kebenaran rumus
- **Kondisi saat ini:** `target_laps_remaining`: min=1, max=68, mean=11,83, median=10, std=8,77; **count==0 = 0** (pit pada lap berjalan tak menghasilkan target 0 karena lap pit terfilter IsAccurate). Persentil 10/25/50/75/90/95/99 = 2/5/10/17/24/28/39.
- **Kebenaran rumus:** cross-check 2000 sampel acak (seed 7) antara `target_laps_remaining` dataset vs rekonstruksi dari pitstop mentah → **0 mismatch (0,00%)**.
- **Label:** **FAKTA** — `audit/a1` [5],[7]; rumus `features.py:432,435` (`berikutnya=[p for p in pits if p>=t]; laps_rem=berikutnya[0]-t`).
- **Dampak:** **rumus target BENAR**. Distribusi lebar (1-68, std 8,77) → MAE ~5-6 lap secara intrinsik sulit diturunkan jauh; right-skew menjelaskan mengapa bucket >15 paling buruk.

### 1.6 Estimasi batas atas akurasi akibat label bising
- **Label:** **HIPOTESIS.**
- **Perkiraan:** jika 28,7% pit dipicu SC/VSC (tak terlihat) dan bersifat reaktif, plafon hit-rate timing-akurat per peristiwa ≈ **70%**; saat ini hit ≤3 lap per-peristiwa (val 2024) = 39,6%, per-baris 41,0%. Artinya masih ada ruang, tapi sebagian error tak tereduksi tanpa fitur SC.
- **Cara uji (Fase B):** perbaiki track-status → re-run CV → ukur ΔMAE/Δhit-rate, terutama di sirkuit high-SC (Monaco, Jeddah, Baku, Singapore). Perkiraan perbaikan maks: MAE val 4,77 → ~4,0-4,4 (lihat §5.1).

---

## 2. PLAFON INFORMASI (Pemeriksaan #2)

### 2.1 Varians target yang dijelaskan fitur sederhana vs model
- **Kondisi saat ini (val 2024):** R² non-parametrik (mean-grup, tanpa training):
  - `tyre_life` saja: **0,139**
  - `(tyre_life, current_compound)`: **0,204**
  - `(stint_number, current_compound)`: **0,070**
  - **Model Run 3 (reg): 0,432** (MAE 4,773)
- **Label:** **FAKTA** — `audit/a2_ceiling_features.py` [C].
- **Dampak:** model menambah **+0,228 R²** di atas tyre_life+compound → model **bukan** sekadar lookup umur ban; ada sinyal nyata. Tapi **57% varians tetap tak terjelaskan** → plafon informasi dari fitur yang ada.
- **Usaha:** — (diagnostik). **Prioritas: —**

### 2.2 Konsistensi dua pembalap satu tim
- **Kondisi saat ini:** |selisih laps_to_pit| antar teammate pada stint_number sama: **n=27** (kecil), median **4,0**, mean 6,11, p90 14,4; 37% ≤2 lap, 63% ≤5 lap.
- **Label:** **TIDAK DAPAT DIPASTIKAN** (sampel terlalu kecil; proxy "situasi serupa" via stint_number kasar).
- **Dampak:** indikasi awal bahwa timing pit bersifat **individual/reaktif**, bukan fungsi kondisi bersama yang sinkron → mendukung adanya plafon akurasi. Perlu Fase B dengan definisi "situasi serupa" lebih ketat (match tyre_life±1, compound sama, lap±2, gap serupa).
- **Cara uji (Fase B):** perbesar sampel dengan matching kondisi eksplisit; hitung varians residual antar-teammate sebagai proxy noise ireduksibel.

---

## 3. KURVA BELAJAR (Pemeriksaan #3 — tanpa training baru)

- **Kondisi saat ini:**
  - K03: cv1 (train 21-22→val 23) MAE **4,851**; cv2 (train 21-23→val 24) MAE **4,773**. K01: cv1 5,070 / cv2 5,080.
  - Run 2 val 2024 (train 2021-23): LSTM MAE 5,36 F1 0,383; XGB MAE 5,02 F1 0,345; Hybrid MAE 5,10 F1 0,281.
- **Label:** **FAKTA** (angka CV/val dari `logs/run3_experiments.jsonl`, `models/metrics.json`; `audit/a2` [E], `audit/a4` [C]).
- **Keterbatasan:** cv1 vs cv2 memakai **tahun val berbeda** → bukan kurva belajar bersih. Bukti bersih berasal dari **ablation Run 2** (val 2024 tetap, train ±2021) yang mencatat penambahan 2021 **memperbaiki** metrik val (LSTM F1 0,349→0,383; arah konsisten).
- **Dampak / kesimpulan:** error **turun saat musim train bertambah**; model **belum jenuh data** (data-limited). Menambah musim (2026 bila tersedia) kemungkinan membantu.
- **Usaha:** — (butuh data baru). **Prioritas: SEDANG** (tergantung ketersediaan data 2026).

---

## 4. KOREKTNES EVALUASI & DEFINISI METRIK (Pemeriksaan #4)

### 4.1 Bug rumus F1-makro (label tidak eksplisit)
- **Komponen:** `app/ml/evaluate.py::_metrik_klasifikasi`.
- **Kondisi saat ini:** baris 76-77 `f1_score(y_true, y_pred, average="macro", zero_division=0)` **tanpa `labels=`**, sedangkan `per_kelas` (baris 68-69) **pakai `labels=range(5)`**. Saat sebuah kelas absen dari y_true∪y_pred (WET di test 2025), sklearn menyimpulkan label hanya dari kelas hadir → WET **dikeluarkan** dari rata-rata macro.
- **Label:** **FAKTA** (terverifikasi numerik).
- **Bukti:**
  ```python
  # evaluate.py:68-69  (per_kelas: BENAR, 5 kelas)
  precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, ...)
  # evaluate.py:76-77  (f1_macro: BUG, labels tidak diteruskan)
  f1_score(y_true, y_pred, average="macro", zero_division=0)
  ```
  - Test 2025: tercatat **0,3752** (= mean per_kelas 4 kelas hadir) vs macro 5-kelas sejati **0,3001** (WET F1=0 ikut dihitung). Inflasi **+0,0751 (+25% relatif)**.
  - Val 2024 & cv1/cv2: **WET hadir** (sup 25/59) → inferred == 5-kelas (0,3575; selisih 0,0000). **Seleksi CV TIDAK tercemar.**
  - Inkonsistensi internal: mean `per_kelas.f1` test = 0,3001 ≠ `f1_macro` 0,3752.
- **Dampak:** (a) angka F1 test **semua run** (1/2/3) menggelembung ~0,075 karena WET absen di test 2025 — **ranking antar-run tetap valid** (setiap run pakai fungsi & test set sama), tapi nilai absolut overstated; (b) README §10 menyebut 0,375 seolah macro 5-kelas.
- **Usaha:** **trivial** (tambah `labels=list(range(5))` di 3 pemanggilan macro). **Prioritas: TINGGI (integritas pelaporan).**

### 4.2 Verifikasi independen metrik regresi
- **Kondisi saat ini:** recompute dari prediksi model final di val 2024: MAE **4,7731**, RMSE **6,2911**, hit≤1 0,1338, ≤3 0,4097, ≤5 0,6346, acc compound 0,6160 — **identik** dengan `models/run3/metrics_run3.json`.
- **Label:** **FAKTA** — `audit/a3_eval_metrics.py` [B]. **Metrik regresi BENAR.**

### 4.3 Metrik gabungan (timing ∧ compound) — per baris vs per peristiwa
- **Kondisi saat ini (val 2024, 717 peristiwa pit, 12.144 baris):**
  - **Per-baris:** k=3 → **24,18%** benar; k=5 → **37,72%**.
  - **Per-peristiwa** (median pred pit lap per stint; compound pada lap terdekat pit): k=3 → **19,67%** (timing≤3 39,6%, compound 59,1%); k=5 → **33,61%** (timing≤5 61,9%).
  - MAE timing per-peristiwa **4,666** vs per-baris **4,773**.
- **Label:** **FAKTA** — `audit/a3` [C].
- **Penjelasan selisih:** per-baris menimbang tiap **lap** (stint panjang menyumbang banyak baris) dan memasukkan horizon jauh (target besar, lebih sulit) → biasanya lebih rendah untuk combined ketat, tapi di sini per-baris combined justru lebih tinggi karena banyak baris horizon-menengah (6-15, MAE terbaik 3,02) mendominasi. Per-peristiwa menimbang tiap **keputusan pit** sama rata dan memakai titik prediksi terdekat-pit → lebih ketat untuk compound (59,1% vs 61,6% semua baris).
- **Dampak:** hanya **~1 dari 5** keputusan pit diprediksi benar **secara gabungan** (lap ±3 DAN compound). Ini metrik paling jujur untuk "aksiabilitas"; belum pernah dilaporkan sebelumnya.
- **Usaha:** kecil (tambah ke evaluasi rutin). **Prioritas: SEDANG.**

### 4.4 Compound metrik hanya 1-5 lap sebelum pit
- **Kondisi saat ini (val 2024):** n=3443, acc **0,5992**, F1-5kelas **0,3628** — **tidak lebih baik** dari semua baris (acc 0,6160, F1 0,3575).
- **Label:** **FAKTA** — `audit/a3` [D].
- **Dampak:** model **tidak makin akurat** soal compound saat pit mendekat → pilihan compound ditentukan strategi (tak terobservasi dari data lap), bukan sekadar jarak. Membatasi nilai aksiabel prediksi compound.

### 4.5 Sampel evaluasi & kebocoran (cek baru saja)
- **Kondisi saat ini:** val 2024 = 12.144 baris, semua `has_target`, season=[2024] saja, target NaN=0; model final dilatih `season<=2023` (`scripts/06_run3_final.py:48-50`). Prediksi per-baris Run 3 memakai fitur lap-berjalan yang semuanya kausal (shift/cummin/merge_asof backward).
- **Label:** **FAKTA** — `audit/a3` [E]. **Tidak ditemukan kebocoran baru** pada jalur evaluasi/prediksi per-baris. (Leakage sequence/truncated-rebuild: rujuk `tests/test_leakage_realdata.py` — tidak diulang sesuai instruksi.)

---

## 5. KUALITAS FITUR (Pemeriksaan #5)

### 5.1 Keluarga fitur status lintasan MATI (temuan terbesar)
- **Komponen:** `app/ml/features.py::_bangun_fitur_inrace` (baris 279-282 filter; 331-351 fitur status).
- **Kondisi saat ini:**
  - `is_sc`, `is_vsc`, `is_red`, `laps_since_sc_vsc`: **nunique=1** (konstan 0 / 10) di seluruh 106.260 baris.
  - `is_green`/`is_yellow`: nyaris konstan (99,6% / 0,4%).
  - **6/44 fitur importance = 0** pada model regresi, dan keenamnya **tepat** keluarga track-status: `['is_green','is_yellow','is_sc','is_vsc','is_red','laps_since_sc_vsc']`.
  - `track_status` hanya berisi GREEN (105.855) & YELLOW (405) — tak ada SC/VSC/RED/ENDED.
- **Label:** **FAKTA** — `audit/a2` [A],[B]; trace filter di bawah.
- **Akar masalah (FAKTA, terverifikasi trace):**
  ```python
  # features.py:279-282 — filter menghapus lap SC/VSC SEBELUM flag dibuat
  if "IsAccurate" in df.columns: df = df[df["IsAccurate"] == True]
  if "Deleted"   in df.columns: df = df[df["Deleted"].isna() | (df["Deleted"] == False)]
  ```
  FastF1 menandai lap di bawah SC/VSC sebagai `IsAccurate=False`. Trace per race:
  - 2021 R1: sebelum filter GREEN 968, **SC 38, VSC 17**, YELLOW 2, ENDED 2 → sesudah filter: GREEN 850, YELLOW 2 (**SC/VSC hilang semua**).
  - 2025 R1: sebelum filter GREEN 624, **SC 286**, YELLOW 17 → sesudah: GREEN 561, YELLOW 7 (**286 SC hilang**).
  Bug sekunder (termask): `features.py:331` `str.extract(r"(\d)")[0]` hanya ambil **digit pertama** dari TrackStatus multi-kode ("126"→"1" GREEN, kehilangan VSC=6).
- **Data kaya yang terbuang:** **112 file `*_trackstatus.parquet`** (kolom Time, Status, Message; kode 1/2/4/6/7) **diekstrak tapi tak pernah dipakai**. Contoh Monaco 2025 (`2025_08_trackstatus.parquet`) memuat periode **VSC (kode 6)** dan "126"/"16" — persis kondisi yang memicu pit dini.
- **Dampak kuantitatif:** model buta terhadap pemicu ~28,7% pit. Korelasi pendukung (FAKTA, `audit/a4` [A]): sirkuit high-SC punya MAE terburuk — **Monaco 10,30; Jeddah 9,48; Marina Bay 6,52** vs sirkuit low-SC **Bahrain 1,92; Catalunya 3,17; Spa 3,19**.
- **Label dampak:** **HIPOTESIS** (kausalitas SC→error): cara uji = perbaiki fitur status (merge_asof interval trackstatus by Time, jangan filter SC/VSC untuk konteks), re-run CV, ukur ΔMAE terutama sirkuit high-SC. **Perkiraan perbaikan maks: MAE val 4,77 → ~4,0-4,4** (±0,4-0,8 lap); compound F1 kasus crossover membaik.
- **Usaha:** **sedang** (re-feature + re-run CV; tanpa ubah target/arsitektur). **Prioritas: TERTINGGI.**

### 5.2 Feature importance & dominasi fitur struktural
- **Kondisi saat ini (gain):**
  - REG top: `total_laps` 0,093 · `tyre_life` 0,088 · `compound_hard` 0,081 · `laps_to_race_end` 0,075 · `delta_pb_so_far` 0,050 · `circuit_id` 0,041.
  - CLF top: `compound_medium` 0,109 · `total_laps` 0,080 · `circuit_len_km` 0,058 · `track_temp` 0,057 · `race_progress` 0,054.
  - BINARY top: `race_progress` **0,277** · `laps_to_race_end` 0,120 · `stint_number` 0,067 · `tyre_life` 0,053.
- **Label:** **FAKTA** — `audit/a2` [B].
- **Dampak:** model bersandar pada fitur **statis/struktural** (posisi balapan, umur ban, sirkuit). Fitur **taktis live** (`gap_to_car_ahead`, `position_delta_prev`, `delta_prev_lap`) importance rendah → konsisten dengan buta-taktik (SC/undercut tak terlihat). Binary head ~40% dari `race_progress`+`laps_to_race_end` → tugasnya memang lebih "struktural" (menjelaskan acc 0,878).
- **Usaha:** — . **Prioritas:** info untuk Fase B (fitur tactical butuh konteks SC).

### 5.3 Fitur Bab III yang diharapkan tapi tidak informatif
- **Label:** **FAKTA.** Keenam fitur status lintasan (§5.1) diharapkan Bab III (subbab 3.2.3: SC/VSC sebagai input) tetapi **nol informasi**. README §6b mengklaim "SC/VSC menjadi fitur input" dan "pit red flag dibuang" → **tidak akurat** terhadap perilaku aktual (mismatch dokumentasi-implementasi).
- **Usaha:** kecil (koreksi README setelah §5.1 diperbaiki). **Prioritas: TINGGI (akurasi dokumentasi skripsi).**

---

## 6. PELANGGARAN ATURAN DUA COMPOUND KERING (Pemeriksaan #6 — hanya diukur)

- **Komponen:** keluaran klasifikasi compound.
- **Definisi (dokumentasikan):** untuk tiap peristiwa pit, himpunan compound kering (SOFT/MEDIUM/HARD) yang dipakai driver **sebelum** pit (dari riwayat `current_compound`). "Pelanggaran potensial" = compound prediksi berikutnya **kering** DAN `union(dry_so_far, prediksi)` berukuran 1 (belum ada 2 spec kering berbeda). Upper bound (stop berikutnya bisa memperkenalkan compound baru).
- **Kondisi saat ini (val 2024, 717 peristiwa):** pelanggaran potensial oleh **PREDIKSI = 57 (7,95%)**; oleh **AKTUAL = 60 (8,37%)**.
- **Label:** **FAKTA** — `audit/a4_segments_rules.py` [B].
- **Dampak:** prediksi **≈** aktual (bahkan sedikit lebih rendah) → model **TIDAK** menghasilkan pilihan compound yang melanggar aturan lebih sering dari realitas. **Tidak ada patologi rule-violation.** (Catatan: angka "aktual" 8,37% mencerminkan batas atas definisi, bukan pelanggaran nyata.)
- **Usaha:** — . **Prioritas: — (temuan negatif yang menenangkan).**

---

## 7. SEGMENTASI ERROR (Pemeriksaan #7 — dengan jumlah sampel)

Semua pada **val 2024** (`audit/a4` [A]); n = jumlah baris.

| Segmen | Kategori | n | MAE | acc compound |
|---|---|---|---|---|
| **Jarak ke pit** | >15 | 3396 | **7,803** | 0,640 |
| | 0-5 | 3443 | 4,481 | 0,599 |
| | 6-15 | 5305 | **3,023** (terbaik) | 0,612 |
| **Umur ban** | 0-5 | 2359 | 5,587 | 0,620 |
| | 6-15 | 5650 | 5,065 | 0,611 |
| | 16-25 | 3091 | 4,051 | 0,596 |
| | >25 | 1044 | **3,492** | 0,696 |
| **Nomor stint** | 1 | 6972 | 5,310 | **0,716** |
| | 2 | 4373 | 3,952 | 0,463 |
| | 3 | 696 | 4,578 | 0,605 |
| | 4 | 103 | 4,612 | 0,447 |
| **Sirkuit (terburuk→terbaik)** | monaco | 273 | **10,304** | 0,582 |
| | jeddah | 222 | 9,479 | 0,459 |
| | interlagos | 411 | 7,102 | **0,000** |
| | marina_bay | 565 | 6,524 | 0,789 |
| | … | | | |
| | catalunya | 720 | 3,170 | 0,662 |
| | bahrain | 605 | **1,924** | 0,228 |
| **Tim (terburuk→terbaik)** | sauber | 1333 | 5,771 | 0,574 |
| | williams | 1036 | 5,275 | 0,566 |
| | … | | | |
| | ferrari | 1204 | **4,101** | 0,720 |
| **Compound saat ini** | INTERMEDIATE | 1253 | 5,888 | **0,104** |
| | MEDIUM | 5751 | 4,927 | 0,764 |
| | HARD | 4353 | 4,408 | 0,586 |
| | SOFT | 771 | 3,849 | 0,530 |
| **Track status** | GREEN 12131 / YELLOW 13; is_sc=is_vsc=is_red=0 | | | **(tak bisa disegmentasi — fitur mati)** |

- **Label:** **FAKTA** (semua angka dari `audit/a4`).
- **Pola penting:**
  1. **Jarak:** >15 lap terburuk (7,80) — horizon jauh sulit; 6-15 terbaik (3,02).
  2. **Sirkuit:** korelasi kuat dengan frekuensi SC — street/high-SC (Monaco, Jeddah, Marina Bay) terburuk; low-SC (Bahrain, Catalunya, Spa) terbaik. **Mendukung §5.1.**
  3. **Compound acc ekstrim rendah** di Interlagos (0,000), Silverstone (0,113), Villeneuve (0,161), INTERMEDIATE (0,104) → balapan variabe/cuaca; strategi compound tak tertangkap.
  4. **Stint 1** compound acc tinggi (0,716) vs **stint 2** (0,463) → stint lanjut lebih tak pasti.
- **Per-season (CV, FAKTA):** cv1 (val 2023) MAE 4,851 / hit≤3 40,9%; cv2 (val 2024) MAE 4,773 / hit≤3 41,0% → **stabil lintas musim**, tak ada degradasi.
- **Usaha/Prioritas:** segmentasi mengarahkan perbaikan ke §5.1 (SC) dan penanganan cuaca/INTER.

---

## 8. PREDIKSI DINAMIS PER-LAP (Pemeriksaan #8)

- **Komponen:** `app/services/predictor.py::_prediksi_xgb_separate`.
- **Kondisi saat ini:** predictor mengambil `provider.get_driver_history(..., until_lap=state.lap)` lalu **hanya baris terakhir** (`hist.iloc[[-1]]`) untuk fitur XGB per-baris. Fitur lap-N dibangun dari operasi kausal: `delta_prev_lap` (shift), `delta_pb_so_far` (cummin), `gap_*` (per-lap grup), cuaca (`merge_asof` backward, tolerance 600s).
- **Label:** **FAKTA** — `predictor.py` (`_prediksi_xgb_separate`, `iloc[[-1]]`); `features.py:289-293,311-315,363-367`.
- **Verifikasi:** Run 3 = model **per-baris tabular** (bukan sequence) → tak ada risiko "melihat lap depan" dari window. Uji truncated-rebuild nyata (`tests/test_leakage_realdata.py`) sudah memverifikasi batas informasi ≤ lap N (dirujuk, tak diulang).
- **Dampak:** **jalur prediksi dinamis BENAR**; tidak ditemukan kebocoran temporal baru. **Bukan** sumber error.
- **Usaha/Prioritas:** — (sudah benar).

---

## 9. ANALISIS KASUS (test 2025 — deskriptif saja)

Seed acak = **20261115**. q conformal = **7,38 lap** (window = pred ± 7 lap). Sumber: `audit/a5_cases.py`.

### 9.A — 5 peristiwa pit acak
| Kasus | Kondisi | Pred (lap terdekat pit) | Aktual | Galat | Catatan teknis |
|---|---|---|---|---|---|
| BOR Belgium R13 | INTERMEDIATE | pit lap 21 | **12** | **+9** | crossover cuaca (INTER→dry dini); model prediksi MEDIUM lebih lambat. Model buta kondisi kering-lintasan. |
| HAD R21 | SOFT (tyre_life 15-20!) | 24 | 18 | +6 | soft umur sangat tua; model kurang tangkap urgensi degradasi. |
| ALO British R12 | INTERMEDIATE | 21 | 11 | **+10** | sama seperti BOR: pit INTER dini (cuaca), tak terprediksi. |
| LEC Monaco R8 | MEDIUM | 37 | 22 | **+15** | Monaco high-SC; model harap stint panjang, realitas pit dini (SC/VSC). |
| HAM Bahrain R4 | MEDIUM | 19 | 17 | **+2** | **kasus baik**: sirkuit low-SC, stop terencana → akurat. |

### 9.B — 3 galat terbesar (TERPISAH dari sampel acak)
**Ketiganya Monaco 2025 R8** — pola sistematis:
| Kasus | Compound | tyre | Pred pit lap | Aktual | Galat | Window conformal |
|---|---|---|---|---|---|---|
| COL | HARD | 7-12 | ~48-51 | **13** | **+38** | [44,58] — **aktual jauh di luar** |
| ALO | HARD | 11-16 | ~46-49 | **16** | **+33** | [39,53] — meleset total |
| STR | HARD | 12-17 | ~48-50 | **17** | **+32** | [41,55] — meleset total |

- **Penjelasan teknis (FAKTA + HIPOTESIS):**
  - **FAKTA:** model memprediksi stint HARD panjang (lap ~48-51) = pola "Monaco satu-stop" yang dipelajari dari 2021-2024. Realitas: pit dini lap 13-17.
  - **FAKTA:** `2025_08_trackstatus.parquet` memuat periode **VSC (Status kode 6)** dan "126"/"16" → ada neutralisasi lintasan di sekitar lap tersebut.
  - **HIPOTESIS (kuat):** pit dini Monaco 2025 dipicu **VSC/SC** (dan/atau perubahan regulasi 2025 yang mendorong multi-stop). Karena `is_vsc=0` (fitur mati), model tak punya sinyal → galat katastropik. **Cara uji:** perbaiki track-status (§5.1), evaluasi ulang Monaco 2025.
  - **FAKTA (conformal):** window ±7 lap memberi coverage marjinal 80% **rata-rata**, tapi **gagal total** pada rezim atipikal ini (aktual lap 13 vs window [44,58]) → coverage **kondisional** buruk di horizon panjang/kejadian langka.
- **Konteks galat peristiwa (test 2025, FAKTA):** median |galat pit lap| per peristiwa **4,50**; p90 **12**; max **37**; hanya **35,3%** peristiwa dalam ≤3 lap.
- **Catatan binary head (FAKTA):** `P(masih pit)` = 0,999-1,000 di semua kasus pertengahan balapan → head biner nyaris selalu yakin "akan pit lagi" di tengah race; diskriminasi efektif terkonsentrasi mendekati finis. **HIPOTESIS:** acc 0,878 sebagian karena mayoritas baris memang "masih pit" (pos_rate 62%); perlu cek kalibrasi (Fase B).

---

## 10. AUDIT RUMUS & PEMBOBOTAN (permintaan khusus pengguna)

### 10.1 Rumus F1-makro — **CACAT** (lihat §4.1)
- **FAKTA:** `evaluate.py:76-77` tanpa `labels=` → macro hanya atas kelas hadir. Test 2025 overstated 0,3752 vs 0,3001. Seleksi CV aman (WET hadir). **Prioritas TINGGI, usaha trivial.**

### 10.2 Objective regresi vs metrik seleksi — **TIDAK SELARAS**
- **Komponen:** `app/ml/run3.py::run_xgb` baris 149.
- **FAKTA:** `XGBRegressor(objective="reg:squarederror")` (meminimumkan **MSE**/mean), padahal metrik seleksi & laporan = **MAE** (optimum = **median**).
- **Label dampak:** **HIPOTESIS** — mengganti ke `reg:absoluteerror` (atau quantile P50) **dengan params hasil tuning** bisa menurunkan MAE. Bukti belum konklusif: K08 (quantile P50) CV MAE 5,07 vs K03 4,81, **tapi K08 pakai params UNTUNED** (n_est 400 dasar) → confounded.
- **Perkiraan perbaikan maks:** kecil, **~0,05-0,25 lap MAE**. **Cara uji (Fase B):** Optuna-tune quantile-P50/absoluteerror, bandingkan CV MAE vs K03. **Prioritas: SEDANG.**

### 10.3 Pembobotan kelas — **bukan bug, tapi batas data**
- **Komponen:** `run3.py::bobot_kelas` (54-71).
- **FAKTA:** `k=len(COMPOUNDS)=5` (hardcoded, benar); sqrt10 = `sqrt(n/(5·c))` cap 10, normalisasi mean-kelas. Distribusi train 2021-23: SOFT 6355 (19%), MEDIUM 9735 (29%), HARD 15056 (45%), INTER 2264 (6,8%), **WET 59 (0,18%)**.
  - Bobot sqrt10: SOFT 0,360 · MEDIUM 0,291 · HARD 0,234 · INTER 0,604 · **WET 3,511**.
  - **Share total bobot** per kelas: SOFT 22,4% · MEDIUM 27,7% · HARD 34,5% · INTER **13,4%** (naik dari 6,8%) · WET **2,0%** (naik dari 0,18%).
- **Dampak:** pembobotan **berhasil** menggeser share gradien ke minoritas (INTER 6,8→13,4%). Tapi **WET tetap 2,0% share dari 59 sampel** → **tak terpelajari karena volume data, bukan karena rumus bobot**. inv20 (Run 2) menekan HARD lebih keras (0,088 vs 0,234) → merusak MEDIUM/HARD F1; sqrt10 (Run 3) lebih seimbang (konsisten F1 Run 3 > Run 2).
- **Label:** **FAKTA** (angka bobot & share). **Kesimpulan:** rumus bobot **wajar**; tidak ada perbaikan berarti yang tersedia dari sisi pembobotan untuk WET/INTER selain tambah data. **Prioritas: RENDAH.**

### 10.4 Rumus conformal — **benar, tapi simetrik pada target heteroskedastik**
- **FAKTA:** `scripts/08_conformal.py` — q = kuantil |residual| level `ceil((n+1)(1-α))/n` dari residual **cv2** (n=12144); q=7,3818; coverage kalibrasi **0,8001** (tepat nominal). Interval `[pred−q, pred+q]` di-clip ≥ lap kini.
- **Label dampak:** **HIPOTESIS** — residual **heteroskedastik** (MAE 3,02 di 6-15 lap vs 7,80 di >15 lap; §7). Interval **simetris konstan** ±7,38 terlalu lebar di dekat pit & terlalu sempit di horizon jauh → coverage **kondisional** buruk (kasus Monaco [44,58] vs aktual 13). **Perbaikan:** normalized/conditional conformal (residual diskala horizon prediksi). **Perkiraan maks:** coverage kondisional lebih merata; tak mengubah MAE titik. **Prioritas: SEDANG.**

### 10.5 Rumus target — **BENAR**
- **FAKTA:** §1.5, 0/2000 mismatch. `laps_remaining = next_pit − t` sesuai Bab III 2.2.7. Tidak ada cacat rumus target.

---

## 11. JAWABAN ATAS 7 PERTANYAAN WAJIB

**(1) Penyebab utama error.**
Urut kontribusi:
- **a. Kehilangan informasi status lintasan (FAKTA, dominan).** 6/44 fitur track-status nol-importance; 28,7% pit saat SC/VSC tak terlihat; sirkuit high-SC = MAE terburuk. Akar: filter `IsAccurate` + file trackstatus interval tak dipakai.
- **b. Sifat reaktif pit (FAKTA + HIPOTESIS).** Cuaca/crossover (INTER acc 0,104), regulasi, keputusan individual → sebagian error ireduksibel dari data lap saja.
- **c. Horizon jauh (FAKTA).** Bucket >15 MAE 7,80; target right-skew (max 68).
- **d. Isu rumus/pelaporan (FAKTA, sekunder).** F1-makro bug (overstate test), objective MSE≠MAE, conformal simetrik. **Tidak** mengubah penyebab dominan, tapi mempengaruhi pelaporan & optimalitas minor.

**(2) Batasnya di arsitektur / dataset / target / training / evaluasi / informasi data?**
- **Utama: INFORMASI DATA** (status lintasan dibuang) — bisa diperbaiki.
- **Kedua: DATASET/target** untuk pit reaktif (label noise red-flag, INTER/WET langka) — sebagian ireduksibel.
- **Evaluasi:** bug F1-makro (pelaporan) — trivial diperbaiki.
- **Training:** objective MSE vs MAE — penyelarasan minor.
- **Arsitektur: BUKAN batas.** XGB terpisah sudah menyamai/mengungguli LSTM & hybrid; menambah kompleksitas arsitektur tanpa memperbaiki informasi **tidak** akan banyak membantu.
- **Label kesimpulan:** didukung bukti → **"cacat pipeline informasi (track-status) + keterbatasan informasi data intrinsik", bukan cacat arsitektur.**

**(3) Apakah hybrid sudah sesuai Bab III?** (rujuk audit lama README §10.3)
- **FAKTA:** arsitektur hybrid **sesuai** Bab III §2.2.5 (ŷ₀ per-timestep, H_t bersama, target terpisah, unidirectional). Ketidakstabilan (seed42 acc 26,5%) dari **detail training yang tak dispesifikasikan Bab III** (loss MSE≫CE tanpa bobot task; OOF→full-model shift; tanpa clipping) — **bukan** pelanggaran desain. Run 3 menghindari ini dengan memisahkan model. **Tidak diulang di sini** (sudah diaudit).

**(4) Apakah ground truth sesuai?**
- **Rumus target: BENAR** (0/2000 mismatch; FAKTA).
- **Dua cacat ground-truth (FAKTA):** (a) pengecualian pit **red-flag dead code** → 354 pit RED tak dibuang (label bising, langgar Bab III 1.4); (b) pit **SC/VSC dipertahankan** (benar) **tetapi konteksnya nol** → model tak bisa pelajari pemicunya. Plus 264 pit berjarak-1-lap (indikasi artifact).
- **Kesimpulan:** ground truth **secara rumus sesuai**, **secara kontekstual cacat** karena fitur kondisi hilang.

**(5) Apakah prediksi dinamis per lap benar?**
- **FAKTA: BENAR.** Per-baris XGB memakai hanya info lap-berjalan; semua fitur kausal (shift/cummin/asof-backward); truncated-rebuild test lulus (dirujuk). **Bukan sumber error.**

**(6) Tiga perbaikan teratas + perkiraan perbaikan maksimal.**
| # | Perbaikan | Jenis | Perkiraan dampak maks | Usaha | Prioritas |
|---|---|---|---|---|---|
| **P1** | **Perbaiki pipeline track-status**: pakai 112 file `*_trackstatus.parquet` via `merge_asof` by Time (independen dari IsAccurate) untuk mengisi `is_sc/is_vsc/is_red/laps_since_sc_vsc`; **jangan** buang lap SC/VSC untuk konteks; perbaiki pengecualian red-flag; perbaiki regex multi-kode. | HIPOTESIS (kausal) | **MAE val 4,77 → ~4,0-4,4** (−0,4 s.d. −0,8); hit≤3 naik; compound crossover membaik; terbesar di sirkuit high-SC | Sedang (re-feature + re-run CV; **tanpa** ubah target/arsitektur) | **TERTINGGI** |
| **P2** | **Perbaiki bug F1-makro** (`labels=range(5)`) + laporkan macro 5-kelas konsisten + tambah **metrik gabungan** (timing∧compound) sebagai headline. | FAKTA (koreksi) | Tidak mengubah model; **koreksi kejujuran**: test F1 0,375→0,300; laporan konsisten | Trivial | **TINGGI** |
| **P3** | **Selaraskan objective regresi ke MAE** (`reg:absoluteerror`/quantile-P50 **tertuning**) + **conformal kondisional** (residual diskala horizon). | HIPOTESIS | MAE −0,05 s.d. −0,25; coverage kondisional conformal merata (kurangi miss Monaco-type) | Rendah-sedang (Fase B CV) | **SEDANG** |

**(7) Urutan eksperimen Fase B (dengan kriteria sukses & batas jumlah, ditetapkan SEBELUM dijalankan).**
**Batas total: maksimum 4 eksperimen.** Semua seleksi via **CV 2021-2024 (grup per race)**; **test 2025 tidak untuk seleksi**; protokol & kriteria ditulis ke file JSON sebelum jalan (pola `selection_criteria_v3.json`).

| Urut | Eksperimen | Perubahan | Kriteria sukses (pra-registrasi) | Gagal jika |
|---|---|---|---|---|
| **B1** | Track-status fix (P1) | merge_asof interval trackstatus → hidupkan is_sc/vsc/red/laps_since; pertahankan lap SC/VSC untuk konteks; perbaiki red-flag exclusion | CV MAE membaik **≥0,20 lap** ATAU hit≤3 CV naik **≥3 poin absolut**, tanpa F1-makro 5-kelas turun >0,010 | perbaikan <0,10 lap MAE DAN <1,5 poin hit≤3 |
| **B2** | Objective MAE (P3a) | Optuna-tune `reg:absoluteerror` & quantile-P50, bandingkan vs K03 | CV MAE **< K03 − 0,05** | CV MAE ≥ K03 − 0,05 |
| **B3** | Conformal kondisional (P3b) | normalized conformal (residual / f(horizon)) | coverage kondisional per bucket jarak dalam **[0,75; 0,85]** DAN lebar interval bucket 0-5 < 7,38 | coverage kondisional tetap <0,70 di bucket mana pun |
| **B4** | (cadangan) re-run evaluasi pasca-B1 | terapkan P2 (metrik) + laporkan ulang Run 1/2/3 dengan macro 5-kelas & metrik gabungan | laporan konsisten (per_kelas mean == f1_macro) | — |

**Aturan berhenti:** hentikan setelah B1 jika tidak ada perbaikan nyata (kriteria gagal B1), karena B1 adalah hipotesis dampak terbesar; B2-B4 opsional. **Evaluasi test 2025 SEKALI** hanya untuk konfigurasi yang lolos kriteria CV, dilaporkan berdampingan Run 1/2/3 dengan macro 5-kelas yang dikoreksi.

---

## LAMPIRAN — Bukti artefak audit
- `audit/a1_ground_truth.py` → komposisi pit, red-flag dead code, consecutive pits, distribusi target, cross-check 0/2000.
- `audit/a2_ceiling_features.py` → fitur mati, importance, R² plafon, konsistensi teammate, kurva belajar.
- `audit/a3_eval_metrics.py` → bug F1-makro, verifikasi metrik regresi, metrik gabungan per-baris/per-peristiwa, compound 1-5 lap.
- `audit/a4_segments_rules.py` → segmentasi (jarak/tyre/stint/sirkuit/tim/compound/musim), rule violation.
- `audit/a5_cases.py` → 5 kasus acak (seed 20261115) + 3 galat terbesar + konteks galat peristiwa.
- Semua skrip **read-only**; tidak ada model/dataset/konfigurasi yang diubah; tidak ada training.
