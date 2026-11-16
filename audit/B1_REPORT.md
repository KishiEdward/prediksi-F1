# LAPORAN B1 — Mengaktifkan Fitur Status Lintasan (FASE B, eksekusi terkontrol)

**Status akhir: B1 DITOLAK sesuai aturan keputusan pra-registrasi (basis CV).**
Protokol pra-registrasi: `models/selection_criteria_b1.json` (ditulis SEBELUM run).
Hasil CV: `models/b1_cv_results.json`. Test sekali: `models/b1_test_once.json`.
Tidak ada model/konfigurasi produksi yang diubah; dataset B1 terpisah di `data/processed/b1/`.

---

## 1. Step 2 — Cek definisi pembersihan data (IsAccurate) di Bab III

**Kesimpulan: mengubah filter `IsAccurate` TIDAK menyimpang dari Bab III; filter saat ini justru KONTRA-Bab III.** (FAKTA)

Bukti (sumber `(4)F1-Bab-III-Prototype.docx`, ekstrak di `audit/bab3_cleaning_extracts.txt`):
- **0 penyebutan** "IsAccurate", "akurat", "menghapus/membuang/filter lap" di seluruh Bab III.
- Para 136/390/403/424/425/436/455/531/865: **SC/VSC WAJIB** jadi variabel in-race.
- Para 533/612/730/867: sistem harus **MENANGANI** data hilang/tidak lengkap (bukan membuang baris pembawa informasi).
- Para 141: Red Flag = kondisi khusus prapemrosesan "**apabila dapat diidentifikasi berdasarkan data yang tersedia**" → memperbaiki track_status justru MENGAKTIFKAN penanganan yang diniatkan Bab III.

Jadi filter `IsAccurate==True` (`features.py:279-282`) adalah pilihan implementasi (kualitas lap-time), **bukan** amanat Bab III, dan efek sampingnya (menghapus semua lap SC/VSC) bertentangan dengan kewajiban menyajikan SC/VSC sebagai fitur.

---

## 2. Implementasi B1 (sesuai pra-registrasi)

Perubahan (di `app/ml/features.py`, **default produksi TIDAK berubah** — semua via flag):
1. **Status all-code** (`status_all_codes=True`): parse SEMUA kode TrackStatus per lap (multi-hot), bukan digit pertama. `track_status` kategorik = kondisi terparah (RED>VSC>SC>YELLOW>GREEN).
2. **Filter longgar** (`keep_inaccurate=True`): pertahankan lap `IsAccurate==False` (SC/VSC/pit/out-lap) agar konteks status tersedia; `lap_time` lap tak-akurat di-NaN-kan agar fitur turunan waktu tidak dirusak nilai tak representatif. Lap `Deleted` tetap dibuang.
3. **Guard target==0**: baris lap-pit (target 0) dibuang → definisi target tetap min=1, identik dengan baseline (no-op di produksi).
4. **Hyperparameter K03/K04 PERSIS sama** (dari `models/run3_optuna_*.json`), tanpa tuning ulang. Bobot sqrt10 tetap. Objective reg tetap `reg:squarederror`.

### Dataset B1 vs produksi (FAKTA, `logs/b1_build.log`)
| | Produksi | B1 |
|---|---|---|
| inference rows | 106.260 | **118.938** (+12.678) |
| training rows | 57.390 | **65.810** (+8.420) |
| track_status | GREEN, YELLOW saja | GREEN 107261, **SC 5216, YELLOW 4016, VSC 2105, RED 340** |
| is_sc / is_vsc / is_red (sum) | 0 / 0 / 0 | **5560 / 2122 / 340** |
| laps_since_sc_vsc | konstan 10 | nunique 11, mean 8.6 |
| red-flag pit exclusion | 0 (dead code) | **313 kejadian dibuang** (AKTIF, sesuai Bab III) |
| WET (train rows) | 84 | 160 |
| target min | 1 | 1 (guard bekerja) |

Fitur status **KINI HIDUP** di dataset B1.

---

## 3. Hasil CV (basis keputusan) — B1 DITOLAK

Fold sama: cv1 (train 21-22→val 23), cv2 (train 21-23→val 24), grup per race.
Sanity check: BASE mereproduksi persis CV K03 Run 3 (cv1 4.8507, cv2 4.7731) → perbandingan valid.

| Basis | Metrik | BASE | B1 | Δ (B1−BASE) |
|---|---|---|---|---|
| **BARIS SAMA** (putusan) | MAE | **4.7563** | **4.8758** | **−0.1195 (memburuk)** |
| BARIS SAMA | F1-makro | 0.3295 | 0.3245 | −0.0050 |
| SELURUH BARIS | MAE | 4.8119 | 5.2208 | −0.4089 (memburuk) |
| SELURUH BARIS | F1-makro | 0.3259 | 0.3223 | −0.0036 |
| BARIS SAMA | bucket >15 | — | memburuk maks +0.5469 | — |

**Aturan keputusan (pra-registrasi):** perbaikan MAE baris-sama harus ≥ +0.10 untuk dipertimbangkan; > +0.20 untuk terima langsung. Hasil: **−0.1195 (negatif)** → **TOLAK B1**. Bucket >15 juga memburuk > 0.30 dan F1 turun, sehingga bahkan band 0.10–0.20 pun tidak terpenuhi.

---

## 4. Diagnosis: MENGAPA B1 tidak membantu (temuan penting)

**Akar struktural (FAKTA, `audit/b1_followup.py` + diagnostik NaN):**
> **Lap SC/VSC ADALAH lap `IsAccurate=False`.** Jadi pada baris "akurat" (baris yang sama
> dengan baseline, tempat prediksi biasa dilakukan), flag `is_sc/is_vsc/is_red` **TETAP NOL**
> bahkan di B1 (terverifikasi: pada 12.032 baris val-2024 sama, is_sc=is_vsc=is_red mean 0.0000).
> Informasi status hanya muncul di baris BARU (lap tak-akurat) yang memang noisy & sulit.

Konsekuensi:
1. **Uji baris-sama secara struktur tidak bisa mendapat manfaat** dari flag status (flag tetap 0 di baris itu), tetapi **malah rugi** karena `delta_prev_lap` NaN naik 3.6%→6.3% (lap tak-akurat ber-lap_time NaN merusak delta lap akurat di sekitarnya). → MAE baris-sama memburuk.
2. **Uji seluruh-baris memburuk** karena populasi val kini memasukkan lap SC/VSC tak-akurat yang intrinsik sulit (pit reaktif) → MAE 5.22.
3. Hanya `laps_since_sc_vsc` yang sedikit hidup di baris akurat (mean 9.52, jarang < 10) — sinyal lemah.

**Implikasi mendalam:** informasi SC/VSC dan lap-time akurat **saling eksklusif di tingkat baris** (FastF1 menandai lap SC sebagai tak akurat). Menambah flag status per-lap TIDAK bisa memberi model konteks SC pada baris prediksi yang akurat. Sisa error di sirkuit high-SC sebagian besar **ireduksibel** dari data lap-akurat: pit reaktif saat SC terjadi pada lap yang justru tak punya baris akurat untuk diprediksi.

**Catatan kejujuran:** desain B1 (NaN-kan lap_time tak akurat) ikut memperburuk delta di baris-sama — sebagian regresi baris-sama adalah artefak desain ini, bukan murni efek fitur status. Namun hasil seluruh-baris dan akar struktural di atas tetap menunjukkan flag status per-lap tidak memberi manfaat seperti yang dihipotesiskan audit.

---

## 5. Causality test fitur baru (WAJIB) — LULUS

Truncated-rebuild pada fitur status B1 (`audit/b1_followup.py`): rebuild fitur dari data dipotong di lap N, bandingkan dengan rebuild penuh untuk lap ≤ N.
- **6 race, 29.519 sel dibandingkan, mismatch = 0 → LULUS (kausal).**
- `is_*` hanya dari TrackStatus lap itu sendiri; `laps_since_sc_vsc` dihitung mundur dalam driver-race. Tidak ada informasi masa depan.

---

## 6. Kasus tersendiri: Monaco 2025 & aturan wajib dua pit

**FAKTA (data pitstop):** jumlah pit stop per pembalap di Monaco:

| Musim | mean stops | max | fraksi ≥2 stop |
|---|---|---|---|
| 2021 | 1.06 | 2 | 6% |
| 2022 | 2.79 | 4 | 100% |
| 2023 | 1.85 | 5 | 60% |
| 2024 | 1.44 | 3 | 38% |
| **2025** | **2.00** | 3 | **90%** |

Detail 3 galat terbesar (test 2025): COL pit [13, 26] (2 stop), ALO pit [16] (1 stop, sangat dini), STR pit [17, 64] (2 stop).

**Kesimpulan:** Monaco 2025 menunjukkan **lonjakan fraksi ≥2 stop (90%) vs 2024 (38%) dan 2021 (6%)**, konsisten dengan **aturan wajib dua pit (mandatory two-stop) Monaco 2025** — regulasi baru FIA untuk memecah prosesi satu-stop. Model dilatih pada 2021-2023 (Monaco didominasi satu-stop, mean 1.06-1.85) sehingga memprediksi stint HARD panjang (lap ~48-51), padahal 2025 memaksa stop dini/lipat. Ditambah VSC (terkonfirmasi di `2025_08_trackstatus.parquet`), ini menjelaskan galat +32 s.d. +38 lap.

**Label:** pola stop = FAKTA; keterkaitan dengan regulasi two-stop = **HIPOTESIS kuat** (konsisten data; regulasi adalah konteks eksternal). Ini **distribution shift regulasi** yang tidak bisa dipelajari model dari data historis — di luar kendali fitur apa pun.

---

## 7. Test 2025 SEKALI — pandangan keempat (B1 DITOLAK, tidak dipromosikan)

Dievaluasi SEKALI setelah keputusan CV dikunci. **TIDAK dipakai untuk seleksi/promosi.**

| Pandangan (test 2025) | MAE | F1-makro | Ket |
|---|---|---|---|
| Run 1 LSTM | 6.049 | 0.374 | (arsip) |
| Run 2 LSTM | 5.991 | 0.337 | (arsip) |
| **Run 3 K03+K04 (TERPILIH)** | **6.084** | **0.375** | model aktif |
| B1 seluruh baris (n=13988) | 6.329 | 0.382 | populasi berbeda (termasuk lap SC) |
| **B1 baris-sama (n=11777)** | **5.940** | 0.378 | **+0.144 vs Run 3** |

**⚠️ Divergensi CV vs test (jujur):** di **baris-sama**, CV (val 2023+2024) bilang B1 **memburuk −0.12**, tetapi test 2025 bilang B1 **membaik +0.14**. Arah berlawanan, magnitudo kecil (~0.1 lap, dalam derau). Sesuai pra-registrasi, **keputusan HANYA dari CV** → **B1 TETAP DITOLAK**. Menggunakan test 2025 untuk mempromosikan B1 akan melanggar protokol (dan test sudah pernah dilihat). Divergensi ini justru menegaskan efeknya **tidak reliabel**.

---

## 8. Step 4 — Bagaimana API menangani lap SC/VSC yang tidak punya baris

Perilaku saat ini (produksi, filter IsAccurate ON; `app/services/providers.py`):

1. **Lap tanpa baris sama sekali** (semua pembalap terfilter, mis. lap SC penuh / lap 1):
   `get_race_state` → `rows.empty` → raise `KeyError` → API balas **404**
   `"Tidak ada data lap N untuk race ..."` (`providers.py:141-142`).
   Bukti nyata: E2E test lap 1 Bahrain 2021 → 404.
2. **Pembalap tertentu tanpa baris di lap itu** (tapi lap ada untuk pembalap lain):
   pembalap itu **tidak muncul** di `drivers` (di-omit diam-diam, bukan error)
   (`providers.py:155-163` membangun drivers hanya dari `rows` lap itu). Jika team filter
   menghasilkan kosong → API balas 404. Bukti: E2E lap 30 Alfa Romeo 2021 → 404.
3. **Field `track_status` pada response TIDAK ANDAL**: dibaca dari `rows["track_status"]`
   (`providers.py:152-153`) yang di produksi hanya pernah GREEN/YELLOW. Jadi selama periode
   SC/VSC nyata, response (bila ada) tetap menampilkan `track_status: GREEN` — menyesatkan.
4. **Run 3 per-baris**: prediksi hanya memakai baris lap-N (`predictor.py` `iloc[[-1]]`);
   tidak ada "stale row" karena pembalap tanpa baris di lap N sudah di-omit.

**Konsekuensi:** justru selama SC/VSC — ketika keputusan pit paling sering terjadi — API replay
paling sering bolong (404) atau salah label status. Ini keterbatasan nyata mode replay.

**Opsi (belum diimplementasikan, butuh persetujuan):**
- (a) **Interpolasi/status-forward**: bila lap N tak punya baris, pakai baris terakhir ≤ N per
  pembalap dan tandai `stale=true` + ambil `track_status` dari file interval trackstatus (bukan
  dari fitur mati). Memperbaiki 404 & label status TANPA mengubah model.
- (b) Sajikan `track_status` dari `*_trackstatus.parquet` (interval, merge by time) di provider,
  independen dari fitur model → label status benar meski model tak memakainya.
- (c) Promosikan dataset B1 (keep_inaccurate) HANYA untuk lapisan penyajian/API (baris tersedia),
  sementara model tetap Run 3 — tapi ini mengubah populasi inference; perlu uji E2E ulang.

---

## 9. Keputusan & rekomendasi

**KEPUTUSAN: B1 DITOLAK** (CV baris-sama −0.1195 lap; aturan pra-registrasi menuntut ≥ +0.10).
Model tetap **Run 3 K03+K04**. Test 2025 yang sekilas menguntungkan B1 (+0.14 baris-sama) TIDAK
dipakai untuk seleksi (melanggar protokol); divergensi CV/test menandakan efek tidak reliabel.

**Yang dipelajari (FAKTA):**
- Fitur status berhasil dihidupkan & kausal, red-flag exclusion aktif (313 pit), tetapi
  **tidak meningkatkan akurasi** karena lap SC/VSC = lap tak-akurat → flag nol di baris prediksi.
- Hipotesis audit "SC/VSC blindness = penyebab dominan" **terkonfirmasi sebagai keterbatasan
  INFORMASI**, tetapi **tidak dapat diperbaiki** sekadar menambah flag status per-lap;
  errornya ireduksibel dari data lap-akurat.

**Rekomendasi Phase B lanjutan (BUTUH PERSETUJUAN, belum dijalankan):**
1. **B1-clean (isolasi)**: pertahankan row-set & fitur waktu BASE persis; tambah HANYA konteks
   SC dari file interval (`laps_since_sc_vsc`, `sc_within_last_N_laps`) via merge_asof by time ke
   baris akurat — tanpa NaN-kan delta, tanpa tambah baris noisy. Uji CV dengan aturan keputusan sama.
   (HIPOTESIS: perbaikan kecil 0–0.1 lap; mungkin tetap di bawah ambang.)
2. **Perbaikan API (non-model)**: opsi 8(a)/8(b) — hilangkan 404 saat SC/VSC & perbaiki label
   `track_status` dari file interval. Tidak mengubah model; murni kualitas penyajian.
3. **P3 (dari audit)**: selaraskan objective regresi ke MAE + conformal kondisional.

**Berhenti di sini sesuai instruksi.** Tidak ada perubahan model produksi; Run 3 tetap aktif.
