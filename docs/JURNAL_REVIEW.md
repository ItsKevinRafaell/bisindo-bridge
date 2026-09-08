# Review & Audit Jurnal BISINDO Bridge

> Dari: Kevin
> Tanggal: 2026-07-21
> File jurnal: `~/Downloads/AIJurnal.pdf`
> Status: Full audit 8 halaman + data verified dari project

---

## Part 1: Penjelasan Project (Buat Semua Orang)

### Project ini ngapain?

Kita bikin sistem yang bisa **mengenali huruf A-Z dari bahasa isyarat BISINDO** (Bahasa Isyarat Indonesia) secara **real-time di browser**.

Cara kerja simpelnya:

1. User angkat tangan depan webcam
2. **MediaPipe** (AI dari Google) detect posisi jari-jari tangan — namanya "landmark" (21 titik per tangan)
3. Titik-titik itu diubah jadi angka (fitur) → **84 angka** untuk 2 tangan
4. **CNN** (model AI kecil) classify jadi huruf A-Z
5. Huruf muncul di layar, bisa dibikin kalimat, bisa di-broadcast pas video call

### Kenapa harus 2 tangan?

BISINDO banyak huruf yang gesture-nya pakai 2 tangan. Misal huruf A, B, D, F, G itu mostly 2 tangan. Kalau sistem cuma detect 1 tangan, banyak huruf yang gak ke-detect.

### Model mana yang dipakai?

**Cuma 2 model yang penting** — sisanya (6 model lain seperti cnn_clean, cnn_quick, dll) itu hasil eksperimen random selama development, gak ada story jelas apa yang beda. **Jangan dimasukin semua ke jurnal**, bikin reviewer bingung.

| Model | Tangan | Fitur | Dataset | Accuracy | Role di Jurnal |
|-------|--------|-------|---------|----------|---------------|
| **cnn_2hand** | 2 | 84 (21×2×2) | 26K balanced | **98.45%** | ★ **MODEL UTAMA — yang di-deploy di web/VPS** |
| cnn_balanced | 1 | 63 (21×3×1) | 208K balanced | 99.12% | Perbandingan — accuracy lebih tinggi tapi cuma 1 tangan, gak practical |

**Kenapa 84 vs 63?**
- 84 = 21 titik × 2 koordinat (x,y) × 2 tangan — gak pakai depth (z)
- 63 = 21 titik × 3 koordinat (x,y,z) × 1 tangan — pakai depth tapi cuma 1 tangan

Trade-off: model 1 tangan lebih akurat secara angka (99.12% vs 98.45%) tapi gak bisa handle 2 tangan yang essential buat BISINDO. Jadi kita pilih model 2 tangan untuk deployment.

### Kenapa model lain (cnn_clean, cnn_quick, etc.) jangan dimasukin?

Karena:
1. Gak ada dokumentasi apa yang beda — cuma nama doang
2. Bukan ablation study (eksperimen sistematis) — cuma trial-error
3. Reviewer bakal nanya: "kenapa ada 7 model? apa bedanya? ablation?"
4. Kalau gak bisa jawab dengan jelas, malah keliatan project-nya gak terstruktur

**Solusi**: Table I di jurnal cukup 2 baris (cnn_2hand + cnn_balanced) + 1 baris optional "CNN baseline (63-dim, 1-hand, canonical data)" kalau mau nunjukin improvement dari baseline.

Atau kalau tim mau tetep masukin banyak model, harus ada penjelasan per model apa yang berubah (contoh di Part 3).

---

## Part 2: Tujuan Jurnal Ini Apa?

**Goals dari jurnal ini** (berdasarkan isinya):

1. **Nunjukin solusi untuk communication barrier** — Deaf community susah komunikasi di online meeting karena gak ada interpreter BISINDO
2. **Ngebuktiin landmark-based approach itu cukup** — gak perlu image-based yang berat, cukup pakai koordinat jari
3. **Bandingin arsitektur CNN** — beberapa variant, mana yang terbaik
4. **Proposal sistem komunikasi** — BISINDO Bridge sebagai fondasi untuk integrasi ke meeting platform

**Apakah ini masuk akal sebagai goals jurnal?**

- ✅ Goal 1 & 2: Bagus, ini contribution yang jelas
- ✅ Goal 4: Bagus, tapi harus jujur — meeting app SUDAH ADA, bukan "future work"
- ⚠️ Goal 3: Kalau cuma 2 model (main + comparison) masih oke. Kalau 7 model tanpa penjelasan, terlalu ribet dan tanda tanya

---

## Part 3: Audit Per Section Jurnal

### Abstract (Halaman 1)

**Isinya**: Problem (communication barrier), approach (MediaPipe + 84 fitur + 1D-CNN), future (meeting platform integration).

**Audit:**

| Poin | Status | Komentar |
|------|--------|----------|
| 84 fitur | ⚠️ Kurang jelas | Gak ada breakdown kenapa 84. Harus: "21×2×2 = 84" |
| "only x and y" | ⚠️ Misleading | Kesannya sengaja buang z. Padahal kita coba juga yang pakai z (63-dim) tapi gak practical |
| Normalization | ⚠️ Cuma disebut sekilas | Harus mention ada 3 step (translate + scale + StandardScaler) |
| Meeting platform | ❌ Underplayed | Dibilang "foundation for future" padahal SUDAH JADI |
| Perlu mention 2-model? | ✅ Ya | Biar jelas ada trade-off accuracy vs practicality |

**Rekomendasi rewrite Abstract** (ada di Part 5).

### Introduction (Halaman 1-2)

**Isinya**: Latar belakang communication barrier, AI untuk sign language, MediaPipe Hands, research gap (BISINDO research limited).

**Audit:**

| Poin | Status | Komentar |
|------|--------|----------|
| Communication barrier | ✅ Bagus | Jelas problem-nya |
| "BISINDO research limited" | ⚠️ Setengah benar | Related work [10]-[13] nunjukin BISINDO research LEBIH aktif dari yang dikira — ada LSTM, MobileNetV2, dll. Gak bisa bilang "limited" lagi kalau udah cite 4 paper BISINDO |
| Research gap | ⚠️ Perlu refine | Gap yang bener: "single-frame 1D-CNN underexplored untuk BISINDO, mostly LSTM/sequence-based" — ini udah ditulis di halaman 3 tapi contradict dengan halaman 1 yang bilang "limited" |
| "84 features" | ⚠️ Muncul tiba-tiba | Harus ada konteks: 2 tangan, 21 landmark, 2 koordinat |

### Related Work (Halaman 2-3)

**Isinya**: Sign language recognition, MediaPipe, CNN, communication systems, research gap.

**Audit:**

| Poin | Status | Komentar |
|------|--------|----------|
| Coverage BISINDO [10]-[13] | ✅ Bagus | Lengkap, ada LSTM, MobileNetV2, CNN+MediaPipe |
| Research gap statement | ✅ Bagus | "single-frame 1D-CNN underexplored for BISINDO" — ini sharp dan valid |
| Struktur | ✅ OK | A. Sign Language, B. MediaPipe, C. CNN, D. Comm Systems, E. Gap — rapi |

**Rekomendasi**: Section ini udah cukup bagus. Cuma perlu konsisten — di Introduction jangan bilang "BISINDO research limited", tapi "dominated by sequence-based architectures, lightweight single-frame underexplored".

### Methodology (Halaman 4-5)

**Isinya**: System architecture, dataset, landmark extraction, normalization, feature representation, CNN, training.

**Audit per subsection:**

#### III.A Overall Architecture

| Poin | Status | Komentar |
|------|--------|----------|
| Flow webcam → MediaPipe → CNN → huruf | ✅ Bagus | Jelas |
| Sentence generation + communication | ⚠️ Gak dijelaskan detail | Ada komponen sentence builder (fist=SPACE, palm=ENTER) yang gak disebut |

#### III.B Dataset Collection

| Poin | Status | Komentar |
|------|--------|----------|
| 138,471 samples | ✅ Match canonical | Tapi ini 1-hand dataset (63-dim). Model utama pakai 2-hand dataset (26,192 rows) — harus mention keduanya |
| "balanced version was generated" | ❌ Gak jelas | Balanced yang mana? Ada landmarks_balanced.csv (208K, 63-dim) dan landmarks_2hands.csv (26K, 84-dim) — yang mana? |
| Contributors | ❌ Gak disebut | Harus mention 9 kontributor, top-3 71% — sebagai limitation |

#### III.C Hand Landmark Extraction

| Poin | Status | Komentar |
|------|--------|----------|
| MediaPipe 21 landmark | ✅ Benar | Good |
| "up to two hands" | ✅ Sesuai main model | Tapi perlu jelasin: kenapa 2 tangan penting buat BISINDO |

#### III.D Coordinate Normalization

| Poin | Status | Komentar |
|------|--------|----------|
| Formula translasi | ⚠️ Cuma step 1 | Yang bener ada 3 step: translate + scale by hand size + StandardScaler |
| Hand size scaling | ❌ Gak disebut | Ini krusial — biar beda ukuran tangan gak ngaruh |
| StandardScaler | ❌ Gak disebut | Pakai scaler.json per model |

#### III.E Feature Representation

| Poin | Status | Komentar |
|------|--------|----------|
| 84 = 21×2×2 | ✅ Benar untuk main model | Tapi harus mention ada variant 63-dim juga |
| Penjelasan compact | ✅ Bagus | "only 84 numerical features vs thousands of pixels" |

#### III.F CNN Architecture

| Poin | Status | Komentar |
|------|--------|----------|
| "Several Conv layers" | ❌ Terlalu vague | Harus ada tabel detail (lihat Part 5) |
| Pooling, FC, Softmax | ⚠️ Disebut tapi gak detail | Berapa filters? kernel size? berapa units? |

#### III.G Model Training

| Poin | Status | Komentar |
|------|--------|----------|
| "Adam + Cross-Entropy" | ⚠️ Minimal info | Harus: lr=0.001, batch=256, epochs=50, split 85/15 seed 42, dropout 0.3 |
| "Adam" — tim tanya ini apa | 📝 Dijelaskan di Part 4 bawah | Adam = optimizer, bukan bagian yang ribet |

### Results (Halaman 6-7)

#### Table I

| Poin | Status | Komentar |
|------|--------|----------|
| 7 model tanpa penjelasan | ❌ Membingungkan | Apa bedanya cnn_clean vs cnn_quick vs cnn_final? Reviewer pasti nanya |
| cnn_balanced 99.12% sebagai "best" | ⚠️ Misleading | 99.12% itu 1-hand, sedangkan fokus paper 2-hand. Harus clear mana main model |
| Accuracy = F1 semua | ⚠️ Aneh | Accuracy dan F1 identik di semua model — perlu jelasin kenapa (balanced dataset, atau memang kebetulan) |

**Rekomendasi**: Table I cukup 2 baris (main + comparison) atau max 3 dengan baseline.

#### Fig. 1 Confusion Matrix

| Poin | Status | Komentar |
|------|--------|----------|
| Placeholder | ❌ Belum ada gambar | File sudah ready: `docs/jurnal-data/cnn_2hand_confusion_matrix.png` (138KB, verified) |
| Disebut "CNN Balanced model" | ⚠️ Salah focus | Harus cnn_2hand untuk main model, atau generate untuk keduanya |

#### Table II Per-Class

| Poin | Status | Komentar |
|------|--------|----------|
| Angka gak verified | ❌ Dari training-time JSON | Data verified ready: `docs/jurnal-data/cnn_2hand_per_class.csv` |
| Letter B, Q, N, E rendah | ✅ Valid observation | Tapi perlu data verified buat support claim ini |

#### Discussion

| Poin | Status | Komentar |
|------|--------|----------|
| Landmark-based cukup | ✅ Valid | Good point |
| Normalization helps | ✅ Valid | Tapi normalization yang dideskripsiin kurang lengkap |
| "CNN Balanced best" | ⚠️ Misleading | Best secara angka, tapi bukan yang di-deploy |

### Conclusion (Halaman 7)

| Poin | Status | Komentar |
|------|--------|----------|
| Summary 84-dim + 99.12% | ⚠️ Inconsistent | 84-dim model accuracy-nya 98.45%, bukan 99.12%. 99.12% itu 63-dim |
| "Foundation for future integration" | ❌ Underplayed | Meeting app SUDAH ADA, bukan future |
| Future work: LSTM, Transformer | ✅ OK | Valid untuk continuous sign language |

---

## Part 4: Istilah yang Tim Tanya

### Adam — itu apa? Perlu masuk jurnal?

**Adam** = **Adaptive Moment Estimation** — ini nama algoritma optimizer. Tugasnya: ngatur gimana model belajar (update bobot/weight) biar semakin akurat.

**Analogi simpel**: Kalau model AI itu kayak orang yang belajar lempar bola ke target, Adam itu kayak "cara ngajarin" biar makin lama makin akurat kena target. Dia ngatur "langkah" belajarnya — kadang kecil, kadang besar, adaptif.

**Perlu masuk jurnal?** Ya, tapi cukup 1 baris: "Adam optimizer (lr=0.001)". Gak perlu jelasin rumus Adam atau detail teknis. Reviewer udah tau Adam itu apa — ini standard optimizer yang dipakai 90% paper deep learning.

**Yang perlu dijawab di jurnal:**
- Optimizer: Adam ✅ (1 baris, gak perlu detail)
- Learning rate: 0.001 ✅ (angka ini penting)
- Loss function: CrossEntropyLoss ✅ (standard buat classification)
- Batch size: 256, Epochs: 50, Split: 85/15, Seed: 42 ✅ (buat reproducibility)

**Yang GAK perlu masuk jurnal:**
- Rumus Adam, detail moment estimation, beta1/beta2 — ini terlalu deep, bukan fokus paper

### Istilah lain yang mungkin membingungkan:

| Istilah | Penjelasan simpel | Perlu di jurnal? |
|---------|-------------------|------------------|
| StandardScaler | Normalisasi biar semua fitur punya mean=0, std=1. Biar model gak bias ke fitur yang angkanya gede | Ya, sebut aja "StandardScaler (zero mean, unit variance)" |
| AdaptiveAvgPool | Cara ngecilin ukuran data di CNN. Pool=1 berarti output selalu 1 value per filter, regardless of input size | Ya, tapi cukup di tabel arsitektur |
| Softmax | Ubah output model jadi probability (0-1) yang sum-nya = 1. Huruf dengan probability tertinggi = prediksi | Ya, 1 baris |
| Dropout(0.3) | Teknik biar model gak overfit — random matiin 30% neuron pas training | Ya, sebut di hyperparameters |
| Stratified split | Bagi data train/test dengan jaga proporsi tiap huruf tetap sama | Ya: "85/15 stratified, random_state=42" |
| ONNX | Format buat convert model PyTorch jadi format yang bisa jalan di browser | Ya, ini contribution — harus ada section |

---

## Part 5: Yang Harus Masuk ke Jurnal (Checklist Goals)

### WAJIB masuk (masuk ke goals jurnal, core contribution):

- [x] **Problem**: communication barrier buat Deaf di online meeting
- [x] **Solution**: BISINDO Bridge — MediaPipe + 1D-CNN + browser deployment
- [x] **Dataset**: 2-hand 26K (main) + mention canonical 138K dan balanced 208K
- [x] **Method**: 84 fitur (breakdown 21×2×2), 3-step normalization, CNN architecture
- [x] **Results**: Table I (2 model), confusion matrix (verified), per-class metrics (verified)
- [x] **Deployment**: ONNX browser inference, meeting app (implemented, bukan future)
- [x] **Limitation**: dataset bias, single split, re-eval limitation

### BOLEH masuk (nice to have, bukan core):

- [ ] Baseline comparison: cnn (63-dim, canonical, 96.04%) — nunjukin improvement dari baseline ke main model
- [ ] Ablation: dengan dan tanpa hand-size scaling — nunjukin normalization step penting
- [ ] Real-time performance: FPS di browser, latency
- [ ] Sentence builder: fist=SPACE, palm=ENTER — kalau ada space di paper

### JANGAN masuk (bikin ribet, tanda tanya):

- [ ] 7 model tanpa penjelasan apa yang beda — ini yang bikin bingung
- [ ] Detail rumus Adam optimizer — ini bukan paper tentang optimizer
- [ ] Detail rumus MediaPipe — cukup cite paper MediaPipe
- [ ] Semua dataset variant (ada 7 CSV: canonical, 2hands, balanced, clean, augmented, dll) — cukup mention 2 yang dipakai (2hands + balanced)
- [ ] Training curves / loss curves — kalau gak ada insight khusus, gak perlu

---

## Part 6: Data yang Sudah Disiapin (Verified)

Semua di `docs/jurnal-data/`:

| File | Size | Untuk apa | Status |
|------|------|-----------|--------|
| `cnn_2hand_confusion_matrix.png` | 138 KB | Fig. 1 (annotated, diagonal bold + error merah) | ✅ Verified, 98.45% |
| `cnn_2hand_confusion_matrix_clean.png` | 104 KB | Fig. 1 alternatif (clean) | ✅ |
| `cnn_2hand_per_class.csv` | 764 B | Table II (26 letters) | ✅ Re-evaluated |
| `cnn_2hand_per_class_chart.png` | 77 KB | Bar chart Precision/Recall/F1 per huruf | ✅ |
| `model_comparison_chart.png` | 96 KB | Bar chart perbandingan semua model | ✅ |
| `project_info.json` | 5.8 KB | Semua metadata (arch, hyperparams, ONNX, dataset) | ✅ |
| `INSTRUKSI_REVISI.md` | 6 KB | Instruksi singkat buat tim | ✅ |
| `cnn_2hand_per_class_eval.json` | 2.9 KB | Per-class JSON | ✅ |
| `cnn_2hand_metrics.json` | 138 B | Training-time metrics | ✅ |
| `cnn_balanced_metrics.json` | 139 B | Baseline comparison | ✅ |

### Per-Class Metrics Verified (cnn_2hand, 98.45%, 3929 test samples):

| Letter | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| A | 0.9944 | 0.9944 | 0.9944 | 179 |
| B | 0.9524 | 0.9333 | 0.9428 | 150 |
| C | 0.9739 | 0.9933 | 0.9835 | 150 |
| D | 1.0000 | 0.9800 | 0.9899 | 150 |
| E | 0.9259 | 1.0000 | 0.9615 | 150 |
| F | 0.9862 | 0.9533 | 0.9695 | 150 |
| G | 0.9934 | 1.0000 | 0.9967 | 150 |
| H | 1.0000 | 1.0000 | 1.0000 | 150 |
| I | 0.9608 | 0.9800 | 0.9703 | 150 |
| J | 0.9934 | 1.0000 | 0.9967 | 150 |
| K | 1.0000 | 0.9867 | 0.9933 | 150 |
| L | 1.0000 | 1.0000 | 1.0000 | 150 |
| M | 0.9669 | 0.9733 | 0.9701 | 150 |
| N | 0.9797 | 0.9667 | 0.9732 | 150 |
| O | 0.9933 | 0.9933 | 0.9933 | 150 |
| P | 0.9804 | 1.0000 | 0.9901 | 150 |
| Q | 0.9934 | 1.0000 | 0.9967 | 150 |
| R | 0.9932 | 0.9667 | 0.9797 | 150 |
| S | 1.0000 | 0.9600 | 0.9796 | 150 |
| T | 0.9933 | 0.9933 | 0.9933 | 150 |
| U | 0.9551 | 0.9933 | 0.9739 | 150 |
| V | 1.0000 | 0.9867 | 0.9933 | 150 |
| W | 0.9934 | 1.0000 | 0.9967 | 150 |
| X | 0.9865 | 0.9733 | 0.9799 | 150 |
| Y | 0.9867 | 0.9867 | 0.9867 | 150 |
| Z | 1.0000 | 0.9800 | 0.9899 | 150 |

Overall: **98.45% accuracy, 98.45% F1 weighted**

Protocol: 85/15 stratified split, random_state=42, 3929 test samples dari 26,192 total (landmarks_2hands.csv).

---

## Part 7: Arsitektur & Hyperparameters (Buat Tabel di Jurnal)

### CNN Architecture — cnn_2hand (Deployed, 84 fitur) ✅ Re-evaluated

| Layer | Operation | Detail |
|-------|-----------|--------|
| Input | — | 84 features, shape (1, 1, 84) |
| Conv1D Block 1 | Conv + ReLU + MaxPool | 64 filters, kernel=3, padding=1, pool=2 |
| Conv1D Block 2 | Conv + ReLU + MaxPool | 128 filters, kernel=3, padding=1, pool=2 |
| Conv1D Block 3 | Conv + ReLU + AdaptiveAvgPool(1) | 64 filters, kernel=3, padding=1, pool=1 |
| Flatten | — | 64 values |
| FC 1 | Linear + ReLU + Dropout(0.3) | 64 → 128 |
| FC 2 | Linear | 128 → 26 (A-Z) |
| Output | Softmax | 26 probabilities |

Total params: ~30K (lightweight, edge-friendly)

### CNN Architecture — cnn_balanced (Comparison, 63 fitur, 99.12%)

Bedanya cuma di pooling dan FC input:

| Layer | Detail |
|-------|--------|
| Input | 63 features, shape (1, 1, 63) |
| Conv1D Block 1-2 | Sama kayak cnn_2hand |
| Conv1D Block 3 | AdaptiveAvgPool(8) — bukan (1) |
| Flatten | 512 values (8×64) — bukan 64 |
| FC 1 | 512 → 128 — bukan 64→128 |
| FC 2, Output | Sama |

### Hyperparameters

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Optimizer | Adam | Standard, dipakai 90% paper |
| Learning rate | 0.001 | Penting buat reproducibility |
| Batch size | 256 | |
| Epochs | 50 | |
| Loss | CrossEntropyLoss | Standard buat classification |
| Dropout | 0.3 | Cegah overfit |
| Split | 85/15 stratified, seed 42 | Penting buat reproducibility |
| Scaler | StandardScaler (mean=0, std=1) | Saved per model |
| Training time | 153s CPU (cnn_2hand) / 3704s (cnn_balanced) | |
| Device | CPU only | Gak pakai GPU |

---

## Part 8: ONNX / Browser Inference (Yang Belum Ada di Jurnal)

Ini **contribution signifikan** tapi gak disebut sama sekali di jurnal. Harus tambah 1 subsection.

### Pipeline:

1. **Convert**: PyTorch → ONNX via `torch.onnx.export` (opset 12) — script: `scripts/convert_to_onnx.py`
2. **ONNX model**: input [batch, 1, 84], output [batch, 26], size 241KB
3. **Browser runtime**: `onnxruntime-web@1.17.1` (CDN) — `ort.InferenceSession.create()`
4. **MediaPipe**: HandLandmarker @0.10.35 float16 (CDN) — detect 21 landmark per tangan
5. **Inference flow di browser**:
   ```
   MediaPipe detect → 21 landmark × 2 tangan (x,y)
   → landmarksToFeatures() → 84 angka (hand1 42 + hand2 42, zero-pad kalau 1 tangan)
   → normalizeFeaturesHandCentric() → translate ke wrist + scale by hand size
   → scaleFeatures() → StandardScaler dari scaler.json
   → predictModel() → ONNX inference → softmax → huruf + confidence
   ```
6. **Verified**: ONNX vs PyTorch output max diff < 0.000004 (identical)
7. **Privacy**: semua inference local di browser, data gak keluar device
8. **Deployment**: `web/` (Vercel) + `meeting/` (Flask-SocketIO + WebRTC mesh + ONNX)

---

## Part 9: Ide Judul Baru

Judul sekarang: *"Real-Time BISINDO Recognition Using MediaPipe Hand Landmark"*

| # | Judul | Kenapa |
|---|-------|--------|
| 1 | **BISINDO Bridge: A Lightweight Dual-Hand 1D-CNN System for Real-Time Indonesian Sign Language Recognition in the Browser** | Paling lengkap — sistem + dual-hand + lightweight + browser |
| 2 | **Dual-Hand BISINDO Fingerspelling Recognition Using Normalized Landmark Coordinates and 1D-CNN** | Emphasize dual-hand novelty |
| 3 | **From Landmarks to Letters: Practical BISINDO Alphabet Recognition with 1D-CNN and On-Device Inference** | Catchy + practical |
| 4 | **BISINDO Bridge: Balancing Accuracy and Usability in Two-Handed Sign Language Recognition with 1D-CNN** | Emphasize trade-off story (99.12% vs 98.45%) — menarik buat reviewer |

Rekomendasi: **#1 atau #4**

---

## Part 10: Checklist Revisi Buat Tim

### Must fix (kalau gak di-fix, paper ditolak):

- [ ] **Abstract**: breakdown 84 fitur (21×2×2), mention 63-fitur comparison, sebutin deployed bukan future
- [ ] **III.D Normalization**: tambah 3-step (translate + scale by hand size + StandardScaler), bukan cuma translasi
- [ ] **III.F Architecture**: tambah tabel CNN (filters, kernel, FC) — sekarang cuma "several layers"
- [ ] **III.G Hyperparameters**: tambah lr=0.001, batch=256, epochs=50, split 85/15 seed 42, dropout 0.3
- [ ] **Table I**: sederhanain jadi 2 baris (cnn_2hand main + cnn_balanced comparison) — JANGAN 7 baris tanpa penjelasan
- [ ] **Fig. 1**: ganti placeholder ke `cnn_2hand_confusion_matrix.png` (verified)
- [ ] **Table II**: pakai `cnn_2hand_per_class.csv` (verified), bukan angka training-time yang gak bisa dibuktiin
- [ ] **Inconsistency**: Abstract bilang 84 fitur tapi best model 99.12% itu 63 fitur — harus dijelasin trade-off

### Must add (yang kurang, tapi bukan error):

- [ ] **1-hand vs 2-hand comparison section**: kenapa pilih 98.45% over 99.12%
- [ ] **ONNX / browser inference subsection**: conversion, deployment, privacy, verified match
- [ ] **Limitation section**: dataset bias (9 contributors), single split, re-eval limitation, webcam only
- [ ] **Meeting app = implemented**: Flask-SocketIO + WebRTC + ONNX, bukan "future work"
- [ ] **Sentence builder = implemented**: fist=SPACE, palm=ENTER

### Nice to have:

- [ ] Judul baru
- [ ] Per-class chart (`cnn_2hand_per_class_chart.png`) sebagai Fig. 2
- [ ] Model comparison chart (`model_comparison_chart.png`)

---

*File ini auto-generated dari audit. Update kalau ada perubahan project.*
