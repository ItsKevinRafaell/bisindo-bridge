# Proposal Outline - BISINDO Bridge

## Judul
**BISINDO Bridge: Perbandingan Metode Klasifikasi untuk Pengenalan Bahasa Isyarat Indonesia**

---

## 1. Pendahuluan

### 1.1 Latar Belakang
- Bahasa Isyarat Indonesia (BISINDO) digunakan oleh komunitas tuna rungu di Indonesia
- Teknologi pengenalan gestur dapat membantu komunikasi
- Perbandingan metode diperlukan untuk menemukan pendekatan optimal

### 1.2 Rumusan Masalah
- Bagaimana performa berbagai metode klasifikasi untuk pengenalan BISINDO?
- Metode mana yang lebih cocok untuk aplikasi real-time?

### 1.3 Tujuan
- Membangun sistem pengenalan BISINDO berbasis landmark
- Membandingkan performa berbagai metode klasifikasi
- Menemukan metode optimal untuk deployment

---

## 2. Tinjauan Pustaka

### 2.1 Hand Landmark Detection
- MediaPipe HandLandmarker
- Ekstraksi fitur landmark tangan

### 2.2 Metode Klasifikasi
- **Traditional ML:** pendekatan berbasis algoritma klasik
- **Deep Learning:** pendekatan berbasis neural network

### 2.3 Related Work
- Studi pengenalan bahasa isyarat
- Perbandingan metode machine learning

---

## 3. Metodologi

### 3.1 Dataset
- Total samples: [jumlah dari dataset]
- Letters: 26 huruf BISINDO (A-Z)
- Fitur: [jumlah] landmark coordinates
- Split: 85% train, 15% test

### 3.2 Pendekatan ML
[Deskripsi pendekatan traditional ML - allgemein, tidak spesifik algoritma]

### 3.3 Pendekatan DL
[Deskripsi pendekatan deep learning - allgemein, tidak spesifik arsitektur]

### 3.4 Evaluasi
- Accuracy
- Precision, Recall, F1-score
- Training time
- Inference time

---

## 4. Eksperimen

### 4.1 Setup
- Hardware: [spesifikasi]
- Software: Python, [libraries yang dipakai]

### 4.2 Hasil

| Metode | Accuracy | Training Time | Inference Time |
|--------|----------|--------------|----------------|
| ML     | [INSERT] | [INSERT]     | [INSERT]       |
| DL     | [INSERT] | [INSERT]     | [INSERT]       |

### 4.3 Analisis Per-Huruf
[Tabel accuracy per huruf]

### 4.4 Analisis Confusion
[Huruf yang sering tertukar]

---

## 5. Pembahasan

### 5.1 Perbandingan Metode
- Perbandingan accuracy
- Perbandingan kompleksitas training
- Perbandingan kecepatan inference
- Kebutuhan resource

### 5.2 Kelebihan & Kekurangan
[Analisis]

### 5.3 Implikasi Praktis
- Kesesuaian untuk aplikasi real-time
- Pertimbangan deployment mobile

---

## 6. Kesimpulan

### 6.1 Temuan
- [Hasil utama]

### 6.2 Rekomendasi
- Metode terbaik untuk pengenalan BISINDO
- Rekomendasi deployment

### 6.3 Future Work
- Ekspansi data
- Word assembly real-time
- Support multi-user

---

## Referensi

[Diisi kemudian]

---

## Appendix: Konfigurasi

### Setup Hardware & Software
[Specs]

### Detail Experiment
[Parameter yang dipakai - diisi dari hasil training]