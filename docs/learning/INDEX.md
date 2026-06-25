# 🎓 BISINDO Landmark-Based Recognition: Learning Path

Selamat datang! Ini adalah panduan belajar lengkap untuk memahami bagaimana sistem BISINDO Bridge bekerja — dari data mentah sampai prediksi huruf.

**Target**: Mahasiswa yang baru belajar machine learning, tapi sudah familiar dengan Python dasar.

**Pendekatan**: Belajar dari kasus nyata (data BISINDO), bukan teori generic.

---

## 📚 Daftar Chapter

### [Chapter 0: Big Picture](00-big-picture.md)
**Kenapa kita ada di sini?**
- Apa masalah yang mau diselesaikan?
- Kenapa pilih landmark-based, bukan image-based?
- Demo data asli: 1 baris CSV = 1 gesture = 63 angka

**Baca ini kalau**: Baru mau paham problem dan konteks.

---

### [Chapter 1: Data Structure](01-data-structure.md)
**Apa yang sebenarnya kita kumpulkan?**
- Struktur CSV: 67 kolom, apa artinya
- 21 landmark tangan: mana yang mana?
- Beda koordinat A vs C vs M (pakai data nyata)
- Eksplorasi: mana yang lebih beda — posisi jempol atau telunjuk?

**Baca ini kalau**: Mau lihat data mentah dan cara membacanya.

---

### [Chapter 2: Feature Extraction](02-feature-extraction.md)
**Dari koordinat ke "ciri"**
- Kenapa koordinat mentah belum cukup?
- Normalisasi relative ke pergelangan tangan
- Buat feature baru: jarak, sudut, rasio
- StandardScaler: bikin semua angka skala sama

**Baca ini kalau**: Penasaran cara ubah landmark jadi feature yang bagus.

---

### [Chapter 3: Decision Tree](03-decision-tree.md)
**Model pertama: 1 pohon keputusan**
- Cara kerja decision tree
- Train 1 pohon, lihat accuracy
- Visualize: "Kalau lm8_y > 0.7 → prediksi A"
- Problem: overfit kalau cuma 1 pohon

**Baca ini kalau**: Mau ngerti dasar classification sebelum random forest.

---

### [Chapter 4: Random Forest](04-random-forest.md)
**Model kedua: banyak pohon**
- Kenapa 1 pohon kurang bagus?
- Bagging: setiap pohon lihat subset data
- Voting: majority vote
- Feature importance: landmark mana paling penting?

**Baca ini kalau**: Siap train Random Forest pertama.

---

### [Chapter 5: Evaluation](05-evaluation.md)
**Apa itu "bagus"?**
- Accuracy: % prediksi benar
- Confusion matrix: huruf mana yang sering ketukar?
- Precision/Recall per huruf
- Kenapa accuracy 95% bisa menipu?

**Baca ini kalau**: Mau evaluasi model dan cari huruf lemah.

---

### [Chapter 6: Iteration](06-iteration.md)
**Collect, retrain, repeat**
- Cek balance data
- Data augmentation: flip, rotate, scale landmark
- Retrain, bandingkan
- Siklus improvement

**Baca ini kalau**: Mau improve accuracy.

---

### [Chapter 7: Deployment](07-deployment.md)
**Model jadi aplikasi**
- Save model: .pkl files
- Load di server
- Input dari webcam → MediaPipe → landmark → prediksi → huruf
- Threshold confidence

**Baca ini kalau**: Mau deploy model ke production.

---

### [Chapter 8: Neural Network vs Random Forest](08-neural-network.md)
**Kapan upgrade?**
- Random Forest: cepat, ringan, CPU
- Neural Network: lebih akurat tapi butuh lebih banyak data + GPU
- Trade-off: speed vs accuracy
- Kapan upgrade worth it?

**Baca ini kalau**: Mau compare dengan deep learning.

---

## 🎯 Rekomendasi Urutan

**Untuk pemula**: Chapter 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

**Untuk yang udah paham dasar**: Skip ke Chapter 4 (Random Forest)

**Untuk yang udah punya model**: Langsung Chapter 5 (Evaluation)

---

## 📊 Data yang Dipakai

Semua contoh pakai data nyata dari:
```
dataset/landmarks_captured_v2.csv
```

**Statistik**:
- Total samples: ~21,000+
- Huruf terbanyak: A (5172), C (2668), B (2138)
- Huruf tersedikit: Z, Y, W (~100)
- Kontributor: legacy, Kevin, dll

**Format**: 67 kolom per baris
- 5 kolom metadata: `letter`, `image_path`, `split`, `num_hands`, `contributor`
- 62 kolom landmark: `lm0_x`, `lm0_y`, `lm0_z`, ..., `lm20_z`

---

## 🛠️ Tools yang Dipakai

- **Python 3.11+**
- **pandas**: baca CSV
- **numpy**: array operations
- **scikit-learn**: machine learning (Random Forest, metrics)
- **matplotlib/seaborn**: visualisasi (opsional)
- **MediaPipe**: extract landmark dari webcam

---

## 💡 Tips Belajar

1. **Baca chapter berurutan** — setiap chapter bangun di atas yang sebelumnya
2. **Coba sendiri** — jangan cuma baca, tapi jalankan kodenya
3. **Eksperimen** — ubah parameter, lihat apa yang terjadi
4. **Catat pertanyaan** — kalau bingung, tulis, nanti tanya
5. **Review** — setelah selesai 1 chapter, ulas ulang poin penting

---

## 🚀 Mulai dari Sini

**Pilih salah satu**:

### Opsi A: Pemula Total
Mulai dari [Chapter 0: Big Picture](00-big-picture.md)

### Opsi B: Udah Paham Konsep Dasar
Langsung ke [Chapter 1: Data Structure](01-data-structure.md)

### Opsi C: Mau Langsung Train Model
Lompat ke [Chapter 4: Random Forest](04-random-forest.md)

---

**Selamat belajar! 🎓**
