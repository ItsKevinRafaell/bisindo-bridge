# Chapter 0: Big Picture — Kenapa Kita Ada di Sini?

> **Tujuan chapter ini**: Kamu paham masalah yang dipecahkan, kenapa pilih pendekatan landmark-based, dan gimana alur kerja keseluruhan sistem.

---

## 🎯 Masalah yang Mau Diselesaikan

Bayangkan situasi ini:

> Seorang penyandang tuna rungu ingin berkomunikasi dengan orang yang tidak bisa bahasa isyarat. Dia menunjukkan **huruf BISINDO** (A-Z) pakai tangan. Tapi lawan bicaranya tidak paham. Butuh **"penerjemah"** yang bisa baca gerakan tangan itu dan ubah jadi teks/suara.

**BISINDO** = Bahasa Isyarat Indonesia. Sistemnya **bukan** one-gesture-one-word (seperti ASL), tapi **ejaan jari** (fingerspelling): tiap huruf punya bentuk tangan sendiri.

**Goal aplikasi**: 
- Input: kamera baca gerakan tangan
- Output: teks huruf A-Z + (opsional) suara

---

## 🚧 Tantangan Utama

Supaya komputer bisa "baca" tangan, ada beberapa problem:

### Problem 1: Gimana komputer "ngerti" tangan?

Komputer ga paham "ini huruf A". Komputer cuma paham **angka**. 
Jadi kita harus **ubah tangan → angka** dulu.

### Problem 2: Tangan bisa beda posisi

Tangan kamu tunjukkan "A":
- Di pojok kiri layar → tetap A
- Di tengah layar → tetap A
- Posisi sedikit miring → tetap A

Tapi kalau kita pakai **koordinat absolute** (posisi di layar), komputer bakal bingung. "A di kiri" dan "A di tengah" bakal keliatan beda padahal sama-sama A.

### Problem 3: Butuh data banyak, tapi training ga boleh lama

Kalau pakai foto (image-based):
- Butuh 500+ foto per huruf × 26 huruf = 13,000 foto
- Training butuh GPU, ~30 menit
- Berat buat dijalankan di laptop/VPS biasa

Kita mau yang **lebih ringan**, bisa jalan di VPS Rp 30,000/bulan.

---

## 💡 Solusi: Landmark-Based (Pendekatan Kita)

Alih-alih simpan **foto** tangan, kita simpan **21 titik koordinat** (landmark) yang MediaPipe extract dari tangan.

### Apa itu 21 Landmark?

MediaPipe (library Google) detect tangan dan tandai 21 titik penting:

```
        8  (ujung telunjuk)
        |
        6
        |
        5
        |
0 ----- 1 --- 2 --- 3 --- 4  (jempol, dari pergelangan ke ujung)
  (wrist)
        |
        9
        |
       10
        |
       12 (ujung jari tengah)
        |
       14
        |
       16 (ujung jari manis)
        |
       18
        |
       20 (ujung kelingking)
```

Setiap titik punya 3 koordinat:
- **x**: posisi horizontal (0 = kiri, 1 = kanan)
- **y**: posisi vertikal (0 = atas, 1 = bawah)
- **z**: kedalaman (negatif = lebih dekat ke kamera)

Jadi: **21 titik × 3 koordinat = 63 angka per tangan**.

### Kenapa Ini Lebih Baik?

| Aspek | Image-Based (foto) | Landmark-Based (kita) |
|-------|--------------------|-----------------------|
| Data per sample | ~300,000 angka (pixel) | **63 angka** |
| Data dibutuhkan | 500+ per huruf | 100+ per huruf |
| Training time | ~30 menit (GPU) | **~2-5 menit (CPU)** |
| Inference speed | ~100ms | **~5ms** |
| Invariant lighting? | ❌ (gelap/terang beda) | ✅ (titik tetap) |
| Invariant background? | ❌ (gedung di belakang beda) | ✅ |
| Storage | besar (MB per foto) | kecil (bytes per sample) |

**Analogi**:
- Image-based = kenalin teman dari **foto muka** (bandingin tiap pixel)
- Landmark-based = kenalin teman dari **ciri khas** (tinggi, berat, rambut, dll)

Ciri khas jauh lebih ringkas dan tetap akurat.

---

## 📊 Demo: Data Asli Kita

Ini 1 baris CSV nyata dari `landmarks_captured_v2.csv`:

```
letter: A
contributor: legacy
lm0_x: 0.262   lm0_y: 0.653   lm0_z: 0.000    ← pergelangan (wrist)
lm1_x: 0.348   lm1_y: 0.663   lm1_z: -0.038   ← jempol sendi bawah
lm2_x: 0.451   lm2_y: 0.624   lm2_z: -0.063
lm3_x: 0.535   lm3_y: ...      lm3_z: ...
lm4_x: 0.596   lm4_y: 0.643                     ← ujung jempol
...
lm8_x: 0.575   lm8_y: 0.204                     ← ujung telunjuk
...
```

**Poin penting**:
- 1 baris = 1 gesture = 1 huruf
- 63 angka = "ciri khas" tangan untuk huruf itu
- `letter` = jawaban yang benar (label)

Saat training, kita kasih model **63 angka + jawaban benar**, suruh model cari pola.

---

## 🔄 Alur Kerja Keseluruhan

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE BISINDO                          │
│                                                             │
│  1. COLLECT (Chapter 1)                                     │
│     Webcam → MediaPipe → 63 angka → CSV                    │
│                                                             │
│  2. PREPARE (Chapter 2)                                     │
│     CSV → normalisasi → features → train/test split        │
│                                                             │
│  3. TRAIN (Chapter 3 & 4)                                   │
│     features → Decision Tree → Random Forest               │
│                                                             │
│  4. EVALUATE (Chapter 5)                                    │
│     test set → accuracy, confusion matrix                   │
│                                                             │
│  5. ITERATE (Chapter 6)                                     │
│     huruf lemah? → collect lagi → augmentasi → retrain     │
│                                                             │
│  6. DEPLOY (Chapter 7)                                      │
│     model.pkl → load di server → prediksi real-time        │
│                                                             │
│  7. UPGRADE (Chapter 8)                                     │
│     RF cukup? → coba Neural Network                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤔 Pertanyaan Refleksi

Sebelum lanjut, coba jawab (ga usah liat jawaban dulu):

1. Kenapa kita simpan 63 angka, bukan foto?
2. Kenapa `letter` kolom penting banget di CSV?
3. Kenapa image-based training lama?
4. Kalau tangan di posisi beda di layar, kenapa landmark tetap bisa kenalan?

<details>
<summary>📝 Klik buat liat jawaban</summary>

1. **63 angka lebih ringkas, cepat dilatih, invariant terhadap lighting/background.** Foto butuh ribuan pixel, sensitif cahaya, dan training lama.
2. **`letter` = label/jawaban benar.** Tanpa label, model ga tau mau belajar apa (supervised learning butuh input + jawaban).
3. **Image-based proses ribuan pixel per foto × ribuan foto = komputasi besar.** Butuh GPU. Landmark cuma 63 angka, CPU cukup.
4. **Karena kita bisa normalisasi landmark ke posisi pergelangan tangan (Chapter 2).** Jadi "A di kiri" dan "A di kanan" punya ciri relative yang sama.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] BISINDO = ejaan jari, tiap huruf = bentuk tangan
- [ ] Tantangan: komputer cuma ngerti angka, tangan bisa beda posisi
- [ ] Solusi: pakai 21 landmark × 3 koordinat = 63 angka per tangan
- [ ] 1 baris CSV = 1 gesture = label + 63 angka
- [ ] Alur: collect → prepare → train → evaluate → iterate → deploy

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 1: Data Structure](01-data-structure.md)** — kita bedah data asli, liat beda koordinat A vs C vs M, dan paham struktur 21 landmark.
