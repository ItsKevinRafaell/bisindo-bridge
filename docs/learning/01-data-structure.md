# Chapter 1: Data Structure — Apa yang Sebenarnya Kita Kumpulkan?

> **Tujuan chapter ini**: Kamu paham struktur CSV, arti 21 landmark, dan beda koordinat tiap huruf.

---

## 📁 File Data Kita

Data kita ada di:

```
dataset/landmarks_captured_v2.csv
```

Ini formatnya **CSV (Comma-Separated Values)**. Bisa dibuka pakai Excel, Notepad, atau Python.

---

## 🔍 Isi File

### 1 Baris = 1 Gesture = 1 Huruf

```
A, cap_A_4, train, 2, legacy, 0.262, 0.653, 0.000, ...
```

Kolom-kolomnya:

| Kolom | Contoh | Arti |
|-------|--------|------|
| `letter` | A | Huruf yang ditunjukkan |
| `image_path` | cap_A_4 | ID/sample name |
| `split` | train | Data untuk training |
| `num_hands` | 2 | Jumlah tangan (1 atau 2) |
| `contributor` | legacy | Siapa yang rekam |
| `lm0_x` | 0.262 | Pergelangan tangan (wrist), koordinat x |
| `lm0_y` | 0.653 | Pergelangan tangan, koordinat y |
| `lm0_z` | 0.000 | Pergelangan tangan, koordinat z |
| ... | ... | ... |
| `lm20_z` | 0.123 | Ujung kelingking, koordinat z |

**Total kolom**: 5 metadata + (21 landmark × 3 koordinat) = **67 kolom**.

---

## ✋ 21 Landmark Tangan

Ini nomor landmarknya dan bagian tangannya:

```
Wrist  = 0   (pergelangan tangan)

Jempol (Thumb):
    1 → pangkal jempol (CMC)
    2 → sendi bawah (MCP)
    3 → sendi tengah (IP)
    4 → ujung jempol (tip)

Telunjuk (Index):
    5 → pangkal (MCP)
    6 → sendi bawah (PIP)
    7 → sendi tengah (DIP)
    8 → ujung telunjuk (tip)

Jari tengah (Middle):
    9  → pangkal (MCP)
    10 → PIP
    11 → DIP
    12 → ujung

Jari manis (Ring):
    13 → pangkal (MCP)
    14 → PIP
    15 → DIP
    16 → ujung

Kelingking (Pinky):
    17 → pangkal (MCP)
    18 → PIP
    19 → DIP
    20 → ujung
```

Visualisasi:

```
         8          (ujung telunjuk)
         |
     7 - 6 - 5
         |
    12 - 10 - 9    (jari tengah)
         |
    16 - 14 - 13   (jari manis)
         |
    20 - 18 - 17   (kelingking)

0 -- 1 -- 2 -- 3 -- 4   (jempol)
(wrist)
```

---

## 📊 Data Nyata: A vs C vs M

Berikut ini landmark dari data asli kita:

### Huruf A

```
Pergelangan tangan (lm0):  x=0.262   y=0.653
Ujung jempol (lm4):        x=0.596   y=0.643
Ujung telunjuk (lm8):      x=0.575   y=0.204
Ujung kelingking (lm20):     x=0.295   y=0.494

Ujung jempol relative terhadap pergelangan:
    Δx = +0.333   (jauh ke kanan)
    Δy = -0.010   (hampir sama tinggi)
```

**Bentuk A**: Jempol memegang telunjuk, telunjuk lurus ke atas. Jadi ujung telunjuk (lm8) tinggi di atas (y kecil = di atas layar), jempol ke kanan.

---

### Huruf C

```
Pergelangan tangan (lm0):  x=0.591   y=0.788
Ujung jempol (lm4):        x=0.384   y=0.804
Ujung telunjuk (lm8):      x=0.366   y=0.565
Ujung kelingking (lm20):     x=0.367   y=0.564

Ujung jempol relative terhadap pergelangan:
    Δx = -0.207   (ke kiri)
    Δy = +0.016   (sedikit ke bawah)
```

**Bentuk C**: Tangan membentuk huruf C, jadi semua ujung jari (jempol, telunjuk, dll) rapat membentuk lengkungan.

---

### Huruf M

```
Pergelangan tangan (lm0):  x=0.640   y=0.657
Ujung jempol (lm4):        x=0.585   y=0.550
Ujung telunjuk (lm8):      x=0.570   y=0.457
Ujung kelingking (lm20):     x=0.662   y=0.411

Ujung jempol relative terhadap pergelangan:
    Δx = -0.055
    Δy = -0.108
```

**Bentuk M**: Tiga jari ke atas (telunjuk, tengah, manis), jempol dan kelingking turun. Kompleks!

---

## 🧐 Kenapa Posisi Absolute Bikin Bingung?

Lihat tabel ini:

| Huruf | lm0_x | lm0_y | lm4_x | lm4_y |
|-------|-------|-------|-------|-------|
| A | 0.262 | 0.653 | 0.596 | 0.643 |
| C | 0.591 | 0.788 | 0.384 | 0.804 |
| M | 0.640 | 0.657 | 0.585 | 0.550 |

Kalau kita lihat koordinat absolute (tanpa relative ke pergelangan), posisi tangan tiap huruf beda-beda.

Tapi ingat: posisi tangan di layar bisa jadi di mana saja. Satu orang tunjukkan A di kiri, orang lain A di tengah. Kalau model hafal koordinat absolute, dia bakal bingung.

**Solusi**: Normalisasi relative ke pergelangan tangan (wrist). Kita bahas di Chapter 2.

---

## 🧮 Latihan: Baca Data Landmark

Coba jawab dari data di bawah:

```
Huruf B:
  lm0_x = 0.400, lm0_y = 0.700   (wrist)
  lm4_x = 0.410, lm4_y = 0.350   (thumb tip)
  lm8_x = 0.420, lm8_y = 0.320   (index tip)
  lm12_x = 0.430, lm12_y = 0.310 (middle tip)
  lm16_x = 0.440, lm16_y = 0.320  (ring tip)
  lm20_x = 0.450, lm20_y = 0.330  (pinky tip)
```

**Pertanyaan**:
1. Koordinat absolute ujung telunjuk (lm8) di mana?
2. Koordinat relative ujung telunjuk terhadap pergelangan?
3. Kenapa semua ujung jari punya y yang hampir sama?

<details>
<summary>📝 Klik buat lihat jawaban</summary>

1. **Absolute lm8**: x=0.420, y=0.320 → agak kanan, di atas
2. **Relative lm8**: Δx = 0.420 - 0.400 = +0.020, Δy = 0.320 - 0.700 = -0.380
3. **Karena huruf B**: semua jari teracung lurus ke atas, jadi y ujung jari mirip (kecil)

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] 1 baris CSV = 1 gesture = 67 kolom
- [ ] 21 landmark × 3 koordinat = 63 angka per tangan
- [ ] `letter` = label/jawaban benar
- [ ] Posisi absolute tangan bisa beda-beda, makanya butuh normalisasi (Chapter 2)
- [ ] Bisa membaca landmark dan membayangkan bentuk tangan

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 2: Feature Extraction](02-feature-extraction.md)** — kita ubah koordinat mentah jadi feature yang invariant dan informatif.
