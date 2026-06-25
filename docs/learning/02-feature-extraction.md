# Chapter 2: Feature Extraction — Dari Koordinat ke "Ciri"

> **Tujuan chapter ini**: Kamu paham kenapa koordinat mentah belum cukup, dan gimana cara ubah jadi feature yang invariant dan informatif.

---

## 🎯 Masalah: Koordinat Absolute Tidak Invariant

Bayangkan situasi ini:

**Situasi 1**: Kamu tunjukkan huruf "A" di pojok kiri layar.
- Pergelangan tangan (wrist) di posisi x=0.2, y=0.7
- Ujung telunjuk di x=0.3, y=0.3

**Situasi 2**: Kamu tunjukkan huruf "A" yang sama, tapi di tengah layar.
- Pergelangan tangan di x=0.5, y=0.5
- Ujung telunjuk di x=0.6, y=0.1

**Pertanyaan**: Ini huruf yang sama (A), tapi koordinatnya beda semua. Gimana caranya model tau ini sama?

**Jawaban**: Kita pakai **koordinat relative**, bukan absolute.

---

## 💡 Solusi 1: Normalisasi Relative ke Pergelangan Tangan

Ide: Kurangi semua koordinat dengan posisi pergelangan tangan (wrist = lm0).

```python
# Sebelum (absolute)
lm0 = (0.2, 0.7)  # wrist di kiri
lm8 = (0.3, 0.3)  # telunjuk

# Sesudah (relative ke wrist)
lm0_rel = (0.0, 0.0)  # wrist jadi origin
lm8_rel = (0.3 - 0.2, 0.3 - 0.7) = (0.1, -0.4)  # telunjuk 0.1 ke kanan, 0.4 ke atas
```

**Keuntungan**:
- Sekarang posisi tangan di layar ga penting
- Yang penting: posisi jari **relative terhadap pergelangan**
- Model bisa generalisasi ke posisi tangan mana saja

---

## 💡 Solusi 2: Feature Engineering Tambahan

Selain relative position, kita bisa bikin feature lain yang lebih informatif:

### 1. Jarak Antar Landmark

```python
# Jarak dari ujung jempol ke ujung telunjuk
distance_thumb_index = sqrt((lm4_x - lm8_x)² + (lm4_y - lm8_y)²)
```

**Kenapa penting?**
- Huruf A: jempol dan telunjuk rapat (jarak kecil)
- Huruf C: jempol dan telunjuk agak renggang (jarak sedang)
- Huruf V: jempol dan telunjuk jauh (jarak besar)

### 2. Sudut Antar Jari

```python
# Sudut antara telunjuk dan jari tengah
angle_index_middle = calculate_angle(lm5, lm9, lm13)
```

**Kenapa penting?**
- Huruf V: sudut besar (telunjuk dan tengah renggang)
- Huruf U: sudut kecil (telunjuk dan tengah rapat)

### 3. Rasio

```python
# Rasio panjang telunjuk terhadap panjang jari tengah
ratio = length_index / length_middle
```

**Kenapa penting?**
- Bisa bedakan huruf yang mirip tapi beda proporsi

---

## 📊 Contoh Nyata: Feature untuk A vs C vs M

Kita hitung feature dari data asli kita:

### Huruf A

```
Pergelangan (lm0):  x=0.262, y=0.653
Ujung jempol (lm4): x=0.596, y=0.643
Ujung telunjuk (lm8): x=0.575, y=0.204

Feature 1: Jempol relative ke pergelangan
  Δx = 0.596 - 0.262 = +0.334
  Δy = 0.643 - 0.653 = -0.010

Feature 2: Telunjuk relative ke pergelangan
  Δx = 0.575 - 0.262 = +0.313
  Δy = 0.204 - 0.653 = -0.449

Feature 3: Jarak jempol-telunjuk
  distance = sqrt((0.596-0.575)² + (0.643-0.204)²)
           = sqrt(0.021² + 0.439²)
           = sqrt(0.0004 + 0.193)
           = 0.439
```

**Interpretasi A**: Jempol ke kanan, telunjuk jauh ke atas, jarak jempol-telunjuk besar.

---

### Huruf C

```
Pergelangan (lm0):  x=0.591, y=0.788
Ujung jempol (lm4): x=0.384, y=0.804
Ujung telunjuk (lm8): x=0.366, y=0.565

Feature 1: Jempol relative ke pergelangan
  Δx = 0.384 - 0.591 = -0.207
  Δy = 0.804 - 0.788 = +0.016

Feature 2: Telunjuk relative ke pergelangan
  Δx = 0.366 - 0.591 = -0.225
  Δy = 0.565 - 0.788 = -0.223

Feature 3: Jarak jempol-telunjuk
  distance = sqrt((0.384-0.366)² + (0.804-0.565)²)
           = sqrt(0.018² + 0.239²)
           = sqrt(0.0003 + 0.057)
           = 0.239
```

**Interpretasi C**: Jempol ke kiri, telunjuk ke kiri dan atas, jarak jempol-telunjuk kecil.

---

### Huruf M

```
Pergelangan (lm0):  x=0.640, y=0.657
Ujung jempol (lm4): x=0.585, y=0.550
Ujung telunjuk (lm8): x=0.570, y=0.457

Feature 1: Jempol relative ke pergelangan
  Δx = 0.585 - 0.640 = -0.055
  Δy = 0.550 - 0.657 = -0.107

Feature 2: Telunjuk relative ke pergelangan
  Δx = 0.570 - 0.640 = -0.070
  Δy = 0.457 - 0.657 = -0.200

Feature 3: Jarak jempol-telunjuk
  distance = sqrt((0.585-0.570)² + (0.550-0.457)²)
           = sqrt(0.015² + 0.093²)
           = sqrt(0.0002 + 0.0086)
           = 0.094
```

**Interpretasi M**: Jempol sedikit ke kiri dan atas, telunjuk ke kiri dan jauh ke atas, jarak jempol-telunjuk sangat kecil.

---

## 📈 Perbandingan Feature

| Huruf | Δx_jempol | Δy_jempol | Δx_telunjuk | Δy_telunjuk | Jarak jempol-telunjuk |
|-------|-----------|-----------|-------------|-------------|----------------------|
| A | +0.334 | -0.010 | +0.313 | -0.449 | 0.439 |
| C | -0.207 | +0.016 | -0.225 | -0.223 | 0.239 |
| M | -0.055 | -0.107 | -0.070 | -0.200 | 0.094 |

**Kesimpulan**:
- **A**: Jempol dan telunjuk ke kanan, telunjuk jauh ke atas, jarak besar
- **C**: Jempol dan telunjuk ke kiri, telunjuk ke atas, jarak sedang
- **M**: Jempol dan telunjuk ke kiri, telunjuk jauh ke atas, jarak sangat kecil

Feature ini **invariant** terhadap posisi tangan di layar, dan **informatif** untuk membedakan huruf.

---

## 🧮 Latihan: Hitung Feature

Diberikan data huruf "B":

```
Pergelangan (lm0):  x=0.400, y=0.700
Ujung jempol (lm4): x=0.410, y=0.350
Ujung telunjuk (lm8): x=0.420, y=0.320
```

**Hitung**:
1. Jempol relative ke pergelangan (Δx, Δy)
2. Telunjuk relative ke pergelangan (Δx, Δy)
3. Jarak jempol-telunjuk

<details>
<summary>📝 Klik buat lihat jawaban</summary>

1. **Jempol relative**: Δx = 0.410 - 0.400 = +0.010, Δy = 0.350 - 0.700 = -0.350
2. **Telunjuk relative**: Δx = 0.420 - 0.400 = +0.020, Δy = 0.320 - 0.700 = -0.380
3. **Jarak**: sqrt((0.410-0.420)² + (0.350-0.320)²) = sqrt(0.01² + 0.03²) = sqrt(0.0001 + 0.0009) = sqrt(0.001) = 0.032

**Interpretasi B**: Jempol dan telunjuk hampir di posisi yang sama (jarak kecil = 0.032), keduanya jauh ke atas dari pergelangan. Ini sesuai bentuk huruf B (jempol dan telunjuk rapat, jari lain teracung).

</details>

---

## 🧠 Kenapa Feature Engineering Penting?

**Prinsip**: Model machine learning cuma bisa belajar dari feature yang kita kasih. Kalau feature jelek, model jelek.

**Contoh buruk**: Pakai koordinat absolute
- Model bingung: "A di kiri" vs "A di kanan" = beda huruf?
- Accuracy rendah

**Contoh bagus**: Pakai koordinat relative + jarak + sudut
- Model fokus: "Oh, jempol ke kanan, telunjuk ke atas, jarak besar = huruf A"
- Accuracy tinggi

---

## 🚀 Apa yang Dilakukan Kode Kita?

Di `train_landmark_model.py`, kita sudah melakukan normalisasi:

```python
# Load data dari CSV
df = pd.read_csv('dataset/landmarks_captured_v2.csv')

# Extract 63 koordinat (21 landmark × 3 coords)
cols = [f'lm{i}_{c}' for i in range(21) for c in ('x', 'y', 'z')]
X = df[cols].values  # shape: (n_samples, 63)

# Handle NaN (kalau ada landmark yang ga terdeteksi)
X = np.nan_to_num(X, nan=0.0)

# Pad ke 126 fitur (hand1 + zero hand2)
X1 = X[:, :63]  # hand 1
X2 = np.zeros_like(X1)  # hand 2 (zero-padded kalau ga ada)
X = np.concatenate([X1, X2], axis=1)  # shape: (n_samples, 126)

# Scale ke mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Catatan**: Kode kita sekarang masih pakai koordinat mentah (belum relative). Ini bisa di-improve nanti. Tapi untuk sekarang, Random Forest cukup robust untuk handle variasi posisi.

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Koordinat absolute tidak invariant terhadap posisi tangan
- [ ] Solusi: normalisasi relative ke pergelangan tangan
- [ ] Feature tambahan: jarak antar landmark, sudut, rasio
- [ ] Feature yang bagus = invariant + informatif
- [ ] Kode kita sudah normalisasi, tapi bisa di-improve dengan feature engineering lebih lanjut

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 3: Decision Tree](03-decision-tree.md)** — kita train model pertama pakai decision tree dan lihat cara kerjanya.
