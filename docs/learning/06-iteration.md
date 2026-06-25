# Chapter 6: Iteration — Collect, Augment, Retrain

> **Tujuan chapter ini**: Kamu paham cara improve model lewat data balancing, augmentation, dan retraining cycle.

---

## 🔄 Siklus Improvement

Machine learning bukan "train sekali selesai". Ini siklus berulang:

```
        Train Model
            ↓
        Evaluate
            ↓
   Identify weak letters
            ↓
   Collect / Augment data
            ↓
        Retrain
            ↓
     (loop back to Evaluate)
```

---

## ⚖️ Masalah 1: Data Imbalanced

Lihat data kita sekarang:

| Huruf | Sample | Status |
|-------|--------|--------|
| A | 5172 | ✅ cukup |
| C | 2668 | ✅ cukup |
| B | 2138 | ✅ cukup |
| F | 2072 | ✅ cukup |
| M | 905 | ⚠️ kurang |
| Z | 100 | ❌ terlalu sedikit |
| Y | 95 | ❌ terlalu sedikit |
| W | 95 | ❌ terlalu sedikit |

**Masalah**: Model akan bias ke huruf dengan data banyak (A, B, C). Huruf dengan data sedikit (Z, Y, W) performancenya jelek.

### Solusi A: Collect Lebih Banyak Data

Paling straightforward: rekam lagi huruf yang kurang.

```bash
# Buka capture page, fokus huruf yang < 500 sample
https://<tunnel-url>/capture
```

Target: minimal **500 sample per huruf**, ideal 1000+.

### Solusi B: Class Weight

Kalau ga sempat collect, kita bisa kasih bobot lebih ke kelas minoritas:

```python
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',  # kelas sedikit → bobot lebih besar
    n_jobs=-1,
    random_state=42
)
```

**Cara kerja `class_weight='balanced'`**:
- Kelas dengan 100 sample → bobot besar
- Kelas dengan 5000 sample → bobot kecil
- Efek: model "peduli" sama-sama ke semua kelas

### Solusi C: Downsampling

Kalau A terlalu banyak (5172), kita bisa ambil subset:

```python
# Ambil max 1000 sample per huruf
df_balanced = df.groupby('letter').apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)
```

**Trade-off**: buang data A → mungkin accuracy A turun, tapi kelas lain naik.

---

## 🎭 Masalah 2: Variasi Data Kurang

**Problem**: Kalau semua data direkam dari 1 orang, 1 posisi, 1 lighting, model ga generalisasi.

**Contoh bad data**:
- 5000 sample A, semua dari Kevin, semua di siang hari
- Pas dipakai orang lain / malam hari → accuracy turun

### Solusi: Variasi Saat Collect

- **Kontributor berbeda**: ajak teman capture
- **Lighting berbeda**: rekam siang, sore, malam
- **Posisi berbeda**: tangan di kiri, kanan, atas, bawah
- **Sudut berbeda**: sedikit miring, dekat, jauh

### Solusi: Data Augmentation

Karena kita pakai **landmark** (bukan gambar), augmentasinya beda dari image-based.

**Augmentasi landmark**:

#### 1. Flip Horizontal (mirror)

```python
def flip_landmarks(landmarks):
    # x' = 1 - x  (mirror terhadap sumbu vertikal tengah)
    flipped = landmarks.copy()
    for i in range(21):
        flipped[i*3] = 1 - flipped[i*3]  # x dibalik
    return flipped
```

**Efek**: tangan kanan → tangan kiri. Bikin data 2x.

#### 2. Rotate

```python
def rotate_landmarks(landmarks, angle_deg):
    # Rotate semua landmark terhadap pergelangan
    angle_rad = np.radians(angle_deg)
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    rotated = landmarks.copy()
    wrist = landmarks[0:3]
    for i in range(21):
        x, y = landmarks[i*3], landmarks[i*3+1]
        # Relative ke wrist
        rx, ry = x - wrist[0], y - wrist[1]
        # Rotate
        new_x = rx * cos - ry * sin + wrist[0]
        new_y = rx * sin + ry * cos + wrist[1]
        rotated[i*3], rotated[i*3+1] = new_x, new_y
    return rotated
```

**Efek**: tangan miring -15°, +15° → model lebih robust terhadap rotasi.

#### 3. Scale

```python
def scale_landmarks(landmarks, scale_factor):
    # Scale jarak jari terhadap pergelangan
    scaled = landmarks.copy()
    wrist = landmarks[0:3]
    for i in range(21):
        for c in range(2):  # x, y
            scaled[i*3+c] = wrist[c] + (landmarks[i*3+c] - wrist[c]) * scale_factor
    return scaled
```

**Efek**: tangan besar/kecil → model invariant terhadap ukuran.

#### 4. Add Noise

```python
def add_noise(landmarks, noise_std=0.01):
    noise = np.random.normal(0, noise_std, len(landmarks))
    return landmarks + noise
```

**Efek**: simulasi landmark yang sedikit "off" → model lebih robust.

### Augmentation Pipeline

```python
def augment_sample(hand, label):
    samples = [(hand, label)]  # original

    # Flip
    samples.append((flip_landmarks(hand), label))

    # Rotate
    for angle in [-15, -10, 10, 15]:
        samples.append((rotate_landmarks(hand, angle), label))

    # Scale
    for scale in [0.9, 1.1]:
        samples.append((scale_landmarks(hand, scale), label))

    # Noise
    samples.append((add_noise(hand, 0.01), label))

    return samples  # 1 original + 9 augmented = 10x data
```

**Hasil**: dari 100 sample → 1000 sample per huruf (virtual).

---

## 📊 Eksperimen: Before vs After Augmentation

### Before (original data)

```python
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print(f"Test accuracy (original): {accuracy_score(y_test, rf.predict(X_test)):.4f}")
# → ~0.88
```

### After (augmented data)

```python
# Augment training set only (jangan augment test set!)
X_train_aug, y_train_aug = [], []
for hand, label in zip(X_train, y_train):
    for aug_hand, aug_label in augment_sample(hand, label):
        X_train_aug.append(aug_hand)
        y_train_aug.append(aug_label)

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)

rf_aug = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf_aug.fit(X_train_aug, y_train_aug)
print(f"Test accuracy (augmented): {accuracy_accuracy(y_test, rf_aug.predict(X_test)):.4f}")
# → ~0.91 (naik 3%)
```

**Kenapa naik?**
- Lebih banyak data training → model belajar variasi lebih kaya
- Augmentasi flip/rotate/scale → model invariant terhadap transformasi itu

---

## ⚠️ Hal-Hal Penting

### 1. Jangan Augment Test Set

**Rule of thumb**: Augment **hanya training set**.

Kenapa? Kalau kita augment test set, kita evaluasi model di data "bikinan" kita sendiri → tidak fair. Test set harus data asli yang belum dimodifikasi.

### 2. Augmentasi Harus Realistis

Jangan augment dengan transformasi yang ga masuk akal:
- ❌ Rotate 90° (tangan ga akan terbalik begitu)
- ❌ Scale 0.5 atau 2.0 (terlalu ekstrem)
- ✅ Rotate ±15°, Scale ±10%, Noise kecil

### 3. Cek Outlier

Sebelum augment, buang data yang jelas salah:

```python
# Cek: ada landmark yang koordinatnya absurd?
# x, y harus di range [0, 1]
# z biasanya di range [-0.5, 0.5]
bad = df[(df['lm0_x'] < 0) | (df['lm0_x'] > 1)]
print(f"Outlier: {len(bad)} sample")
```

---

## 🧠 Latihan Pemahaman

1. Kenapa data imbalanced bermasalah?
2. Apa itu data augmentation untuk landmark?
3. Kenapa augmentasi hanya di training set?
4. Sebut 3 jenis augmentasi landmark dan efeknya.
5. Kalau huruf A punya 5000 sample, Z cuma 100, apa solusi cepat?

<details>
<summary>📝 Jawaban</summary>

1. Model bias ke kelas mayoritas, kelas minoritas performance jelek.
2. Modifikasi landmark (flip, rotate, scale, noise) untuk bikin data tambahan tanpa rekam ulang.
3. Supaya evaluasi fair. Test set harus data asli, kalau augment → evaluasi di data bikinan = menipu diri sendiri.
4. Flip (mirror), Rotate (miring), Scale (besar/kecil), Noise (sedikit off). Tiap jenis bikin model invariant terhadap transformasi itu.
5. Pakai `class_weight='balanced'` atau augmentasi data Z, atau downsample A.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Siklus: train → evaluate → identify → augment → retrain
- [ ] Data imbalanced → bias ke kelas mayoritas
- [ ] Solusi imbalance: collect, class_weight, downsampling
- [ ] Augmentasi landmark: flip, rotate, scale, noise
- [ ] Augmentasi HANYA training set, bukan test set
- [ ] Augmentasi harus realistis (±15°, ±10%, noise kecil)

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 7: Deployment](07-deployment.md)** — kita save model dan deploy ke server untuk prediksi real-time.
