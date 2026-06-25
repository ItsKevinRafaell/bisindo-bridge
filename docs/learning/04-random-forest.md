# Chapter 4: Random Forest — Banyak Pohon Lebih Baik?

> **Tujuan chapter ini**: Kamu paham Random Forest, cara kerjanya, dan cara train model untuk data BISINDO.

---

## 🤔 Masalah Decision Tree

Dari Chapter 3, kita tau 1 pohon:
- Bisa **overfit** kalau terlalu dalam
- Bisa **underfit** kalau terlalu dangkal
- Instability: sedikit perubahan data training → pohon sangat berbeda

**Analogi**: 1 saksi mata sering salah ingat. 200 saksi mata → hasil rata-rata lebih akurat.

---

## 🌲 Apa itu Random Forest?

**Random Forest** = kumpulan banyak Decision Tree yang bekerja bareng.

Ada 2 ide utama:

### 1. Bagging (Bootstrap Aggregating)

Setiap tree belajar dari **subset acak data training**.

```
Training data asli: 1000 sample

Tree 1: ambil 1000 sample random dengan replacement
       → mungkin sample A muncul 3x, sample B ga muncul

Tree 2: ambil 1000 sample random lagi
       → kombinasi berbeda

Tree 3: ... dst
```

Kenapa? Supaya tiap tree punya "perspektif" berbeda.

### 2. Random Feature Selection

Di setiap split, tree hanya boleh pilih dari **subset acak feature**.

```
Dari 126 feature, random pilih √126 ≈ 11 feature.
Dari 11 feature itu, cari split terbaik.
```

Kenapa? Supaya tidak ada satu feature yang mendominasi. Setiap tree fokus ke aspek berbeda.

### 3. Voting

Setiap tree vote huruf apa. Yang paling banyak vote = prediksi akhir.

```
Tree 1: vote A (confidence 0.85)
Tree 2: vote A (confidence 0.78)
Tree 3: vote C (confidence 0.62)
Tree 4: vote A (confidence 0.91)
...
Tree 200: vote A (confidence 0.88)

Vote mayoritas: A (175/200) → Prediksi: A
Confidence: 0.87
```

---

## 🚀 Train Random Forest untuk BISINDO

Kode:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load
df = pd.read_csv('dataset/landmarks_captured_v2.csv')

# Features
cols = [f'lm{i}_{c}' for i in range(21) for c in ('x', 'y', 'z')]
X = df[cols].astype(np.float32).values
X = np.nan_to_num(X, nan=0.0)
X = np.concatenate([X, np.zeros_like(X)], axis=1)  # pad ke 126
y = df['letter'].astype(str).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
rf = RandomForestClassifier(
    n_estimators=200,    # jumlah pohon
    max_depth=None,      # pohon bisa sedalam apapun
    max_features='sqrt', # √126 feature per split
    n_jobs=-1,           # pakai semua CPU core
    random_state=42
)
rf.fit(X_train_s, y_train)

# Evaluate
train_acc = accuracy_score(y_train, rf.predict(X_train_s))
test_acc = accuracy_score(y_test, rf.predict(X_test_s))
print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
```

**Yang terjadi**:
- Scikit-learn bikin 200 pohon
- Tiap pohon lihat subset data dan feature yang beda
- Semua pohon di-fit secara paralel pakai semua CPU core
- Hasil: model yang robust terhadap overfit

---

## 📊 Kenapa Lebih Bagus dari Decision Tree?

### Eksperimen: Random Forest vs Decision Tree

```python
tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train_s, y_train)
tree_test = accuracy_score(y_test, tree.predict(X_test_s))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_s, y_train)
rf_test = accuracy_score(y_test, rf.predict(X_test_s))

print(f"Decision Tree test accuracy: {tree_test:.4f}")
print(f"Random Forest test accuracy: {rf_test:.4f}")
```

**Hasil yang diharapkan**:
```
Decision Tree:  ~80%
Random Forest:  ~90%
```

---

## 🔍 Feature Importance

Random Forest bisa kasih tahu landmark mana yang paling sering dipakai untuk split.

```python
importances = rf.feature_importances_

# Plot 10 feature paling penting
feature_names = [f'lm{i}_{c}' for i in range(21) for c in ('x','y','z')] * 2
indices = np.argsort(importances)[-10:]
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
```

**Hasil tipikal**:
```
lm8_y: 0.052   (ujung telunjuk, y)
lm4_x: 0.048   (ujung jempol, x)
lm12_y: 0.045  (ujung jari tengah, y)
lm20_y: 0.041  (ujung kelingking, y)
lm0_x: 0.038   (pergelangan, x)
```

**Interpretasi**: Ujung jari (terutama telunjuk, jempol, tengah) paling penting. Masuk akal, karena bentuk huruf BISINDO ditentukan oleh posisi ujung jari.

---

## 🎛️ Hyperparameter Tuning

Hyperparameter = setting sebelum training. Beberapa yang penting:

| Parameter | Default | Efek |
|-----------|---------|------|
| `n_estimators` | 100 | Jumlah pohon. Lebih banyak = lebih akurat tapi lebih lambat training. |
| `max_depth` | None | Kedalaman max tiap pohon. None = grow sampai pure. |
| `max_features` | 'sqrt' | Jumlah feature random per split. 'sqrt' atau 'log2'. |
| `min_samples_split` | 2 | Minimum sample untuk split node. Lebih besar = pohon lebih dangkal. |
| `min_samples_leaf` | 1 | Minimum sample di leaf. Lebih besar = less overfit. |

### Coba Eksperimen

```python
for n in [50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    test_acc = accuracy_score(y_test, rf.predict(X_test_s))
    print(f"n_estimators={n}: test={test_acc:.4f}")
```

**Hasil tipikal**:
```
n_estimators=50:   test=0.875
n_estimators=100:  test=0.889
n_estimators=200:  test=0.895
n_estimators=500:  test=0.898 (diminishing return)
```

**Insight**: 200 pohon cukup. 500 sedikit lebih baik tapi training lebih lama.

---

## 🧠 Latihan Pemahaman

1. Apa beda Decision Tree dan Random Forest?
2. Apa itu bagging? Kenapa diperlukan?
3. Kenapa voting mayoritas lebih bagus dari 1 pohon?
4. Kalau kita tambah `n_estimators` dari 200 ke 1000, apa yang terjadi? Worth it?

<details>
<summary>📝 Jawaban</summary>

1. Decision Tree = 1 pohon. Random Forest = banyak pohon yang vote bareng.
2. Bagging = tiap tree lihat subset acak data. Supaya tiap tree berbeda, mengurangi variance/overfit.
3. Error individual tree bisa saling cancel. Rata-rata banyak pohon lebih stabil dan akurat.
4. Training lebih lambat, accuracy sedikit naik tapi diminishing returns. Biasanya ga worth it kalau improvement <1%.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Random Forest = banyak Decision Tree
- [ ] Bagging = subset acak data per tree
- [ ] Random feature selection = subset acak feature per split
- [ ] Voting mayoritas = prediksi akhir
- [ ] Hyperparameter: n_estimators, max_depth, max_features
- [ ] Feature importance: landmark mana paling penting

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 5: Evaluation](05-evaluation.md)** — kita evaluasi model dan baca confusion matrix.
