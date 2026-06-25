# Chapter 3: Decision Tree — Model Pertama

> **Tujuan chapter ini**: Kamu paham cara kerja decision tree, train model pertama, dan liat masalahnya.

---

## 🌳 Apa itu Decision Tree?

**Decision Tree** = pohon keputusan. Model ini bikin keputusan lewat serangkaian pertanyaan ya/tidak.

**Analogi**: Bayangkan kamu main "Tebak Huruf" dengan temanmu. Kamu kasih dia 63 angka landmark, dan dia harus tebak hurufnya. Cara dia tebak: tanya ya/tidak.

```
Apakah ujung telunjuk (lm8) tinggi di atas?
├─ YA (lm8_y < 0.4)
│   └─ Apakah ujung jempol (lm4) dekat pergelangan?
│       ├─ YA (lm4_y - lm0_y < -0.3)
│       │   └─ Apakah jarak jempol-telunjuk besar?
│       │       ├─ YA → Prediksi: A
│       │       └─ TIDAK → Prediksi: M
│       └─ TIDAK → Prediksi: C
└─ TIDAK (lm8_y >= 0.4)
    └─ Apakah jempol ke kiri?
        ├─ YA → Prediksi: D
        └─ TIDAK → Prediksi: B
```

Tiap **node** = pertanyaan. Tiap **leaf** = prediksi.

---

## 🧮 Cara Decision Tree Belajar

Saat training, decision tree cari **split terbaik** di tiap node:

1. **Untuk setiap feature** (63 koordinat):
   - Coba berbagai threshold (misal: lm8_y < 0.3, < 0.4, < 0.5, ...)
   - Hitung seberapa "pure" hasil split-nya

2. **Pilih split terbaik**: yang bikin child node paling pure (homogen)
   - Pure = semua sample di node itu huruf yang sama
   - Impure = campuran A, B, C, ...

3. **Ulangi** untuk tiap child node sampai:
   - Node sudah pure (semua huruf sama), atau
   - Mencapai `max_depth` (batas kedalaman)

### Metrik "Purity"

**Gini Impurity** (yang dipakai scikit-learn):

```
Gini = 1 - Σ(p_i)²

di mana p_i = proporsi kelas i di node itu
```

**Contoh**:
- Node isinya 100 A, 0 lain → Gini = 1 - (1.0)² = 0 (pure!)
- Node isinya 50 A, 50 C → Gini = 1 - (0.5² + 0.5²) = 0.5 (impure)
- Node isinya 33 A, 33 C, 33 M → Gini = 1 - (0.11+0.11+0.11) = 0.67 (sangat impure)

**Goal**: Bikin Gini serendah mungkin di tiap leaf.

---

## 🚀 Train Decision Tree Pertama Kita

Kode:

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('dataset/landmarks_captured_v2.csv')

# Extract features
cols = [f'lm{i}_{c}' for i in range(21) for c in ('x', 'y', 'z')]
X = df[cols].astype(np.float32).values
X = np.nan_to_num(X, nan=0.0)
# Pad to 126
X = np.concatenate([X, np.zeros_like(X)], axis=1)
y = df['letter'].astype(str).values

# Split
X_train, X_test, y_train, y_test = train_tree_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train decision tree
tree = DecisionTreeClassifier(
    max_depth=10,          # batas kedalaman
    random_state=42
)
tree.fit(X_train_s, y_train)

# Evaluate
pred = tree.predict(X_test_s)
acc = accuracy_score(y_test, pred)
print(f"Decision Tree accuracy: {acc:.4f}")
```

**Hasil yang diharapkan**: ~70-85% accuracy.

---

## 🔍 Visualisasi Decision Tree

Kalau kita export pohonnya, bakal keliatan kayak gini (disederhanakan):

```
lm8_y_scaled <= 0.5      [gini=0.85, samples=18000, semua huruf]
├─ True
│  └─ lm4_x_scaled <= -0.3  [gini=0.6, samples=5000, A,C,M dominan]
│     ├─ True → C (gini=0.2, samples=2000)
│     └─ False → A (gini=0.3, samples=3000)
└─ False
   └─ lm0_y_scaled <= 0.6 [gini=0.7, samples=13000]
      ├─ ...
```

**Cara baca**:
1. Pertama cek: `lm8_y` (ujung telunjuk) tinggi?
2. Kalau ya, cek `lm4_x` (jempol) posisinya
3. dst sampai leaf (prediksi huruf)

---

## ⚠️ Masalah Decision Tree: Overfitting

**Overfitting** = model terlalu "hafal" data training, sampai detail noise pun dihafal.

**Gejala**:
- Training accuracy: 99% (hampir sempurna)
- Test accuracy: 70% (turun banyak)

**Kenapa?** Kalau `max_depth` terlalu dalam, tree bikin leaf untuk tiap sample individu. Tiap sample jadi "aturan khusus". Pas di data baru, aturan itu ga cocok.

### Coba Eksperimen max_depth

```python
for depth in [3, 5, 10, 20, None]:  # None = unlimited
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train_s))
    test_acc = accuracy_score(y_test, tree.predict(X_test_s))
    print(f"depth={depth}: train={train_acc:.3f} test={test_acc:.3f}")
```

**Hasil tipikal**:

| max_depth | Train acc | Test acc |
|-----------|-----------|----------|
| 3 | 0.55 | 0.52 (underfit) |
| 5 | 0.75 | 0.70 |
| 10 | 0.90 | 0.80 |
| 20 | 0.98 | 0.75 (overfit) |
| None | 1.00 | 0.70 (parah overfit) |

**Pelajaran**: Tree terlalu dalam = overfit. Tree terlalu dangkal = underfit.

---

## 💡 Solusi: Random Forest (Spoiler Chapter 4)

1 tree rentan overfit. Solusinya: pakai **banyak tree** dan gabungkan dengan voting. Itu yang dibahas di Chapter 4.

---

## 🧐 Latihan Pemahaman

1. Decision tree bikin keputusan lewat apa?
2. Apa itu Gini Impurity? Range nilainya berapa?
3. Kenapa max_depth terlalu besar bikin overfit?
4. Kalau test accuracy jauh lebih rendah dari train accuracy, apa yang terjadi?

<details>
<summary>📝 Jawaban</summary>

1. Serangkaian pertanyaan ya/tidak tentang feature (threshold split).
2. Ukuran impurity node. Range 0 (pure) sampai ~0.67 (3 kelas sama rata) atau mendekati 1 (banyak kelas tercampur). Gini rendah = baik.
3. Tree terlalu dalam bikin leaf khusus per-sample → hafal noise → gagal generalisasi ke data baru.
4. Overfitting. Model hafal data training tapi ga bisa prediksi data baru.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Decision tree = serangkaian pertanyaan ya/tidak
- [ ] Tree belajar dengan cari split yang minimasi Gini impurity
- [ ] max_depth kontrol kompleksitas: terlalu dalam = overfit
- [ ] 1 tree rentan overfit → solusinya Random Forest (Chapter 4)

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 4: Random Forest](04-random-forest.md)** — kita pakai banyak pohon dan liat kenapa lebih bagus.
