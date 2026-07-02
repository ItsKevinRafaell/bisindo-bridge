# Prerequisites: NumPy & Math Basics

> Sebelum belajar CNN, kita perlu ngerti dasar-dasar NumPy dan operasi matematika. Ini penting banget buat ngerti gimana data landmark diproses.

---

## Daftar Isi

1. [NumPy Arrays](#numpy-arrays)
2. [Reshape: Flat ↔ Matrix](#reshape-flat--matrix)
3. [Dot Product](#dot-product)
4. [Visualisasi Tangan](#visualisasi-tangan)
5. [Latihan](#latihan)

---

## NumPy Arrays

NumPy itu library Python buat komputasi numerik. Array NumPy itu kayak list, tapi bisa doing operasi matematika lebih efisien.

```python
import numpy as np

# List Python (bisa, tapi lambat)
my_list = [1, 2, 3, 4, 5]

# NumPy array (lebih cepat)
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Output: [1 2 3 4 5]
```

### Operasi Dasar

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr + 10)      # [11 12 13 14 15]  ← semua elemen +10
print(arr * 2)       # [2 4 6 8 10]     ← semua elemen ×2
print(arr.sum())     # 15               ← jumlah semua
print(arr.mean())    # 3.0              ← rata-rata
print(arr.std())     # 1.414...         ← standar deviasi
```

---

## Reshape: Flat ↔ Matrix

Ini **KONSEP KUNCI** buat ngerti landmark data.

### Apa itu Reshape?

Data landmark di CSV disimpan flat (1 baris, 63 kolom):
```
[lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ..., lm20_z]
  ↓        ↓        ↓        ↓        ↓        ↓
  0        1        2        3        4        5       62
```

Tapi enak kalo kita reshape jadi matrix 21×3:
```
landmark_0: [lm0_x, lm0_y, lm0_z]
landmark_1: [lm1_x, lm1_y, lm1_z]
...
landmark_20: [lm20_x, lm20_y, lm20_z]
```

### Kode Reshape

```python
# Data flat: 63 angka
flat_data = np.array([0.64, 0.72, 0.0, 0.58, 0.65, 0.0, ...])  # 63 elements

# Reshape ke 21×3 (21 landmark, 3 koordinat)
reshaped = flat_data.reshape(21, 3)
print(reshaped.shape)  # (21, 3)

print(reshaped)
# [[0.64 0.72 0.0 ]
#  [0.58 0.65 0.0 ]
#  ...
#  [0.51 0.83 0.0 ]]
```

### Kenapa Penting?

```
Flat (63,)                          Reshaped (21×3)
┌────────────────────────────┐
│ [x₀,y₀,z₀,x₁,y₁,z₁,...]    │    ┌─landmark_0: [x₀,y₀,z₀]
│                             │    ├─landmark_1: [x₁,y₁,z₁]
│                             │    ├─landmark_2: [x₂,y₂,z₂]
│                             │    └─landmark_20: [x₂₀,y₂₀,z₂₀]
└────────────────────────────┘
```

Reshaped lebih intuitif buat visualisasi dan memahami posisi tangan.

---

## Dot Product

Dot product (product dalam) itu operasi matrix yang **SANGAT PENTING** di neural networks.

### Definisi

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product: 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
result = np.dot(a, b)
print(result)  # 32
```

### Visualisasi

```
a = [1, 2, 3]
b = [4, 5, 6]
         ↓
┌──────────────────────────────────────┐
│ a[0]×b[0] + a[1]×b[1] + a[2]×b[2]   │
│   1×4    +   2×5    +   3×6         │
│     4    +    10    +    18        │
│              = 32                   │
└──────────────────────────────────────┘
```

### Dot Product di Neural Network

Setiap neuron menghitung:
```
output = dot(weights, inputs) + bias
```

Contoh dengan 3 input:
```python
inputs = np.array([0.64, 0.72, 0.58])   # 3 fitur
weights = np.array([0.5, -0.3, 0.8])    # 3 weight
bias = 0.1

output = np.dot(weights, inputs) + bias
print(output)  # 0.5×0.64 + (-0.3)×0.72 + 0.8×0.58 + 0.1 = ...
```

---

## Visualisasi Tangan

Mari kita visualisasi 21 landmark tangan.

### Landmark Positions

```
           4
           |
    0 ─────┼───── 1
     \     |     /
      \    |    /
       2---3----   ← Thumb joint
        \  |  /
         5-6-7     ← Index finger (5,6,7,8)
           |
           9-10-11  ← Middle finger
           |
          13-14-15  ← Ring finger
           |
          17-18-19   ← Pinky finger
           |
          (wrist center = 0)
```

- **Landmark 0**: Wrist (pergelangan)
- **Landmark 1-4**: Thumb (ibu jari)
- **Landmark 5-8**: Index (telunjuk)
- **Landmark 9-12**: Middle (tengah)
- **Landmark 13-16**: Ring (manis)
- **Landmark 17-20**: Pinky (kelingking)

### Plot dengan Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulasi data landmark (21×3)
landmarks = np.random.rand(21, 3)  # 21 points, x,y,z

# Plot 2D (x, y)
plt.figure(figsize=(8, 8))
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100)

# Label setiap landmark
for i in range(21):
    plt.annotate(str(i), (landmarks[i, 0], landmarks[i, 1]), 
                 fontsize=8, ha='center', va='bottom')

plt.title("21 Hand Landmarks")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()
```

### Koneksi antar Landmark

```python
# Definisikan koneksi tulang
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# Plot connections
for start, end in connections:
    plt.plot([landmarks[start, 0], landmarks[end, 0]],
             [landmarks[start, 1], landmarks[end, 1]], 'b-', linewidth=2)
```

---

## Latihan

### Latihan 1: Load CSV & Extract Features

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("dataset/landmarks_captured_v2.csv")

# Ambil 63 fitur landmark
feature_cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
X = df[feature_cols].values

print(f"Shape: {X.shape}")  # (n_samples, 63)
print(f"First sample: {X[0]}")  # 63 angka pertama
```

### Latihan 2: Reshape ke 21×3

```python
# Ambil 1 sample
single_sample = X[0]  # shape: (63,)

# Reshape ke 21×3
hand = single_sample.reshape(21, 3)
print(f"Hand shape: {hand.shape}")  # (21, 3)
print(f"Landmark 0 (wrist): {hand[0]}")
print(f"Landmark 5 (index): {hand[5]}")
```

### Latihan 3: Normalisasi Sederhana

```python
# Translate: kurangkan wrist position dari semua landmark
wrist = hand[0]
hand_centered = hand - wrist

print(f"Wrist after centering: {hand_centered[0]}")  # [0, 0, 0]

# Scale: bagi dengan max distance dari wrist
distances = np.linalg.norm(hand_centered, axis=1)
max_dist = distances.max()
hand_normalized = hand_centered / max_dist

print(f"Max distance from wrist: {max_dist}")
```

---

## Ringkasan

| Konsep | Fungsi |
|--------|--------|
| NumPy Array | Menyimpan data numerik efisien |
| Reshape | Ubah format data (flat↔matrix) |
| Dot Product | Operasi dasar neural network |
| Visualisasi | Bantu ngerti data landmark |

**Next:** [02-what-is-ml.md](02-what-is-ml.md) - Apa itu Machine Learning?

---

## Referensi

- NumPy docs: https://numpy.org/doc/
- Matplotlib tutorial: https://matplotlib.org/stable/tutorials/index.html
