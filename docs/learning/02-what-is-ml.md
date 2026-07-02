# Apa Itu Machine Learning?

> Penjelasan Machine Learning tanpa jargon teknis. Setelah baca ini, kamu akan ngerti kenapa kita pakai ML buat recognize BISINDO letters.

---

## Daftar Isi

1. [Function Approximation](#function-approximation)
2. [Supervised Learning](#supervised-learning)
3. [Training vs Programming](#training-vs-programming)
4. [Kenapa ML untuk BISINDO?](#kenapa-ml-untuk-bisindo)
5. [Jenis-Jenis ML](#jenis-jenis-ml)
6. [ML vs Traditional Programming](#ml-vs-traditional-programming)

---

## Function Approximation

### Apa itu Approximation?

**Approximation** = mencari perkiraan yang "cukup bagus".

Contoh sehari-hari:
```
Jam analog: jarum pendek = jam, jarum panjang = menit
           ↓
         APPROXIMATION dari waktu

Thermometer: tinggi air = temperatur
             ↓
           APPROXIMATION dari panas
```

### Function dalam Matematika

Function itu "mesin" yang terima input, keluarin output:

```
┌─────────────┐
│   Function  │  f(x) = x²
├─────────────┤
│  Input: 3   │  →  Output: 9
│  Input: -2   │  →  Output: 4
│  Input: 0.5  │  →  Output: 0.25
└─────────────┘
```

### Function Approximation dalam ML

Di ML, kita **tidak tahu** function-nya. Kita cuma punya:
- Input data
- Output yang diharapkan

Kita cari function yang **mendekati** mapping dari input ke output.

```
┌────────────────────────────────────────┐
│              ML Problem                  │
├────────────────────────────────────────┤
│                                         │
│   Input: [0.64, 0.72, 0.0, ...]      │  ← 63 angka
│           ↓                            │
│        ???                              │  ← kita cari function-nya
│           ↓                            │
│   Output: "M"                          │  ← salah satu dari 26 huruf
│                                         │
└────────────────────────────────────────┘
```

---

## Supervised Learning

### Definisi

**Supervised Learning** = ML dengan data berlabel (labeled data).

```
┌────────────────────────────────────────────┐
│         Supervised Learning                 │
├────────────────────────────────────────────┤
│                                             │
│   Data:                                      │
│   ┌────────────────────────────────────┐   │
│   │ Input          │ Label/Output       │   │
│   ├────────────────┼────────────────────┤   │
│   │ [0.64, 0.72..] │ "A"               │   │
│   │ [0.58, 0.65..] │ "B"               │   │
│   │ [0.71, 0.83..] │ "C"               │   │
│   │ ...             │ ...               │   │
│   └────────────────────────────────────┘   │
│                                             │
│   Goal: Learn mapping input → label         │
│                                             │
└────────────────────────────────────────────┘
```

### Analogi: Ajari Anak Kenal Huruf

```python
# Kamu kasih contoh ke anak:
contoh = [
    ("Ini huruf A", "A"),      # input + label
    ("Ini huruf B", "B"),
    ("Ini huruf A lagi", "A"),
    # ...
]
```

Dari contoh-contoh itu, anak belajar pattern:
- "Huruf A punya bentuk..."
- "Huruf B punya bentuk..."

ML itu sama:
```python
# Dataset = contoh-contoh
dataset = [
    ([landmark_A], "A"),  # input + label
    ([landmark_B], "B"),
    # ...
]

# Model = "anak" yang belajar dari contoh
model = train(dataset)

# Setelah training, model bisa prediksi huruf baru
prediksi = model.predict(new_landmark)
```

---

## Training vs Programming

### Traditional Programming

```python
# Kamu tulis RULES explicit
def classify_hand(landmarks):
    # RULES yang kamu tentukan
    if finger_1_extended and not finger_2_extended:
        return "V"
    elif all_fingers_curled:
        return "O"
    else:
        return "UNKNOWN"

# Problem: terlalu banyak edge cases!
```

### Machine Learning Programming

```python
# Kamu kasih DATA, bukan rules
dataset = load_landmarks()  # 26,000+ examples

# Model OTOMATIS belajar rules dari data
model = train(dataset)

# Model figure out rules sendiri
prediction = model.predict(new_hand)
```

### Perbandingan Langsung

| Aspek | Traditional Programming | Machine Learning |
|-------|------------------------|------------------|
| **Input** | Rules + Data | Data + Labels |
| **Output** | Answers | Model (learned rules) |
| **Proses** | Human writes rules | Computer learns rules |
| **Edge Cases** | Manual handling | Learned from data |
| **Scaling** | Hard | Easier |

---

## Kenapa ML untuk BISINDO?

### Masalah dengan Rule-Based

Coba bayangin mau tulis rules untuk huruf A-Z:

```python
# Attempts menulis rules
if thumb_up() and index_curved():
    return "C"  # Hmm, tapi kalau jariagnya beda?

# Trying more...
if thumb_touches_index() and other_fingers_straight():
    return "A"  # Tapi orang lain mungkin beda?

# Still missing...
# Berapa banyak rules kita perlu?
# A-Z = 26 huruf × ~10 edge cases = 260 rules?
# Belum lagi kombinasi iluminasi, jarak, sudut!
```

### Kenapa ML Lebih Cocok?

```
Masalah BISINDO:
┌────────────────────────────────────────────┐
│                                             │
│  Input variation:                           │
│  • Jarak tangan berbeda (dekat/jauh)        │
│  • Sudut berbeda (miring, miring)           │
│  • Ukuran tangan berbeda (besar/kecil)       │
│  • Iluminasi berbeda (terang/gelap)         │
│  • Posisi berbeda (kiri/kanan/center)      │
│                                             │
│  = millions of possible inputs              │
│                                             │
└────────────────────────────────────────────┘
```

ML bisa handle variation ini karena:
1. **Learn dari banyak contoh** → capture semua variation
2. **Automatic feature learning** → figure out apa yang penting
3. **Generalization** → bisa prediksi inputs baru yang belum pernah dilihat

---

## Jenis-Jenis ML

### 1. Supervised Learning (Yang Kita Pakai)

```
Training: Input + Label → Learn mapping
Prediction: Input baru → Predicted label
```

Contoh BISINDO:
- Input: landmark coordinates
- Label: huruf (A-Z)
- Task: Classification (26 classes)

### 2. Unsupervised Learning

```
Training: Input only → Learn patterns/structure
```

Contoh:
- Clustering: group similar hands together
- Dimensionality reduction: compress 63 features → 2-3 for visualization

### 3. Reinforcement Learning

```
Training: Agent takes actions → gets rewards → learns optimal policy
```

Contoh:
- Game playing (Chess, Go)
- Robot control

---

## ML vs Traditional Programming

### Decision Tree (RF base)

```python
# ML model learns this tree automatically
# dari data, bukan dari kamu menulis if-else

# Contoh learned rules:
if landmark_5_y > 0.5:
    if landmark_8_x > 0.6:
        return "K"
    else:
        return "P"
else:
    if landmark_12_z < 0.1:
        return "M"
    else:
        return "N"
```

### Neural Network (CNN/MLP base)

```python
# ML model learns NUMERIC WEIGHTS
# bukan explicit if-else rules

weights = {
    'layer1': [[0.12, -0.34, ...], ...],  # 63 → 128
    'layer2': [[0.56, -0.78, ...], ...],  # 128 → 64
    'output': [[0.91, 0.23, ...], ...],     # 64 → 26
}

# Prediction: matrix multiplication, bukan if-else
output = forward_pass(landmarks, weights)
```

### Kenapa NN Bisa Tanpa If-Else?

```
Perceptron (neuron):
┌──────────────────────────────────────┐
│                                      │
│   w₀ x₀ ─┐                           │
│   w₁ x₁ ─┤                           │
│   w₂ x₂ ─┤── Σ(wᵢxᵢ) + b → ReLU → y │
│   ...    ─┤                           │
│   w₆₂x₆₂┘                            │
│                                      │
└──────────────────────────────────────┘

Dengan billions of connections, network bisa
represent complex decision boundaries
tanpa explicit rules.
```

---

## Ringkasan

```
┌────────────────────────────────────────────┐
│              ML Core Concepts               │
├────────────────────────────────────────────┤
│                                             │
│  1. Function Approximation                  │
│     Cari function yang map input → output   │
│                                             │
│  2. Supervised Learning                     │
│     Learn dari data + labels                │
│                                             │
│  3. Training vs Programming                 │
│     ML: learn from examples                 │
│     Traditional: write explicit rules       │
│                                             │
│  4. Why ML for BISINDO?                     │
│     Too many variations untuk rules         │
│                                             │
└────────────────────────────────────────────┘
```

**Next:** [03-neural-networks-basics.md](03-neural-networks-basics.md) - Neural Networks dari Nol

---

## Referensi

- Andrew Ng's ML Course (Coursera)
- 3Blue1Brown Neural Networks series (YouTube)
- Fast.ai Practical Deep Learning
