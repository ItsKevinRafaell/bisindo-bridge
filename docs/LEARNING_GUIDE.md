# BISINDO Bridge - Learning Guide

> Panduan pembelajaran Machine Learning & Deep Learning melalui proyek nyata: Sistem Pengenalan Bahasa Isyarat Indonesia (BISINDO).

---

## Daftar Isi

1. [Pengenalan Sign Language Recognition](#chapter-1-pengenalan-sign-language-recognition)
2. [Pengumpulan Data & Ekstraksi Fitur](#chapter-2-pengumpulan-data--ekstraksi-fitur)
3. [Preprocessing & Normalisasi Data](#chapter-3-preprocessing--normalisasi-data)
4. [Pendekatan Machine Learning](#chapter-4-pendekatan-machine-learning)
5. [Pendekatan Deep Learning](#chapter-5-pendekatan-deep-learning)
6. [Evaluasi Model](#chapter-6-evaluasi-model)
7. [Jebakan Augmentasi Data](#chapter-7-jebakan-augmentasi-data)
8. [Deployment ke Dunia Nyata](#chapter-8-deployment-ke-dunia-nyata)
9. [Pelajaran & Best Practices](#chapter-9-pelajaran--best-practices)

---

## Chapter 1: Pengenalan Sign Language Recognition

### Apa itu BISINDO?

BISINDO (Bahasa Isyarat Indonesia) adalah bahasa isyarat yang digunakan oleh komunitas Tuli di Indonesia. Berbeda dengan SIBI (Sistem Isyarat Bahasa Indonesia) yang dibuat pemerintah, BISINDO berkembang secara alami di komunitas Tuli.

BISINDO memiliki 26 huruf alfabet (A-Z), masing-masing dengan bentuk tangan yang unik. Beberapa huruf menggunakan 1 tangan (C, E, I, L, O, R, U, V, Y), beberapa menggunakan 2 tangan (A, B, D, F, G, H, J, K, M, N, P, Q, S, T, W, X, Z).

### Mengapa Landmark-Based?

Ada dua pendekatan utama untuk sign language recognition:

| Pendekatan | Cara Kerja | Kelebihan | Kekurangan |
|------------|-----------|-----------|------------|
| **Image-based** | Input gambar/frame langsung ke CNN | Kaya informasi visual | Butuh GPU besar, sensitif lighting/background |
| **Landmark-based** | Ekstrak titik-titik tangan dulu, baru klasifikasi | Ringan, cepat, robust | Kehilangan info tekstur/warna |

Kita pilih **landmark-based** karena:
- Lebih ringan → bisa jalan di CPU/laptop biasa
- Tidak terpengaruh warna kulit, background, lighting
- Cocok untuk real-time webcam inference

### MediaPipe HandLandmarker

Google MediaPipe menyediakan model pre-trained yang bisa mendeteksi **21 landmark** pada setiap tangan:

```
Landmark 0: Wrist (pergelangan tangan)
Landmark 1-4: Thumb (ibu jari)
Landmark 5-8: Index finger (telunjuk)
Landmark 9-12: Middle finger (jari tengah)
Landmark 13-16: Ring finger (jari manis)
Landmark 17-20: Pinky (jari kelingking)
```

Setiap landmark punya koordinat **(x, y, z)** yang dinormalisasi ke range 0-1 relatif terhadap frame kamera.

### Mengapa Bandingkan ML vs DL?

Proyek ini adalah **studi komparatif**:

- **ML (Random Forest, SVM)**: Algoritma klasik, interpretable, cepat training
- **DL (MLP, CNN)**: Model neural network, bisa belajar pattern kompleks

Pertanyaan riset: **Mana yang lebih baik untuk sign language recognition?**

---

## Chapter 2: Pengumpulan Data & Ekstraksi Fitur

### Bagaimana MediaPipe Mengekstrak Landmark

Saat kamu menunjukkan tangan ke webcam, MediaPipe melakukan:

1. **Deteksi tangan** → temukan bounding box tangan di frame
2. **Landmark estimation** → prediksi 21 titik pada setiap tangan
3. **Output**: array of (x, y, z) untuk setiap landmark (di projek ini tidak menggunakan z axis)

```python
# Contoh output MediaPipe
hand_landmarks = [
    (0.5, 0.3, 0.1),   # Landmark 0: Wrist
    (0.52, 0.28, 0.09), # Landmark 1: Thumb CMC
    (0.54, 0.26, 0.08), # Landmark 2: Thumb MCP
    ...
]
```

Koordinat **dinormalisasi** ke range 0-1:
- x = 0 → kiri frame, x = 1 → kanan frame
- y = 0 → atas frame, y = 1 → bawah frame
- z = kedalaman relatif (semakin negatif = semakin dekat ke kamera)

### Format Data: 1-Hand vs 2-Hand

**1-Hand Format (63 features):**
```
21 landmarks × 3 coordinates (x, y, z) = 63 features
```

**2-Hand Format (84 features):**
```
Hand 1: 21 landmarks × 2 coordinates (x, y) = 42 features
Hand 2: 21 landmarks × 2 coordinates (x, y) = 42 features
Total: 84 features
```

**Kenapa 2-hand pakai xy saja (bukan xyz)?**
- z-coordinate dari MediaPipe kurang akurat untuk 2 tangan
- xy sudah cukup untuk membedakan bentuk tangan
- Lebih sedikit fitur → model lebih cepat

### Struktur Dataset

File: `dataset/landmarks_2hands.csv`

```csv
letter,path,split,num_hands,contributor,lm0_x,lm0_y,...,lm20_x,lm20_y,h2_lm0_x,h2_lm0_y,...,h2_lm20_x,h2_lm20_y
A,A_0_1782718269,train,2,capture,0.278,0.791,...,0.283,0.700,0.835,0.756,...,0.640,0.818
B,B_0_1782718270,train,2,capture,0.828,0.704,...,0.576,0.999,0.312,0.725,...,0.316,0.631
```

**Kolom:**
- `letter`: Huruf yang direpresentasikan (A-Z)
- `path`: Identifier unik untuk sample
- `split`: train/test split
- `num_hands`: Jumlah tangan terdeteksi (1 atau 2)
- `contributor`: Siapa yang collect data
- `lm0_x` sampai `lm20_y`: 42 fitur untuk Hand 1
- `h2_lm0_x` sampai `h2_lm20_y`: 42 fitur untuk Hand 2 (zeros jika hanya 1 tangan)

### Script Pengumpulan Data

File: `capture_fast2.py`

```bash
# Collect 1000 samples untuk huruf A
python3 capture_fast2.py A --count 1000

# Batch collection (Q-Z)
for letter in Q R S T U V W X Y Z; do
  python3 capture_fast2.py $letter --count 1000
done
```

**Cara kerja:**
1. Buka webcam
2. Deteksi tangan dengan MediaPipe
3. Tampilkan skeleton overlay (hijau = hand 1, biru = hand 2)
4. Simpan landmark ke CSV setiap 0.05 detik (20 samples/detik)
5. Ulangi sampai target count tercapai

### Pelajaran: Konsistensi Format

**Masalah yang pernah terjadi:**
- Training data: 1-hand format (63 features)
- Inference: 2-hand format (84 features)
- Result: Model gagal total karena input size mismatch

**Solusi:** Selalu gunakan `capture_fast2.py` untuk consistency.

---

## Chapter 3: Preprocessing & Normalisasi Data

### Mengapa Raw Coordinates Tidak Bisa Langsung Dipakai?

Bayangkan kamu collect data dengan tangan **dekat** ke kamera:
```
Landmark positions: (0.3, 0.5), (0.35, 0.48), ...
```

Lalu inference dengan tangan **jauh** dari kamera:
```
Landmark positions: (0.45, 0.55), (0.47, 0.54), ...
```

Meskipun bentuk tangan **sama persis**, koordinatnya **berbeda** karena:
1. **Posisi di frame** berbeda (tangan di kiri vs kanan)
2. **Ukuran** berbeda (tangan besar vs kecil)
3. **Jarak dari kamera** berbeda

Model akan bingung dan accuracy drop drastis.

### StandardScaler

StandardScaler menormalisasi distribusi fitur:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Apa yang dilakukan:**
- Hitung mean (μ) dan standard deviation (σ) untuk setiap fitur
- Transform: `x_scaled = (x - μ) / σ`
- Result: mean = 0, std = 1 untuk setiap fitur

**Kenapa penting:**
- Fitur dengan range besar tidak mendominasi fitur dengan range kecil
- Model converge lebih cepat
- Regularization bekerja lebih efektif

### Hand-Centric Normalization

Ini adalah **game-changer** untuk distance-invariant recognition.

**Konsep:**
1. **Translate ke wrist** → jadikan wrist sebagai origin (0, 0)
2. **Scale by hand size** → normalize berdasarkan ukuran tangan

```python
def normalize_hand(hand_coords):
    wrist = hand_coords[0]  # Landmark 0 = wrist
    centered = hand_coords - wrist  # Translate ke origin
    
    # Scale by max distance from wrist (hand size)
    distances = np.linalg.norm(centered, axis=1)
    max_dist = distances.max()
    if max_dist > 0:
        centered = centered / max_dist
    
    return centered
```

**Visualisasi:**

Sebelum normalisasi:
```
Tangan dekat:  wrist=(0.3, 0.5), thumb=(0.35, 0.45)
Tangan jauh:   wrist=(0.6, 0.7), thumb=(0.62, 0.68)
```

Setelah normalisasi:
```
Tangan dekat:  wrist=(0, 0), thumb=(0.5, -0.5)
Tangan jauh:   wrist=(0, 0), thumb=(0.5, -0.5)  ← SAMA!
```

**Hasil:**
- Tanpa normalisasi: accuracy 99% (dekat) → 20% (jauh)
- Dengan normalisasi: accuracy 95%+ untuk semua jarak

### Pipeline Normalisasi Lengkap

```
Raw landmarks (84 features)
    ↓
Hand-centric normalization (wrist-centered, size-normalized)
    ↓
StandardScaler (mean=0, std=1)
    ↓
Model input
```

**Penting:** Pipeline yang sama harus dipakai di **training** dan **inference**.

---

## Chapter 4: Pendekatan Machine Learning

### Random Forest

**Konsep:**
- Ensemble dari banyak Decision Tree
- Setiap tree dilatih pada subset data yang berbeda (bootstrap sampling)
- Final prediction = majority vote dari semua tree

**Analogi:**
Bayangkan kamu tanya 100 orang "ini huruf apa?". Setiap orang punya expertise berbeda (ada yang fokus jari telunjuk, ada yang fokus posisi tangan). Final answer = jawaban yang paling banyak dipilih.

**Kelebihan:**
- Tidak perlu feature scaling (tapi tetap membantu)
- Robust terhadap overfitting
- Feature importance bisa diinterpretasi
- Cepat training

**Kekurangan:**
- Model besar (banyak tree)
- Kurang fleksibel untuk pattern kompleks

**Implementasi:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,      # 300 trees
    max_depth=None,        # Unlimited depth
    max_features="sqrt",   # sqrt(n_features) per split
    random_state=42
)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
```

### Support Vector Machine (SVM)

**Konsep:**
- Cari hyperplane yang **memaksimalkan margin** antara classes
- Support vectors = data points yang paling dekat ke decision boundary
- Kernel trick: map ke higher dimension untuk linear separation

**Visualisasi:**
```
Class A: ● ● ●
Class B: ○ ○ ○

Decision boundary: garis yang memisahkan ● dan ○ dengan margin terbesar
```

**Kelebihan:**
- Efektif untuk high-dimensional data
- Memory efficient (hanya simpan support vectors)
- Bisa handle non-linear dengan kernel

**Kekurangan:**
- Lambat untuk dataset besar
- Kurang interpretable
- Sensitif terhadap feature scaling

**Implementasi:**
```python
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",          # Radial Basis Function
    C=1.0,                 # Regularization parameter
    gamma="scale",         # Kernel coefficient
    probability=True       # Enable probability estimates
)
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)
```

### Feature Engineering untuk ML

ML models (terutama SVM) sangat bergantung pada feature quality:

1. **StandardScaler** → wajib untuk SVM
2. **Hand-centric normalization** → membuat features distance-invariant
3. **Feature selection** → hapus fitur yang tidak informatif (optional)

### Training Pipeline (sklearn)

File: `train/train_ml.py`

```python
# 1. Load data
df = pd.read_csv("dataset/landmarks_2hands.csv")
X = df[feature_cols].values
y = df["letter"].values

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
accuracy = model.score(X_test, y_test)

# 6. Save
joblib.dump(model, "models/ml/rf_model.pkl")
```

---

## Chapter 5: Pendekatan Deep Learning

### MLP vs CNN untuk Landmark Data

**MLP (Multi-Layer Perceptron):**
- Fully connected layers
- Treat features sebagai flat vector
- Tidak ada spatial awareness

**CNN (Convolutional Neural Network):**
- Convolutional layers untuk extract local patterns
- Pooling layers untuk reduce dimensionality
- **1D CNN** cocok untuk sequential data (landmarks adalah sequence)

**Kenapa 1D CNN?**

Landmarks adalah **sequence terstruktur**:
```
Landmark 0 (wrist) → Landmark 1-4 (thumb) → Landmark 5-8 (index) → ...
```

1D CNN bisa belajar:
- Pattern lokal (bentuk jari individu)
- Pattern global (konfigurasi seluruh tangan)
- Hierarchical features (simple → complex)

### Arsitektur CNN

File: `train/train_dl.py`

```python
class CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        self.conv = nn.Sequential(
            # Block 1: Extract low-level features
            nn.Conv1d(1, 64, 3, padding=1),   # 84 → 84
            nn.ReLU(),
            nn.MaxPool1d(2),                   # 84 → 42
            
            # Block 2: Extract mid-level features
            nn.Conv1d(64, 128, 3, padding=1),  # 42 → 42
            nn.ReLU(),
            nn.MaxPool1d(2),                   # 42 → 21
            
            # Block 3: Extract high-level features
            nn.Conv1d(128, 64, 3, padding=1),  # 21 → 21
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)            # 21 → 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                      # 64×8 = 512
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                   # Prevent overfitting
            nn.Linear(128, num_classes)        # 26 classes
        )
```

**Penjelasan Layer:**

| Layer | Input | Output | Fungsi |
|-------|-------|--------|--------|
| Conv1d(1, 64, 3) | (batch, 1, 84) | (batch, 64, 84) | 64 filters, kernel size 3 |
| MaxPool1d(2) | (batch, 64, 84) | (batch, 64, 42) | Downsample 2× |
| Conv1d(64, 128, 3) | (batch, 64, 42) | (batch, 128, 42) | 128 filters |
| MaxPool1d(2) | (batch, 128, 42) | (batch, 128, 21) | Downsample 2× |
| Conv1d(128, 64, 3) | (batch, 128, 21) | (batch, 64, 21) | 64 filters |
| AdaptiveAvgPool1d(8) | (batch, 64, 21) | (batch, 64, 8) | Fixed output size |
| Linear(512, 128) | (batch, 512) | (batch, 128) | Fully connected |
| Linear(128, 26) | (batch, 128) | (batch, 26) | Output: 26 classes |

**Total parameters:** ~116K (untuk 84 features, 26 classes)

### Training Pipeline (PyTorch)

```python
# 1. Prepare data
X_tensor = torch.FloatTensor(X_train).reshape(-1, 1, 84)
y_tensor = torch.LongTensor(y_train)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 2. Initialize model
model = CNN(input_dim=84, num_classes=26)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(50):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    
    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        accuracy = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch+1}: accuracy={accuracy:.4f}")

# 4. Save model
torch.save(model.state_dict(), "models/dl/cnn_2hand_model.pt")
```

### Hyperparameters

| Parameter | Value | Alasan |
|-----------|-------|--------|
| Learning rate | 0.001 | Default Adam, stabil |
| Batch size | 256 | Balance speed & stability |
| Epochs | 50 | Cukup converge, tidak overfit |
| Dropout | 0.3 | Prevent overfitting |
| Optimizer | Adam | Adaptive learning rate |

---

## Chapter 6: Evaluasi Model

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
```

**Stratification:**
- Memastikan distribusi class sama di train dan test
- Contoh: jika A = 10% di dataset, maka A = 10% di train dan test
- Penting untuk dataset imbalanced

**Split ratio:**
- 85% training (22,263 samples)
- 15% testing (3,929 samples)

### Metrics

**Accuracy:**
```
Accuracy = (Benar) / (Total)
```
- Mudah dipahami
- Tapi misleading untuk imbalanced dataset

**Precision, Recall, F1:**
```
Precision = TP / (TP + FP)  → "Kalau model bilang A, seberapa yakin?"
Recall    = TP / (TP + FN)  → "Dari semua A, berapa yang tertangkap?"
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

**Contoh:**
- Model prediksi "A" 100 kali, 90 benar → Precision = 90%
- Ada 100 sample "A" di test, model tangkap 95 → Recall = 95%
- F1 = 2 × (0.9 × 0.95) / (0.9 + 0.95) = 92.4%

### Confusion Matrix

File: `eval/confusion.py`

```bash
python3 eval/confusion.py --model cnn_2hand
```

**Cara baca:**
```
              Predicted
              A    B    C    ...
Actual  A   [177]  1    0    ...
        B     0  [147]  2    ...
        C     1    0  [150]  ...
```

- **Diagonal** = prediksi benar
- **Off-diagonal** = prediksi salah
- **Row** = actual class
- **Column** = predicted class

**Contoh dari proyek kita:**
```
Most confused pairs:
  N → M: 7   (N salah prediksi jadi M)
  M → N: 6   (M salah prediksi jadi N)
  B → E: 2   (B salah prediksi jadi E)
```

**Insight:**
- M dan N sering tertukar → bentuk tangan mirip
- B dan E kadang tertukar → jari ditekuk sama-sama

### Pelajaran: Validation Accuracy ≠ Real-World Accuracy

**Kasus nyata:**
- Validation accuracy: 99.1%
- Webcam test (tangan jauh): 20% accuracy

**Penyebab:**
- Validation data: kondisi sama dengan training (jarak, lighting)
- Webcam test: kondisi berbeda (jarak bervariasi, lighting berubah)

**Solusi:**
- Hand-centric normalization
- Test di berbagai kondisi
- Jangan terlalu percaya validation metrics

---

## Chapter 7: Jebakan Augmentasi Data

### Apa itu Data Augmentation?

Data augmentation = membuat variasi baru dari data yang ada:

```python
# Contoh augmentasi
rotated = rotate_landmarks(hand, angle=10°)
scaled = scale_landmarks(hand, factor=1.1)
noisy = add_noise(hand, std=0.01)
```

**Tujuan:**
- Tambah jumlah training data
- Buat model lebih robust
- Prevent overfitting

### Mengapa Tampaknya Ide Bagus?

**Logika:**
- Dataset kita: 26,000 samples
- Setelah augmentation 6×: 156,000 samples
- Lebih banyak data = model lebih baik?

**Hasil awal:**
- Validation accuracy: 99% → 100%
- "Wah, augmentation berhasil!"

### Mengapa Gagal?

**Masalah 1: Synthetic vs Real**

Augmentation membuat pola yang **tidak realistis**:
```
Original:  jari telunjuk lurus, jari lain ditekuk
Augmented: jari telunjuk lurus + rotate 15° + scale 1.2

Real webcam: jari telunjuk lurus (tapi tidak persis sama dengan augmented)
```

Model belajar pattern dari augmented data, tapi real webcam input berbeda.

**Masalah 2: MediaPipe Limitation**

MediaPipe hanya detect 1 hand ketika 2 tangan overlap. Augmenting 1-hand patterns tidak membantu 2-hand recognition.

**Masalah 3: Distribution Shift**

Augmented data punya distribusi berbeda dari real data:
- Training distribution: augmented patterns
- Test distribution: real webcam patterns
- Mismatch → performance drop

### Hasil Eksperimen

| Approach | Validation Accuracy | Webcam Performance |
|----------|-------------------|-------------------|
| Original data | 99.1% | ✅ Best |
| Simple augmentation (2×) | 95% | ⚠️ Worse |
| Aggressive augmentation (6-8×) | 100% |  Worst |

**Insight:** Validation accuracy bisa naik, tapi real-world performance turun.

### Kapan Augmentation Berguna?

Augmentation **berguna** ketika:
- Variasi yang dibuat **realistis**
- Match dengan kondisi deployment
- Contoh: rotate ±5° untuk handle camera angle variation

Augmentation **berbahaya** ketika:
- Membuat pola yang tidak mungkin terjadi di real world
- Terlalu agresif (6-8× augmentation)
- Tidak test di real deployment

---

## Chapter 8: Deployment ke Dunia Nyata

### Webcam Inference Pipeline

File: `inference_webcam.py`

```
Webcam Feed (30 FPS)
    ↓
MediaPipe HandLandmarker
    ↓
Extract 84 features (hand 1 + hand 2)
    ↓
Hand-centric normalization
    ↓
StandardScaler
    ↓
CNN prediction
    ↓
Post-processing (C/S disambiguation)
    ↓
Display result
```

**Implementation:**
```python
while True:
    ret, frame = cap.read()
    
    # 1. Detect hands
    lm, num_hands = extract(frame, detector)
    
    # 2. Predict
    if lm:
        result = clf.predict(lm, num_hands)
        
        # 3. Display
        frame = draw(frame, lm, num_hands, result)
        cv2.putText(frame, f"{result['letter']} ({result['confidence']:.2f})", ...)
    
    cv2.imshow("BISINDO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### C/S Disambiguation

**Masalah:**
- C dan S punya bentuk tangan **sangat mirip**
- Perbedaan: C = 1 tangan, S = 2 tangan

**Solusi:**
```python
# Post-processing logic
if pred_letter == 'S' and num_hands == 1:
    pred_letter = 'C'  # 1 hand = C
elif pred_letter == 'C' and num_hands == 2:
    if probs['S'] > 0.3:
        pred_letter = 'S'  # 2 hands + high S probability = S
```

**Kenapa threshold 0.3?**
- Empirical testing menunjukkan ini balance antara false positive dan false negative
- Bisa di-tune berdasarkan use case

### MediaPipe Limitations

**1. Overlapping Hands**

Ketika 2 tangan overlap, MediaPipe hanya detect 1 hand:
```
Expected: 2 hands detected
Actual: 1 hand detected
Result: S (2-hand letter) → salah prediksi jadi C
```

**2. Lighting Conditions**

- Terlalu gelap → landmark detection gagal
- Terlalu terang (backlight) → silhouette only, no details
- Ideal: diffused lighting, tidak ada shadow keras

**3. Fast Movement**

- Motion blur → landmark jitter
- Solution: temporal smoothing (average last N frames)

**4. Hand Size**

- Tangan terlalu kecil di frame (< 50px) → detection unreliable
- Tangan terlalu besar (> 80% frame) → some landmarks out of frame

### Performance Optimization

**FPS Target:** 30 FPS untuk smooth experience

**Bottlenecks:**
1. MediaPipe detection (~10-15ms)
2. Model inference (~5ms untuk CNN)
3. Display rendering (~5ms)

**Optimizations:**
- Run MediaPipe setiap frame, model setiap 3 frames
- Use GPU untuk inference (jika available)
- Reduce display resolution

---

## Chapter 9: Pelajaran & Best Practices

### Key Lessons

**1. Data Format Consistency is Critical**

Training data 63 features + inference 84 features = model gagal total.

**Best practice:**
- Standardize format dari awal
- Document schema dengan jelas
- Validate input shape sebelum training

**2. Hand-Centric Normalization is Essential**

Tanpa normalization: accuracy 99% (dekat) → 20% (jauh).
Dengan normalization: accuracy 95%+ untuk semua jarak.

**Best practice:**
- Selalu apply hand-centric normalization untuk landmark-based recognition
- Test di berbagai jarak dan posisi

**3. Validation Accuracy ≠ Real-World Accuracy**

Model bisa 100% di validation tapi gagal di production.

**Best practice:**
- Test di real deployment environment
- Collect diverse data (jarak, lighting, angles)
- Monitor performance post-deployment

**4. Augmentation Can Hurt Performance**

Lebih banyak data ≠ lebih baik jika data tidak realistis.

**Best practice:**
- Augment only with realistic variations
- Always test augmented model on real data
- Prefer quality over quantity

**5. Batch Data Collection Works Better**

Collect semua 26 huruf sekaligus → retrain →才发现 ada masalah.

**Best practice:**
- Collect in batches (A-K, L-Z)
- Retrain dan verify setelah setiap batch
- Catch issues early

### Data Collection Workflow

**Recommended approach:**

```bash
# Phase 1: Collect A-K
for letter in A B C D E F G H I J K; do
  python3 capture_fast2.py $letter --count 1000
done

# Retrain dan verify
python3 train/train_dl.py --data dataset/landmarks_2hands.csv --epochs 50 --name cnn_2hand
python3 eval/confusion.py --model cnn_2hand

# Phase 2: Collect L-Z
for letter in L M N O P Q R S T U V W X Y Z; do
  python3 capture_fast2.py $letter --count 1000
done

# Final retrain
python3 train/train_dl.py --data dataset/landmarks_2hands.csv --epochs 50 --name cnn_2hand_full
```

**Target:** 1000 samples per huruf, balance dataset.

### When to Retrain vs Finetune

**Retrain from scratch:**
- Tambah huruf baru (L-Z)
- Ganti arsitektur model
- Dataset berubah signifikan
- **Waktu:** 2-5 menit

**Finetune:**
- Tambah data untuk huruf yang sama
- Minor adjustments
- **Waktu:** 30-60 detik
- **Risk:** Catastrophic forgetting

**Recommendation:** Retrain from scratch untuk sign language recognition.

### Architecture Decisions

**1. Why PyTorch over TensorFlow?**

- Lebih Pythonic
- Dynamic computation graph
- Better debugging experience
- Research community preference

**2. Why 1D CNN over 2D CNN?**

- Landmarks adalah 1D sequence
- 2D CNN untuk image data
- 1D CNN lebih efficient untuk sequential features

**3. Why 84 features (xy) over 126 features (xyz)?**

- z-coordinate dari MediaPipe kurang akurat
- xy sudah sufficient untuk shape recognition
- Lebih sedikit features = model lebih cepat

**4. Why AdaptiveAvgPool1d?**

- Handle variable input sizes
- Fixed output dimension untuk FC layer
- Flexible untuk different feature counts

### Troubleshooting Guide

**Problem: Low accuracy on specific letters**

Check:
1. Confusion matrix → huruf mana yang sering tertukar?
2. Data quality → cukup variasi di training data?
3. Hand position → huruf ini butuh 1 atau 2 tangan?

Solution:
- Recollect dengan pose yang lebih jelas
- Tambah samples (2000+ instead of 1000)
- Check num_hands consistency

**Problem: Model works close but not far**

Check:
1. Hand-centric normalization applied?
2. Training data diverse distances?

Solution:
- Apply hand-centric normalization
- Collect data at various distances

**Problem: C/S confusion**

Check:
1. num_hands detection working?
2. Post-processing logic implemented?

Solution:
- Use num_hands untuk disambiguation
- Tune probability threshold

### Next Steps

**Untuk improvement:**

1. **Collect more diverse data**
   - Different hand sizes
   - Various lighting conditions
   - Multiple contributors

2. **Try ensemble methods**
   - Combine ML + DL predictions
   - Majority voting untuk stability

3. **Add temporal smoothing**
   - Average predictions over last N frames
   - Reduce jitter

4. **Deploy to mobile**
   - TensorFlow Lite conversion
   - On-device inference

5. **Expand to words/sentences**
   - Sequence modeling (LSTM/Transformer)
   - Context-aware recognition

---

## Appendix: Command Reference

### Data Collection
```bash
# Single letter
python3 capture_fast2.py A --count 1000

# Batch
for letter in A B C D E F G H I J K; do
  python3 capture_fast2.py $letter --count 1000
done
```

### Training
```bash
# ML models
python3 train/train_ml.py --model both --epochs 50

# DL models
python3 train/train_dl.py --data dataset/landmarks_2hands.csv --epochs 50 --name cnn_2hand
```

### Evaluation
```bash
# Confusion matrix
python3 eval/confusion.py --model cnn_2hand

# ML vs DL comparison
python3 eval/compare.py
```

### Inference
```bash
# Webcam
python3 inference_webcam.py --model models/dl/cnn_2hand_model.pt
```

### Git Workflow
```bash
# Create feature branch
git checkout -b dl/cnn-2hand-full

# Commit changes
git add .
git commit -m "feat: add hand-centric normalization"

# Push
git push origin dl/cnn-2hand-full

# Merge to main
git checkout main
git merge dl/cnn-2hand-full
git push origin main
```

---

## Credits

**Project:** BISINDO Bridge  
**Team:** 5-person team (AI Lead, Data Lead, Frontend Dev, Automator, Tester)  
**Timeline:** June 2026  
**Tech Stack:** Python, PyTorch, scikit-learn, MediaPipe, OpenCV

**Key Files:**
- `train/train_dl.py` - Deep learning training
- `train/train_ml.py` - Machine learning training
- `inference_webcam.py` - Real-time inference
- `capture_fast2.py` - Data collection
- `eval/confusion.py` - Model evaluation
- `dataset/landmarks_2hands.csv` - Training dataset (26,192 samples)

---

*Guide ini dibuat berdasarkan pengalaman nyata mengembangkan sistem BISINDO recognition. Semua lesson learned berasal dari trial and error di proyek ini.*
