# Chapter 8: Neural Network vs Random Forest — Kapan Upgrade?

> **Tujuan chapter ini**: Kamu paham Neural Network sebagai alternatif, kapan worth it untuk upgrade, dan trade-off yang perlu dipertimbangkan.

---

## 🤔 Kenapa Pertimbangkan Neural Network?

Random Forest udah bagus (accuracy 85-90%, cepat, ringan). Tapi ada batas:

- Sulit tangkap pola **non-linear kompleks**
- Tidak belajar **representasi** data (hanya split berdasarkan feature yang dikasih)
- Kalau feature engineering jelek, model ikut jelek

Neural Network bisa:
- Belajar **feature otomatis** dari data mentah
- Tangkap pola non-linear kompleks
- Bisa pakai data mentah (tanpa feature engineering banyak)

---

## 🧠 Apa Itu Neural Network?

**Neural Network** = jaringan neuron buatan yang terinspirasi otak.

### Arsitektur Sederhana

```
Input (126 features)
      ↓
  [Layer 1: 256 neurons] → ReLU
      ↓
  [Layer 2: 128 neurons] → ReLU
      ↓
  [Layer 3: 26 neurons] → Softmax
      ↓
Output (26 kelas, probability tiap huruf)
```

**Cara kerja**:
1. Input 126 angka masuk ke layer pertama
2. Tiap neuron hitung: sum(weighted input) → aktivasi (ReLU)
3. Pass ke layer berikutnya
4. Layer terakhir: 26 output = probability tiap huruf A-Z
5. Pilih huruf dengan probability tertinggi

### Contoh Layer

**Hidden Layer 1** (256 neuron):
```
neuron_1 = ReLU(w1·lm0_x + w2·lm0_y + w3·lm0_z + ... + b1)
neuron_2 = ReLU(w1·lm0_x + w2·lm0_y + ... + b2)
...
neuron_256 = ReLU(...)
```

Tiap neuron punya weight sendiri. **Training** = cari weight terbaik.

---

## 🚀 Train MLP untuk BISINDO

MLP = Multi-Layer Perceptron, tipe neural network paling dasar.

```python
from tensorflow import keras

# Convert labels ke one-hot
from tensorflow.keras.utils import to_categorical
letter_classes = sorted(df['letter'].unique())
y_train_oh = to_categorical(
    [letter_classes.index(l) for l in y_train],
    num_classes=len(letter_classes)
)
y_test_oh = to_categorical(
    [letter_classes.index(l) for l in y_test],
    num_classes=len(letter_classes)
)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(126,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(letter_classes), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train_s, y_train_oh,
    validation_data=(X_test_s, y_test_oh),
    epochs=40,
    batch_size=256
)

test_loss, test_acc = model.evaluate(X_test_s, y_test_oh)
print(f"MLP test accuracy: {test_acc:.4f}")
```

**Hasil tipikal**:
- Accuracy: 88-92% (sedikit lebih baik dari RF)
- Training time: 5-15 menit (CPU) atau 1-3 menit (GPU)

---

## 📊 Head-to-Head: Random Forest vs Neural Network

| Aspek | Random Forest | Neural Network |
|-------|---------------|----------------|
| **Training time** (CPU) | 2-5 menit | 5-15 menit |
| **Inference speed** | ~5ms | ~10-50ms |
| **Data dibutuhkan** | 50-100 per huruf | 500+ per huruf |
| **Feature engineering** | Perlu (kualitas tergantung feature) | Minimal (bisa raw data) |
| **Interpretability** | Tinggi (feature importance) | Rendah (black box) |
| **Accuracy** | 85-90% | 88-95% |
| **Overfit risk** | Rendah | Sedang-tinggi (butuh regularization) |
| **Deployment size** | 5-50MB | 10-100MB |
| **Dependencies** | scikit-learn | TensorFlow/PyTorch + dependencies |
| **VPS Rp 30rb** | ✅ Jalan mulus | ⚠️ Bisa tapi lambat training |

---

## 🎯 Kapan Upgrade ke Neural Network?

### Stay dengan Random Forest kalau:
- ✅ Dataset < 5000 sample per huruf
- ✅ Butuh training cepat (< 5 menit)
- ✅ Butuh interpretability (tahu feature mana penting)
- ✅ Infrastruktur terbatas (VPS kecil, CPU only)
- ✅ Accuracy 85%+ udah cukup

### Upgrade ke Neural Network kalau:
- ✅ Dataset > 5000 sample per huruf
- ✅ Butuh accuracy > 90%
- ✅ Ada GPU (training lebih cepat)
- ✅ Feature engineering mentok
- ✅ Butuh model yang lebih "learnable"

### Case BISINDO Sekarang:
- Data: ~1000 sample/huruf rata-rata
- Infrastruktur: VPS 3GB RAM (CPU only)
- Accuracy target: > 85%

**Kesimpulan**: **Random Forest cukup** untuk sekarang.

---

## 🔬 Eksperimen: Bandingkan RF vs MLP

Kode perbandingan:

```python
# Train both
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_s, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_s))

model = keras.Sequential([...])  # seperti di atas
model.fit(X_train_s, y_train_oh, epochs=40, batch_size=256, verbose=0)
nn_acc = model.evaluate(X_test_s, y_test_oh, verbose=0)[1]

print(f"Random Forest: {rf_acc:.4f}")
print(f"Neural Network: {nn_acc:.4f}")
print(f"Difference: {(nn_acc - rf_acc):+.4f}")
```

**Hasil yang diharapkan** dengan data sekarang:
```
Random Forest: 0.8850
Neural Network: 0.8950
Difference: +0.0100
```

**Interpretasi**: NN cuma naik 1%. Belum worth it karena:
- Training lebih lama
- Lebih banyak dependency
- Lebih sulit interpret
- Data belum cukup

---

## 🌐 Export ke Browser (TF.js)

Kalau kita tetap mau pakai Neural Network untuk **inference di browser**, TensorFlow.js memungkinkan:

```python
import tensorflowjs as tfjs

# Save Keras model
model.save('models/mlp_model.h5')

# Convert to TF.js format
tfjs.converters.save_keras_model(model, 'web/models/')
```

Hasilnya:
- `web/models/model.json` (metadata)
- `web/models/group1-shard*.bin` (weights)
- Bisa load dan predict di browser pakai TensorFlow.js

**Keuntungan**:
- Inference di client → tidak perlu server
- Privacy (data ga ke server)
- Low latency

**Tapi**: Tetap butuh training di backend (Python) karena TF.js training lambat.

---

## 💡 Rekomendasi untuk BISINDO

### Fase Sekarang (2026 Q2)
- ✅ Pakai Random Forest
- ✅ Fokus collect data sampai 5000+ per huruf
- ✅ Iterasi: augmentasi, tuning, evaluasi

### Fase Next (2026 Q3-4)
- 🔄 Kalau dataset udah 10k+ per huruf, coba train MLP
- 🔄 Bandingkan accuracy RF vs MLP
- 🔄 Kalau MLP lebih bagus > 3%, upgrade

### Fase Future (2027+)
- 🔮 Pertimbangkan **Vision Transformer (ViT)** untuk raw image
- 🔮 Pertimbangkan **3D hand model** untuk gesture yang bergerak (bukan statis)
- 🔮 Pertimbangkan **fine-tuning** dari model pretrained

---

## 🧠 Latihan Pemahaman

1. Apa kelebihan Neural Network dibanding Random Forest?
2. Apa kekurangan Neural Network?
3. Kapan upgrade ke Neural Network worth it?
4. Kenapa TF.js berguna untuk BISINDO?
5. Dengan data sekarang (~1000/huruf), model mana yang kamu pilih?

<details>
<summary>📝 Jawaban</summary>

1. Bisa belajar feature otomatis dari data mentah, tangkap pola non-linear kompleks, accuracy bisa lebih tinggi dengan data banyak.
2. Training lebih lama, butuh data banyak, sulit interpret (black box), butuh lebih banyak resources (GPU ideal).
3. Kalau dataset > 5000/huruf, butuh accuracy > 90%, punya GPU, feature engineering udah mentok, dan butuh accuracy yang lebih tinggi.
4. Bisa inference di browser tanpa server, privacy terjaga, low latency.
5. Random Forest. Data belum cukup untuk Neural Network, training cepat, accuracy 85%+ udah cukup untuk fase sekarang.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Neural Network = jaringan neuron buatan
- [ ] MLP = neural network paling dasar (dense layers)
- [ ] RF vs NN: RF lebih ringan, NN lebih akurat (dengan data banyak)
- [ ] Kapan upgrade: data cukup + butuh accuracy lebih
- [ ] TF.js untuk inference di browser
- [ ] Untuk BISINDO sekarang: Random Forest cukup

---

## 🎓 Akhir Perjalanan

Selamat! Kamu udah sampai di akhir learning path. Sekarang kamu paham:

1. ✅ Kenapa pakai landmark (Chapter 0)
2. ✅ Struktur data (Chapter 1)
3. ✅ Feature engineering (Chapter 2)
4. ✅ Decision Tree dasar (Chapter 3)
5. ✅ Random Forest (Chapter 4)
6. ✅ Evaluation metrics (Chapter 5)
7. ✅ Iteration & augmentation (Chapter 6)
8. ✅ Deployment real-time (Chapter 7)
9. ✅ Neural Network sebagai alternatif (Chapter 8)

**Next steps**:
- 🎯 Collect data sampai 5000+ per huruf
- 🎯 Train model pertama pakai RF
- 🎯 Evaluate, identify weak letters
- 🎯 Iterate: collect, augment, retrain
- 🎯 Deploy, monitor, improve

**Semoga sukses dengan BISINDO!** 🤟

---

## ⏭️ Kembali ke Index

[Balik ke Index](INDEX.md)
