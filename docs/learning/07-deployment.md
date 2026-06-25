# Chapter 7: Deployment — Model Jadi Aplikasi

> **Tujuan chapter ini**: Kamu paham cara save model, load model, dan pakai untuk prediksi real-time di server.

---

## 💾 Save Model

Setelah training, kita perlu **save** supaya bisa dipakai lagi tanpa retrain.

### Kenapa Save 3 File?

```
models/landmark_classifier.pkl        → Random Forest model
models/landmark_classifier_scaler.pkl → StandardScaler (normalisasi)
models/landmark_classifier_labels.pkl → Mapping label → huruf
```

### Kenapa 3 File, Bukan 1?

Bayangkan kamu mau prediksi gesture baru:

1. **Model** butuh input yang sudah di-scale → perlu **scaler**
2. **Scaler** harus sama persis dengan yang dipakai saat training
3. **Labels** mapping hasil prediksi (index 0, 1, 2) ke huruf (A, B, C)

Kalau scaler beda, prediksi jadi salah.

### Kode Save

```python
import pickle

# Save
with open('models/landmark_classifier.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('models/landmark_classifier_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/landmark_classifier_labels.pkl', 'wb') as f:
    pickle.dump({'classes': sorted(rf.classes_), 'n_classes': len(rf.classes_)}, f)

print("Model saved!")
```

---

## 📂 Struktur Model

Setelah save:

```
models/
├── landmark_classifier.pkl          # Random Forest (bisa 50-200 MB)
├── landmark_classifier_scaler.pkl   # StandardScaler (kecil)
└── landmark_classifier_labels.pkl     # Dictionary labels (kecil)
```

---

## 🚀 Load Model dan Prediksi

```python
import pickle
import numpy as np

# Load
with open('models/landmark_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/landmark_classifier_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/landmark_classifier_labels.pkl', 'rb') as f:
    labels = pickle.load(f)['classes']

# Prediksi satu sample (63 angka dari MediaPipe)
landmarks = [...]  # 63 floats

# Step 1: Pad ke 126 (hand1 + zero hand2)
features = landmarks + [0.0] * 63  # 126 floats

# Step 2: Scale
features_scaled = scaler.transform([features])

# Step 3: Predict
prediction_index = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

letter = labels[prediction_index]
confidence = probabilities[prediction_index]

print(f"Predicted: {letter}, Confidence: {confidence:.2f}")
```

---

## 🔄 Inference Pipeline End-to-End

```
Webcam Frame
    ↓
MediaPipe HandLandmarker
    ↓
21 Landmarks (x,y,z) → 63 angka
    ↓
Pad to 126 features (hand1 + zero hand2)
    ↓
Scale with StandardScaler
    ↓
Random Forest predict
    ↓
Letter + Confidence
    ↓
Display on web
```

**Latency total**: ~5-10ms (MediaPipe) + ~5ms (RF inference) = **~15ms**. Cukup cepat untuk realtime.

---

## 🛠️ Integrasi ke Flask Server

Di `meeting/app.py`, ada kode ini:

```python
from src.landmark_classifier import LandmarkClassifier

paths = {
    'model': os.path.join(MODEL_DIR, 'landmark_classifier.pkl'),
    'scaler': os.path.join(MODEL_DIR, 'landmark_classifier_scaler.pkl'),
    'labels': os.path.join(MODEL_DIR, 'landmark_classifier_labels.pkl'),
}

classifier = LandmarkClassifier(paths['model'], paths['scaler'], paths['labels'])
```

Class ini mempermudah load dan predict.

### Prediksi via SocketIO

```python
@socketio.on('predict_landmarks')
def on_predict_landmarks(data):
    if not classifier:
        return

    landmarks = data.get('landmarks', [])
    if len(landmarks) != 63:
        return

    # 63 → 126
    feats = list(landmarks) + [0.0] * 63

    # Predict
    prediction = classifier.predict(feats)

    if prediction['confidence'] > 0.5:
        emit('prediction', {
            'letter': prediction['letter'],
            'confidence': prediction['confidence'],
        })
```

---

## 🔒 Confidence Threshold

**Kenapa perlu threshold?**

Kalau confidence rendah, mending model bilang "ga yakin" daripada ngawur.

```python
THRESHOLD = 0.6

if confidence > THRESHOLD:
    display(letter)
else:
    display("?")  # or ignore
```

### Memilih Threshold

| Threshold | Recall | Precision |
|-----------|--------|-----------|
| 0.5 | Tinggi | Sedikit lebih banyak false positive |
| 0.7 | Sedang | Lebih sedikit false positive |
| 0.9 | Rendah | Hampir semua prediksi benar |

**Trade-off**: Threshold tinggi → model lebih jarang prediksi, tapi lebih akurat ketika prediksi.

---

## 📈 Model Monitoring

Setelah deploy, kita perlu monitor:

1. **Confidence distribution**: Prediksi mayoritas confidence > 0.7?
2. **Error patterns**: Huruf mana yang sering salah?
3. **Latency**: Inference < 50ms?

```python
# Log setiap prediksi untuk analisis nanti
import json

log_entry = {
    'timestamp': datetime.now().isoformat(),
    'landmarks': landmarks,
    'prediction': letter,
    'confidence': confidence
}

with open('prediction_log.jsonl', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

---

## 🧠 Latihan Pemahaman

1. Kenapa perlu 3 file untuk model? Apa fungsi masing-masing?
2. Kenapa input harus di-scale dengan scaler yang sama dari training?
3. Apa fungsi confidence threshold?
4. Kalau threshold terlalu tinggi, apa efeknya?

<details>
<summary>📝 Jawaban</summary>

1. Model = Random Forest. Scaler = normalisasi (mean/std training). Labels = mapping index → huruf.
2. Kalau scaler beda, feature input tidak dalam skala yang sama dengan training. Model belajar pattern dari feature scaled tertentu, jadi input harus diskala sama persis.
3. Memutuskan apakah prediksi cukup yakin untuk ditampilkan. Confidence rendah → lebih baik ignore daripada salah.
4. Model jarang prediksi, tapi ketika prediksi biasanya benar. Bisa membuat UI "lambat" karena sering ga yakin.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Save model → 3 file: model, scaler, labels
- [ ] Load model → predict input baru
- [ ] Pipeline inference: landmarks → pad → scale → predict
- [ ] Confidence threshold untuk filter prediksi tidak yakin
- [ ] Integrasi ke Flask server via SocketIO
- [ ] Monitoring model setelah deploy

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 8: Neural Network vs Random Forest](08-neural-network.md)** — kita bandingkan RF dengan deep learning dan diskusi kapan upgrade.
