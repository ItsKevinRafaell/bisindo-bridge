# Chapter 5: Evaluation — Gimana Cara Tau Model Bagus?

> **Tujuan chapter ini**: Kamu paham metrics evaluasi (accuracy, precision, recall, F1, confusion matrix) dan cara baca hasil untuk improve model.

---

## 🎯 Kenapa Evaluation Penting?

Setelah train model, kita perlu tahu:
- Seberapa bagus modelnya?
- Huruf mana yang bagus diprediksi?
- Huruf mana yang sering salah?
- Apa yang perlu diperbaiki?

**Accuracy 95%** kedengeran bagus, tapi bisa menipu. Kenapa? Mari kita lihat.

---

## 📊 Metrics Dasar

### 1. Accuracy

```
Accuracy = Jumlah prediksi benar / Total prediksi
```

**Contoh**:
- 1000 sample test
- 850 prediksi benar
- Accuracy = 850/1000 = 85%

**Kapan misleading?** Kalau dataset **imbalanced** (kelas tidak seimbang).

**Contoh misleading**:
- Dataset: 950 A, 50 B
- Model malas: prediksi semua A
- Accuracy = 950/1000 = 95% (terdengar bagus!)
- Tapi: prediksi B semua salah → useless

**Solusi**: Pakai precision, recall, F1, confusion matrix.

---

### 2. Confusion Matrix

**Tabel yang menunjukkan prediksi vs actual**.

```
              Predicted
              A    B    C    D    ...
Actual  A  |  80   10   5    3   ...
        B  |  5    75   15   5   ...
        C  |  10   8    80   2   ...
        D  |  2    5    3    90  ...
        ...
```

**Cara baca**:
- Diagonal (80, 75, 80, 90) = prediksi benar
- Off-diagonal = prediksi salah

**Contoh**:
- Baris B: 75 benar, 25 salah (5 dikira A, 15 dikira C, 5 dikira D)
- Kolom C: 100 diprediksi C, tapi 20 salah (5 actual A, 15 actual B)

**Insight dari confusion matrix**:
- B sering dikira C (15 kasus) → B dan C mirip? Perlu lebih banyak data?
- A sering dikira A (80/98) → A bagus
- D jarang dikira salah (90/100) → D bagus

---

### 3. Precision

**Dari semua yang diprediksi X, berapa yang beneran X?**

```
Precision = True Positives / (True Positives + False Positives)
```

**Contoh untuk huruf B**:
- Model prediksi B untuk 90 sample
- 75 beneran B (True Positive)
- 15 bukan B (False Positive)
- Precision = 75 / (75 + 15) = 75/90 = 83%

**Interpretasi**: Kalau model bilang "ini B", 83% kemungkinan benar.

---

### 4. Recall

**Dari semua yang beneran X, berapa yang terdeteksi?**

```
Recall = True Positives / (True Positives + False Negatives)
```

**Contoh untuk huruf B**:
- Actual B: 100 sample
- 75 terdeteksi sebagai B (True Positive)
- 25 tidak terdeteksi (False Negative)
- Recall = 75 / (75 + 25) = 75/100 = 75%

**Interpretasi**: Dari semua B asli, model hanya deteksi 75%.

---

### 5. F1 Score

**Harmonic mean dari precision dan recall**.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Kenapa harmonic mean?**
- Arithmetic mean: (0.83 + 0.75) / 2 = 0.79
- Harmonic mean: 2 × (0.83 × 0.75) / (0.83 + 0.75) = 0.79

Tapi kalau salah satu rendah, harmonic mean lebih rendah.

**Contoh**:
- Precision = 0.9, Recall = 0.1
- Arithmetic mean = 0.5
- Harmonic mean = 0.18 (lebih rendah, menunjukkan masalah)

**F1 bagus kalau precision dan recall dua-duanya bagus**.

---

## 🚀 Generate Classification Report

```python
from sklearn.metrics import classification_report

# Predict
y_pred = rf.predict(X_test_s)

# Generate report
report = classification_report(y_test, y_pred, target_names=df['letter'].unique())
print(report)
```

**Output**:
```
              precision    recall  f1-score   support

           A       0.95      0.96      0.95      1000
           B       0.83      0.75      0.79       200
           C       0.88      0.85      0.86       300
           D       0.90      0.92      0.91       400
         ...

    accuracy                           0.91      2600
   macro avg       0.89      0.87      0.88      2600
weighted avg       0.91      0.91      0.91      2600
```

**Cara baca**:
- **A**: Precision 95%, Recall 96%, F1 95% → bagus
- **B**: Precision 83%, Recall 75%, F1 79% → perlu improvement
- **accuracy**: overall accuracy (91%)
- **macro avg**: rata-rata sederhana (anggap semua kelas sama penting)
- **weighted avg**: rata-rata tertimbang (kelas besar lebih penting)

---

## 📈 Visualize Confusion Matrix

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**Apa yang dicari**:
- Diagonal gelap = bagus (prediksi benar banyak)
- Off-diagonal terang = problem (banyak prediksi salah)
- Pattern: huruf mana sering dikira huruf lain?

---

## 🔍 Identifikasi Masalah dari Evaluation

### Masalah 1: Huruf Tertentu Recall Rendah

**Contoh**: B recall = 75%

**Artinya**: Dari 100 sample B asli, 25 tidak terdeteksi sebagai B.

**Solusi**:
- Collect lebih banyak data B
- Cek apakah B sering dikira huruf lain (lihat confusion matrix)
- Mungkin B mirip dengan huruf lain (C, D)

---

### Masalah 2: Huruf Tertentu Precision Rendah

**Contoh**: C precision = 70%

**Artinya**: Model prediksi C untuk 100 sample, tapi 30 bukan C.

**Solusi**:
- Cek huruf apa yang sering dikira C (lihat confusion matrix)
- Mungkin huruf itu mirip C
- Collect lebih banyak data untuk huruf itu

---

### Masalah 3: Kelas Imbalanced

**Contoh**:
- A: 1000 sample
- Z: 100 sample

**Efek**: Model bias ke A, Z performance jelek.

**Solusi**:
- Collect lebih banyak Z
- Atau: pakai class_weight di model
  ```python
  rf = RandomForestClassifier(class_weight='balanced')
  ```
- Atau: data augmentation (Chapter 6)

---

## 🧮 Latihan: Baca Classification Report

Diberikan report:

```
              precision    recall  f1-score   support

           A       0.95      0.96      0.95      1000
           B       0.70      0.60      0.65       200
           C       0.85      0.80      0.82       300
           D       0.92      0.94      0.93       500

    accuracy                           0.90      2000
```

**Pertanyaan**:
1. Huruf mana yang performancenya paling bagus?
2. Huruf mana yang paling bermasalah?
3. Kalau model prediksi B, berapa kemungkinan benar?
4. Dari 200 sample B asli, berapa yang terdeteksi sebagai B?

<details>
<summary>📝 Jawaban</summary>

1. **D** (F1 = 0.93, precision 0.92, recall 0.94)
2. **B** (F1 = 0.65, precision 0.70, recall 0.60)
3. 70% (precision B = 0.70)
4. 120 (recall B = 0.60, jadi 0.60 × 200 = 120)

</details>

---

## 💡 Practical Tips

### 1. Target Performance

Untuk BISINDO:
- **Accuracy overall**: target ≥85%
- **F1 per huruf**: target ≥80%
- Kalau <70%: perlu improvement (more data, better features)

### 2. Iterate Based on Evaluation

```
Train → Evaluate → Identify weak letters → Collect more data → Retrain → Evaluate lagi
```

### 3. Save Evaluation Results

```python
import json

# Save classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
with open('models/evaluation_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)

# Save confusion matrix
np.save('models/confusion_matrix.npy', cm)
```

---

## 🧠 Latihan Pemahaman

1. Kenapa accuracy 95% bisa misleading?
2. Apa beda precision dan recall?
3. Kapan pakai F1 score?
4. Apa yang dicari di confusion matrix?
5. Kalau recall huruf X = 60%, apa artinya?

<details>
<summary>📝 Jawaban</summary>

1. Karena dataset bisa imbalanced. Model malas yang prediksi kelas mayoritas bisa dapat accuracy tinggi tapi useless.
2. Precision: dari prediksi X, berapa yang benar? Recall: dari actual X, berapa yang terdeteksi?
3. F1 = harmonic mean precision dan recall. Pakai saat kita butuh balance antara keduanya.
4. Diagonal (prediksi benar) dan off-diagonal (prediksi salah, huruf mana sering dikira apa).
5. Dari 100 sample X asli, hanya 60 yang terdeteksi sebagai X. 40 tidak terdeteksi.

</details>

---

## ✅ Yang Harus Kamu Paham Setelah Chapter Ini

- [ ] Accuracy bisa misleading kalau imbalanced
- [ ] Precision = dari prediksi X, berapa yang benar
- [ ] Recall = dari actual X, berapa yang terdeteksi
- [ ] F1 = harmonic mean precision dan recall
- [ ] Confusion matrix = tabel prediksi vs actual
- [ ] Cara identify weak letters dari evaluation
- [ ] Iteration cycle: train → evaluate → improve → retrain

---

## ⏭️ Selanjutnya

Lanjut ke **[Chapter 6: Iteration](06-iteration.md)** — kita improve model dengan data augmentation dan retraining.
