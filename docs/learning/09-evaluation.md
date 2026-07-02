# Evaluation Metrics

> Penjelasan lengkap cara mengevaluasi model ML: train/test split, accuracy, precision, recall, F1 score, dan confusion matrix.

---

## Daftar Isi

1. [Train/Test Split](#traintest-split)
2. [Accuracy](#accuracy)
3. [Precision, Recall, F1](#precision-recall-f1)
4. [Confusion Matrix](#confusion-matrix)
5. [Classification Report](#classification-report)
6. [Per-Letter Analysis](#per-letter-analysis)
7. [Common Pitfalls](#common-pitfalls)
8. [Evaluation Pipeline](#evaluation-pipeline)

---

## Train/Test Split

### Kenapa Split Data?

```
┌────────────────────────────────────────────────────────────────┐
│                    Train/Test Split Concept                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset: 26,000 samples                                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                         ALL DATA                           │ │
│  │                                                           │ │
│  │  ┌────────────────────┐    ┌────────────────────┐        │ │
│  │  │                    │    │                    │        │ │
│  │  │   TRAINING SET     │    │    TEST SET        │        │ │
│  │  │   (85%)           │    │    (15%)           │        │ │
│  │  │   22,100 samples  │    │    3,900 samples   │        │ │
│  │  │                    │    │                    │        │ │
│  │  │  Used for:         │    │  Used for:          │        │ │
│  │  │  - Learning weights│    │  - Final evaluation │        │ │
│  │  │  - Validation      │    │  - NEVER seen by   │        │ │
│  │  │  - Tuning          │    │    model during    │        │ │
│  │  │                    │    │    training!        │        │ │
│  │  └────────────────────┘    └────────────────────┘        │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Stratified Split

```python
from sklearn.model_selection import train_test_split

# Stratified split: maintains class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,        # 15% untuk test
    random_state=42,       # Reproducibility
    stratify=y             # Maintain class balance
)

print(f"Training: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# Check class distribution
print("\nClass distribution:")
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    train_count = (y_train == letter).sum()
    test_count = (y_test == letter).sum()
    print(f"{letter}: train={train_count}, test={test_count}")
```

### Validation Set

```python
# Split train further: train vs validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,          # 10% of training data
    random_state=42,
    stratify=y_train
)

print(f"Training: {len(X_tr)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Final split:
# Train: 76.5%
# Val: 8.5%
# Test: 15%
```

---

## Accuracy

### Apa itu Accuracy?

```
Accuracy = (Correct predictions) / (Total predictions)

┌────────────────────────────────────────────────────────────────┐
│                        Accuracy Calculation                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Predictions vs Actual:                                          │
│  ┌─────────┬────────┬─────────┐                                │
│  │ Sample  │ Predicted│ Actual │ Correct?                    │
│  ├─────────┼────────┼─────────┤                                │
│  │    1    │    A   │    A   │  ✓                           │
│  │    2    │    B   │    C   │  ✗                           │
│  │    3    │    M   │    M   │  ✓                           │
│  │    4    │    K   │    K   │  ✓                           │
│  │    5    │    S   │    S   │  ✓                           │
│  └─────────┴────────┴─────────┘                                │
│                                                                 │
│  Accuracy = 4/5 = 0.80 = 80%                                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Accuracy di PyTorch

```python
import torch
import torch.nn as nn

# Model prediction
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, dim=1)  # Class dengan prob tertinggi

# Calculate accuracy
correct = (predicted == y_test_tensor).sum().item()
total = y_test_tensor.size(0)
accuracy = correct / total

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
# Output: Accuracy: 0.9750 (97.50%)
```

### Accuracy Limitations

```
┌────────────────────────────────────────────────────────────────┐
│                   Accuracy Limitations                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Accuracy bisa MENIPU kalau dataset imbalanced:                  │
│                                                                 │
│  Example:                                                       │
│  Dataset: 100 samples                                           │
│  - Class A: 95 samples                                         │
│  - Class B: 5 samples                                          │
│                                                                 │
│  Model prediksi SEMUA A:                                        │
│  - Accuracy = 95/100 = 95%!                                    │
│  - Tapi model ga bisa deteksi B sama sekali!                     │
│                                                                 │
│  Solution: Lihat Precision, Recall, F1!                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Precision, Recall, F1

### Confusion Matrix Basics

```
┌────────────────────────────────────────────────────────────────┐
│                    Confusion Matrix 2-Class                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      Predicted                                   │
│                  ┌──────────┬──────────┐                       │
│                  │   NEG    │   POS    │                       │
│  ┌───────┬──────┴──────────┴──────────┴───────┐               │
│  │  NEG  │         │         │              │               │
│  │       │   TN    │    FP   │              │               │
│  ├───────┼─────────┼─────────┼──────────────┤               │
│  │  POS  │   FN    │    TP   │              │               │
│  │       │         │         │              │               │
│  └───────┴─────────┴─────────┴──────────────┘               │
│                                                                 │
│  TN = True Negative  (Prediksi NEG, actual NEG)               │
│  FP = False Positive (Prediksi POS, actual NEG) ← "False alarm"
│  FN = False Negative (Prediksi NEG, actual POS) ← "Missed"   │
│  TP = True Positive  (Prediksi POS, actual POS)               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Precision

```
Precision = TP / (TP + FP) = "Dari semua yang diprediksi POS, berapa yang benar?"

┌────────────────────────────────────────────────────────────────┐
│                         Precision Concept                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Precision tinggi = "Kalau model bilang POS, percaya aja"       │
│                                                                 │
│  Example:                                                      │
│  Model prediksi 100 kali "A"                                   │
│  - 95 kali benar (TP = 95)                                    │
│  - 5 kali salah (FP = 5)                                       │
│                                                                 │
│  Precision = 95 / (95 + 5) = 95%                              │
│                                                                 │
│  Interpretation: "Kalau model bilang A, 95% kemungkinan bener"  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Recall

```
Recall = TP / (TP + FN) = "Dari semua yang ACTUAL POS, berapa yang ketangkep?"

┌────────────────────────────────────────────────────────────────┐
│                          Recall Concept                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Recall tinggi = "Model nemu hampir semua yang POS"             │
│                                                                 │
│  Example:                                                      │
│  Ada 100 sample huruf A di test set                             │
│  Model prediksi 90 kali "A"                                     │
│  - 90 kali benar (TP = 90)                                    │
│  - 10 kali salah/hilang (FN = 10)                              │
│                                                                 │
│  Recall = 90 / (90 + 10) = 90%                                │
│                                                                 │
│  Interpretation: "Model nemu 90% dari semua huruf A"            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

┌────────────────────────────────────────────────────────────────┐
│                          F1 Score Concept                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  F1 = Harmonic mean dari Precision dan Recall                   │
│                                                                 │
│  Kenapa harmonic mean?                                          │
│  - Precision = 100%, Recall = 50% → Arithmetic mean = 75%      │
│  - F1 = 2×(1.0×0.5)/(1.0+0.5) = 2/3 = 66.7% ← Lebih jujur!  │
│                                                                 │
│  Precision dan Recall perlu HIGH together untuk F1 tinggi!       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Precision │ Recall │ F1        │ Interpretation       │  │
│  ├───────────┼────────┼───────────┼──────────────────────┤  │
│  │    0.90   │  0.90  │   0.90   │ Good balance         │  │
│  │    1.00   │  0.50  │   0.67   │ Biased toward POS    │  │
│  │    0.50   │  1.00  │   0.67   │ Biased toward NEG    │  │
│  │    0.80   │  0.80  │   0.80   │ Good                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Precision/Recall untuk BISINDO

```python
from sklearn.metrics import precision_recall_fscore_support

# Calculate per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred,
    average=None,           # Per-class
    labels=classes          # A-Z
)

# Print for letter 'A'
letter_idx = 0  # A
print(f"Letter A:")
print(f"  Precision: {precision[letter_idx]:.4f}")
print(f"  Recall:    {recall[letter_idx]:.4f}")
print(f"  F1:        {f1[letter_idx]:.4f}")
print(f"  Support:   {support[letter_idx]} samples")
```

---

## Confusion Matrix

### Apa itu Confusion Matrix?

```
┌────────────────────────────────────────────────────────────────┐
│                     Confusion Matrix: 26 Classes                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Predicted                                              Actual │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │     │  A  │  B  │  C  │  D  │ ... │  Z  │              │ │
│  │ ────┼─────┼─────┼─────┼─────┼─────┼─────┤              │ │
│  │  A  │ 95  │  1  │  0  │  0  │ ... │  0  │              │ │
│  │  B  │  2  │ 89  │  1  │  0  │ ... │  0  │              │ │
│  │  C  │  0  │  0  │ 92  │  1  │ ... │  0  │              │ │
│  │  D  │  0  │  0  │  2  │ 88  │ ... │  0  │              │ │
│  │ ... │ ... │ ... │ ... │ ... │ ... │ ... │              │ │
│  │  Z  │  0  │  0  │  0  │  0  │ ... │ 91  │              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Diagonal = Correct predictions (makin gelap makin banyak)     │
│  Off-diagonal = Misclassifications                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Confusion Matrix di Python

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: BISINDO CNN')
plt.tight_layout()
plt.show()
```

### Normalized Confusion Matrix

```python
# Normalized by actual class (row normalization)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix (by row)')
plt.tight_layout()
plt.show()
```

### Finding Most Confused Pairs

```python
# Find most confused letter pairs
import numpy as np

# Set diagonal to 0 (we don't count correct predictions)
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)

# Find top confused pairs
confused_pairs = []
for i in range(26):
    for j in range(26):
        if i != j and cm_copy[i, j] > 0:
            confused_pairs.append({
                'actual': chr(65 + i),
                'predicted': chr(65 + j),
                'count': cm_copy[i, j]
            })

# Sort by count
confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)

# Print top 10
print("Top 10 Most Confused Pairs:")
print("-" * 40)
for i, pair in enumerate(confused_pairs[:10], 1):
    print(f"{i:2}. {pair['actual']} → {pair['predicted']}: {pair['count']} samples")
```

---

## Classification Report

### scikit-learn Classification Report

```python
from sklearn.metrics import classification_report

# Generate report
report = classification_report(
    y_test, y_pred,
    target_names=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    digits=4
)

print(report)
```

### Sample Output

```
              precision    recall  f1-score   support

           A     0.9632    0.9853    0.9741       152
           B     0.9383    0.9512    0.9447       148
           C     0.9712    0.9324    0.9514       151
           D     0.9495    0.9672    0.9583       149
           ...
           Z     0.9558    0.9421    0.9489       147

   micro avg     0.9612    0.9612    0.9612      3900
   macro avg     0.9605    0.9598    0.9601      3900
weighted avg     0.9612    0.9612    0.9611      3900
```

### Understanding the Report

```
┌────────────────────────────────────────────────────────────────┐
│                    Classification Report Explained               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Per-Class Metrics:                                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Letter │ Precision │ Recall │ F1-Score │ Support          │ │
│  ├────────┼───────────┼────────┼───────────┼─────────────────┤ │
│  │   A    │  0.96    │  0.99  │   0.97   │ 152 (samples)   │ │
│  │   B    │  0.94    │  0.95  │   0.94   │ 148             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  - Precision: "Dari yang diprediksi A, 96% bener"             │
│  - Recall: "Dari semua A, model nemu 99%"                     │
│  - F1: Harmonic mean, balance metric                          │
│  - Support: Jumlah sample di test set                          │
│                                                                 │
│  Summary Metrics:                                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ micro avg: Treat each sample equally                     │ │
│  │            TP/(TP+FP+FN)                               │ │
│  │                                                            │ │
│  │ macro avg: Average of per-class metrics                  │ │
│  │            mean(P_recall for all classes)               │ │
│  │                                                            │ │
│  │ weighted avg: Macro avg weighted by support             │ │
│  │               Accounts for class imbalance                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Per-Letter Analysis

### Best and Worst Performing Letters

```python
import pandas as pd

# Calculate per-letter metrics
metrics = []
for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    # TP, FP, FN dari confusion matrix
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.append({
        'Letter': letter,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Correct': tp,
        'Total': int(cm[i, :].sum())
    })

df = pd.DataFrame(metrics)
df = df.sort_values('F1')

# Best 5 letters
print("\n🏆 TOP 5 Letters:")
print(df.tail(5)[['Letter', 'F1', 'Precision', 'Recall']].to_string(index=False))

# Worst 5 letters
print("\n⚠️ BOTTOM 5 Letters (Need Improvement):")
print(df.head(5)[['Letter', 'F1', 'Precision', 'Recall']].to_string(index=False))
```

### Why Are Some Letters Hard?

```
┌────────────────────────────────────────────────────────────────┐
│                   Common BISINDO Confusion Pairs                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  M ↔ N:                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  M: Semua jari ditekuk, ibu jari di samping              │  │
│  │  N: Semua jari ditekuk, ibu jari di depan              │  │
│  │  Perbedaan: posisi ibu jari aja!                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  B ↔ E:                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  B: Jari telunjuk lurus, lain ditekuk                   │  │
│  │  E: Semua jari ditekuk ke telapak                        │  │
│  │  Perbedaan: telunjuk lurus vs ditekuk                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  K ↔ P:                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  K: Telunjuk + tengah lurus, ibu jari di antara         │  │
│  │  P: Telunjuk + middle +戒指 + pinky ditekuk, ibu jari di│  │
│  │     antara index & middle, arm up                      │  │
│  │  Perbedaan: ring & pinky fingers, arm position          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Solution: Collect more diverse samples for confused letters!    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Common Pitfalls

### Pitfall 1: Data Leakage

```
┌────────────────────────────────────────────────────────────────┐
│                      Data Leakage Warning                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BAD PRACTICE:                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  # Include preprocessing dalam training!                 │  │
│  │  scaler.fit(X_all)  # WRONG! Menggunakan test data!     │  │
│  │  X_train = scaler.transform(X_train)                    │  │
│  │  X_test = scaler.transform(X_test)                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  GOOD PRACTICE:                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  scaler.fit(X_train)   # Fit HANYA pada training data  │  │
│  │  X_train = scaler.transform(X_train)                    │  │
│  │  X_test = scaler.transform(X_test)  # Transform, NOT fit │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Same applies untuk:                                             │
│  - LabelEncoder                                                 │
│  - Any feature selection                                        │
│  - Any statistics-based preprocessing                           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Pitfall 2: Not Using Stratified Split

```python
# BAD: Random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
# Risk: Some letters might not appear in train OR test!

# GOOD: Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y
)
# Guarantee: Each letter appears proportionally in both sets
```

### Pitfall 3: Trusting Validation Accuracy Too Much

```
┌────────────────────────────────────────────────────────────────┐
│          Validation Accuracy ≠ Real-World Performance             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Our experience in BISINDO project:                             │
│                                                                 │
│  Validation accuracy: 99.1%                                    │
│  Webcam test (far hand): 20%!!!                                │
│                                                                 │
│  Why?                                                           │
│  - Validation data: collected by same person, same conditions  │
│  - Webcam test: different person, different distances         │
│                                                                 │
│  Solution:                                                      │
│  ✓ Use hand-centric normalization                                │
│  ✓ Collect diverse training data (various distances, persons)  │
│  ✓ Test on real webcam before declaring "done"                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Pipeline

### Complete Evaluation Script

```python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

def evaluate_model(model, X_test, y_test, classes):
    """Complete model evaluation."""
    
    # Set model to evaluation mode
    model.eval()
    
    # Predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, dim=1)
        y_pred = y_pred.numpy()
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=classes
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Results
    results = {
        'accuracy': accuracy,
        'per_class': {
            letter: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            for i, letter in enumerate(classes)
        },
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    
    return results

def print_evaluation_report(results, classes):
    """Print formatted evaluation report."""
    
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # Overall accuracy
    print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        results['y_pred'],  # PyTorch predictions
        results['y_pred'], # Already predicted
        target_names=classes
    ))
    
    # Per-class summary
    print("\nPer-Class Summary (sorted by F1):")
    sorted_classes = sorted(
        results['per_class'].items(),
        key=lambda x: x[1]['f1']
    )
    
    print("-" * 50)
    print(f"{'Letter':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    
    for letter, metrics in sorted_classes:
        print(f"{letter:<8} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Worst performers
    print("\n⚠️ WORST PERFORMING LETTERS (need more data/attention):")
    print("-" * 50)
    for letter, metrics in sorted_classes[:5]:
        print(f"  {letter}: F1={metrics['f1']:.4f} "
              f"(P={metrics['precision']:.2f}, R={metrics['recall']:.2f})")

# Usage
results = evaluate_model(model, X_test, y_test, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
print_evaluation_report(results, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Summary                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Train/Test Split                                            │
│     - 85% train, 15% test                                       │
│     - Stratified split: maintain class balance                  │
│     - Validation set: for hyperparameter tuning                  │
│                                                                  │
│  2. Accuracy                                                    │
│     - (Correct) / (Total)                                       │
│     - Good for balanced datasets                                │
│     - Can mislead for imbalanced data                           │
│                                                                  │
│  3. Precision, Recall, F1                                        │
│     - Precision: TP/(TP+FP) = "how accurate are positive preds"  │
│     - Recall: TP/(TP+FN) = "how many positives did we find"    │
│     - F1: Harmonic mean of precision and recall                 │
│                                                                  │
│  4. Confusion Matrix                                            │
│     - Visualize all predictions vs actuals                       │
│     - Find confused letter pairs                                 │
│     - Identify letters needing improvement                       │
│                                                                  │
│  5. Common Pitfalls                                             │
│     - Data leakage (fit on test data!)                          │
│     - Non-stratified split                                      │
│     - Trusting validation accuracy too much                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [10-exercises.md](10-exercises.md) - Hands-on Exercises

---

## Referensi

- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Classification metrics: https://scikit-learn.org/stable/modules/classes.html#classification-metrics
- Confusion matrix visualization: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
