#!/usr/bin/env python3
"""
Evaluate BISINDO model performance in detail
Identify problematic letters and confusion patterns
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load model
print("Loading model...")
with open('models/landmark_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/landmark_classifier_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/landmark_classifier_labels.pkl', 'rb') as f:
    labels_data = pickle.load(f)

if isinstance(labels_data, dict) and 'classes' in labels_data:
    labels = labels_data['classes']
else:
    labels = labels_data

print(f"Model loaded: {len(labels)} classes")
print(f"Classes: {labels}")

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('dataset/landmarks_captured.csv')
print(f"Total samples: {len(df)}")

# Check data distribution
print("\n" + "="*60)
print("DATA DISTRIBUTION")
print("="*60)
letter_counts = df['letter'].value_counts().sort_index()
print(letter_counts)
print(f"\nMean samples per letter: {letter_counts.mean():.1f}")
print(f"Min samples: {letter_counts.min()} ({letter_counts.idxmin()})")
print(f"Max samples: {letter_counts.max()} ({letter_counts.idxmax()})")
print(f"Std dev: {letter_counts.std():.1f}")

# Check hand count distribution
print("\n" + "="*60)
print("HAND COUNT DISTRIBUTION")
print("="*60)
hand_counts = df['num_hands'].value_counts().sort_index()
print(hand_counts)

# Prepare test data (last 20% of each letter)
print("\n" + "="*60)
print("PREPARING TEST DATA")
print("="*60)

# Split: 80% train, 20% test (stratified by letter)
from sklearn.model_selection import train_test_split

features = [col for col in df.columns if col.startswith('lm')]
X = df[features].values
y = df['letter'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
print("\nPredicting on test set...")
y_pred = classifier.predict(X_test_scaled)
y_proba = classifier.predict_proba(X_test_scaled)

# Overall accuracy
accuracy = (y_pred == y_test).sum() / len(y_test)
print(f"\nOverall accuracy: {accuracy:.2%}")

# 1. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

# Print formatted report
for letter in labels:
    if letter in report:
        metrics = report[letter]
        print(f"{letter:2s}: precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}  support={int(metrics['support'])}")

print(f"\nOverall: precision={report['weighted avg']['precision']:.3f}  recall={report['weighted avg']['recall']:.3f}  f1={report['weighted avg']['f1-score']:.3f}")

# 2. Per-letter accuracy
print("\n" + "="*60)
print("PER-LETTER ACCURACY")
print("="*60)
letter_accuracies = {}
for letter in labels:
    mask = y_test == letter
    if mask.sum() > 0:
        acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum()
        letter_accuracies[letter] = acc
        n_samples = mask.sum()
        print(f"{letter:2s}: {acc:6.2%} ({n_samples:3d} samples)")

# Sort by accuracy
print("\n" + "="*60)
print("LETTERS BY ACCURACY (LOWEST TO HIGHEST)")
print("="*60)
sorted_letters = sorted(letter_accuracies.items(), key=lambda x: x[1])
for letter, acc in sorted_letters:
    status = "❌" if acc < 0.70 else ("⚠️" if acc < 0.85 else "✅")
    print(f"{status} {letter:2s}: {acc:6.2%}")

# 3. Confusion Matrix
print("\n" + "="*60)
print("CONFUSION MATRIX (Most Confused Letters)")
print("="*60)
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Find most confused pairs
confusion_pairs = []
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true': true_label,
                'pred': pred_label,
                'count': cm[i, j]
            })

# Sort by confusion count
confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

print("Top 20 most confused pairs:")
for pair in confusion_pairs[:20]:
    print(f"  {pair['true']} → {pair['pred']}: {pair['count']} samples")

# 4. Confidence analysis
print("\n" + "="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)
max_confidences = np.max(y_proba, axis=1)
print(f"Mean confidence: {max_confidences.mean():.3f}")
print(f"Min confidence: {max_confidences.min():.3f}")
print(f"Max confidence: {max_confidences.max():.3f}")
print(f"Std dev: {max_confidences.std():.3f}")

# Confidence by correctness
correct_mask = y_pred == y_test
correct_conf = max_confidences[correct_mask]
incorrect_conf = max_confidences[~correct_mask]

print(f"\nCorrect predictions:")
print(f"  Mean confidence: {correct_conf.mean():.3f}")
print(f"  Count: {len(correct_conf)}")

print(f"\nIncorrect predictions:")
print(f"  Mean confidence: {incorrect_conf.mean():.3f}")
print(f"  Count: {len(incorrect_conf)}")

# 5. Plot confusion matrix
print("\nSaving confusion matrix plot...")
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - BISINDO Letter Recognition', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Letter', fontsize=12, fontweight='bold')
plt.ylabel('Actual Letter', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_evaluation.png', dpi=150)
print("Saved: confusion_matrix_evaluation.png")

# 6. Plot per-letter accuracy
print("Saving per-letter accuracy plot...")
plt.figure(figsize=(14, 6))
letters_sorted = sorted(letter_accuracies.keys())
accuracies = [letter_accuracies[l] for l in letters_sorted]

bars = plt.bar(letters_sorted, accuracies, color='steelblue', alpha=0.7)
plt.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Target (90%)')
plt.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Minimum (70%)')

# Color bars by accuracy
for bar, acc in zip(bars, accuracies):
    if acc < 0.70:
        bar.set_color('red')
    elif acc < 0.85:
        bar.set_color('orange')
    elif acc >= 0.90:
        bar.set_color('green')

plt.xlabel('Letter', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Per-Letter Accuracy', fontsize=16, fontweight='bold')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('per_letter_accuracy.png', dpi=150)
print("Saved: per_letter_accuracy.png")

# 7. Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

problematic_letters = [l for l, acc in letter_accuracies.items() if acc < 0.85]
if problematic_letters:
    print(f"\n❌ Problematic letters (accuracy <85%):")
    for letter in problematic_letters:
        acc = letter_accuracies[letter]
        n_samples = (y_test == letter).sum()
        print(f"  {letter}: {acc:.2%} ({n_samples} test samples)")
        print(f"    → Collect more data for letter {letter}")
else:
    print("✅ All letters have accuracy ≥85%")

# Check data imbalance
print(f"\nData distribution:")
for letter in labels:
    n_samples = letter_counts.get(letter, 0)
    status = "✅" if n_samples >= 1000 else ("⚠️" if n_samples >= 500 else "❌")
    print(f"  {status} {letter}: {n_samples:4d} samples")

print("\nRecommendations:")
print("1. If accuracy <90% overall, collect more data")
print("2. For letters with <1000 samples, collect at least 500 more")
print("3. For letters with accuracy <85%, collect 1000 more samples")
print("4. If confusion is high between specific letters, collect more distinguishing samples")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)