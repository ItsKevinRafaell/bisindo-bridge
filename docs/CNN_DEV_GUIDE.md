# CNN Development Guide - BISINDO Bridge

## Table of Contents
1. [Data Format](#data-format)
2. [Data Collection](#data-collection)
3. [Training Pipeline](#training-pipeline)
4. [Inference Pipeline](#inference-pipeline)
5. [Key Lessons](#key-lessons)
6. [Troubleshooting](#troubleshooting)

---

## Data Format

### Format Evolution

| Version | Features | Format | Status |
|---------|----------|--------|--------|
| Original | 63 | 1-hand xyz | Deprecated |
| 2-hand xy | 84 | 2-hand xy | **Current** |
| 2-hand xyz | 126 | 2-hand xyz | Available |

### Current Format: 84 Features (2-hand xy)

```
Columns: letter, path, split, num_hands, contributor,
         lm0_x, lm0_y, ..., lm20_x, lm20_y,        # Hand 1 (42 values)
         h2_lm0_x, h2_lm0_y, ..., h2_lm20_x, h2_lm20_y  # Hand 2 (42 values)
```

**Hand 2 = zeros** jika hanya 1 tangan terdeteksi.

### Auto-Detection in `train_dl.py`

```python
# Checks in order:
1. 2-hand xyz (h1_lm* + h2_lm*) → 126 features
2. 2-hand xy (lm* + h2_lm*) → 84 features
3. 1-hand xyz (lm*) → 63 features
```

---

## Data Collection

### Capture Scripts

| Script | Features | Hands | Output |
|--------|----------|-------|--------|
| `capture_fast.py` | 63 (xyz) | 1 | `landmarks_xy.csv` |
| `capture_fast2.py` | 84 (xy) | 2 | `landmarks_2hands.csv` |

### Recommended: `capture_fast2.py`

```bash
# Single letter
python3 capture_fast2.py A --count 1000

# Batch (Q-Z example)
for letter in Q R S T U V W X Y Z; do
  python3 capture_fast2.py $letter --count 1000
done
```

### Data Schema

```python
CSV_HEADER = ["letter", "path", "split", "num_hands", "contributor"]
CSV_HEADER += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y")]
CSV_HEADER += [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
```

### Merging Data from Multiple Sources

When combining data from different capture scripts:

```python
import pandas as pd

# Load main dataset (2-hand format)
df_main = pd.read_csv('dataset/landmarks_2hands.csv')

# Load 1-hand data
df_1hand = pd.read_csv('dataset/landmarks_xy.csv')

# Pad with zeros for hand 2
h2_cols = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
for col in h2_cols:
    df_1hand[col] = 0.0

# Rename columns to match
df_1hand = df_1hand.rename(columns={"image_path": "path"})

# Reorder and merge
cols = df_main.columns.tolist()
df_1hand = df_1hand[cols]
merged = pd.concat([df_main, df_1hand], ignore_index=True)
```

---

## Training Pipeline

### Basic Training

```bash
python3 train/train_dl.py --data dataset/landmarks_2hands.csv --epochs 50 --name cnn_2hand
```

### What Happens During Training

1. **Load data** → auto-detect format (63/84/126 features)
2. **Hand-centric normalization** → distance-invariant features
3. **StandardScaler** → normalize distribution
4. **Train/test split** → 85/15 with stratification
5. **Train CNN** → 1D convolutional network
6. **Save artifacts** → model, scaler, labels, metrics

### Hand-Centric Normalization

Makes model **distance-invariant** by:

```python
def normalize_hand(hand_coords):
    wrist = hand_coords[0]  # Landmark 0 = wrist
    centered = hand_coords - wrist  # Translate to origin

    # Scale by hand size (max distance from wrist)
    distances = np.linalg.norm(centered, axis=1)
    max_dist = distances.max()
    if max_dist > 0:
        centered = centered / max_dist

    return centered
```

**Why this matters:**
- Hand close to camera → large coordinates
- Hand far from camera → small coordinates
- After normalization → same pattern regardless of distance

### Model Architecture

```python
class CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64*8, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
```

**Params:** ~116K for 84 features, 26 classes

---

## Inference Pipeline

### Webcam Inference

```bash
python3 inference_webcam.py
```

### How It Works

1. **MediaPipe detects hands** → returns landmarks
2. **Extract features** → 84 values (hand 1 + hand 2 or zeros)
3. **Hand-centric normalization** → same as training
4. **StandardScaler** → normalize distribution
5. **CNN prediction** → probabilities for each letter
6. **Post-processing** → C/S disambiguation

### C/S Disambiguation

C and S look similar but differ in hand count:

```python
# Post-processing logic
if pred_letter == 'S' and num_hands == 1:
    pred_letter = 'C'  # 1 hand = C
elif pred_letter == 'C' and num_hands == 2:
    if probs['S'] > 0.3:
        pred_letter = 'S'  # 2 hands + high S prob = S
```

### Real-Time Display

```
┌─────────────────────────────────────┐
│  A (0.95)                           │  ← Prediction
│                                     │
│  [Hand skeleton overlay]            │  ← Green = hand 1, Blue = hand 2
│                                     │
│  2 hand(s) -> A (0.95)              │  ← Status bar
│  FPS: 30                            │
└─────────────────────────────────────┘
```

---

## Key Lessons

### 1. Data Format Matters

**Problem:** Training data was 1-hand (63 features), but webcam uses 2-hand (84 features).

**Solution:** Always collect with `capture_fast2.py` for consistent 84-feature format.

### 2. Hand-Centric Normalization is Critical

**Without normalization:**
- Hand close → 99% accuracy
- Hand far → 20% accuracy

**With normalization:**
- Hand close → 99% accuracy
- Hand far → 95% accuracy

### 3. C/S Disambiguation Needs Hand Count

C and S have similar hand shapes but different hand counts:
- C = 1 hand
- S = 2 hands

Use `num_hands` from MediaPipe to disambiguate.

### 4. Augmentation Degrades Performance

| Approach | Test Accuracy | Webcam Performance |
|----------|--------------|-------------------|
| Original data | 99.1% | Best |
| Aggressive augmentation | 100% | Worse |

**Why:** Augmentation creates unrealistic patterns that don't match real webcam input.

### 5. MediaPipe Limitations

- Detects 1 hand when hands overlap
- Requires good lighting and clear background
- May miss fast movements

### 6. Collect Data in Batches

Efficient workflow:
1. Collect A-K → retrain → verify
2. Collect L-Z → retrain → verify
3. Fix problematic letters → retrain

Don't collect all 26 letters at once — catch issues early.

---

## Troubleshooting

### Problem: Low accuracy on specific letters

**Symptoms:** M/N confused, B/E confused

**Causes:**
- Similar hand shapes
- Insufficient training data
- Poor hand separation in data collection

**Solutions:**
1. Recollect with clearer hand positions
2. Increase samples (2000+ instead of 1000)
3. Check confusion matrix for patterns

### Problem: Model works close but not far

**Cause:** Missing hand-centric normalization

**Solution:** Ensure `normalize_features()` is applied in both training and inference.

### Problem: 1-hand letters detected as 2-hand

**Cause:** MediaPipe detects phantom second hand

**Solution:** Add confidence threshold or require sustained detection.

### Problem: Data format mismatch

**Symptoms:** `input_dim` error or wrong number of features

**Solution:** Check CSV headers match expected format:
```bash
head -1 dataset/landmarks_2hands.csv
# Should show: letter,path,split,num_hands,contributor,lm0_x,lm0_y,...,h2_lm20_x,h2_lm20_y
```

---

## Quick Reference

### Commands

```bash
# Collect data
python3 capture_fast2.py A --count 1000

# Train model
python3 train/train_dl.py --data dataset/landmarks_2hands.csv --epochs 50 --name cnn_2hand

# Test webcam
python3 inference_webcam.py

# Confusion matrix
python3 eval/confusion.py --model cnn_2hand

# Merge 1-hand data into 2-hand CSV
python3 << 'EOF'
import pandas as pd
df_main = pd.read_csv('dataset/landmarks_2hands.csv')
df_1hand = pd.read_csv('dataset/landmarks_xy.csv')
h2_cols = [f"h2_lm{i}_{c}" for i in range(21) for c in ("x", "y")]
for col in h2_cols:
    df_1hand[col] = 0.0
df_1hand = df_1hand.rename(columns={"image_path": "path"})
cols = df_main.columns.tolist()
df_1hand = df_1hand[cols]
merged = pd.concat([df_main, df_1hand], ignore_index=True)
merged.to_csv('dataset/landmarks_2hands.csv', index=False)
EOF
```

### File Locations

| File | Purpose |
|------|---------|
| `train/train_dl.py` | Training script |
| `inference_webcam.py` | Webcam inference |
| `capture_fast2.py` | Data collection |
| `eval/confusion.py` | Confusion matrix |
| `dataset/landmarks_2hands.csv` | Training data |
| `models/dl/cnn_2hand_*` | Model artifacts |

### Expected Performance

- **16 letters (A-P):** 99.1% accuracy
- **26 letters (A-Z):** ~95-97% accuracy (estimated)
- **Real webcam:** 90-95% with good lighting and hand position

---

## Version History

| Date | Change |
|------|--------|
| 2026-06-28 | Initial CNN setup |
| 2026-06-29 | 2-hand format support, hand-centric normalization |
| 2026-06-30 | C/S disambiguation, full 26-letter support |
