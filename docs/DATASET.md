# Dataset Card — BISINDO Hand Landmarks

> HuggingFace-style data card for `dataset/landmarks_captured_v2.csv`.

## Dataset Summary

A collection of hand-landmark vectors for the 26 static letters of the BISINDO (Bahasa Isyarat Indonesia / Indonesian Sign Language) alphabet. Each sample consists of 21 hand-landmark coordinates (x, y, z) extracted from images via MediaPipe Hands v0.10.

- **Size**: 138,471 samples
- **Language**: Indonesian Sign Language (BISINDO)
- **Task**: Static hand-gesture classification (26 classes A–Z)
- **Format**: CSV (one row per sample)
- **License**: MIT (matches repo `LICENSE`)

## Supported Tasks

- Image-to-gesture classification (via landmark extraction)
- Hand-pose regression (research)
- Cross-locale signer robustness study (research)

## Languages

Indonesian Sign Language (BISINDO) — the official sign language of Indonesia. This dataset covers the **fingerspelling alphabet** (A–Z), not the full lexical BISINDO vocabulary.

## Dataset Structure

### Files

| File | Rows | Purpose |
|---|---|---|
| `landmarks_captured_v2.csv` | 138,471 | **Canonical** — current production dataset |
| `landmarks_original.csv` | ~138k | Original capture (identical to v2) |
| `landmarks_clean.csv` | 177,748 | Cleaned variant |
| `landmarks_balanced.csv` | 208,001 | Rebalanced (~5000 per letter) |
| `landmarks_augmented.csv` | 553,884 | Augmented with rotations + noise |
| `landmarks_2hands.csv` | 26,193 | 2-hand subset (84-dim) |
| `landmarks_xy.csv` | ~50k | xy-only format (42-dim) |

### Schema (canonical: `landmarks_captured_v2.csv`)

| Column | Type | Description |
|---|---|---|
| `letter` | string | Target class — one of A–Z |
| `image_path` | string | Synthetic ID: `<source>_<letter>_<idx>` |
| `split` | string | `train` (137,956) or `test` (515) |
| `num_hands` | int | Number of hands detected: 1 (13,817) or 2 (124,654) |
| `contributor` | string | Capturer pseudonym (9 unique) |
| `lm0_x` … `lm20_z` | float | 63 features = 21 landmarks × 3 coords (x, y, z) |

### Splits

| Split | Rows |
|---|---|
| `train` | 137,956 (99.6%) |
| `test` | 515 (0.4%) |

> **Note**: The canonical `split` column is mostly `train`. For model evaluation, the recommended split is a stratified 85/15 over the full dataset using `random_state=42` (yields 117,700 train / 20,771 test). This is what `eval/compare.py` uses.

## Class Distribution

26 letters, **min 5015, max 6388, mean 5326, std 381** — moderately balanced.

| Letter | Count | | Letter | Count | | Letter | Count |
|---|---|---|---|---|---|---|---|
| A | 6,388 | | J | 5,085 | | S | 5,187 |
| B | 5,087 | | K | 5,159 | | T | 5,282 |
| C | 5,064 | | L | 5,023 | | U | 5,802 |
| D | 5,264 | | M | 5,325 | | V | 5,071 |
| E | 6,104 | | N | 5,199 | | W | 5,064 |
| F | 5,094 | | O | 5,469 | | X | 5,109 |
| G | 5,085 | | P | 5,068 | | Y | 5,683 |
| H | 5,015 | | Q | 5,031 | | Z | 5,150 |
| I | 5,616 | | R | 6,047 | | | |

## Contributors

9 unique contributors captured data. Top 5:

| Contributor | Samples | Share |
|---|---|---|
| Jouxing | 34,373 | 24.8% |
| Kevin | 32,196 | 23.3% |
| felis | 32,046 | 23.2% |
| anjay | 20,171 | 14.6% |
| legacy | 8,514 | 6.1% |

Other 4 contributors account for the remaining ~8%. **Capture-bias risk**: top-3 contributors produce 71% of all samples — generalization to new signers may be limited.

## Hand Detection

- **1 hand**: 13,817 samples (10%)
- **2 hands**: 124,654 samples (90%) — most samples capture both hands for context, even when only one is signing

The 1-hand subset is useful for signer self-occlusion studies; the 2-hand subset is the default for training.

## Data Collection

- **Capture client**: `web/capture.html` (browser-based, MediaPipe Hands)
- **Capture prompt**: live webcam, letter shown on screen, contributor performs the gesture
- **Per-letter target**: 5000 samples
- **Storage**: server-side CSV append via `meeting/app.py → /api/sample`

## Preprocessing

The following pipeline is applied at training time (see `train/train_dl.py`):

1. **Hand-centric normalization**: translate to wrist (landmark 0), scale by max distance from wrist (hand size)
2. **Standard scaling**: zero mean, unit variance (scaler saved per-model as `*_scaler.json`)
3. **Stratified 85/15 split**, `random_state=42`

The 1-hand xyz format (63 features) is the default. A 2-hand xy format (84 features) is also supported — see `cnn_2hand_model.pt`.

## Intended Use

- ✅ Research on BISINDO fingerspelling recognition
- ✅ Education / accessibility tooling prototypes
- ✅ Cross-locale signer robustness studies
- ❌ **Not** a clinical-grade sign language translator
- ❌ **Not** a production system for legal / medical contexts

## Limitations

- **Static letters only** — no temporal / dynamic gesture support (e.g. J and Z require motion)
- **Single camera angle** — all samples captured from front-facing webcam
- **Controlled lighting** — no low-light or harsh-shadow samples
- **Limited demographic diversity** — 9 contributors, mostly from same locale
- **Class collapse risk** — letters with similar shapes (M/N, U/V) may be confused by any model
- **No motion data** — gesture dynamics are lost

## Citation

```
BISINDO Bridge — Hand landmark dataset for Indonesian Sign Language fingerspelling.
138,471 samples, 26 classes A–Z. MediaPipe Hands v0.10 extraction.
License: MIT.
```