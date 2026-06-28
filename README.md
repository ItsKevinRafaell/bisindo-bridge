# BISINDO Bridge

Two-way Bahasa Isyarat Indonesia (BISINDO) translator using landmark-based gesture recognition.

## Project Overview

BISINDO Bridge membandingkan dua pendekatan machine learning untuk pengenalan gestur BISINDO:
- **ML Team**: Random Forest, SVM (traditional ML)
- **DL Team**: MLP, CNN (deep learning)

Dataset: ~108,000 samples, 26 huruf (A-Z)

## Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd bisindo-bridge
pip install -r requirements.txt
```

### 2. Data
Data sudah tersedia di `dataset/landmarks_captured_v2.csv` (108k rows).

### 3. Training

**ML Team:**
```bash
python train/train_ml.py --model both
```

**DL Team:**
```bash
pip install tensorflow tensorflowjs
python train/train_dl.py --model both --epochs 50
```

### 4. Evaluation
```bash
python eval/compare.py
```

Output: `models/comparison_report.md`

## Project Structure

```
bisindo-bridge/
├── train/               # Training scripts
│   ├── train_ml.py     # ML Team: RF, SVM
│   └── train_dl.py     # DL Team: MLP, CNN
├── eval/                # Evaluation scripts
│   └── compare.py       # ML vs DL comparison
├── models/              # Trained models (gitignored)
│   ├── ml/              # ML team outputs
│   └── dl/              # DL team outputs
├── meeting/             # Meeting app server
├── web/                 # Frontend (test, capture)
├── docs/                # Documentation
└── dataset/             # Dataset (gitignored)
    └── landmarks_captured_v2.csv
```

## Team Assignment

| Team | Role | Files |
|------|------|-------|
| **ML Team** | Random Forest, SVM | train/train_ml.py, models/ml/ |
| **DL Team** | MLP, CNN | train/train_dl.py, models/dl/ |
| **Proposal** | Documentation | docs/PROPOSAL_OUTLINE.md |

## Requirements

- Python 3.9+
- scikit-learn
- pandas, numpy
- tensorflow (DL team only)
- tensorflowjs (DL team only)

## Timeline

| Week | Goal |
|------|------|
| 1-2 | Data collection (Sprint 1) |
| 3-4 | Model training (Sprint 2) |
| 5-6 | Evaluation & demo (Sprint 3) |
| 7 | Final polish |
| 10 | Deadline |
