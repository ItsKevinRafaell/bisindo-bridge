# BISINDO Bridge

Two-way Bahasa Isyarat Indonesia (BISINDO) translator using landmark-based gesture recognition.

## Project Overview

BISINDO Bridge membandingkan dua pendekatan machine learning untuk pengenalan gestur BISINDO:
- **ML Team**: Traditional ML methods
- **DL Team**: Deep learning methods

Dataset: ~138,000 samples, 26 huruf (A-Z)

## Project Structure

```
bisindo-bridge/
├── train/               # Training scripts
│   ├── train_ml.py     # ML Team
│   └── train_dl.py     # DL Team
├── eval/                # Evaluation scripts
│   └── compare.py       # Model comparison
├── models/              # Trained models (gitignored)
│   ├── ml/              # ML team outputs
│   └── dl/              # DL team outputs
├── meeting/             # Meeting app server
├── web/                 # Frontend
├── docs/                # Documentation
│   ├── TEAM_GUIDE.md
│   ├── PROPOSAL_OUTLINE.md
│   └── BRANCH_STRATEGY.md
└── dataset/             # Dataset (gitignored)
    └── landmarks_captured_v2.csv
```

## Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd bisindo-bridge
pip install -r requirements.txt
```

### 2. Training

**ML Team:**
```bash
python train/train_ml.py --model rf  # atau model lain
```

**DL Team:**
```bash
pip install tensorflow tensorflowjs
python train/train_dl.py --model mlp  # atau arsitektur lain
```

### 3. Evaluation
```bash
python eval/compare.py
```

## Team Assignment

| Team | Role | Files |
|------|------|-------|
| **ML Team (2)** | Traditional ML | train/train_ml.py, models/ml/ |
| **DL Team (2)** | Deep Learning | train/train_dl.py, models/dl/ |
| **Proposal (1)** | Documentation | docs/PROPOSAL_OUTLINE.md |

## Requirements

- Python 3.9+
- scikit-learn
- pandas, numpy
- tensorflow (DL team only)
- tensorflowjs (DL team only)

## Documentation

Lihat `docs/TEAM_GUIDE.md` untuk panduan lengkap tim.