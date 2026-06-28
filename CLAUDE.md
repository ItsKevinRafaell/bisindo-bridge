# BISINDO Bridge - Claude Code Instructions

## Project Overview

Two-way BISINDO (Bahasa Isyarat Indonesia) translator using landmark-based gesture recognition.
Comparison study: ML (Random Forest, SVM) vs DL (MLP, CNN).

## Architecture

```
bisindo-bridge/
├── train/              # Training scripts (per team)
│   ├── train_ml.py     # ML Team: Random Forest, SVM
│   └── train_dl.py     # DL Team: MLP, CNN
├── eval/               # Evaluation
│   └── compare.py      # ML vs DL comparison
├── models/             # Model outputs (gitignored)
│   ├── ml/             # ML team: rf_model.pkl, svm_model.pkl
│   └── dl/             # DL team: mlp_model.h5, cnn_model.h5
├── meeting/            # Flask-SocketIO meeting server
├── web/                # Static frontend (Vercel-deployable)
├── dataset/            # Data (gitignored)
│   └── landmarks_captured_v2.csv  # 108k rows, 63 features
└── docs/               # Documentation
    ├── TEAM_GUIDE.md
    └── PROPOSAL_OUTLINE.md
```

## Team Assignment

| Team | Role | Training Command |
|------|------|------------------|
| ML (2) | Random Forest, SVM | `python train/train_ml.py --model both` |
| DL (2) | MLP, CNN | `python train/train_dl.py --model both --epochs 50` |
| Proposal (1) | Documentation | Use docs/PROPOSAL_OUTLINE.md |

## Data Schema

CSV: `landmarks_captured_v2.csv`
- Columns: letter, image_path, split, num_hands, contributor, lm0_x..lm20_z
- 63 features (21 landmarks x 3 coordinates)
- ~108,000 samples, 26 letters (A-Z)

## Key Commands

```bash
# ML Team
python train/train_ml.py --model rf        # RF only
python train/train_ml.py --model svm       # SVM only
python train/train_ml.py --model both      # Both

# DL Team
pip install tensorflow tensorflowjs        # First time
python train/train_dl.py --model mlp       # MLP only
python train/train_dl.py --model cnn       # CNN only
python train/train_dl.py --model both      # Both

# Evaluation
python eval/compare.py                     # Generate comparison report
```

## Meeting Server

```bash
cd meeting
python app.py
```

Akses: `http://localhost:5000` atau via Cloudflare tunnel

## Branch Strategy

```
main (production)
├── ml/rf-dev, ml/svm-dev   # ML team
├── dl/mlp-dev, dl/cnn-dev  # DL team
└── docs/proposal           # Proposal
```

See `docs/BRANCH_STRATEGY.md` for workflow details.

## Last Updated
2026-06-28
