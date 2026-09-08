# BISINDO Bridge

> **Research prototype / proof-of-concept** — not production-grade.
> CNN-based static-gesture recognition for the 26-letter BISINDO (Bahasa Isyarat Indonesia) fingerspelling alphabet.

A 1D-CNN classifier over 21 hand-landmark coordinates extracted via MediaPipe Hands. Trained on 138,471 samples contributed by 9 signers. Best model achieves **99.12%** accuracy on a held-out 15% test split.

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd bisindo-bridge
pip install -r requirements.txt
```

### 2. Reproduce the champion CNN

```bash
python train/train_dl.py --name cnn_balanced --epochs 50 --batch 256
# Output: models/dl/cnn_balanced_model.pt + metrics + scaler + labels
```

Expected outcome: accuracy ≈ 0.99 on the held-out test split (random_state=42).

### 3. Generate comparison report

```bash
python eval/compare.py
# Output:
#   models/comparison_report.md       (Markdown table + champion rationale)
#   models/comparison_data.json       (machine-readable)
#   models/dl/comparison_chart.png    (visual comparison)
```

### 4. (Optional) Re-evaluate from checkpoint

```bash
python eval/compare.py --reeval
# Attempts independent inference. Succeeds for ~2/8 models; others use
# training-time metrics from *_metrics.json (see LIMITATIONS.md).
```

### 5. Run the meeting app (demo only)

```bash
cd meeting
python app.py
# → http://localhost:4500
```

## Project Structure

```
bisindo-bridge/
├── train/                  # Training scripts
│   ├── train_ml.py         # sklearn baselines (out of scope for journal)
│   └── train_dl.py         # PyTorch CNN training
├── eval/
│   └── compare.py          # Model comparison report generator
├── models/
│   ├── comparison_report.md      ← auto-generated, latest results
│   ├── comparison_data.json      ← auto-generated, machine-readable
│   └── dl/                # 8 CNN checkpoints + per-model metrics
├── meeting/                # Flask-SocketIO meeting server (demo-quality)
├── web/                    # Static frontend (Vercel-deployable)
├── docs/
│   ├── RESULTS.md          ← model selection rationale (read this for paper)
│   ├── DATASET.md          ← data card (sample counts, contributors, schema)
│   ├── LIMITATIONS.md      ← honest constraints (read for paper)
│   ├── TEAM_GUIDE.md
│   ├── PROPOSAL_OUTLINE.md
│   └── BRANCH_STRATEGY.md
├── dataset/
│   └── landmarks_captured_v2.csv  # canonical, 138,471 rows
├── STATE.md                # live project state (loop-engineering ritual)
├── LOOP.md                 # planned loops
└── .claude/
    ├── skills/             # reusable prompt patterns (cnn-train, cnn-eval, …)
    └── budget.md           # token tracking
```

## Results

Champion: **`cnn_balanced`** — accuracy **0.9912**, F1 **0.9912**, train time 3704s.

8 CNN variants compared. See:

- [`docs/RESULTS.md`](docs/RESULTS.md) — selection rationale + tier breakdown
- [`models/comparison_report.md`](models/comparison_report.md) — raw report
- [`models/dl/comparison_chart.png`](models/dl/comparison_chart.png) — visual comparison

## Documentation

| Doc | Purpose |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | **For the paper** — champion rationale + comparison table |
| [`docs/DATASET.md`](docs/DATASET.md) | **For the paper** — data card (schema, distribution, contributors) |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **For the paper** — honest constraints |
| `docs/TEAM_GUIDE.md` | Team onboarding (legacy) |
| `docs/PROPOSAL_OUTLINE.md` | Original proposal outline |
| `docs/BRANCH_STRATEGY.md` | Git workflow |
| `docs/MEETING_ARCHITECTURE.md` | Meeting app design (demo only) |
| `docs/LEARNING_GUIDE.md` + `docs/learning/` | 10-chapter ML/DL tutorial |

## Requirements

- Python 3.11+
- PyTorch 2.x (CPU is fine; CUDA optional)
- scikit-learn, pandas, numpy, matplotlib
- MediaPipe Hands v0.10 (for data collection + optional inference)

## License

MIT — see `LICENSE`.

## Status

This is a **research prototype**. The CNN models are reproducible and well-evaluated (see `RESULTS.md`). The companion meeting app is demo-quality (see `LIMITATIONS.md` §6) and not suitable for production deployment.