# Branch Strategy - BISINDO Bridge

## Overview
GitHub Flow (simple, effective for small teams)

## Branch Structure

```
main (production-ready)
├── ml/rf-dev          # ML Team: Random Forest
├── ml/svm-dev         # ML Team: SVM
├── dl/mlp-dev         # DL Team: MLP
├── dl/cnn-dev         # DL Team: CNN
└── docs/proposal      # Proposal writing
```

## Rules

1. **Branch dari `main`** — selalu start dari fresh main
2. **Naming convention** — `team/feature-description`
3. **PR ke `main`** — setelah testing lokal berhasil
4. **Squash merge** — history bersih
5. **Delete branch** — setelah merge

## Workflow

### ML Team (2 orang)

```bash
# Setup
git checkout main
git pull origin main
git checkout -b ml/rf-dev

# Development
# ... edit train/train_ml.py, experiment ...
python train/train_ml.py --model rf

# Commit & Push
git add .
git commit -m "feat(ml): optimize RF hyperparameters"
git push origin ml/rf-dev

# Create PR via GitHub UI
# After merge:
git checkout main
git pull origin main
git branch -d ml/rf-dev
```

### DL Team (2 orang)

```bash
# Setup
git checkout main
git pull origin main
git checkout -b dl/mlp-dev

# Development
pip install tensorflow tensorflowjs
python train/train_dl.py --model mlp

# Commit & Push
git add .
git commit -m "feat(dl): initial MLP training"
git push origin dl/mlp-dev

# Create PR via GitHub UI
```

### Proposal Team (1 orang)

```bash
# Setup
git checkout main
git pull origin main
git checkout -b docs/proposal

# Write proposal
# ... edit docs/PROPOSAL_OUTLINE.md ...

# Commit & Push
git add .
git commit -m "docs: proposal draft v1"
git push origin docs/proposal
```

## Integration (All Teams)

After all teams finish training:

```bash
# Run comparison
python eval/compare.py

# Update proposal with results
# Fill in accuracy tables in docs/PROPOSAL_OUTLINE.md

# Final commit & merge
git add .
git commit -m "docs: add final comparison results"
git push origin docs/proposal
```

## Conflict Resolution

If conflicts occur:
1. `git fetch origin`
2. `git merge origin/main` into your branch
3. Resolve conflicts manually
4. `git add .` and `git commit`
5. Push

## Tips

- Commit often with clear messages
- Keep branches short-lived (< 1 week)
- Always test locally before PR
- Delete old branches to stay clean