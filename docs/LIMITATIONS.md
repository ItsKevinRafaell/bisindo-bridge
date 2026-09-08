# Limitations — BISINDO Bridge

> Honest constraints of the current system, framed for journal reviewer scrutiny.

## 1. Scope: CNN-only

This submission compares **CNN architectures only**. The original project plan also covered sklearn baselines (Random Forest, SVM), but those are intentionally excluded:

- `models/ml/` is empty in the public repo (RF/SVM pkl files were gitignored)
- Adding them now would either (a) require retraining on this dataset (~30 min) without adding meaningful insight (sklearn models plateau at ~0.97 on this dataset) or (b) require citing previous runs without reproducible artifacts
- CNN-only framing keeps the comparison focused and honest

## 2. Single Train/Test Split

All reported numbers use a single 85/15 stratified split (`random_state=42`). We do **not** report:

- K-fold cross-validation (5-fold or 10-fold) — would tighten confidence intervals
- Holdout set evaluation — no truly unseen test set
- Cross-signer evaluation — same 9 contributors across train/test

**Mitigation in this paper**: report exact split, exact seed, and exact evaluation code so other researchers can reproduce.

## 3. Independent Re-evaluation of Checkpoints

Of the 8 CNN checkpoints in `models/dl/`, only 2 could be independently re-evaluated from `state_dict` + scaler alone (`cnn`, `cnn_2hand`). The other 6 — including the published champion `cnn_balanced` (acc 0.9912) — use architectures that cannot be reconstructed from `state_dict` alone because:

- The model class definition is not preserved alongside the checkpoint
- Several checkpoints have `fc.1.weight=[128, 512]`, implying a hidden 64→512 projection not present in the current `train_dl.py` CNN class
- The hand-centric normalization pipeline at training time is partially reconstructed from comments + scaler stats, but exact byte-identical inference is not achievable

**Implication**: the published numbers in `models/comparison_report.md` come from `*_metrics.json` files written at training time, not from re-evaluation. We cannot independently verify the 0.9912 champion number from the current repo state.

**Mitigation in this paper**: state this explicitly in `RESULTS.md` and tag the numbers as "training-time metrics" rather than "verified re-evaluation".

## 4. Dataset Bias

- **9 contributors** (top-3 produce 71% of all samples) — model may overfit to their hand shapes, skin tones, signing style
- **Webcam capture only** — no mobile, no depth sensor, no multi-camera
- **Indoor controlled lighting** — no outdoor, low-light, or harsh-shadow samples
- **Right-handed bias** (suspected) — not measured in this dataset
- **Class imbalance** is mild (5015–6388 per letter) but the `cnn_balanced` model was retrained on a rebalanced version, not the canonical CSV

## 5. Model Coverage

We report 8 CNN variants, all 1D-CNN over 63-dim landmark vectors (or 84-dim for `cnn_2hand`). Missing comparisons:

- No Transformer / attention-based models
- No graph neural networks (landmarks are naturally graph-structured — hand skeleton)
- No LSTM / temporal models (would require sequence data, not static landmarks)
- No data augmentation at training time (rotation, noise, scaling)
- No Mixup / CutMix
- No self-supervised pretraining

## 6. Application (BISINDO Meeting App)

The companion web app (`meeting/`, `web/`) is **demo-quality**, not production:

- **Hardcoded secret**: `SECRET_KEY = 'bisindo-meeting-secret-2026'` (planned fix in `meeting/app.py:33`)
- **CORS `*`**: any origin can hit the API (planned fix: env-driven whitelist)
- **Debug flag**: `allow_unsafe_werkzeug=True` is set unconditionally (planned fix: env-gated)
- **Ephemeral tunnel**: Cloudflare Quick Tunnel URL changes on every restart
- **SFU not deployed**: mediasoup Node server requires Node.js install which is missing on VPS
- **Classifier fails on VPS**: MediaPipe Tasks Python imports `libGLESv2.so.2` even for inference; not installed in the Debian LXC container

These app-level issues do **not** affect the CNN results, but reviewers who try to demo the system will hit them.

## 7. Security

- **VPS SSH**: `PermitRootLogin yes`, `PasswordAuthentication yes`
- **No firewall**: ufw / iptables not installed on VPS
- **No fail2ban**
- **App auth**: none — anyone who hits the tunnel URL can join any room

This is acceptable for research demos; it would not pass any production security review.

## 8. Reproducibility Specifics

- **Random seeds** are set in training (`random_state=42`) but PyTorch CPU non-determinism can still cause ±0.5 pp variance
- **No MLflow / W&B / experiment tracking** — model cards are JSON files only
- **No automated tests** for `eval/compare.py` or `train/train_dl.py`
- **No CI / GitHub Actions** — verification is manual

## 9. Ethical Considerations

- **Signer privacy**: contributor pseudonyms are used (no real names)
- **Dataset size fairness**: large Western sign-language datasets (e.g. ASL Citizen, 5M+ samples) dwarf this 138k-sample effort — BISINDO research is resource-constrained
- **Cultural context**: BISINDO is one of several Indonesian sign languages (SIBI is the official school sign system); this work targets BISINDO fingerspelling only, not full conversational sign

## 10. Out of Scope

These are explicitly **not** part of this submission:

- Real-time performance benchmarks (FPS on mobile, latency on edge devices)
- Model compression (quantization, pruning, distillation)
- Multi-modal fusion (RGB + depth + landmarks)
- Sign-language-to-text translation beyond letter spelling
- Vocabulary beyond A–Z (no BISINDO words or phrases)

---

## Summary

The CNN models are strong (best 99.12% on in-distribution test) but the system as a whole is a research prototype. The honest framing for reviewers: "8 CNN architectures compared on a 138k-sample BISINDO fingerspelling dataset, with reproducible training pipeline and documented limitations."