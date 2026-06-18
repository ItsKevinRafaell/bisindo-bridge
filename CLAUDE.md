# BISINDO Two-Way Communication Bridge

## 🎯 Project Overview

Aplikasi penerjemah BISINDO (Bahasa Isyarat Indonesia) **dua arah** menggunakan pendekatan **Landmark-Based Gesture Recognition**.

### Strategy Comparison

| Approach | Data Needed | Training Time | Accuracy | Speed |
|----------|-------------|---------------|----------|-------|
| **Image-Based** (backup) | 500+ images | ~30 min (GPU) | 91.35% | ~100ms |
| **Landmark-Based** (current) | 50-100 per letter | ~2 min (CPU) | TBD | ~5ms |

---

## 🏗️ New Architecture (Landmark-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                    LANDMARK-BASED PIPELINE                    │
│                                                             │
│  Gambar → MediaPipe Hands → 21 Landmarks → Features → Model │
│              ↓                                               │
│         [(x,y,z) × 21 = 63 values]                         │
│              ↓                                               │
│         Normalized & Scaled                                 │
│              ↓                                               │
│         Random Forest / Neural Network                       │
│              ↓                                               │
│         Prediksi Letter (A-Z)                               │
└─────────────────────────────────────────────────────────────┘
```

### Why Landmark-Based?

1. **Invariant** terhadap lighting, background, hand size
2. **Compact features** - hanya 63 nilai per sample
3. **Fast training** - bisa di CPU dalam hitungan menit
4. **Interpretable** - bisa visualize pola tangan

---

## 📁 Project Structure

```
/home/kevin/bisindo-bridge/
├── backup-v1-image-based/     # Previous strategy (for report)
├── dataset/                   # BISINDO dataset
│   ├── train/
│   ├── test/
│   └── landmarks/              # Extracted landmarks (to be generated)
├── models/                    # Trained models
├── src/
│   ├── landmark_extractor.py  # MediaPipe → 63 features
│   ├── landmark_trainer.py    # Training script
│   ├── landmark_classifier.py  # Inference model
│   ├── gesture_detector.py    # Real-time detection
│   └── tts_engine.py          # Text-to-Speech
├── app.py                    # Main Streamlit app
├── requirements.txt
└── CLAUDE.md
```

---

## 🔧 Implementation Plan

### Phase 1: Landmark Extraction
- [ ] Extract 63 features (21 landmarks × 3 coords) dari dataset
- [ ] Build landmark dataset CSV
- [ ] Validate extraction quality

### Phase 2: Training
- [ ] Train classifier (Random Forest / Simple NN)
- [ ] Compare accuracy vs image-based
- [ ] Optimize hyperparameters

### Phase 3: Real-time Detection
- [ ] Integrate dengan app.py
- [ ] Add letter buffering
- [ ] Performance testing

### Phase 4: Data Collection
- [ ] Auto-capture dengan landmarks
- [ ] Incremental learning

---

## 📊 Expected Results

- **Training time**: ~2-5 min (vs 30 min)
- **Accuracy**: Target > 85%
- **Inference speed**: < 10ms (vs ~100ms)
- **Dataset size needed**: 50-100 per letter (vs 500+)

---

## ✅ Meeting Server - LIVE

**Flask-SocketIO Video Meeting dengan BISINDO Recognition**

```bash
cd meeting
python app.py
```

Akses: `http://localhost:5000` atau via Cloudflare tunnel

**Features:**
- Video relay via server (server-side processing)
- BISINDO landmark extraction + prediction
- Real-time letter detection (A-Z)
- TTS pronunciation via gTTS
- Multiple users in room

**Last Updated**: 2026-06-18
**Status**: Phase 1 - Meeting MVP Working ✅
