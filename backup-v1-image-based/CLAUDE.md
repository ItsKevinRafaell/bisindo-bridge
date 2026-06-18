# BISINDO Two-Way Communication Bridge

## 🎯 Project Overview

Aplikasi penerjemah BISINDO (Bahasa Isyarat Indonesia) **dua arah** yang menjembatani komunikasi antara orang Tuli dan orang dengar.

### Core Features
- **Sisi Kiri (Gesture → Audio)**: Deteksi gesture BISINDO dari kamera → translate ke text + audio
- **Sisi Kanan (Speech → Gesture Guide)**: Speech-to-text → tampilkan panduan gesture BISINDO
- **Two-Way Communication**: Real-time bidirectional translation

### Innovation
- Bukan sekadar deteksi gesture, tapi **communication bridge** dengan dampak sosial nyata
- Belum ada di Indonesia untuk BISINDO
- Target: membantu orang Tuli berkomunikasi dengan orang dengar dalam situasi sehari-hari

---

## 📊 Dataset

### Source
**Repository**: https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset

**Paper**: "BISINDO Hand-Sign Detection Using Transfer Learning" (IEEE ICRAIE 2023)

### Dataset Stats
| Metric | Value |
|--------|-------|
| **Total Images** | 520 |
| **Train** | 416 images (80%) |
| **Test** | 104 images (20%) |
| **Classes** | 26 (A-Z) |
| **Images per Class** | 20 (16 train + 4 test) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TWO-WAY BRIDGE APP                        │
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │   SISI KIRI      │          │   SISI KANAN     │        │
│  │   (Gesture →     │          │   (Speech →      │        │
│  │    Audio)        │          │    Gesture Guide)│        │
│  │                  │          │                  │        │
│  │  📷 Camera       │          │  🎤 Microphone   │        │
│  │      ↓           │          │      ↓           │        │
│  │  MediaPipe       │          │  Speech-to-Text │        │
│  │  Hand Detection  │          │  (Whisper)      │        │
│  │      ↓           │          │      ↓           │        │
│  │  MobileNetV2    │          │  NLP            │        │
│  │  Predict huruf   │          │  → BISINDO     │        │
│  │      ↓           │          │      ↓           │        │
│  │  Spell → kata    │          │  Show gesture   │        │
│  │  + TTS Audio    │          │  guide          │        │
│  └──────────────────┘          └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web app, UI |
| **Camera** | OpenCV | Video processing |
| **Hand Detection** | MediaPipe Hands | Lightweight hand detection |
| **Gesture Recognition** | MobileNetV2 | Image classification A-Z |
| **Text-to-Speech** | gTTS | Indonesian audio output |

---

## 📋 Implementation Roadmap

### Phase 1: Setup & Dataset ✅
- [x] Clone BISINDO dataset dari GitHub
- [x] Setup project directory
- [x] Create documentation

### Phase 2: Gesture Detection Core ✅
- [x] Train MobileNetV2 on BISINDO dataset (91.35% accuracy)
- [x] Implement hand detection dengan MediaPipe
- [x] Implement letter buffering (H-A-L-O → HALO)

### Phase 3: Gesture → Audio ✅
- [x] Dictionary matching (huruf → kata Indonesia)
- [x] Integrate Text-to-Speech (gTTS Indonesian)
- [x] Streamlit UI dengan 2-column layout

### Phase 4: Speech → Gesture Guide ⏳
- [ ] Implement Speech-to-Text (Whisper)
- [ ] Build gesture guide database (A-Z BISINDO)
- [ ] Display gesture guide (text + image/video)

### Phase 5: Data Collection ⏳
- [ ] Auto-capture system untuk kumpulin dataset
- [ ] Target letter selector (A-Z)
- [ ] Batch management
- [ ] Progress tracking

### Phase 6: Polish & Testing
- [ ] Error handling & edge cases
- [ ] Performance tuning
- [ ] User testing

---

## 📁 Project Structure

```
/home/kevin/bisindo-bridge/
├── dataset/                    # BISINDO dataset (520 images)
│   ├── train/                # 416 training images + XML
│   ├── test/                 # 104 testing images + XML
│   └── collectedimages/       # Original images
├── models/                   # Trained model files
│   ├── bisindo_mobilenetv2_v4_best.keras
│   └── label_map_v4.json
├── src/                      # Source code
│   ├── gesture_detector.py   # Hand detection + gesture recognition
│   ├── speech_processor.py   # Speech-to-text
│   ├── gesture_guide_db.py   # BISINDO gesture guides
│   └── tts_engine.py        # Text-to-Speech
├── docs/                     # Documentation
│   └── COLAB_TRAINING_GUIDE.md
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
└── CLAUDE.md                 # This file
```

---

## 🎯 Success Criteria

- ✅ Gesture recognition accuracy > 90%
- ⏳ Speech-to-text accuracy > 85%
- ⏳ End-to-end latency < 2 seconds
- ✅ UI responsive dan intuitive
- ⏳ Demo video 2-3 menit yang menunjukkan two-way communication

---

## 📝 Notes

- Dataset ini lebih kecil (520 images) dibanding dataset Kaggle agungmrf (11,470 images)
- Bisa augmentasi data untuk improve accuracy
- Whisper API butuh internet, bisa switch ke Whisper.cpp untuk offline mode
- Untuk production, pertimbangkan pakai dataset yang lebih besar

---

## 🔗 References

1. **BISINDO Dataset**: https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset
2. **MediaPipe Hands**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
3. **OpenAI Whisper**: https://platform.openai.com/docs/guides/speech-to-text

---

**Last Updated**: 2026-06-18
**Status**: Phase 3 Complete - Ready for Phase 4 & 5
