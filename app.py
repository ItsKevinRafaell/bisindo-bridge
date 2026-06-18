"""
BISINDO Bridge - Integrated Web App with Webcam
Tabs: Detection | Data Collection | Training
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="BISINDO Bridge",
    page_icon="🤟",
    layout="wide"
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Custom CSS
st.markdown("""
<style>
    .stApp { background: #0F172A; }
    [data-testid="stSidebar"] { background: #1E293B; }
    .section-box {
        background: #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .progress-cell {
        background: #334155;
        border-radius: 4px;
        padding: 4px;
        text-align: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8;
    }
    .chat-bubble {
        background: #3B82F6;
        color: white;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 4px 0;
        max-width: 85%;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'capture_letter' not in st.session_state:
    st.session_state.capture_letter = 'A'
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 50
if 'samples' not in st.session_state:
    st.session_state.samples = []
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.selectbox("Model", ["Enhanced (170 features)", "Basic (63 features)"], key="model_select")

    st.divider()

    st.slider("Confidence", 0.1, 0.99, 0.5, key="confidence")
    st.slider("Hold Time (sec)", 0.5, 3.0, 1.0, key="hold_time")

    st.divider()

    st.markdown("**Dataset Info**")
    collected = 0
    if Path("dataset/collected").exists():
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_dir = Path(f"dataset/collected/{letter}")
            if letter_dir.exists():
                collected += len(list(letter_dir.glob("*.jpg")))
    st.metric("Collected Samples", collected)

    if Path("models/enhanced_classifier.pkl").exists():
        st.success("✅ Model trained")
    else:
        st.warning("⚠️ Model belum ada")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📷 Data Collection", "🧠 Training"])

# ============================================================
# TAB 1: DETECTION
# ============================================================
with tab1:
    st.markdown("## 🔍 Gesture Detection")
    st.caption("Deteksi gesture BISINDO real-time")

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.session_state.camera_running:
            st.markdown("### 📹 Camera Active")
            st.markdown("Tunjukkan gesture BISINDO ke kamera...")

            frame_window = st.image([])
            cap = cv2.VideoCapture(0)

            if st.button("Stop Camera", type="primary"):
                st.session_state.camera_running = False
                cap.release()
                st.rerun()

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_window.image(frame, channels="BGR")
                time.sleep(0.03)
        else:
            st.info("Klik Start Camera untuk deteksi real-time")

            if st.button("Start Camera", type="primary", use_container_width=True):
                st.session_state.camera_running = True
                st.rerun()

    with col2:
        st.markdown("### 📝 Riwayat")

# ============================================================
# TAB 2: DATA COLLECTION
# ============================================================
with tab2:
    st.markdown("## 📷 Data Collection")
    st.caption("Kumpulin sample gesture BISINDO untuk training model")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📋 Settings")

        # Letter selection
        letter_cols = st.columns([1, 1])
        with letter_cols[0]:
            selected_letter = st.selectbox("Huruf", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), index=0)
        with letter_cols[1]:
            target_count = st.number_input("Target", 10, 100, 50)

        st.divider()

        # Camera preview + capture
        if st.button("📹 Start Capture", type="primary", use_container_width=True):
            st.session_state.capture_letter = selected_letter
            st.session_state.capture_count = target_count
            st.session_state.samples = []
            st.session_state.camera_running = True

        if st.session_state.camera_running:
            st.success(f"📹 Capturing: {st.session_state.capture_letter} ({len(st.session_state.samples)}/{st.session_state.capture_count})")

            # Webcam preview
            frame_window = st.image([])
            cap = cv2.VideoCapture(0)

            # Initialize MediaPipe
            import mediapipe as mp
            model_path = "/tmp/hand_landmarker.task"
            if not os.path.exists(model_path):
                st.info("Downloading MediaPipe model...")

            while st.session_state.camera_running and len(st.session_state.samples) < st.session_state.capture_count:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_window.image(frame, channels="BGR")

                if len(st.session_state.samples) >= st.session_state.capture_count:
                    break

            cap.release()
            st.session_state.camera_running = False
            st.rerun()
        else:
            st.info("Pilih huruf → Klik Start Capture untuk mulai webcam")

    with col2:
        st.markdown("### 📊 Progress A-Z")

        for i in range(0, 26, 4):
            cols = st.columns(4)
            for j in range(4):
                if i+j < 26:
                    letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i+j]
                    letter_dir = Path(f"dataset/collected/{letter}")
                    n = len(list(letter_dir.glob("*.jpg"))) if letter_dir.exists() else 0
                    with cols[j]:
                        color = "#10B981" if n >= 50 else "#F59E0B" if n >= 20 else "#64748B"
                        st.markdown(f'<div class="progress-cell" style="background: {color}20; color: {color};">{letter}: {n}</div>', unsafe_allow_html=True)

        st.divider()

        if st.button("🗑️ Clear All Data"):
            import shutil
            if Path("dataset/collected").exists():
                shutil.rmtree("dataset/collected")
            Path("dataset/collected").mkdir(parents=True, exist_ok=True)
            st.rerun()

# ============================================================
# TAB 3: TRAINING
# ============================================================
with tab3:
    st.markdown("## 🧠 Training")
    st.caption("Train model dengan data yang sudah dikumpulkan")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Count samples
        total = 0
        letter_counts = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_dir = Path(f"dataset/collected/{letter}")
            n = len(list(letter_dir.glob("*.jpg"))) if letter_dir.exists() else 0
            letter_counts[letter] = n
            total += n

        st.metric("Total Samples", total)
        st.metric("Letters Covered", sum(1 for n in letter_counts.values() if n > 0))

        if total < 50:
            st.warning("⚠️ Sample terlalu sedikit. Kumpulin minimal 50 sample.")

        st.divider()

        if st.button("🚀 Train Model", type="primary", disabled=(total < 50)):
            with st.spinner("Training..."):
                from src.landmark_extractor import LandmarkExtractor
                from src.landmark_trainer import LandmarkTrainer

                all_data = []
                extractor = LandmarkExtractor(static_image_mode=True)

                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    letter_dir = Path(f"dataset/collected/{letter}")
                    if not letter_dir.exists():
                        continue
                    for img_path in letter_dir.glob("*.jpg"):
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        landmarks = extractor.extract_landmarks(img)
                        if landmarks:
                            all_data.append({
                                'letter': letter,
                                'landmarks': landmarks
                            })

                if len(all_data) >= 50:
                    X = np.array([d['landmarks'] for d in all_data])
                    y = np.array([d['letter'] for d in all_data])

                    trainer = LandmarkTrainer()
                    trainer.label_encoder.fit(y)
                    y_enc = trainer.label_encoder.transform(y)
                    results = trainer.train(X, y_enc)

                    trainer.save(
                        'models/enhanced_classifier.pkl',
                        'models/enhanced_scaler.pkl',
                        'models/enhanced_labels.pkl'
                    )

                    st.session_state.training_done = True
                    st.session_state.accuracy = results['test_accuracy']
                    st.success(f"✅ Training selesai! Accuracy: {results['test_accuracy']:.1%}")

    with col2:
        if st.session_state.training_done:
            st.metric("Accuracy", f"{st.session_state.accuracy:.1%}")
            if st.button("🔄 Retrain"):
                st.session_state.training_done = False
                st.rerun()
        else:
            st.info("Belum ada model trained")
