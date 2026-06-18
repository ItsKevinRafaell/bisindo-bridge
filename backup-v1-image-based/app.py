"""
BISINDO Two-Way Communication Bridge - Ultra Minimalist v4
=========================================================
Minimalist UI using native Streamlit components

Run: streamlit run app.py --server.port 8000
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import os

# Page config
st.set_page_config(
    page_title="Teman Bicara Kamu",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "bisindo_mobilenetv2_v4_best.keras")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map_v4.json")

# ============================================================
# CUSTOM CSS - Ultra Minimalist
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* Page background */
    .stApp {
        background: #0F172A !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1E293B !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #1E293B; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 2px; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Progress grid */
    .progress-grid {
        display: grid;
        grid-template-columns: repeat(13, 1fr);
        gap: 4px;
        margin-top: 8px;
    }

    .progress-cell {
        text-align: center;
        padding: 4px;
        background: #334155;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #94A3B8;
    }

    /* Chat bubble */
    .chat-bubble {
        background: #3B82F6;
        color: white;
        padding: 10px 14px;
        border-radius: 12px;
        border-bottom-right-radius: 4px;
        font-size: 0.9rem;
        max-width: 85%;
        margin-left: auto;
        margin-bottom: 8px;
    }

    /* Section containers */
    .section-container {
        background: #1E293B;
        border-radius: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .section-header {
        padding: 14px 20px;
        border-bottom: 1px solid #334155;
        font-weight: 600;
        color: #F8FAFC;
        font-size: 0.9rem;
    }

    .section-body {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px;
    }

    .section-body-scroll {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
    }

    .section-footer {
        padding: 14px 20px;
        border-top: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'tts' not in st.session_state:
    st.session_state.tts = None
if 'completed_words' not in st.session_state:
    st.session_state.completed_words = []
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.30
if 'letter_hold_time' not in st.session_state:
    st.session_state.letter_hold_time = 1.0
if 'auto_speak' not in st.session_state:
    st.session_state.auto_speak = True
if 'last_spoken_word' not in st.session_state:
    st.session_state.last_spoken_word = ""

# ============================================================
# LAZY LOADERS
# ============================================================
@st.cache_resource
def load_detector(model_path, label_map_path):
    from src.gesture_detector import GestureDetector
    return GestureDetector(
        model_path=model_path,
        label_map_path=label_map_path,
        confidence_threshold=st.session_state.confidence_threshold
    )

@st.cache_resource
def load_tts():
    from src.tts_engine import TTSEngine
    return TTSEngine(language='id')

# ============================================================
# SIDEBAR: Settings (Always visible)
# ============================================================
with st.sidebar:
    st.header("Settings")

    st.divider()

    # Data Collection
    st.subheader("Data Collection")
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    collect_letter = st.selectbox("Target Letter", letters, index=0)
    batch = st.number_input("Batch Number", min_value=1, value=1, step=1)
    target = st.slider("Target Count", min_value=5, max_value=50, value=10, step=5)
    auto_capture = st.toggle("Auto-capture")

    st.divider()

    # Detection Settings
    st.subheader("Detection Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.99,
        value=st.session_state.confidence_threshold,
        step=0.05
    )
    st.session_state.confidence_threshold = threshold

    hold_time = st.slider(
        "Letter Hold Time (sec)",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.letter_hold_time,
        step=0.25
    )
    st.session_state.letter_hold_time = hold_time

    auto_speak = st.toggle("Auto-speak words", value=st.session_state.auto_speak)
    st.session_state.auto_speak = auto_speak

    st.divider()

    # Progress
    st.subheader("Progress (A-Z)")
    cols = st.columns(13)
    for i, letter in enumerate(letters):
        with cols[i % 13]:
            st.markdown(f'<div class="progress-cell">{letter}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("Phase 2-3 - Gesture to Audio")

# ============================================================
# TOP HEADER
# ============================================================
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("🤟")
with col2:
    st.markdown("### Teman Bicara Kamu")

st.divider()

# ============================================================
# MAIN LAYOUT (2 columns)
# ============================================================
col_left, col_right = st.columns([7, 3])

# ============================================================
# LEFT COLUMN: CAMERA (70%)
# ============================================================
with col_left:
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)

        # Header
        st.markdown('<div class="section-header">Live Camera Feed</div>', unsafe_allow_html=True)

        # Body
        st.markdown('<div class="section-body">', unsafe_allow_html=True)

        if st.session_state.camera_running:
            st.success("Camera Active - Processing video feed...")
        else:
            st.info("**Camera Off**  \nTunjukkan isyarat BISINDO ke kamera untuk memulai")

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer - Start/Stop button ONLY
        st.markdown('<div class="section-footer">', unsafe_allow_html=True)

        if st.session_state.camera_running:
            if st.button("Stop Camera", type="primary", use_container_width=True):
                st.session_state.camera_running = False
                if st.session_state.detector:
                    st.session_state.detector.reset()
                st.rerun()
        else:
            if st.button("Start Camera", type="primary", use_container_width=True):
                st.session_state.camera_running = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RIGHT COLUMN: HISTORY (30%)
# ============================================================
with col_right:
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)

        # Header
        st.markdown('<div class="section-header">Riwayat Terjemahan</div>', unsafe_allow_html=True)

        # Body
        st.markdown('<div class="section-body-scroll">', unsafe_allow_html=True)

        if st.session_state.completed_words:
            for word in st.session_state.completed_words:
                st.markdown(f"""
                <div class="chat-bubble">
                    {word}
                    <div style="font-size: 0.65rem; color: rgba(255,255,255,0.6); margin-top: 4px; text-align: right;">{time.strftime('%H:%M')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; color: #94A3B8; padding: 40px 20px;">
                <div>Mulai berbicara dengan isyarat...</div>
                <div style="color: #64748B; font-size: 0.8rem; margin-top: 8px;">Aktifkan kamera dan tunjukkan gesture BISINDO</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer - Clear & Speak All
        st.markdown('<div class="section-footer">', unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Clear", use_container_width=True):
                if st.session_state.detector:
                    st.session_state.detector.reset()
                st.session_state.completed_words = []
                st.rerun()

        with col_btn2:
            if st.button("Speak All", use_container_width=True):
                if st.session_state.completed_words:
                    all_text = " ".join(st.session_state.completed_words)
                    try:
                        tts = load_tts()
                        audio = tts.text_to_audio(all_text)
                        if audio:
                            st.audio(audio, format='audio/mp3')
                    except:
                        pass

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# CAMERA LOOP
# ============================================================
if st.session_state.camera_running:
    # Check model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)):
        st.error("Model files not found!")
        st.session_state.camera_running = False
        st.stop()

    try:
        detector = load_detector(MODEL_PATH, LABEL_MAP_PATH)
        detector.confidence_threshold = st.session_state.confidence_threshold
        detector.letter_hold_time = st.session_state.letter_hold_time
        st.session_state.detector = detector

        tts = load_tts()
        st.session_state.tts = tts
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.session_state.camera_running = False
        st.stop()

    # Video placeholder
    video_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open camera!")
        st.session_state.camera_running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera loop
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result = detector.process_frame(frame)

        # Convert for display
        display_frame = cv2.cvtColor(result['annotated_frame'], cv2.COLOR_BGR2RGB)

        # Display video
        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

        # Check for completed word
        if (detector.current_word == "" and
            hasattr(detector, 'last_word') and
            detector.last_word and
            detector.last_word != st.session_state.last_spoken_word):

            completed_word = detector.last_word
            st.session_state.completed_words.append(completed_word)
            st.session_state.last_spoken_word = completed_word
            detector.last_word = ""

            # Auto-speak
            if st.session_state.auto_speak and st.session_state.tts:
                audio = st.session_state.tts.text_to_audio(completed_word)
                if audio:
                    st.audio(audio, format='audio/mp3')

            st.rerun()

        time.sleep(0.03)

    cap.release()
