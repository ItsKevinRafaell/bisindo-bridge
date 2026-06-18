"""
BISINDO Gesture Detector
========================
Real-time hand gesture recognition using:
- MediaPipe Hands for hand detection
- MobileNetV2 (V2 trained model) for gesture classification
- Letter buffering to spell words (H-A-L-O -> HALO)
"""

import cv2
import numpy as np
import json
import os
import time
from collections import deque

# Lazy imports for heavy dependencies
_tf = None
_mediapipe = None
_preprocess_input = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def _get_mediapipe():
    global _mediapipe
    if _mediapipe is None:
        import mediapipe as mp
        _mediapipe = mp
    return _mediapipe


def _get_preprocess_input():
    global _preprocess_input
    if _preprocess_input is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        _preprocess_input = preprocess_input
    return _preprocess_input


class GestureDetector:
    """Real-time BISINDO gesture detector."""

    def __init__(self, model_path, label_map_path, confidence_threshold=0.60):
        """
        Initialize the gesture detector.

        Args:
            model_path: Path to the trained .keras model file
            label_map_path: Path to label_map_v2.json
            confidence_threshold: Minimum confidence for prediction (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.img_size = (224, 224)

        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)

        self.classes = self.label_map['class_names']
        self.idx_to_class = {int(k): v for k, v in self.label_map['idx_to_class'].items()}
        self.num_classes = self.label_map['num_classes']

        # Load model (lazy)
        self.model = None
        self.model_path = model_path

        # Initialize MediaPipe (lazy)
        self.hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.frame_timestamp_ms = 0
        self.last_word = ""

        # Letter buffer for spelling words
        self.letter_buffer = deque(maxlen=10)  # Store last 10 predictions
        self.current_word = ""
        self.last_letter = ""
        self.last_letter_time = 0
        self.letter_hold_time = 1.0  # seconds to hold before adding to word
        self.word_timeout = 3.0  # seconds of no detection to reset word

        # Smoothing
        self.prediction_history = deque(maxlen=5)  # Smooth over last 5 frames
        self.last_detection_time = 0
        self.missed_frames = 0  # Grace period counter for hand detection misses
        self.missed_frames_threshold = 5  # Only clear history after this many misses

        # Performance profiling
        self.timing_stats = {
            'detect_hand': [],
            'crop_preprocess': [],
            'predict': [],
            'total': []
        }
        self.frame_count = 0
        self.profile_interval = 30  # Print stats every N frames

    def _load_model(self):
        """Lazy load the TensorFlow model."""
        if self.model is None:
            tf = _get_tf()
            self.model = tf.keras.models.load_model(self.model_path)

    def _load_mediapipe(self):
        """Lazy load MediaPipe."""
        if self.hands is None:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Download hand landmarker model if not exists
            import urllib.request
            model_path = "/tmp/hand_landmarker.task"
            if not os.path.exists(model_path):
                print("Downloading MediaPipe hand landmarker model...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                    model_path
                )

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hands = HandLandmarker.create_from_options(options)
            # Drawing done manually with cv2 (mp.solutions removed in MediaPipe 0.10.35)
            self.mp_drawing = None
            self.mp_drawing_styles = None

    def detect_hand(self, frame):
        """
        Detect hand in frame using MediaPipe.

        Args:
            frame: BGR image from OpenCV

        Returns:
            hand_landmarks: MediaPipe hand landmarks or None
            annotated_frame: Frame with hand landmarks drawn
        """
        import mediapipe as mp

        self._load_mediapipe()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands (increment timestamp for VIDEO mode)
        self.frame_timestamp_ms += 1
        result = self.hands.detect_for_video(mp_image, self.frame_timestamp_ms)

        annotated_frame = frame.copy()

        if result.hand_landmarks:
            # API baru return list of list (per hand)
            landmarks = result.hand_landmarks[0]

            # Draw landmarks
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

            # Return landmarks dalam format yang compatible dengan crop_hand_region
            return landmarks, annotated_frame

        return None, annotated_frame

    def crop_hand_region(self, frame, hand_landmarks, padding=0.15):
        """
        Crop hand region from frame using MediaPipe landmarks.

        Args:
            frame: BGR image
            hand_landmarks: MediaPipe hand landmarks (list of NormalizedLandmark)
            padding: Extra padding around hand bounding box

        Returns:
            cropped: Cropped and resized hand image (224x224)
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        h, w, _ = frame.shape

        # Get bounding box from landmarks (API baru: list of NormalizedLandmark)
        x_coords = [lm.x * w for lm in hand_landmarks]
        y_coords = [lm.y * h for lm in hand_landmarks]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        pad_x = int(bbox_w * padding)
        pad_y = int(bbox_h * padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        # Crop
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None, (0, 0, 0, 0)

        # Resize to model input size
        cropped = cv2.resize(cropped, self.img_size)

        return cropped, (x1, y1, x2, y2)

    def preprocess(self, image):
        """
        Preprocess image for model input (matching V2 training).

        Args:
            image: BGR image (224x224)

        Returns:
            processed: Preprocessed image ready for model (1, 224, 224, 3)
        """
        preprocess_fn = _get_preprocess_input()

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MobileNetV2 preprocessing (same as V2 training)
        image = image.astype(np.float32)
        image = preprocess_fn(image)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, hand_image):
        """
        Predict gesture from cropped hand image.

        Args:
            hand_image: Cropped hand image (224x224)

        Returns:
            letter: Predicted letter (A-Z) or None
            confidence: Prediction confidence (0-1)
            all_probs: All class probabilities
        """
        self._load_model()

        # Preprocess
        processed = self.preprocess(hand_image)
        print(f"[DEBUG] Preprocessed image shape: {processed.shape}, range: [{processed.min():.3f}, {processed.max():.3f}]")

        # Predict
        predictions = self.model.predict(processed, verbose=0)
        probs = predictions[0]

        # Get top prediction
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]

        print(f"[DEBUG] Raw prediction: idx={predicted_idx}, confidence={confidence:.4f}")
        print(f"[DEBUG] Top 3 predictions:")
        top3 = np.argsort(probs)[::-1][:3]
        for idx in top3:
            letter = self.idx_to_class[idx]
            print(f"   {letter}: {probs[idx]:.4f}")

        # Add to history for smoothing
        self.prediction_history.append(predicted_idx)

        # DISABLED: Smooth prediction (majority vote over last N frames)
        # Temporarily disabled to see raw predictions
        # if len(self.prediction_history) >= 3:
        #     from collections import Counter
        #     counter = Counter(self.prediction_history)
        #     smoothed_idx = counter.most_common(1)[0][0]
        #     # Use confidence from the smoothed prediction (not last frame)
        #     confidence = probs[smoothed_idx]
        #     predicted_idx = smoothed_idx
        #     print(f"[DEBUG] After smoothing: idx={predicted_idx}, confidence={confidence:.4f}")

        letter = self.idx_to_class[predicted_idx]
        print(f"[DEBUG] Final prediction: {letter} with confidence {confidence:.4f}")

        if confidence < self.confidence_threshold:
            print(f"[DEBUG] Confidence {confidence:.4f} below threshold {self.confidence_threshold}, returning None")
            return None, confidence, probs

        print(f"[DEBUG] Returning letter: {letter}")
        return letter, confidence, probs

    def update_word(self, letter):
        """
        Update word buffer with new letter prediction.

        Logic:
        - If same letter detected for `letter_hold_time`, add to word
        - If no detection for `word_timeout`, reset word
        - Prevent adding same letter twice in a row

        Args:
            letter: Predicted letter or None
        """
        current_time = time.time()

        if letter is not None:
            self.last_detection_time = current_time
            self.missed_frames = 0  # Reset miss counter

            if letter == self.last_letter:
                # Same letter - check hold time
                if current_time - self.last_letter_time >= self.letter_hold_time:
                    # Only add if different from last letter in word (prevent duplicates)
                    if not self.current_word or self.current_word[-1] != letter:
                        self.current_word += letter
                    self.last_letter_time = current_time
                    self.last_letter = letter
            else:
                # New letter
                self.last_letter = letter
                self.last_letter_time = current_time
        else:
            # Grace period — don't clear history immediately on miss
            self.missed_frames += 1

            if self.missed_frames >= self.missed_frames_threshold:
                # Truly lost hand — clear prediction history
                self.prediction_history.clear()
                self.last_letter = ""

            # No detection - check word timeout
            if (current_time - self.last_detection_time > self.word_timeout and
                    len(self.current_word) > 0):
                # Save completed word
                self.last_word = self.current_word
                self.current_word = ""

    def process_frame(self, frame):
        """
        Process a single frame end-to-end.

        Args:
            frame: BGR image from camera

        Returns:
            result: dict with keys:
                - annotated_frame: Frame with landmarks and UI
                - letter: Current predicted letter (or None)
                - confidence: Prediction confidence
                - current_word: Word being spelled
                - bbox: Hand bounding box
                - hand_detected: Whether hand was detected
        """
        total_start = time.time()

        # Detect hand
        t0 = time.time()
        hand_landmarks, annotated_frame = self.detect_hand(frame)
        t_detect = time.time() - t0

        if hand_landmarks is None:
            print(f"[DEBUG] No hand detected")
            self.update_word(None)
            self._record_timing(t_detect, 0, 0, time.time() - total_start)
            return {
                'annotated_frame': annotated_frame,
                'letter': None,
                'confidence': 0.0,
                'current_word': self.current_word,
                'bbox': None,
                'hand_detected': False
            }

        print(f"[DEBUG] Hand detected, processing prediction...")

        # Crop hand region
        t0 = time.time()
        hand_image, bbox = self.crop_hand_region(frame, hand_landmarks)
        t_crop = time.time() - t0
        print(f"[DEBUG] Crop complete, bbox: {bbox}, image shape: {hand_image.shape if hand_image is not None else 'None'}")

        if hand_image is None:
            print(f"[DEBUG] Crop failed, returning")
            self.update_word(None)
            self._record_timing(t_detect, t_crop, 0, time.time() - total_start)
            return {
                'annotated_frame': annotated_frame,
                'letter': None,
                'confidence': 0.0,
                'current_word': self.current_word,
                'bbox': None,
                'hand_detected': True
            }

        # Predict gesture
        t0 = time.time()
        letter, confidence, probs = self.predict(hand_image)
        t_predict = time.time() - t0

        print(f"[DEBUG] Prediction: {letter}, confidence: {confidence:.3f}, threshold: {self.confidence_threshold}")

        # Update word buffer
        self.update_word(letter)

        # Draw UI on frame
        self._draw_ui(annotated_frame, letter, confidence, bbox)

        t_total = time.time() - total_start
        self._record_timing(t_detect, t_crop, t_predict, t_total)

        return {
            'annotated_frame': annotated_frame,
            'letter': letter,
            'confidence': confidence,
            'current_word': self.current_word,
            'bbox': bbox,
            'hand_detected': True
        }

    def _draw_ui(self, frame, letter, confidence, bbox):
        """Draw prediction UI on frame."""
        h, w, _ = frame.shape

        # Draw bounding box
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if letter else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw prediction text
        if letter:
            text = f"{letter} ({confidence:.0%})"
            color = (0, 255, 0)
        else:
            text = "Detecting..."
            color = (0, 255, 255)

        # Text background
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color, 3)

        # Draw current word
        if self.current_word:
            word_text = f"Word: {self.current_word}"
            (word_w, word_h), _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(frame, (10, 60), (20 + word_w, 70 + word_h + 10), (0, 0, 0), -1)
            cv2.putText(frame, word_text, (15, 65 + word_h), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 2)

        # Draw last letter indicator
        if self.last_letter and self.last_letter != letter:
            hold_progress = min(1.0, (time.time() - self.last_letter_time) / self.letter_hold_time)
            bar_width = int(200 * hold_progress)
            cv2.rectangle(frame, (10, h - 30), (10 + bar_width, h - 10), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, h - 30), (210, h - 10), (255, 255, 255), 2)

    def _record_timing(self, t_detect, t_crop, t_predict, t_total):
        """Record timing stats and print summary periodically."""
        self.timing_stats['detect_hand'].append(t_detect)
        self.timing_stats['crop_preprocess'].append(t_crop)
        self.timing_stats['predict'].append(t_predict)
        self.timing_stats['total'].append(t_total)
        self.frame_count += 1

        if self.frame_count % self.profile_interval == 0:
            self._print_timing_summary()

    def _print_timing_summary(self):
        """Print average timing for the last N frames."""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                recent = times[-self.profile_interval:]
                stats[key] = {
                    'avg': sum(recent) / len(recent) * 1000,  # ms
                    'min': min(recent) * 1000,
                    'max': max(recent) * 1000,
                }

        print(f"\n{'='*55}")
        print(f"⚡ Performance Profile (last {self.profile_interval} frames)")
        print(f"{'='*55}")
        print(f"{'Stage':<20} {'Avg (ms)':>10} {'Min':>8} {'Max':>8} {'% of total':>10}")
        print(f"{'-'*55}")

        total_avg = stats.get('total', {}).get('avg', 0)
        for key in ['detect_hand', 'crop_preprocess', 'predict', 'total']:
            if key in stats:
                pct = f"{stats[key]['avg']/total_avg*100:.0f}%" if total_avg and key != 'total' else "—"
                print(f"  {key:<18} {stats[key]['avg']:>8.1f} {stats[key]['min']:>8.1f} {stats[key]['max']:>8.1f} {pct:>10}")

        fps = 1000 / total_avg if total_avg else 0
        print(f"{'-'*55}")
        print(f"  Est. FPS: {fps:.1f}")
        print(f"{'='*55}\n")

        # Clear for next batch
        for key in self.timing_stats:
            self.timing_stats[key] = []

    def reset(self):
        """Reset word buffer and prediction history."""
        self.current_word = ""
        self.last_letter = ""
        self.last_letter_time = 0
        self.prediction_history.clear()

    def release(self):
        """Release resources."""
        if self.hands:
            self.hands.close()
