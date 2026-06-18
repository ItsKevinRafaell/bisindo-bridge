"""
Gesture Detector - BISINDO Landmark-Based Approach
Uses extracted landmarks (63 features) for fast gesture recognition
- MediaPipe Hands for hand detection
- Random Forest for classification
- Letter buffering to spell words (H-A-L-O -> HALO)
"""

import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
import urllib.request
import time

class GestureDetector:
    """Real-time BISINDO gesture detector using landmark-based approach."""

    def __init__(self, model_path=None, scaler_path=None, label_path=None,
                 confidence_threshold=0.5, letter_hold_time=1.0):
        """
        Initialize gesture detector

        Args:
            model_path: Path to Random Forest model
            scaler_path: Path to StandardScaler
            label_path: Path to LabelEncoder
            confidence_threshold: Minimum confidence to accept prediction
            letter_hold_time: Seconds to hold letter before accepting
        """
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.confidence_threshold = confidence_threshold
        self.letter_hold_time = letter_hold_time

        # State
        self.current_letter = ""
        self.letter_start_time = 0
        self.last_word = ""
        self.word_buffer = []
        self.frame_timestamp_ms = 0
        self.hands = None

        # Load model if paths provided
        if model_path and scaler_path and label_path:
            self.load_model(model_path, scaler_path, label_path)

    def load_model(self, model_path, scaler_path, label_path):
        """Load trained landmark classifier"""
        print(f"Loading landmark model from {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(label_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        print(f"  Loaded {len(self.label_encoder.classes_)} classes")

    def _load_mediapipe(self):
        """Lazy load MediaPipe"""
        if self.hands is not None:
            return

        # Download model if needed
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
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = HandLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        """Extract 63 features (21 landmarks × 3 coords) from frame"""
        self._load_mediapipe()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        self.frame_timestamp_ms += 1
        result = self.hands.detect_for_video(mp_image, self.frame_timestamp_ms)

        if not result.hand_landmarks:
            return None

        # Extract 63 features
        landmarks = []
        hand = result.hand_landmarks[0]

        for landmark in hand:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return landmarks

    def predict(self, landmarks):
        """Predict letter from landmarks"""
        if self.model is None or self.scaler is None:
            return None, 0

        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = self.scaler.transform(landmarks)

        prediction = self.model.predict(landmarks_scaled)
        probabilities = self.model.predict_proba(landmarks_scaled)[0]

        letter = self.label_encoder.inverse_transform(prediction)[0]
        confidence = float(probabilities[prediction[0]])

        return letter, confidence

    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        h, w, c = frame.shape
        landmark_list = []

        for i in range(21):
            x = int(landmarks[i*3] * w)
            y = int(landmarks[i*3+1] * h)
            landmark_list.append((x, y))

        # Draw connections (static list since mp.solutions removed in MediaPipe 0.10+)
        connections = [
            (0,1),(1,2),(2,3),(3,4),        # thumb
            (0,5),(5,6),(6,7),(7,8),        # index
            (0,9),(9,10),(10,11),(11,12),    # middle
            (0,13),(13,14),(14,15),(15,16),  # ring
            (0,17),(17,18),(18,19),(19,20),  # pinky
            (5,9),(9,13),(13,17),(0,17)       # palm
        ]

        for start_idx, end_idx in connections:
            start = landmark_list[start_idx]
            end = landmark_list[end_idx]
            cv2.line(frame, start, end, (0, 255, 0), 2)

        # Draw points
        for point in landmark_list:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def process_frame(self, frame):
        """Process video frame and update word buffer"""
        annotated_frame = frame.copy()
        h, w, c = frame.shape

        # Extract landmarks
        landmarks = self.extract_landmarks(frame)

        if landmarks is None:
            # No hand detected
            cv2.putText(annotated_frame, "No hand detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.current_letter = ""
            return {
                'annotated_frame': annotated_frame,
                'current_letter': None,
                'current_word': "",
                'confidence': 0
            }

        # Predict
        letter, confidence = self.predict(landmarks)

        # Draw landmarks
        self._draw_landmarks(annotated_frame, landmarks)

        # Show prediction
        color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
        cv2.putText(annotated_frame, f"Letter: {letter} ({confidence:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(annotated_frame, f"Word: {''.join(self.word_buffer)}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Letter buffering logic
        current_time = time.time()

        if letter and confidence >= self.confidence_threshold:
            if letter == self.current_letter:
                # Same letter - check if held long enough
                if current_time - self.letter_start_time >= self.letter_hold_time:
                    # Add to word buffer if new letter
                    if not self.word_buffer or self.word_buffer[-1] != letter:
                        self.word_buffer.append(letter)
                        self.letter_start_time = current_time
            else:
                # New letter detected
                self.current_letter = letter
                self.letter_start_time = current_time
        else:
            # Low confidence - reset
            self.current_letter = ""
            self.letter_start_time = current_time

        return {
            'annotated_frame': annotated_frame,
            'current_letter': letter,
            'current_word': ''.join(self.word_buffer),
            'confidence': confidence
        }

    def reset(self):
        """Reset word buffer"""
        self.current_letter = ""
        self.letter_start_time = 0
        self.last_word = ""
        self.word_buffer = []


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test landmark gesture detector')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    # Load detector
    detector = GestureDetector(
        model_path='models/landmark_classifier.pkl',
        scaler_path='models/landmark_classifier_scaler.pkl',
        label_path='models/landmark_classifier_labels.pkl',
        confidence_threshold=0.5,
        letter_hold_time=1.0
    )

    # Open camera
    cap = cv2.VideoCapture(args.camera)

    print("\n🎥 Running real-time gesture recognition (Landmark-Based)")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process
        result = detector.process_frame(frame)

        # Display
        cv2.imshow('Gesture Recognition', result['annotated_frame'])

        if result['current_word']:
            print(f"\rWord: {result['current_word']}", end='')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
