"""
Landmark Classifier - Inference model for real-time prediction
Uses extracted landmarks for fast, accurate gesture recognition
Updated for MediaPipe Tasks Python API
"""

import os
import pickle
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LandmarkClassifier:
    def __init__(self, model_path=None, scaler_path=None, label_path=None):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hand_landmarker = None

        if model_path and scaler_path and label_path:
            self.load(model_path, scaler_path, label_path)

    def load(self, model_path, scaler_path, label_path):
        """Load trained model and encoders"""
        print(f"📂 Loading model from: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(label_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Initialize MediaPipe HandLandmarker
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        base_options = python.BaseOptions(model_asset_path=self._get_hand_model_path())
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        print(f"   ✅ Model loaded: {len(self.label_encoder.classes_)} classes")

    def _get_hand_model_path(self):
        """Get path to hand landmarker model"""
        model_path = os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task")
        if os.path.exists(model_path):
            return model_path
        return None  # Will try bundled model

    def extract_landmarks(self, frame):
        """Extract landmarks from video frame"""
        if self.hand_landmarker is None:
            return None

        # Convert OpenCV frame (BGR) to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        result = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Extract 63 values (21 landmarks × 3 coords)
        landmarks = []
        hand = result.hand_landmarks[0]

        for landmark in hand:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return landmarks

    def extract_landmarks_from_image(self, image_path):
        """Extract landmarks from image file"""
        if self.hand_landmarker is None:
            return None

        image = cv2.imread(image_path)
        return self.extract_landmarks(image)

    def predict(self, landmarks):
        """Predict letter from landmarks"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded!")

        landmarks = np.array(landmarks).reshape(1, -1)

        # Scale
        landmarks_scaled = self.scaler.transform(landmarks)

        # Predict
        prediction = self.model.predict(landmarks_scaled)
        probabilities = self.model.predict_proba(landmarks_scaled)[0]

        letter = self.label_encoder.inverse_transform(prediction)[0]
        confidence = float(probabilities[prediction[0]])

        return {
            'letter': letter,
            'confidence': confidence,
            'probabilities': {
                self.label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }

    def predict_top_k(self, landmarks, k=3):
        """Get top-k predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded!")

        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = self.scaler.transform(landmarks)

        prediction = self.model.predict(landmarks_scaled)
        probabilities = self.model.predict_proba(landmarks_scaled)[0]

        # Get top-k indices
        top_k_idx = np.argsort(probabilities)[-k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append({
                'letter': self.label_encoder.classes_[idx],
                'confidence': float(probabilities[idx])
            })

        return results

    def draw_landmarks(self, frame, landmarks, prediction=None):
        """Draw landmarks on frame"""
        if landmarks is None:
            return frame

        h, w, c = frame.shape
        landmark_list = []

        for i in range(21):
            x = int(landmarks[i*3] * w)
            y = int(landmarks[i*3+1] * h)
            landmark_list.append((x, y))

        # Draw connections (MediaPipe hand skeleton)
        connections = [
            (0,1),(1,2),(2,3),(3,4),        # thumb
            (0,5),(5,6),(6,7),(7,8),        # index
            (0,9),(9,10),(10,11),(11,12),    # middle
            (0,13),(13,14),(14,15),(15,16),  # ring
            (0,17),(17,18),(18,19),(19,20),  # pinky
            (5,9),(9,13),(13,17)             # palm
        ]

        for start_idx, end_idx in connections:
            start = landmark_list[start_idx]
            end = landmark_list[end_idx]
            cv2.line(frame, start, end, (0, 255, 0), 2)

        # Draw points
        for point in landmark_list:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Draw prediction
        if prediction:
            cv2.putText(frame, f"{prediction['letter']} ({prediction['confidence']:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def close(self):
        """Clean up resources"""
        if self.hand_landmarker:
            self.hand_landmarker.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test landmark classifier')
    parser.add_argument('--model', default='models/landmark_classifier.pkl',
                        help='Model path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    args = parser.parse_args()

    # Load classifier
    base_path = args.model.replace('.pkl', '')
    classifier = LandmarkClassifier(
        f'{base_path}.pkl',
        f'{base_path}_scaler.pkl',
        f'{base_path}_labels.pkl'
    )

    # Open camera
    cap = cv2.VideoCapture(args.camera)

    print("\n🎥 Running real-time gesture recognition...")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks
        landmarks = classifier.extract_landmarks(frame)

        if landmarks:
            # Predict
            prediction = classifier.predict(landmarks)
            print(f"\rPredicted: {prediction['letter']} ({prediction['confidence']:.2f})", end='')

            # Draw
            frame = classifier.draw_landmarks(frame, landmarks, prediction)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Landmark Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    classifier.close()


if __name__ == '__main__':
    main()
