"""
Auto-Capture Tool - Kumpulin data BISINDO dengan cepat
Capture landmark-based samples secara otomatis dengan stability check
"""

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
import urllib.request
from pathlib import Path
import json

class AutoCapture:
    def __init__(self, target_letter='A', target_count=50, output_dir='dataset/collected'):
        self.target_letter = target_letter.upper()
        self.target_count = target_count
        self.output_dir = Path(output_dir) / self.target_letter
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.samples = []
        self.stability_buffer = []
        self.cap = None

        # MediaPipe
        self.hands = None
        self._load_mediapipe()

    def _load_mediapipe(self):
        """Load MediaPipe"""
        model_path = "/tmp/hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe model...")
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
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.hands = HandLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        """Extract 63 features dari frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hands.detect_for_video(mp_image, int(cv2.getTickCount()))

        if not result.hand_landmarks:
            return None, None

        landmarks = []
        hand = result.hand_landmarks[0]

        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])

        return landmarks, hand

    def check_stability(self, landmarks, threshold=0.02):
        """Check apakah tangan stabil (tidak tremor)"""
        if len(self.stability_buffer) < 5:
            self.stability_buffer.append(landmarks)
            return False

        # Remove oldest
        self.stability_buffer.pop(0)
        self.stability_buffer.append(landmarks)

        # Calculate variance
        buffer_array = np.array(self.stability_buffer)
        variances = np.var(buffer_array, axis=0)

        return np.mean(variances) < threshold

    def draw_landmarks(self, frame, hand, captured=False):
        """Draw landmarks on frame"""
        h, w = frame.shape[:2]
        landmark_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

        # Connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        for start, end in connections:
            cv2.line(frame, landmark_list[start], landmark_list[end],
                    (0, 255, 0) if not captured else (0, 255, 0), 2)

        for point in landmark_list:
            color = (0, 0, 255) if captured else (255, 0, 0)
            cv2.circle(frame, point, 5, color, -1)

        return frame

    def run(self):
        """Run auto-capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera!")
            return

        print(f"\n{'='*50}")
        print(f"AUTO-CAPTURE - Letter: {self.target_letter}")
        print(f"Target: {self.target_count} samples")
        print(f"Output: {self.output_dir}")
        print(f"{'='*50}")
        print("\nInstructions:")
        print("  - Tunjukkan tangan ke kamera")
        print("  - Tahan posisi ~0.5 detik untuk auto-capture")
        print("  - Tekan 'q' untuk quit dan simpan")
        print("  - Tekan 'r' untuk reset buffer")
        print()

        frame_count = 0
        last_capture_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            landmarks, hand = self.extract_landmarks(frame)

            # Status text
            status = "Mencari tangan..."
            color = (128, 128, 128)

            if landmarks and hand:
                stable = self.check_stability(landmarks)
                current_time = cv2.getTickCount() / cv2.getTickFrequency()

                # Auto-capture if stable
                if stable and current_time - last_capture_time > 0.5:
                    if len(self.samples) < self.target_count:
                        # Save sample
                        sample_path = self.output_dir / f"{self.target_letter}_{len(self.samples):04d}.jpg"
                        cv2.imwrite(str(sample_path), frame)

                        self.samples.append({
                            'path': str(sample_path),
                            'landmarks': landmarks
                        })
                        last_capture_time = current_time
                        print(f"  Captured {len(self.samples)}/{self.target_count} - {sample_path.name}")

                # Visual feedback
                if stable:
                    status = f"STABIL - {len(self.samples)}/{self.target_count}"
                    color = (0, 255, 0)
                    frame = self.draw_landmarks(frame, hand, captured=True)
                else:
                    status = f"Stabilizing... {len(self.samples)}/{self.target_count}"
                    color = (0, 255, 255)
                    frame = self.draw_landmarks(frame, hand, captured=False)
            else:
                status = f"Mencari tangan... ({len(self.samples)}/{self.target_count})"
                color = (0, 0, 255)

            # Draw UI
            cv2.rectangle(frame, (0, 0), (320, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Target: {self.target_letter}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {len(self.samples)}/{self.target_count}",
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Auto-Capture', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.stability_buffer = []
                print("Buffer reset!")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Save metadata
        if self.samples:
            self._save_samples()

        print(f"\n{'='*50}")
        print(f"Selesai! {len(self.samples)} samples disimpan ke {self.output_dir}")
        print(f"{'='*50}")

    def _save_samples(self):
        """Save samples metadata"""
        metadata = {
            'letter': self.target_letter,
            'count': len(self.samples),
            'samples': [{'path': s['path']} for s in self.samples]
        }

        meta_path = self.output_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved: {meta_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto-capture BISINDO gestures')
    parser.add_argument('--letter', default='A', help='Target letter (A-Z)')
    parser.add_argument('--count', type=int, default=50, help='Target sample count')
    parser.add_argument('--output', default='dataset/collected', help='Output directory')
    args = parser.parse_args()

    capturer = AutoCapture(
        target_letter=args.letter,
        target_count=args.count,
        output_dir=args.output
    )
    capturer.run()


if __name__ == '__main__':
    main()
