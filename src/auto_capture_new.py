"""
BISINDO Auto-Capture
Capture hand landmarks automatically from webcam
"""

import cv2
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Import MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode

# Config
OUTPUT_DIR = 'dataset'
OUTPUT_FILE = f'{OUTPUT_DIR}/landmarks_captured.csv'
COUNT_PER_LETTER = 50  # Target capture per letter

# Letter target (huruf yang mau di-capture)
TARGET_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class AutoCapture:
    def __init__(self):
        # Initialize MediaPipe
        print("📦 Loading MediaPipe...")
        base_options = python.BaseOptions(model_asset_path=os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task"))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # State
        self.current_letter = 'A'
        self.captured = {letter: 0 for letter in TARGET_LETTERS}
        self.last_hand_time = 0
        self.steady_count = 0
        self.steady_threshold = 30  # frames steady = capture
        self.last_landmarks = None

        # Load existing data
        self.load_existing()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def load_existing(self):
        """Load existing captured data"""
        if os.path.exists(OUTPUT_FILE):
            self.df = pd.read_csv(OUTPUT_FILE)
            # Count per letter
            for letter in TARGET_LETTERS:
                self.captured[letter] = len(self.df[self.df['letter'] == letter])
            print(f"📂 Loaded existing: {len(self.df)} samples")
        else:
            self.df = None
            print("📂 New dataset")

    def save_sample(self, letter, landmarks):
        """Save a landmark sample"""
        row = {
            'letter': letter,
            'image_path': f'captured_{letter}_{self.captured[letter]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'split': 'train',
        }
        for i, val in enumerate(landmarks):
            row[f'lm{i}_x'] = val[0]
            row[f'lm{i}_y'] = val[1]
            row[f'lm{i}_z'] = val[2]

        if self.df is None:
            self.df = pd.DataFrame([row])
        else:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        self.captured[letter] += 1

    def process_frame(self, frame):
        """Process frame and detect hand"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = [[p.x, p.y, p.z] for p in hand]
            self.last_hand_time = time.time()
            return landmarks, hand
        else:
            return None, None

    def draw_hand(self, frame, hand_landmarks, letter, captured_count):
        """Draw hand skeleton and info"""
        h, w = frame.shape[:2]

        if hand_landmarks:
            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]

            for start, end in connections:
                pt1 = hand_landmarks[start]
                pt2 = hand_landmarks[end]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw points
            for pt in hand_landmarks:
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # UI
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Letter: {letter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {captured_count}/{COUNT_PER_LETTER}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Hold hand steady to capture", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Progress bar
        progress = captured_count / COUNT_PER_LETTER
        cv2.rectangle(frame, (20, 115), (300, 130), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 115), (int(20 + 280 * progress), 130), (0, 255, 0), -1)

        # Instructions
        cv2.putText(frame, "Q/A-Z: Change letter | SPACE: Manual capture | ESC: Quit",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def run(self):
        """Main loop"""
        print("\n" + "="*50)
        print("🖐️ BISINDO Auto-Capture")
        print("="*50)
        print(f"Target: {COUNT_PER_LETTER} per letter")
        print("Controls:")
        print("  Q/A-Z: Change target letter")
        print("  SPACE: Manual capture")
        print("  S: Save and quit")
        print("  ESC: Quit without saving")
        print("="*50 + "\n")

        steady_frames = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process
            landmarks, hand_pts = self.process_frame(frame)

            # Check if hand is steady
            if landmarks:
                if self.last_landmarks is not None:
                    # Calculate movement
                    diff = np.mean(np.abs(np.array(landmarks) - np.array(self.last_landmarks)))
                    if diff < 0.01:  # Very steady
                        steady_frames += 1
                    else:
                        steady_frames = 0

                self.last_landmarks = landmarks
            else:
                steady_frames = 0
                self.last_landmarks = None

            # Auto capture if steady enough and need more
            current_count = self.captured[self.current_letter]
            if steady_frames >= self.steady_threshold and current_count < COUNT_PER_LETTER:
                self.save_sample(self.current_letter, landmarks)
                print(f"✅ Captured {self.current_letter}: {self.captured[self.current_letter]}/{COUNT_PER_LETTER}")
                steady_frames = 0

                # Auto advance if done
                if self.captured[self.current_letter] >= COUNT_PER_LETTER:
                    # Find next letter
                    for letter in TARGET_LETTERS:
                        if self.captured[letter] < COUNT_PER_LETTER:
                            self.current_letter = letter
                            break

            # Draw
            frame = self.draw_hand(frame, hand_pts, self.current_letter, current_count)
            cv2.imshow('BISINDO Auto-Capture', frame)

            # Key events
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\n❌ Quit without saving")
                break
            elif key == ord('s') or key == ord('S'):  # Save and quit
                self.save()
                break
            elif key == 32:  # SPACE - manual capture
                if landmarks and current_count < COUNT_PER_LETTER:
                    self.save_sample(self.current_letter, landmarks)
                    print(f"✅ Manual capture: {self.current_letter} {self.captured[self.current_letter]}")
            elif key in [ord(c) for c in TARGET_LETTERS] or key == ord('q'):  # Change letter
                if key == ord('q'):
                    self.current_letter = 'A'
                else:
                    self.current_letter = chr(key).upper()
                print(f"📌 Target letter: {self.current_letter} ({self.captured[self.current_letter]} captured)")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

    def save(self):
        """Save collected data"""
        if self.df is not None and len(self.df) > 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.df.to_csv(OUTPUT_FILE, index=False)
            print(f"\n💾 Saved {len(self.df)} samples to {OUTPUT_FILE}")
        else:
            print("\n❌ No data to save")


def main():
    capture = AutoCapture()
    capture.run()


if __name__ == '__main__':
    main()
