"""
BISINDO Auto-Capture - 2 Hand Support
Capture hand landmarks from webcam (1 or 2 hands)
"""

import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

OUTPUT_DIR = 'dataset'
OUTPUT_FILE = f'{OUTPUT_DIR}/landmarks_captured_v2.csv'
TARGET_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# 67-col schema: letter, image_path, split, num_hands, contributor, lm0_x..lm20_z
NUM_FEATURES = 63  # store hand 1 only; hand 2 not persisted

def main():
    print("📦 Loading MediaPipe...")

    base_options = python.BaseOptions(
        model_asset_path=os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task")
    )
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_letter = 'A'
    captured = {l: 0 for l in TARGET_LETTERS}
    df = None
    timestamp = 0
    last_capture_time = 0
    CAPTURE_DELAY_MS = 50  # Capture max every 50ms (20fps) - MUCH FASTER

    print("\n" + "="*50)
    print("🖐️ BISINDO Auto-Capture - 2 HAND MODE")
    print("="*50)
    print("Detects 1 or 2 hands simultaneously")
    print("A-Z = Ganti huruf | S = Save | ESC = Quit")
    print("="*50 + "\n")

    # Hand skeleton connections
    connections = [
        (0,1),(1,2),(2,3),(3,4),       # thumb
        (0,5),(5,6),(6,7),(7,8),       # index
        (0,9),(9,10),(10,11),(11,12),  # middle
        (0,13),(13,14),(14,15),(15,16),# ring
        (0,17),(17,18),(18,19),(19,20),# pinky
        (5,9),(9,13),(13,17)           # palm
    ]

    # Colors for each hand
    hand_colors = [(0, 255, 0), (0, 165, 255)]  # green, orange
    hand_labels = ['L', 'R']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 33
        result = detector.detect_for_video(mp_image, timestamp)

        h, w = frame.shape[:2]
        num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

        # Build 126 features (2 hands × 63 features)
        all_landmarks = None
        if num_hands > 0:
            all_landmarks = []
            for hand_idx in range(min(num_hands, 2)):
                hand = result.hand_landmarks[hand_idx]
                for p in hand:
                    all_landmarks.extend([p.x, p.y, p.z])
                # Draw skeleton
                color = hand_colors[hand_idx]
                for s, e in connections:
                    p1 = hand[s]
                    p2 = hand[e]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)),
                            (int(p2.x*w), int(p2.y*h)), color, 2)
                for pt in hand:
                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 4, (0,0,255), -1)
                # Label hand
                wrist = hand[0]
                cv2.putText(frame, hand_labels[hand_idx],
                           (int(wrist.x*w)-15, int(wrist.y*h)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Pad second hand with zeros if only 1 hand detected
            if num_hands == 1:
                all_landmarks.extend([0.0] * 63)

        count = captured[current_letter]

        # Capture every 200ms when hands detected
        current_time = timestamp
        if all_landmarks and (current_time - last_capture_time) >= CAPTURE_DELAY_MS:
            row = {
                'letter': current_letter,
                'image_path': f'cap_{current_letter}_{count}',
                'split': 'train',
                'num_hands': 2,
                'contributor': os.environ.get('BISINDO_CONTRIBUTOR', 'desktop')
            }
            for i in range(21):
                # Hand 1 (only hand stored in 67-col schema)
                row[f'lm{i}_x'] = all_landmarks[i*3]
                row[f'lm{i}_y'] = all_landmarks[i*3+1]
                row[f'lm{i}_z'] = all_landmarks[i*3+2]

            if df is None:
                df = pd.DataFrame([row])
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            captured[current_letter] += 1
            print(f"✅ {current_letter}: {captured[current_letter]} (hands={num_hands})")
            last_capture_time = current_time

        # Draw UI overlay
        cv2.rectangle(frame, (10,10), (350,110), (0,0,0), -1)
        cv2.putText(frame, f"Letter: {current_letter}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Captured: {count}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        hands_text = f"Hands: {num_hands}/2" if num_hands > 0 else "No hand"
        hands_color = (0,255,0) if num_hands >= 1 else (0,0,255)
        cv2.putText(frame, hands_text, (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hands_color, 1)
        progress = min(count / 100, 1.0)
        cv2.rectangle(frame, (20,90), (200,105), (50,50,50), -1)
        cv2.rectangle(frame, (20,90), (int(20+180*progress),105), (0,255,0), -1)
        cv2.putText(frame, "A-Z=letter | S=save", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

        cv2.imshow('BISINDO Auto-Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            if df is not None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                df.to_csv(OUTPUT_FILE, index=False)
                print(f"\n💾 Saved {len(df)} samples")
            break
        elif (65 <= key <= 90) or (97 <= key <= 122):
            # A-Z (65-90) atau a-z (97-122)
            current_letter = chr(key).upper()
            print(f"🔤 Switched to: {current_letter}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
