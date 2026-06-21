#!/usr/bin/env python3
"""
Fast BISINDO Data Collector - Terminal Version
Collect 2000 samples per letter directly to CSV
"""

import cv2
import csv
import os
import time
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# Configuration
CSV_PATH = '/home/kevin/bisindo-bridge/dataset/landmarks_captured_v2.csv'
TARGET_SAMPLES_PER_LETTER = 50000
COLLECTION_INTERVAL_MS = 1  # 1ms = 1000fps (max speed)

# MediaPipe setup (Tasks API)
def get_hand_model_path():
    """Get path to hand landmarker model"""
    model_path = os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task")
    if os.path.exists(model_path):
        return model_path
    return None

base_options = python.BaseOptions(model_asset_path=get_hand_model_path())
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

def get_csv_fields():
    """67-col schema (2-hand only; hand 2 not stored, padded at inference time)."""
    fields = ['letter', 'image_path', 'split', 'num_hands', 'contributor']
    for i in range(21):
        fields.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
    return fields

def collect_data():
    """Main collection loop"""
    # Get contributor name
    contributor = input("Enter your name (contributor): ").strip() or "Anonymous"

    # Check if CSV exists
    file_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0

    # Open CSV for appending
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_fields())

        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            print(f"Created new CSV: {CSV_PATH}")
        else:
            print(f"Opened existing CSV: {CSV_PATH}")

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*60)
        print("BISINDO Fast Data Collector")
        print("="*60)
        print(f"Contributor: {contributor}")
        print(f"Target: {TARGET_SAMPLES_PER_LETTER} samples per letter")
        print(f"Interval: {COLLECTION_INTERVAL_MS}ms")
        print("\nControls:")
        print("  a-z: Select letter to collect")
        print("  SPACE: Start/Stop collection")
        print("  q: Quit")
        print("="*60)

        current_letter = 'A'
        is_collecting = False
        samples_collected = {chr(i): 0 for i in range(65, 91)}  # A-Z

        # Count existing samples
        if file_exists:
            with open(CSV_PATH, 'r') as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    letter = row.get('letter')
                    if letter and letter in samples_collected:
                        samples_collected[letter] += 1

        print(f"\nExisting samples: {sum(samples_collected.values())} total")
        for letter in sorted(samples_collected.keys()):
            if samples_collected[letter] > 0:
                print(f"  {letter}: {samples_collected[letter]}")

        print(f"\nStarting collection. Select letter (a-z) and press SPACE to begin...")

        last_capture_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame!")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw landmarks (manual drawing since we're not using drawing_utils)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw connections
                    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
                    for start, end in connections:
                        start_point = (int(hand_landmarks[start].x * frame.shape[1]), int(hand_landmarks[start].y * frame.shape[0]))
                        end_point = (int(hand_landmarks[end].x * frame.shape[1]), int(hand_landmarks[end].y * frame.shape[0]))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                    # Draw points
                    for lm in hand_landmarks:
                        point = (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                        cv2.circle(frame, point, 4, (0, 0, 255), -1)

            # Display info
            num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
            cv2.putText(frame, f"Letter: {current_letter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected[current_letter]}/{TARGET_SAMPLES_PER_LETTER}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {'COLLECTING' if is_collecting else 'PAUSED'}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if is_collecting else (0, 0, 255), 2)
            cv2.putText(frame, f"Hands: {num_hands}",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('BISINDO Collector', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                is_collecting = not is_collecting
                print(f"\n{'Started' if is_collecting else 'Stopped'} collection for letter {current_letter}")
            elif 97 <= key <= 122:  # a-z
                current_letter = chr(key).upper()
                print(f"\nSelected letter: {current_letter}")
                print(f"Progress: {samples_collected[current_letter]}/{TARGET_SAMPLES_PER_LETTER}")

            # Collect data
            if is_collecting and results.hand_landmarks:
                current_time = time.time() * 1000  # Convert to ms

                if current_time - last_capture_time >= COLLECTION_INTERVAL_MS:
                    # Check if we've reached target
                    if samples_collected[current_letter] >= TARGET_SAMPLES_PER_LETTER:
                        print(f"\n✓ Letter {current_letter} complete! ({TARGET_SAMPLES_PER_LETTER} samples)")
                        is_collecting = False
                        continue

                    # Extract landmarks
                    num_hands = len(results.hand_landmarks)

                    # Hand 1
                    hand1 = results.hand_landmarks[0]
                    hand1_data = []
                    for lm in hand1:
                        hand1_data.extend([lm.x, lm.y, lm.z])

                    # Hand 2 (if exists)
                    hand2_data = []
                    if num_hands > 1:
                        hand2 = results.hand_landmarks[1]
                        for lm in hand2:
                            hand2_data.extend([lm.x, lm.y, lm.z])
                    else:
                        hand2_data = [0.0] * 63  # Fill with zeros

                    # Create row (67-col schema, 2-hand only, hand1 stored, hand2 implied)
                    row = {
                        'letter': current_letter,
                        'image_path': f'cap_{current_letter}_{samples_collected[current_letter]}',
                        'split': 'train',
                        'num_hands': 2,
                        'contributor': contributor
                    }

                    for i in range(21):
                        row[f'lm{i}_x'] = hand1_data[i*3]
                        row[f'lm{i}_y'] = hand1_data[i*3+1]
                        row[f'lm{i}_z'] = hand1_data[i*3+2]

                    # Write to CSV
                    writer.writerow(row)
                    samples_collected[current_letter] += 1
                    last_capture_time = current_time

                    # Print progress every 100 samples
                    if samples_collected[current_letter] % 100 == 0:
                        print(f"  {current_letter}: {samples_collected[current_letter]}/{TARGET_SAMPLES_PER_LETTER}")

        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()

        print("\n" + "="*60)
        print("Collection Summary")
        print("="*60)
        total = sum(samples_collected.values())
        print(f"Total samples collected: {total}")
        print(f"Samples per letter:")
        for letter in sorted(samples_collected.keys()):
            if samples_collected[letter] > 0:
                print(f"  {letter}: {samples_collected[letter]}")
        print("="*60)

if __name__ == "__main__":
    collect_data()
