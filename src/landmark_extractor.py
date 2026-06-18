"""
Landmark Extractor - BISINDO Landmark-Based Approach
Extract 63 features (21 landmarks × 3 coordinates) from images
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
import json
import urllib.request

class LandmarkExtractor:
    def __init__(self, static_image_mode=True, max_hands=1):
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.hands = None
        self.frame_timestamp_ms = 0

    def _load_mediapipe(self):
        """Lazy load MediaPipe to avoid import error"""
        if self.hands is not None:
            return

        import mediapipe as mp

        # Download model if needed
        model_path = "/tmp/hand_landmarker.task"
        if not os.path.exists(model_path):
            print("📥 Downloading MediaPipe hand landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path
            )
            print("✅ Model downloaded!")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if self.static_image_mode:
            running_mode = VisionRunningMode.IMAGE
        else:
            running_mode = VisionRunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hands = HandLandmarker.create_from_options(options)

    def extract_landmarks(self, image):
        """Extract 63 features from image (21 landmarks × 3 coordinates)"""
        import mediapipe as mp

        self._load_mediapipe()

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect hands
        if self.static_image_mode:
            result = self.hands.detect(mp_image)
        else:
            self.frame_timestamp_ms += 1
            result = self.hands.detect_for_video(mp_image, self.frame_timestamp_ms)

        if not result.hand_landmarks:
            return None

        # Extract 63 features (21 landmarks × 3 coordinates)
        landmarks = []
        hand = result.hand_landmarks[0]

        for landmark in hand:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return landmarks

    def extract_from_dataset(self, dataset_path, output_path):
        """Extract landmarks from BISINDO dataset"""
        print(f"📂 Extracting landmarks from: {dataset_path}")

        all_data = []
        letter_counts = {}

        # Process train and test directories
        for split in ['train', 'test']:
            split_path = Path(dataset_path) / split
            if not split_path.exists():
                print(f"⚠️  Directory not found: {split_path}")
                continue

            # Get all jpg files in this directory
            # Structure: train/A.275ba73c-....jpg (letter is first char before '.')
            images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))

            print(f"   Processing {split}: {len(images)} images...")

            for img_path in images:
                # Extract letter from filename (e.g., "A.275ba73c.jpg" → "A")
                filename = img_path.stem
                letter = filename[0].upper()

                if not letter.isalpha():
                    continue

                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"⚠️  Failed to read: {img_path}")
                    continue

                # Extract landmarks
                landmarks = self.extract_landmarks(image)

                if landmarks:
                    all_data.append({
                        'letter': letter,
                        'image_path': str(img_path),
                        'landmarks': landmarks,
                        'split': split
                    })
                    letter_counts[letter] = letter_counts.get(letter, 0) + 1
                else:
                    print(f"⚠️  No hand detected: {img_path}")

        print(f"\n📊 Extraction Results:")
        print(f"   Total samples extracted: {len(all_data)}")
        print(f"   Letters detected: {len(letter_counts)}")

        for letter in sorted(letter_counts.keys()):
            print(f"   {letter}: {letter_counts[letter]} samples")

        if len(all_data) == 0:
            print("\n❌ No landmarks extracted! Check if MediaPipe model downloaded correctly.")
            return []

        # Save to CSV
        self.save_to_csv(all_data, output_path)

        # Save metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_samples': len(all_data),
                'letter_counts': letter_counts,
                'features_per_sample': 63,
                'source': 'BISINDO Dataset'
            }, f, indent=2)

        print(f"\n✅ Saved to: {output_path}")
        print(f"   Metadata: {metadata_path}")

        return all_data

    def save_to_csv(self, data, output_path):
        """Save extracted landmarks to CSV file"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['letter', 'image_path', 'split']
            for i in range(21):
                header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
            writer.writerow(header)

            # Data rows
            for item in data:
                row = [item['letter'], item['image_path'], item['split']]
                row.extend(item['landmarks'])
                writer.writerow(row)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract landmarks from BISINDO dataset')
    parser.add_argument('--dataset', default='dataset', help='Dataset path')
    parser.add_argument('--output', default='dataset/landmarks.csv', help='Output CSV path')
    args = parser.parse_args()

    # Create extractor
    extractor = LandmarkExtractor(static_image_mode=True)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Extract landmarks
    extractor.extract_from_dataset(args.dataset, args.output)


if __name__ == '__main__':
    main()
