"""
Enhanced Landmark Extractor with Feature Engineering
Adds distance, angle, and ratio features for better accuracy
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
import json
import urllib.request
import mediapipe as mp

class EnhancedLandmarkExtractor:
    """Extract enhanced features from hand landmarks"""

    # Landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        self.hands = None
        self.frame_timestamp_ms = 0
        self._load_mediapipe()

    def _load_mediapipe(self):
        """Load MediaPipe model"""
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
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hands = HandLandmarker.create_from_options(options)

    def extract(self, image):
        """Extract enhanced features from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.hands.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Get 21 landmarks (63 base features)
        hand = result.hand_landmarks[0]
        base_features = []
        for lm in hand:
            base_features.extend([lm.x, lm.y, lm.z])

        # Extract enhanced features
        enhanced = self._extract_enhanced_features(hand)

        return base_features + enhanced

    def _extract_enhanced_features(self, hand):
        """Extract additional features from landmarks"""
        features = []

        # Convert to list for easier access
        lm = [(hand[i].x, hand[i].y, hand[i].z) for i in range(21)]

        # === DISTANCE FEATURES ===
        # Finger lengths
        features.append(self._dist(lm[0], lm[4]))   # Wrist to thumb tip
        features.append(self._dist(lm[0], lm[8]))   # Wrist to index tip
        features.append(self._dist(lm[0], lm[12]))  # Wrist to middle tip
        features.append(self._dist(lm[0], lm[16]))  # Wrist to ring tip
        features.append(self._dist(lm[0], lm[20]))  # Wrist to pinky tip

        # Finger segment lengths
        for mcp, pip, dip, tip in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
            features.append(self._dist(lm[mcp], lm[pip]))  # MCP-PIP
            features.append(self._dist(lm[pip], lm[dip]))  # PIP-DIP
            features.append(self._dist(lm[dip], lm[tip]))  # DIP-TIP
            features.append(self._dist(lm[mcp], lm[tip]))  # MCP-TIP

        # Thumb segments
        features.append(self._dist(lm[1], lm[2]))   # CMC-MCP
        features.append(self._dist(lm[2], lm[3]))   # MCP-IP
        features.append(self._dist(lm[3], lm[4]))   # IP-TIP

        # === ANGLE FEATURES ===
        # Finger angles (using atan2)
        for tip, mcp in [(8,5), (12,9), (16,13), (20,17)]:
            features.append(self._angle_xy(lm[tip], lm[mcp]))

        # Thumb angle
        features.append(self._angle_xy(lm[4], lm[1]))

        # === RATIO FEATURES ===
        # Hand span
        max_x = max(lm[i][0] for i in range(21))
        min_x = min(lm[i][0] for i in range(21))
        max_y = max(lm[i][1] for i in range(21))
        min_y = min(lm[i][1] for i in range(21))
        width = max_x - min_x
        height = max_y - min_y

        features.append(width)   # Hand width
        features.append(height)  # Hand height
        features.append(width / (height + 1e-6))  # Aspect ratio

        # Normalized distances
        palm_size = self._dist(lm[0], lm[9])  # Wrist to middle MCP
        for i in [4, 8, 12, 16, 20]:
            features.append(self._dist(lm[0], lm[i]) / (palm_size + 1e-6))

        # === PALM FEATURES ===
        # Palm center
        palm_center = (
            (lm[0][0] + lm[5][0] + lm[9][0] + lm[13][0] + lm[17][0]) / 5,
            (lm[0][1] + lm[5][1] + lm[9][1] + lm[13][1] + lm[17][1]) / 5,
        )

        # Distance from palm center to each finger tip (normalized)
        for tip in [8, 12, 16, 20]:
            features.append(self._dist_xy(lm[tip], palm_center) / (palm_size + 1e-6))

        # Finger spread angles
        features.append(self._angle_xy(lm[8], lm[12]))   # Index to middle
        features.append(self._angle_xy(lm[12], lm[16]))  # Middle to ring
        features.append(self._angle_xy(lm[16], lm[20]))  # Ring to pinky

        return features

    def _dist(self, p1, p2):
        """Euclidean distance"""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def _dist_xy(self, p1, p2):
        """XY distance only"""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _angle_xy(self, p1, p2):
        """Angle in XY plane"""
        return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])

    def extract_from_dataset(self, dataset_path, output_path):
        """Extract enhanced features from BISINDO dataset"""
        print(f"Extracting enhanced features from: {dataset_path}")

        all_data = []
        letter_counts = {}

        for split in ['train', 'test']:
            split_path = Path(dataset_path) / split
            if not split_path.exists():
                continue

            images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))
            print(f"  {split}: {len(images)} images...")

            for img_path in images:
                letter = img_path.stem[0].upper()
                if not letter.isalpha():
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                features = self.extract(image)
                if features:
                    all_data.append({
                        'letter': letter,
                        'image_path': str(img_path),
                        'split': split,
                        'features': features
                    })
                    letter_counts[letter] = letter_counts.get(letter, 0) + 1

        print(f"\nExtracted {len(all_data)} samples")
        print(f"Features per sample: {63 + len(all_data[0]['features']) if all_data else 0}")

        self.save_to_csv(all_data, output_path)
        return all_data

    def save_to_csv(self, data, output_path):
        """Save to CSV"""
        n_features = len(data[0]['features']) if data else 0

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['letter', 'image_path', 'split']
            for i in range(21):
                header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
            for i in range(n_features):
                header.append(f'feat{i}')

            writer.writerow(header)

            for item in data:
                row = [item['letter'], item['image_path'], item['split']]
                row.extend(item['features'])
                writer.writerow(row)

        print(f"Saved to: {output_path}")


if __name__ == '__main__':
    extractor = EnhancedLandmarkExtractor()
    data = extractor.extract_from_dataset('dataset', 'dataset/enhanced_landmarks.csv')
