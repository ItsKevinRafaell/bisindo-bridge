"""
BISINDO Landmark Augmentation
Generate augmented samples from existing landmarks
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import json

def augment_landmark(landmark, noise_scale=0.02, scale_range=(0.9, 1.1), rotation_range=(-15, 15)):
    """
    Augment a single landmark sample with:
    - Random noise
    - Random scale
    - Random rotation (applied to x,y only)
    """
    lm = np.array(landmark).reshape(21, 3)

    # Apply rotation
    angle = np.random.uniform(*rotation_range) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    lm = lm @ rotation_matrix.T

    # Apply scale
    scale = np.random.uniform(*scale_range)
    lm = lm * scale

    # Apply noise
    noise = np.random.normal(0, noise_scale, lm.shape)
    lm = lm + noise

    return lm.flatten()

def augment_dataset(input_csv, output_csv, n_augments=4):
    """
    Augment dataset and save to new CSV
    """
    print(f"📊 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    # Get landmark columns
    lm_cols = [c for c in df.columns if c.startswith('lm')]
    feature_cols = ['letter', 'image_path', 'split'] + lm_cols

    print(f"   Original samples: {len(df)}")
    print(f"   Letters: {df['letter'].nunique()}")

    # Create augmented samples
    augmented_rows = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"   Processing {idx}/{len(df)}...")

        letter = row['letter']
        image_path = row['image_path']
        split = row['split']
        landmarks = row[lm_cols].values

        # Original sample
        augmented_rows.append({
            'letter': letter,
            'image_path': image_path,
            'split': split,
            **{lm_cols[i]: landmarks[i] for i in range(len(lm_cols))}
        })

        # Augmented samples
        for aug_idx in range(n_augments):
            aug_landmarks = augment_landmark(landmarks)
            augmented_rows.append({
                'letter': letter,
                'image_path': f"{image_path}_aug{aug_idx}",
                'split': split,
                **{lm_cols[i]: aug_landmarks[i] for i in range(len(lm_cols))}
            })

    # Create new DataFrame
    df_aug = pd.DataFrame(augmented_rows)

    print(f"\n✅ Augmented dataset: {len(df_aug)} samples")
    print(f"   Per letter distribution:")
    print(df_aug['letter'].value_counts().sort_index().head(10))

    # Save
    df_aug.to_csv(output_csv, index=False)
    print(f"   Saved to: {output_csv}")

    return df_aug

def split_dataset(df, test_ratio=0.2):
    """
    Split dataset into train/test
    """
    train_rows = []
    test_rows = []

    for letter in df['letter'].unique():
        letter_df = df[df['letter'] == letter]
        n_test = max(1, int(len(letter_df) * test_ratio))

        indices = letter_df.index.tolist()
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        test_rows.extend(test_indices)
        train_rows.extend(train_indices)

    df_train = df.loc[train_rows].copy()
    df_test = df.loc[test_rows].copy()

    # Update split column
    df_train['split'] = 'train'
    df_test['split'] = 'test'

    return df_train, df_test

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Augment BISINDO landmark dataset')
    parser.add_argument('--input', default='dataset/landmarks.csv',
                        help='Input CSV file')
    parser.add_argument('--output', default='dataset/landmarks_augmented.csv',
                        help='Output CSV file')
    parser.add_argument('--n-augments', type=int, default=4,
                        help='Number of augmentations per sample')
    args = parser.parse_args()

    print("=" * 50)
    print("🖐️ BISINDO Landmark Augmentation")
    print("=" * 50)

    # Augment
    df_aug = augment_dataset(args.input, args.output, args.n_augments)

    # Split
    print("\n📊 Splitting dataset...")
    df_train, df_test = split_dataset(df_aug)

    # Save split datasets
    train_path = args.output.replace('.csv', '_train.csv')
    test_path = args.output.replace('.csv', '_test.csv')

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\n✅ Saved:")
    print(f"   Train: {train_path} ({len(df_train)} samples)")
    print(f"   Test:  {test_path} ({len(df_test)} samples)")

if __name__ == '__main__':
    main()
