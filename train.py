#!/usr/bin/env python3
"""
Main Training Script - BISINDO Landmark-Based Approach
Extract landmarks from dataset and train classifier
"""

import os
import sys

def main():
    print("=" * 60)
    print("BISINDO LANDMARK-BASED TRAINING")
    print("=" * 60)
    print()

    # Step 1: Extract landmarks
    print("\n[STEP 1] Extracting landmarks from dataset...")
    print("-" * 60)

    from src.landmark_extractor import LandmarkExtractor

    extractor = LandmarkExtractor()

    dataset_path = 'dataset'
    output_path = 'dataset/landmarks.csv'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = extractor.extract_from_dataset(dataset_path, output_path)

    if len(data) == 0:
        print("\n❌ No landmarks extracted. Please check your dataset.")
        return

    # Step 2: Train model
    print("\n[STEP 2] Training classifier...")
    print("-" * 60)

    from src.landmark_trainer import LandmarkTrainer

    trainer = LandmarkTrainer()
    X, y, df = trainer.load_data(output_path)
    results = trainer.train(X, y, test_size=0.2)

    # Save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    base_path = os.path.join(model_dir, 'landmark_classifier')

    trainer.save(
        f'{base_path}.pkl',
        f'{base_path}_scaler.pkl',
        f'{base_path}_labels.pkl'
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
Results:
  - Total samples: {results['n_samples']}
  - Features: {results['n_features']}
  - Classes: {results['n_classes']}
  - Train accuracy: {results['train_accuracy']:.2%}
  - Test accuracy: {results['test_accuracy']:.2%}

Output files:
  - Dataset: {output_path}
  - Model: {base_path}.pkl
  - Scaler: {base_path}_scaler.pkl
  - Labels: {base_path}_labels.pkl

Next steps:
  1. Run: streamlit run app.py
  2. Test real-time gesture recognition
  3. Collect more data if accuracy is low
    """)

if __name__ == '__main__':
    main()
