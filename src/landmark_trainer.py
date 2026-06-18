"""
Landmark Trainer - Train classifier using extracted landmarks
Fast training using Random Forest / Neural Network
"""

import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from datetime import datetime

class LandmarkTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self, csv_path):
        """Load extracted landmarks from CSV"""
        print(f"📂 Loading data from: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} samples")
        print(f"   Columns: {len(df.columns)}")

        # Extract features and labels
        feature_cols = [col for col in df.columns if col.startswith('lm')]
        X = df[feature_cols].values
        y = df['letter'].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"   Features shape: {X.shape}")
        print(f"   Labels: {len(self.label_encoder.classes_)} classes")
        print(f"   Classes: {list(self.label_encoder.classes_)}")

        return X, y_encoded, df

    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the classifier"""
        print("\n🧠 Training Landmark Classifier...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Check for NaN/Inf
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
            print("   ⚠️  Warning: NaN or Inf found in training data!")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        # Train Random Forest
        print("   Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"\n📊 Results:")
        print(f"   Train Accuracy: {train_score:.4f}")
        print(f"   Test Accuracy:  {test_score:.4f}")

        # Detailed report
        y_pred = self.model.predict(X_test_scaled)
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Confusion matrix summary
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 Confusion Matrix (first 5 classes):")
        print(cm[:5, :5])

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_)
        }

    def save(self, model_path, scaler_path, label_path):
        """Save trained model and encoders"""
        print(f"\n💾 Saving model to: {model_path}")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save label encoder
        with open(label_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Save metadata
        metadata = {
            'n_features': 63,
            'model_type': 'RandomForest',
            'n_estimators': self.model.n_estimators,
            'classes': list(self.label_encoder.classes_),
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✅ Model saved: {model_path}")
        print(f"   ✅ Scaler saved: {scaler_path}")
        print(f"   ✅ Labels saved: {label_path}")
        print(f"   ✅ Metadata: {metadata_path}")

    def predict(self, landmarks):
        """Predict letter from landmarks"""
        if self.model is None:
            raise ValueError("Model not loaded!")

        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = self.scaler.transform(landmarks)
        prediction = self.model.predict(landmarks_scaled)
        probabilities = self.model.predict_proba(landmarks_scaled)

        letter = self.label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[0][prediction[0]]

        return letter, confidence, probabilities[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train landmark classifier')
    parser.add_argument('--data', default='dataset/landmarks/landmarks.csv',
                        help='Path to landmarks CSV')
    parser.add_argument('--output', default='models/landmark_model.pkl',
                        help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set ratio')
    args = parser.parse_args()

    # Create trainer
    trainer = LandmarkTrainer()

    # Load data
    X, y, df = trainer.load_data(args.data)

    # Train
    results = trainer.train(X, y, test_size=args.test_size)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save model
    base_path = args.output.replace('.pkl', '')
    trainer.save(
        f'{base_path}.pkl',
        f'{base_path}_scaler.pkl',
        f'{base_path}_labels.pkl'
    )

    print("\n✅ Training complete!")
    print(f"   Test Accuracy: {results['test_accuracy']:.2%}")


if __name__ == '__main__':
    main()
