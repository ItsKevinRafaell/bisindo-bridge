"""
BISINDO Landmark Trainer (Enhanced)
Train classifier on augmented landmark data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import time

def load_dataset(csv_path):
    """Load and prepare dataset"""
    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Get landmark columns
    lm_cols = [c for c in df.columns if c.startswith('lm')]
    X = df[lm_cols].values
    y = df['letter'].values

    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")

    return X, y, lm_cols

def train_model(X_train, y_train, model_type='rf', n_estimators=300):
    """Train classifier"""
    print(f"\n🤖 Training {model_type}...")

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    start = time.time()
    model.fit(X_train, y_train)
    print(f"   Training time: {time.time() - start:.2f}s")

    return model

def evaluate_model(model, X, y, label_encoder):
    """Evaluate model"""
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    return accuracy, y_pred

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train BISINDO landmark classifier')
    parser.add_argument('--data', default='dataset/landmarks_augmented_train.csv',
                        help='Training data CSV')
    parser.add_argument('--test-data', default='dataset/landmarks_augmented_test.csv',
                        help='Test data CSV')
    parser.add_argument('--output', default='models/landmark_classifier_v2',
                        help='Output model prefix')
    parser.add_argument('--model', default='rf', choices=['rf', 'gb'],
                        help='Model type')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of trees (RF) or estimators (GB)')
    args = parser.parse_args()

    print("=" * 60)
    print("🖐️ BISINDO Landmark Training (Enhanced)")
    print("=" * 60)

    # Load training data
    X_train, y_train, lm_cols = load_dataset(args.data)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = train_model(X_train_scaled, y_train_enc, args.model, args.n_estimators)

    # Cross-validation on training set
    print("\n📊 Cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train_enc, cv=5, n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Evaluate on training set
    train_acc, _ = evaluate_model(model, X_train_scaled, y_train_enc, le)
    print(f"   Train Accuracy: {train_acc:.4f}")

    # Evaluate on test set if available
    test_acc = None
    if args.test_data and os.path.exists(args.test_data):
        print(f"\n📊 Evaluating on test set...")
        X_test, y_test, _ = load_dataset(args.test_data)
        X_test_scaled = scaler.transform(X_test)
        y_test_enc = le.transform(y_test)

        test_acc, y_pred = evaluate_model(model, X_test_scaled, y_test_enc, le)
        print(f"   Test Accuracy: {test_acc:.4f}")

        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Save model
    print(f"\n💾 Saving model to {args.output}...")

    with open(f"{args.output}.pkl", 'wb') as f:
        pickle.dump(model, f)

    with open(f"{args.output}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    with open(f"{args.output}_labels.pkl", 'wb') as f:
        pickle.dump(le, f)

    # Save metadata
    metadata = {
        'n_features': len(lm_cols),
        'model_type': args.model,
        'n_estimators': args.n_estimators,
        'classes': le.classes_.tolist(),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc) if test_acc else None,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(f"{args.output}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model saved!")
    print(f"   Model:     {args.output}.pkl")
    print(f"   Scaler:    {args.output}_scaler.pkl")
    print(f"   Labels:    {args.output}_labels.pkl")
    print(f"   Metadata:  {args.output}_metadata.json")

if __name__ == '__main__':
    main()
