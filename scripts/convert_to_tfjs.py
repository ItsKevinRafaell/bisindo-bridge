#!/usr/bin/env python3
"""
Convert PyTorch CNN model to TensorFlow.js format for browser inference.

Usage:
    python scripts/convert_to_tfjs.py [model_name]

Example:
    python scripts/convert_to_tfjs.py cnn_2hand

This will:
1. Load the PyTorch model from models/dl/{model_name}_model.pt
2. Convert to TensorFlow SavedModel format
3. Convert to TF.js Layers format in web/models/
4. Copy scaler.json and labels.json to web/models/
"""

import os
import sys
import json
import shutil
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_webcam import CNN


def convert_pytorch_to_tfjs(model_name='cnn_2hand'):
    """Convert PyTorch model to TF.js format."""

    # Paths
    pt_model_path = f'models/dl/{model_name}_model.pt'
    scaler_path = f'models/dl/{model_name}_scaler.json'
    labels_path = f'models/dl/{model_name}_labels.json'

    tfjs_output_dir = 'web/models'
    temp_saved_model = '/tmp/bisindo_saved_model'

    print(f"🔄 Converting {model_name} to TF.js format...")

    # Load PyTorch model
    print("📦 Loading PyTorch model...")
    device = torch.device('cpu')
    pt_model = CNN(input_dim=84, num_classes=26)
    pt_model.load_state_dict(torch.load(pt_model_path, map_location=device))
    pt_model.eval()

    # Load scaler and labels
    print("📊 Loading scaler and labels...")
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    # Build equivalent TensorFlow/Keras model
    print("🏗️  Building TensorFlow model...")
    tf_model = keras.Sequential([
        layers.Input(shape=(84,), name='input'),
        layers.Reshape((84, 1), name='reshape'),

        # Conv block 1
        layers.Conv1D(64, 3, padding='same', name='conv1d_1'),
        layers.ReLU(name='relu_1'),
        layers.MaxPooling1D(2, name='maxpool_1'),

        # Conv block 2
        layers.Conv1D(128, 3, padding='same', name='conv1d_2'),
        layers.ReLU(name='relu_2'),
        layers.MaxPooling1D(2, name='maxpool_2'),

        # Conv block 3
        layers.Conv1D(64, 3, padding='same', name='conv1d_3'),
        layers.ReLU(name='relu_3'),
        layers.GlobalAveragePooling1D(name='adaptive_avg_pool'),

        # FC layers
        layers.Dense(128, name='dense_1'),
        layers.ReLU(name='relu_4'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(26, name='dense_2'),
        layers.Softmax(name='softmax')
    ])

    # Transfer weights from PyTorch to TensorFlow
    print("⚙️  Transferring weights...")

    # Extract PyTorch weights
    pt_weights = pt_model.state_dict()

    # Map PyTorch layers to TensorFlow layers
    weight_mapping = {
        'conv.0.weight': 'conv1d_1',  # Conv1d(1, 64, 3)
        'conv.0.bias': 'conv1d_1',
        'conv.3.weight': 'conv1d_2',  # Conv1d(64, 128, 3)
        'conv.3.bias': 'conv1d_2',
        'conv.6.weight': 'conv1d_3',  # Conv1d(128, 64, 3)
        'conv.6.bias': 'conv1d_3',
        'fc.1.weight': 'dense_1',     # Linear(64*8, 128)
        'fc.1.bias': 'dense_1',
        'fc.4.weight': 'dense_2',     # Linear(128, 26)
        'fc.4.bias': 'dense_2',
    }

    # Set weights for each layer
    tf_weights = []

    # Conv1D layers (PyTorch: [out_channels, in_channels, kernel_size])
    # TF: [kernel_size, in_channels, out_channels]
    for i, (pt_key, tf_name) in enumerate(weight_mapping.items()):
        if 'conv' in pt_key and 'weight' in pt_key:
            # Conv1D weight: transpose from [out, in, kernel] to [kernel, in, out]
            w = pt_weights[pt_key].numpy()
            w = np.transpose(w, (2, 1, 0))
            tf_weights.append(w)
        elif 'conv' in pt_key and 'bias' in pt_key:
            tf_weights.append(pt_weights[pt_key].numpy())
        elif 'dense' in pt_key and 'weight' in pt_key:
            # Dense weight: transpose from [out, in] to [in, out]
            w = pt_weights[pt_key].numpy()
            w = np.transpose(w, (1, 0))
            tf_weights.append(w)
        elif 'dense' in pt_key and 'bias' in pt_key:
            tf_weights.append(pt_weights[pt_key].numpy())

    tf_model.set_weights(tf_weights)

    # Test the model
    print("🧪 Testing converted model...")
    test_input = np.random.randn(1, 84).astype(np.float32)
    tf_output = tf_model.predict(test_input, verbose=0)
    print(f"   Output shape: {tf_output.shape}")
    print(f"   Sum of probabilities: {np.sum(tf_output):.4f}")

    # Save as TensorFlow SavedModel
    print("💾 Saving TensorFlow SavedModel...")
    if os.path.exists(temp_saved_model):
        shutil.rmtree(temp_saved_model)
    tf_model.save(temp_saved_model, save_format='tf')

    # Convert to TF.js using tensorflowjs_converter
    print("🔄 Converting to TF.js format...")
    if os.path.exists(tfjs_output_dir):
        shutil.rmtree(tfjs_output_dir)
    os.makedirs(tfjs_output_dir)

    # Use tensorflowjs_converter CLI
    import subprocess
    cmd = [
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_layers_model',
        temp_saved_model,
        tfjs_output_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Conversion failed:\n{result.stderr}")
        sys.exit(1)

    # Copy scaler and labels
    print("📋 Copying scaler and labels...")
    shutil.copy(scaler_path, os.path.join(tfjs_output_dir, 'scaler.json'))
    shutil.copy(labels_path, os.path.join(tfjs_output_dir, 'labels.json'))

    # Cleanup
    shutil.rmtree(temp_saved_model)

    print(f"✅ Conversion complete!")
    print(f"   Model: {tfjs_output_dir}/model.json")
    print(f"   Scaler: {tfjs_output_dir}/scaler.json")
    print(f"   Labels: {tfjs_output_dir}/labels.json")
    print(f"\n📝 Next steps:")
    print(f"   1. Start the meeting server: cd meeting && python app.py")
    print(f"   2. Open http://localhost:4500 in your browser")
    print(f"   3. Test the model in the virtual meeting!")


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'cnn_2hand'

    if not os.path.exists(f'models/dl/{model_name}_model.pt'):
        print(f"❌ Model not found: models/dl/{model_name}_model.pt")
        print(f"   Available models:")
        for f in os.listdir('models/dl'):
            if f.endswith('_model.pt'):
                print(f"   - {f.replace('_model.pt', '')}")
        sys.exit(1)

    convert_pytorch_to_tfjs(model_name)
