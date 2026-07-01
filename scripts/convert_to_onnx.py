#!/usr/bin/env python3
"""
Convert PyTorch CNN model to ONNX format for browser inference.

Usage:
    python scripts/convert_to_onnx.py [model_name]

Example:
    python scripts/convert_to_onnx.py cnn_2hand

This will:
1. Load the PyTorch model from models/dl/{model_name}_model.pt
2. Export to ONNX format in web/models/
3. Copy scaler.json and labels.json to web/models/
"""

import os
import sys
import json
import shutil
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_webcam import CNN


def convert_pytorch_to_onnx(model_name='cnn_2hand'):
    """Convert PyTorch model to ONNX format."""

    # Paths
    pt_model_path = f'models/dl/{model_name}_model.pt'
    scaler_path = f'models/dl/{model_name}_scaler.json'
    labels_path = f'models/dl/{model_name}_labels.json'

    output_dir = 'web/models'
    onnx_model_path = os.path.join(output_dir, 'model.onnx')

    print(f"🔄 Converting {model_name} to ONNX format...")

    # Check if model exists
    if not os.path.exists(pt_model_path):
        print(f"❌ Model not found: {pt_model_path}")
        return False

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create dummy input (batch_size=1, channels=1, features=84)
    print("🔧 Creating dummy input...")
    dummy_input = torch.randn(1, 1, 84, dtype=torch.float32)

    # Export to ONNX using legacy mode (TorchScript-based)
    print("💾 Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            pt_model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False  # Use legacy TorchScript exporter
        )

    print(f"✅ ONNX model exported to: {onnx_model_path}")

    # Copy scaler and labels
    print("📋 Copying scaler and labels...")
    shutil.copy(scaler_path, os.path.join(output_dir, 'scaler.json'))
    shutil.copy(labels_path, os.path.join(output_dir, 'labels.json'))

    # Verify the model
    print("🔍 Verifying ONNX model...")
    try:
        import onnx
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX model is valid!")

        # Print model info
        file_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Input shape: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"   Output shape: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in model.graph.output[0].type.tensor_type.shape.dim]}")

    except ImportError:
        print("⚠️  ONNX package not installed, skipping verification")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"⚠️  Verification warning: {e}")

    print(f"\n✅ Conversion complete!")
    print(f"   Model: {onnx_model_path}")
    print(f"   Scaler: {output_dir}/scaler.json")
    print(f"   Labels: {output_dir}/labels.json")
    print(f"\n📝 Next steps:")
    print(f"   1. Start the meeting server: cd meeting && python app.py")
    print(f"   2. Open http://localhost:4500 in your browser")
    print(f"   3. Test gesture detection with ONNX Runtime Web!")

    return True


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'cnn_2hand'

    # List available models
    if model_name == '--list':
        print("Available models:")
        for f in sorted(os.listdir('models/dl')):
            if f.endswith('_model.pt'):
                name = f.replace('_model.pt', '')
                print(f"  - {name}")
        sys.exit(0)

    success = convert_pytorch_to_onnx(model_name)
    sys.exit(0 if success else 1)
