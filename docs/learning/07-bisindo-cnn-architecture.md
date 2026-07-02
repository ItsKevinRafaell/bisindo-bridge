# Arsitektur CNN BISINDO

> Penjelasan detail arsitektur CNN yang kita pakai di proyek BISINDO. Layer-by-layer breakdown dengan visualisasi dan perbandingan.

---

## Daftar Isi

1. [Arsitektur Overview](#arsitektur-overview)
2. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
3. [Input Preprocessing](#input-preprocessing)
4. [Convolutional Block](#convolutional-block)
5. [Fully Connected Block](#fully-connected-block)
6. [Parameter Analysis](#parameter-analysis)
7. [Code Implementation](#code-implementation)
8. [Comparison dengan MLP](#comparison-dengan-mlp)
9. [Variants dan Alternatives](#variants-dan-alternatives)
10. [Latihan](#latihan)

---

## Arsitektur Overview

### Complete Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   BISINDO CNN Architecture                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input (84 features)                                          │
│   ├── Hand 1: 21 landmarks × 2 coords (x, y) = 42 values      │
│   └── Hand 2: 21 landmarks × 2 coords (x, y) = 42 values      │
│                                                                 │
│   Reshape: (batch, 1, 84)                                     │
│          │                                                     │
│          ▼                                                     │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │                    CONV BLOCK                           │ │
│   │  ┌─────────────────────────────────────────────────┐   │ │
│   │  │ Conv1d(1→64, k=3, p=1) + ReLU + MaxPool(2)   │   │ │
│   │  │ (1, 84) → (64, 42)                            │   │ │
│   │  └─────────────────────────────────────────────────┘   │ │
│   │                          │                              │ │
│   │  ┌─────────────────────────────────────────────────┐   │ │
│   │  │ Conv1d(64→128, k=3, p=1) + ReLU + MaxPool(2)│   │ │
│   │  │ (64, 42) → (128, 21)                          │   │ │
│   │  └─────────────────────────────────────────────────┘   │ │
│   │                          │                              │ │
│   │  ┌─────────────────────────────────────────────────┐   │ │
│   │  │ Conv1d(128→64, k=3, p=1) + ReLU + GAP(1)    │   │ │
│   │  │ (128, 21) → (64, 1)                           │   │ │
│   │  └─────────────────────────────────────────────────┘   │ │
│   └─────────────────────────────────────────────────────────┘ │
│          │                                                     │
│          ▼ Flatten                                             │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │                   FC BLOCK                              │ │
│   │  Linear(64→128) + ReLU + Dropout(0.3)                  │ │
│   │                          │                              │ │
│   │  Linear(128→26)          │                              │ │
│   └─────────────────────────────────────────────────────────┘ │
│          │                                                     │
│   Output: (batch, 26) - logits untuk 26 huruf               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Stats

| Metric | Value |
|--------|-------|
| **Total Parameters** | ~61,000 |
| **Conv Layers** | 3 |
| **FC Layers** | 2 |
| **Input Size** | 84 features |
| **Output Size** | 26 classes |
| **Training Epochs** | 50 |
| **Batch Size** | 256 |
| **Learning Rate** | 0.001 |

---

## Layer-by-Layer Breakdown

### Layer 1: Input

```python
# Input shape: (batch, 84)
# Layout: [hand1_x0, hand1_y0, hand1_x1, ... , hand2_x0, hand2_y0, ...]

# Reshape untuk Conv1d: (batch, 1, 84)
x = x.view(batch_size, 1, 84)
```

### Layer 2: Conv1d Block 1

```python
# Conv1d(1, 64, kernel_size=3, padding=1)
# Input:  (batch, 1, 84)
# Output: (batch, 64, 84)

conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
x = conv1(x)

# ReLU activation
x = torch.relu(x)  # shape: (batch, 64, 84)

# MaxPool1d(2)
# Output: (batch, 64, 42)
x = nn.MaxPool1d(2)(x)
```

**Apa yang terjadi:**
- 1 input channel → 64 output channels
- 64 filters mendeteksi 64 different patterns
- MaxPool reduce length 84 → 42

### Layer 3: Conv1d Block 2

```python
# Conv1d(64, 128, kernel_size=3, padding=1)
# Input:  (batch, 64, 42)
# Output: (batch, 128, 42)

conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
x = conv2(x)

# ReLU activation
x = torch.relu(x)  # shape: (batch, 128, 42)

# MaxPool1d(2)
# Output: (batch, 128, 21)
x = nn.MaxPool1d(2)(x)
```

**Apa yang terjadi:**
- 64 channels → 128 channels (2x more patterns)
- More complex feature combinations
- Length reduce 42 → 21

### Layer 4: Conv1d Block 3

```python
# Conv1d(128, 64, kernel_size=3, padding=1)
# Input:  (batch, 128, 21)
# Output: (batch, 64, 21)

conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
x = conv3(x)

# ReLU activation
x = torch.relu(x)  # shape: (batch, 64, 21)

# AdaptiveAvgPool1d(1)
# Output: (batch, 64, 1)
x = nn.AdaptiveAvgPool1d(1)(x)
```

**Apa yang terjadi:**
- Compress 128 → 64 (keep most important)
- AdaptiveAvgPool: 21 positions → 1 value per channel
- Global summary of entire hand pattern

### Layer 5: Flatten

```python
# Flatten: (batch, 64, 1) → (batch, 64)
x = x.view(x.size(0), -1)
```

### Layer 6: FC Block 1

```python
# Linear: (batch, 64) → (batch, 128)
fc1 = nn.Linear(64, 128)
x = fc1(x)

# ReLU activation
x = torch.relu(x)  # shape: (batch, 128)

# Dropout(0.3)
x = nn.Dropout(0.3)(x)
```

### Layer 7: FC Block 2 (Output)

```python
# Linear: (batch, 128) → (batch, 26)
fc2 = nn.Linear(128, 26)
x = fc2(x)

# Output: (batch, 26) - logits for 26 letters
return x
```

---

## Input Preprocessing

### Hand-Centric Normalization

```python
def normalize_hand(hand_coords):
    """
    Normalize hand: translate to wrist, scale by hand size.
    
    Args:
        hand_coords: array of shape (21, 3) - 21 landmarks with (x, y, z)
    
    Returns:
        Normalized coordinates
    """
    wrist = hand_coords[0]  # Landmark 0 = wrist
    
    # Translate: subtract wrist position
    centered = hand_coords - wrist
    
    # Scale: normalize by max distance from wrist
    distances = np.linalg.norm(centered, axis=1)
    max_dist = distances.max()
    
    if max_dist > 0:
        centered = centered / max_dist
    
    return centered

def normalize_features(X, input_dim):
    """
    Apply hand-centric normalization to all samples.
    
    Args:
        X: Raw features of shape (n_samples, 84) or (n_samples, 63)
        input_dim: 84 for 2-hand, 63 for 1-hand
    
    Returns:
        Normalized features
    """
    X_norm = X.copy()
    
    if input_dim == 84:  # 2-hand xy
        for i in range(X.shape[0]):
            # Hand 1 (first 42 values)
            h1 = X[i, :42].reshape(21, 2)
            h1_norm = normalize_hand(h1)
            X_norm[i, :42] = h1_norm.flatten()
            
            # Hand 2 (next 42 values)
            h2 = X[i, 42:].reshape(21, 2)
            if np.any(h2 != 0):  # Only if hand 2 exists
                h2_norm = normalize_hand(h2)
                X_norm[i, 42:] = h2_norm.flatten()
    
    return X_norm
```

### Why Normalization Matters

```
┌────────────────────────────────────────────────────────────────┐
│            Before vs After Hand-Centric Normalization           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BEFORE (raw coordinates):                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Sample 1 (hand close):  wrist=(0.3, 0.5), thumb=(0.35,0.45)│
│  │ Sample 2 (hand far):     wrist=(0.6, 0.7), thumb=(0.62,0.68)│
│  │                                                                 │
│  │ Same letter, DIFFERENT coordinates!                         │
│  │ Model thinks they're different letters!                      │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  AFTER (normalized):                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Sample 1: wrist=(0,0), thumb=(0.05, -0.05)              │
│  │ Sample 2: wrist=(0,0), thumb=(0.02, -0.02)             │
│  │                                                                 │
│  │ Same letter, SAME relative positions!                      │
│  │ Model correctly identifies same letter!                    │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# StandardScaler: (x - mean) / std
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Purpose:
# - Features with large range don't dominate
# - Model converges faster
# - Regularization works better
```

### Complete Preprocessing Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                Complete Preprocessing Pipeline                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw CSV Data                                                  │
│      │                                                         │
│      ▼                                                         │
│  Extract 84 features (hand1 + hand2)                           │
│      │                                                         │
│      ▼                                                         │
│  Hand-Centric Normalization                                     │
│  ├── Translate to wrist (origin)                               │
│  └── Scale by hand size (max distance)                        │
│      │                                                         │
│      ▼                                                         │
│  StandardScaler                                                 │
│  ├── Compute mean & std per feature                           │
│  └── Transform: (x - mean) / std                              │
│      │                                                         │
│      ▼                                                         │
│  Reshape to (batch, 1, 84)                                     │
│      │                                                         │
│      ▼                                                         │
│  CNN Forward Pass                                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Convolutional Block

### Block Design Philosophy

```
┌────────────────────────────────────────────────────────────────┐
│                    Convolutional Block Design                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pattern: Conv → ReLU → Pool (repeat)                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Conv: Extract local features                            │  │
│  │ ↑ "What patterns exist here?"                           │  │
│  │                                                          │  │
│  │ ReLU: Add non-linearity                                 │  │
│  │ ↑ "Which patterns are important?"                       │  │
│  │                                                          │  │
│  │ Pool: Reduce spatial size                               │  │
│  │ ↑ "Focus on most important patterns"                    │  │
│  │                                                          │  │
│  │ Repeat: Build hierarchical features                     │  │
│  │ ↑ "Combine simple patterns → complex patterns"           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Channel Progression

```python
# Channel progression: 1 → 64 → 128 → 64
# 
# Why increase then decrease?
#
# 1. INCREASE (1→64→128): 
#    - Start with generic features
#    - Expand to capture many patterns
#    - More channels = more feature combinations
#
# 2. DECREASE (128→64):
#    - Compress to most important features
#    - Reduce computation in FC layer
#    - Prevent overfitting

# Alternative progressions:
# - 1 → 32 → 64 → 32 (smaller, faster)
# - 1 → 128 → 256 → 128 (larger, more capacity)
# - 1 → 64 → 64 → 64 (constant width)
```

### Spatial Reduction

```
┌────────────────────────────────────────────────────────────────┐
│                  Spatial Reduction via Pooling                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: 84 positions                                            │
│      │                                                         │
│      ▼                                                         │
│  After Conv1: (84) + ReLU                                      │
│      │                                                         │
│      ▼ Pool(2)                                                 │
│  42 positions                                                  │
│      │                                                         │
│      ▼                                                         │
│  After Conv2: (42) + ReLU                                      │
│      │                                                         │
│      ▼ Pool(2)                                                 │
│  21 positions                                                  │
│      │                                                         │
│      ▼                                                         │
│  After Conv3: (21) + GAP(1)                                    │
│      │                                                         │
│      ▼                                                         │
│  1 position (global summary)                                    │
│                                                                 │
│  Total reduction: 84 → 1 (84× smaller)                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Fully Connected Block

### FC Design

```python
# Flatten output: (batch, 64, 1) → (batch, 64)
x = x.view(batch, -1)

# FC1: 64 → 128
# Combine all 64 extracted features
# Learn complex combinations of features

# FC2: 128 → 26
# Final classification layer
# Output: logits for 26 letters
```

### Why Two FC Layers?

```
┌────────────────────────────────────────────────────────────────┐
│              Why Two FC Layers, Not One?                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  One FC Layer (64 → 26):                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  f(x) = W₁x + b                                        │  │
│  │                                                          │  │
│  │  Limitation: Linear combination only                     │  │
│  │  Can only learn linear decision boundaries               │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Two FC Layers (64 → 128 → 26):                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  h = ReLU(W₁x + b₁)                                    │  │
│  │  y = W₂h + b₂                                          │  │
│  │                                                          │  │
│  │  Benefit: Non-linear transformation                      │  │
│  │  Can learn non-linear decision boundaries!              │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Dropout

```python
# Dropout(0.3): randomly "turn off" 30% of neurons during training
x = nn.Dropout(0.3)(x)

# Why dropout?
#
# During training:
# - 30% of neurons → output 0
# - Remaining 70% → work harder
#
# Effect:
# - Prevents overfitting
# - Forces network to be robust
# - Acts like ensemble of smaller networks
```

### Dropout Visualization

```
┌────────────────────────────────────────────────────────────────┐
│                        Dropout Effect                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WITHOUT DROPOUT:                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │   ○──○──○──○  All neurons active                       │  │
│  │   ╲╱╲╱╲╱╲╱    Every epoch                               │  │
│  │   ○──○──○──○  Network memorizes training data!          │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  WITH DROPOUT (p=0.3):                                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Epoch 1:        Epoch 2:        Epoch 3:               │  │
│  │  ○──○──○──○      ○──○──○──○      ○──○──○──○            │  │
│  │   ╲╱  ╲╱╲╱      ╲╱╲╱    ╲╱      ╲╱╲╱╲╱╲╱              │  │
│  │   ○──○──○──○      ○──○──○──○      ○──○──○──○            │  │
│  │                                                          │  │
│  │  Different neurons "die" each epoch!                    │  │
│  │  Network learns more robust features                    │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Parameter Analysis

### Detailed Parameter Count

```python
# Layer-by-layer parameters
import torch.nn as nn

# Conv1d(1, 64, k=3, p=1)
# Parameters = (in_channels × kernel_size + bias) × out_channels
#           = (1 × 3 + 1) × 64 = 256
conv1_params = (1 * 3 + 1) * 64
print(f"Conv1: {conv1_params}")

# Conv1d(64, 128, k=3, p=1)
# Parameters = (64 × 3 + 1) × 128 = 24,576
conv2_params = (64 * 3 + 1) * 128
print(f"Conv2: {conv2_params}")

# Conv1d(128, 64, k=3, p=1)
# Parameters = (128 × 3 + 1) × 64 = 24,640
conv3_params = (128 * 3 + 1) * 64
print(f"Conv3: {conv3_params}")

# FC1: 64 → 128
# Parameters = input × output + bias
#            = 64 × 128 + 128 = 8,320
fc1_params = 64 * 128 + 128
print(f"FC1: {fc1_params}")

# FC2: 128 → 26
fc2_params = 128 * 26 + 26
print(f"FC2: {fc2_params}")

total = conv1_params + conv2_params + conv3_params + fc1_params + fc2_params
print(f"\nTotal: {total:,}")
```

### Parameter Breakdown Table

| Layer | Shape | Parameters | % of Total |
|-------|-------|------------|------------|
| Conv1d(1→64) | 1×64×3 + 64 | 256 | 0.4% |
| Conv1d(64→128) | 64×128×3 + 128 | 24,576 | 40.1% |
| Conv1d(128→64) | 128×64×3 + 64 | 24,640 | 40.2% |
| Linear(64→128) | 64×128 + 128 | 8,320 | 13.6% |
| Linear(128→26) | 128×26 + 26 | 3,354 | 5.5% |
| **Total** | | **61,146** | 100% |

### Parameter Efficiency

```
┌────────────────────────────────────────────────────────────────┐
│                   Parameter Efficiency                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CNN Parameters: ~61,000                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Compare with:                                          │  │
│  │                                                          │  │
│  │  MLP(84→128→64→26):                                    │  │
│  │  84×128 + 128×64 + 64×26 + biases                     │  │
│  │  = 10,752 + 8,192 + 1,664 + ~200                        │  │
│  │  = ~20,800 parameters                                  │  │
│  │                                                          │  │
│  │  CNN has MORE parameters but:                          │  │
│  │  ✓ Shared weights (filters slide)                       │  │
│  │  ✓ Better feature learning                              │  │
│  │  ✓ Captures spatial relationships                       │  │
│  │  ✓ Usually better accuracy                              │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Code Implementation

### Complete Model Class

```python
import torch
import torch.nn as nn

class BISINDO_CNN(nn.Module):
    """
    CNN for BISINDO letter recognition.
    
    Architecture:
    - Conv Block: 3 Conv1d layers with pooling
    - FC Block: 2 fully connected layers
    
    Args:
        input_dim: 84 for 2-hand, 63 for 1-hand
        num_classes: 26 for A-Z
    """
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, input_dim) e.g., (32, 84)
        x = x.unsqueeze(1)  # (batch, 1, input_dim) → (32, 1, 84)
        
        x = self.conv(x)    # (batch, 64, 1)
        x = self.fc(x)       # (batch, num_classes)
        
        return x
    
    def predict(self, x):
        """Return predicted class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x):
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
```

### Alternative: Module-by-Module

```python
import torch
import torch.nn as nn

class BISINDO_CNN_Modular(nn.Module):
    """Modular version for easier debugging."""
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Reshape: (batch, 84) → (batch, 1, 84)
        x = x.unsqueeze(1)
        
        # Conv block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.gap(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC block
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### Usage Example

```python
import torch

# Create model
model = BISINDO_CNN(input_dim=84, num_classes=26)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Forward pass
batch_size = 32
x = torch.randn(batch_size, 84)  # Random landmark data
output = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")  # (32, 26)

# Predictions
predictions = model.predict(x)
print(f"Predictions: {predictions.shape}")  # (32,)

# Probabilities
probs = model.predict_proba(x)
print(f"Probabilities: {probs.shape}")  # (32, 26)
print(f"Sum of probs: {probs[0].sum():.4f}")  # Should be ~1.0
```

---

## Comparison dengan MLP

### MLP Architecture

```python
class BISINDO_MLP(nn.Module):
    """MLP baseline for comparison."""
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### CNN vs MLP Comparison

| Aspect | CNN | MLP |
|--------|-----|-----|
| **Architecture** | Conv + FC | FC only |
| **Feature Learning** | Automatic via conv filters | Implicit in weights |
| **Spatial Awareness** | Yes (consecutive positions) | No (treats all equal) |
| **Parameters** | ~61,000 | ~18,000 |
| **Conv Blocks** | 3 | 0 |
| **FC Blocks** | 2 | 3 |
| **Pooling** | Yes (MaxPool, GAP) | No |
| **Dropout** | 0.3 | None |

### Feature Learning Comparison

```
┌────────────────────────────────────────────────────────────────┐
│               Feature Learning: CNN vs MLP                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MLP (Flattened View):                                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  [x₀,y₀,x₁,y₁,...,x₂₀,y₂₀, x₀',y₀',...,x₂₀',y₂₀']    │  │
│  │  ├───┴───┴───┴───────┴─────┴───┴───┴───────────┘        │  │
│  │    All 84 values treated equally, no spatial structure     │  │
│  │                                                          │  │
│  │  Weight matrix: 84 × 128                                 │  │
│  │  Each output neuron sees ALL inputs equally               │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  CNN (Convolutional View):                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  [x₀,y₀,x₁,y₁,x₂,y₂|x₃,y₃,x₄,y₄,x₅,y₅|...]           │  │
│  │  └──────┘└──────┘└──────┘ └──────┘└──────┘└──────┘      │  │
│  │    Filter 1    Filter 2    Filter 3  (kernel=3)          │  │
│  │                                                          │  │
│  │  Local receptive field: each output sees only 3 values   │  │
│  │  Weight sharing: same filter used across all positions    │  │
│  │  Translation equivariance: pattern detected anywhere!    │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### When CNN Outperforms MLP

```
┌────────────────────────────────────────────────────────────────┐
│             When to Use CNN vs MLP for BISINDO                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CNN outperforms MLP when:                                      │
│  ✓ Letters have consistent local patterns                      │
│    (e.g., thumb shape, finger positions)                       │
│  ✓ Patterns can appear at different positions                  │
│    (e.g., hand closer/farther from camera)                     │
│  ✓ Dataset is large (10K+ samples)                            │
│  ✓ Multiple hand configurations per letter                     │
│                                                                 │
│  MLP is competitive when:                                       │
│  ✓ Limited data (< 10K samples)                               │
│  ✓ CPU-only training (CNN slower)                              │
│  ✓ Simple patterns (letters very distinct)                     │
│                                                                 │
│  Our BISINDO results:                                          │
│  - MLP accuracy: ~95%                                          │
│  - CNN accuracy: ~98%                                          │
│  - CNN wins by 3% on test set                                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Variants dan Alternatives

### Variant 1: Larger CNN

```python
class BISINDO_CNN_Large(nn.Module):
    """Larger version with more capacity."""
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # More dropout for larger model
            nn.Linear(256, num_classes)
        )

# Parameters: ~250,000 (4x more)
# Pros: More capacity, may capture more patterns
# Cons: Slower, more prone to overfitting
```

### Variant 2: Residual CNN

```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = torch.relu(out + residual)  # Skip connection
        return out

class BISINDO_CNN_Residual(nn.Module):
    """CNN with residual connections."""
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(128)
        
        self.conv_final = nn.Conv1d(64, 128, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
```

### Variant 3: 1D ResNet-like

```python
class BISINDO_ResNet(nn.Module):
    """Simplified ResNet for 1D."""
    
    def __init__(self, num_classes=26):
        super().__init__()
        
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_ch, out_ch, blocks):
        layers = [ResidualBlock(in_ch if i == 0 else out_ch) 
                  for i in range(blocks)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)
```

---

## Latihan

### Latihan 1: Parameter Count

```python
def count_params_cnn(input_dim=84, num_classes=26):
    """Calculate total parameters."""
    # Conv layers
    c1 = (1 * 3 + 1) * 64      # Conv1d(1, 64, 3)
    c2 = (64 * 3 + 1) * 128    # Conv1d(64, 128, 3)
    c3 = (128 * 3 + 1) * 64    # Conv1d(128, 64, 3)
    
    # FC layers
    f1 = 64 * 128 + 128        # Linear(64, 128)
    f2 = 128 * 26 + 26         # Linear(128, 26)
    
    return c1 + c2 + c3 + f1 + f2

print(f"Total params: {count_params_cnn():,}")
```

### Latihan 2: Forward Pass Debug

```python
import torch

def debug_forward(model, x):
    """Debug forward pass with shape printing."""
    print(f"Input: {x.shape}")
    
    x = x.unsqueeze(1)
    print(f"After unsqueeze: {x.shape}")
    
    # Conv1
    x = model.conv[0](x)  # Conv1d
    print(f"After Conv1: {x.shape}")
    x = torch.relu(x)
    x = model.conv[2](x)    # MaxPool
    print(f"After Pool1: {x.shape}")
    
    # Conv2
    x = model.conv[3](x)
    print(f"After Conv2: {x.shape}")
    x = torch.relu(x)
    x = model.conv[5](x)
    print(f"After Pool2: {x.shape}")
    
    # Conv3
    x = model.conv[6](x)
    print(f"After Conv3: {x.shape}")
    x = torch.relu(x)
    x = model.conv[8](x)
    print(f"After GAP: {x.shape}")
    
    # FC
    x = x.view(x.size(0), -1)
    print(f"After flatten: {x.shape}")
    
    return x

model = BISINDO_CNN()
x = torch.randn(4, 84)
debug_forward(model, x)
```

### Latihan 3: Compare CNN vs MLP

```python
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def compare_models(X_train, y_train, X_test, y_test):
    """Compare CNN vs MLP performance."""
    
    # MLP
    mlp = nn.Sequential(
        nn.Linear(84, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 26)
    )
    
    # CNN
    cnn = BISINDO_CNN()
    
    # Training code would go here...
    # (same training loop for both)
    
    # Evaluation
    mlp.eval()
    cnn.eval()
    
    with torch.no_grad():
        mlp_pred = mlp(X_test).argmax(1)
        cnn_pred = cnn(X_test).argmax(1)
    
    mlp_acc = accuracy_score(y_test, mlp_pred)
    cnn_acc = accuracy_score(y_test, cnn_pred)
    
    print(f"MLP accuracy: {mlp_acc:.4f}")
    print(f"CNN accuracy: {cnn_acc:.4f}")
    print(f"CNN improvement: {(cnn_acc - mlp_acc)*100:.2f}%")
```

### Latihan 4: Feature Map Visualization

```python
import matplotlib.pyplot as plt

def visualize_feature_maps(model, x):
    """Visualize intermediate feature maps."""
    
    # Get intermediate activations
    hooks = []
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv[0].register_forward_hook(get_activation('conv1'))
    model.conv[3].register_forward_hook(get_activation('conv2'))
    model.conv[6].register_forward_hook(get_activation('conv3'))
    
    # Forward pass
    output = model(x.unsqueeze(1))
    
    # Visualize
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    
    for i, (name, feat) in enumerate(activations.items()):
        ax = axes[i // 4, i % 4]
        feat = feat[0]  # First sample
        ax.imshow(feat.cpu().numpy()[:16], aspect='auto', cmap='viridis')
        ax.set_title(f'{name}: {feat.shape}')
    
    plt.tight_layout()
    plt.show()
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                  BISINDO CNN Architecture Summary                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT → CONV BLOCK → FC BLOCK → OUTPUT                        │
│                                                                  │
│  Conv Block (Feature Extraction):                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv1d(1→64) → ReLU → Pool →                           │   │
│  │ Conv1d(64→128) → ReLU → Pool →                         │   │
│  │ Conv1d(128→64) → ReLU → GAP(1)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  FC Block (Classification):                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Flatten → Linear(64→128) → ReLU → Dropout(0.3) →     │   │
│  │ Linear(128→26) → Output (26 classes)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Total Parameters: ~61,000                                      │
│  Accuracy: ~98% on test set                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [08-training-and-hyperparams.md](08-training-and-hyperparams.md) - Training & Hyperparameters

---

## Referensi

- PyTorch nn.Module docs: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Delving Deep into Rectifiers: https://arxiv.org/abs/1502.01852 (Xavier/He init)
- Dropout paper: https://arxiv.org/abs/1207.0580
