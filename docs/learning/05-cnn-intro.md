# CNN Introduction: Convolutional Neural Networks

> Di file sebelumnya kita udah kenal MLP. Sekarang kita belajar CNN - arsitektur yang revolusioner untuk computer vision. Penjelasan pakai analogi dan visualisasi.

---

## Daftar Isi

1. [Masalah dengan MLP](#masalah-dengan-mlp)
2. [Intuisi Convolutional Neural Network](#intuisi-convolutional-neural-network)
3. [Convolution Operation](#convolution-operation)
4. [Filters/Kernels](#filterskernels)
5. [Feature Maps](#feature-maps)
6. [Pooling](#pooling)
7. [Conv2d di PyTorch](#conv2d-di-pytorch)
8. [CNN Architecture untuk Images](#cnn-architecture-untuk-images)
9. [Kenapa CNN Bagus untuk Images?](#kenapa-cnn-bagus-untuk-images)
10. [Latihan](#latihan)

---

## Masalah dengan MLP

### MLP: Flatten Everything

```
MLP untuk image 28x28:
┌────────────────────────────────────┐
│                                    │
│   28 pixels × 28 pixels = 784     │
│                                    │
│   Setiap pixel = 1 input           │
│                                    │
│   MLP: 784 → 128 → 64 → 10        │
│                                    │
└────────────────────────────────────┘

Problems:
❌ Lost spatial information!
  - Pixel di kiri atas ga相关的 dengan pixel di kanan bawah
  - Tapi MLP treat semua sama

❌ Too many parameters!
  - 784 × 128 = 100,352 weights (layer 1 alone)
  - Mahal compute, mudah overfit

❌ Tidak shift-invariant!
  - Kalau gambar geser 1 pixel, ML harus relearn
```

### Spatial Relationships

MLP treat input sebagai "bag of pixels":
```
Image (MLP view):
┌───────┬───────┬───────┐
│ Pixel │ Pixel │ Pixel │
│   0   │   1   │   2   │
├───────┼───────┼───────┤
│ Pixel │ Pixel │ Pixel │   ← Semua dianggap independent
│   3   │   4   │   5   │
├───────┼───────┼───────┤
│ Pixel │ Pixel │ Pixel │
│   6   │   7   │   8   │
└───────┴───────┴───────┘

Real image:
┌───────┬───────┬───────┐
│  Eye  │  Eye  │ Nose  │   ← Spatial relationships MATTER!
├───────┼───────┼───────┤
│  Ear  │ Mouth │ Chin  │
├───────┼───────┼───────┤
│ Neck  │ Shirt │ Collar│
└───────┴───────┴───────┘
```

---

## Intuisi Convolutional Neural Network

### Gimana Mata Manusia Lihat Gambar?

```
┌─────────────────────────────────────────────────────────────────┐
│                     How You Scan an Image                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Your eyes gak scan semua pixel sekaligus                     │
│                                                                  │
│  2. Your eyes FOCUS PADA局部 (local patches)                    │
│                                                                  │
│  3. Move scanning window across image:                          │
│                                                                  │
│      ┌─────────────────┐                                        │
│      │                 │  ← Window 1: "Edge detected"           │
│      └─────────────────┘                                        │
│           ┌─────────────────┐                                    │
│           │                 │  ← Window 2: "Texture detected"     │
│           └─────────────────┘                                    │
│                ┌─────────────────┐                               │
│                │                 │  ← Window 3: "Another edge"   │
│                └─────────────────┘                                │
│                                                                  │
│  4. Brain combines local patterns → whole object                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CNN Mimics This Process

```
┌─────────────────────────────────────────────────────────────────┐
│                     CNN Processing                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image (28×28)                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │ Conv Layer  │  ← Scan dengan sliding window (kernel)        │
│  │ (Filters)   │  ← Extract local features (edges, textures)   │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │ Pool Layer  │  ← Downsample (reduce size)                   │
│  │ (Pooling)   │  ← Keep important info, discard redundancy    │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │ Conv Layer  │  ← Extract higher-level features              │
│  │ (Filters)   │  ← Combine edges → shapes → objects           │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │   FC Layer  │  ← Final classification                       │
│  │   (MLP)     │                                                │
│  └─────────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Convolution Operation

### Apa itu Convolution?

**Convolution** = operasi sliding window dengan filter/kernel.

```
┌────────────────────────────────────────────────────────────────┐
│                     Convolution Operation                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input (5×5)                    Filter/Kernel (3×3)           │
│   ┌───────────┐                  ┌───────────┐               │
│   │ 1 2 3 4 5 │                  │ 0 1 0     │               │
│   │ 6 7 8 9 10│      ★           │ 1-1 1     │ = Convolution │
│   │11 12 13 14│                  │ 0 1 0     │               │
│   │16 17 18 19│                  └───────────┘               │
│   └───────────┘                                                │
│                        │                                        │
│                        ▼                                        │
│                 Output (3×3)                                     │
│                 ┌───────────┐                                  │
│                 │  ?  ?  ?  │                                  │
│                 │  ?  ?  ?  │                                  │
│                 │  ?  ?  ?  │                                  │
│                 └───────────┘                                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Convolution

```
Input: 5×5 matrix              Filter: 3×3
        [[1, 2, 3, 4, 5],              [[0, 1, 0],
         [6, 7, 8, 9, 10],       ★       [1, -1, 1],
         [11,12,13,14,15],               [0, 1, 0]]
         [16,17,18,19,20],
         [21,22,23,24,25]]

Step 1: Place filter at top-left of input
┌─────────────────────────────────────────┐
│  ┌─────────┐                            │
│  │ 1 2 3   │ ← Input region             │
│  │ 6 7 8   │   overlaid by filter       │
│  │11 12 13 │                            │
│  └─────────┘                            │
│  ┌───────────┐                           │
│  │ 0 1 0    │ ← Filter (slide this)    │
│  │ 1-1 1    │                           │
│  │ 0 1 0    │                           │
│  └───────────┘                           │
│                                         │
│  Compute element-wise multiplication:   │
│  1×0 + 2×1 + 3×0 +                      │
│  6×1 + 7×(-1) + 8×1 +                   │
│  11×0 + 12×1 + 13×0                     │
│                                         │
│  = 0 + 2 + 0 + 6 - 7 + 8 + 0 + 12 + 0  │
│  = 21  → Output[0,0] = 21               │
└─────────────────────────────────────────┘

Step 2: Slide filter 1 step to the right
┌─────────────────────────────────────────┐
│      ┌─────────┐                        │
│      │ 2 3 4   │ ← New region           │
│      │ 7 8 9   │                        │
│      │12 13 14 │                        │
│      └─────────┘                        │
│                                         │
│  Compute:                               │
│  2×0 + 3×1 + 4×0 +                      │
│  7×1 + 8×(-1) + 9×1 +                   │
│  12×0 + 13×1 + 14×0                     │
│                                         │
│  = 0 + 3 + 0 + 7 - 8 + 9 + 0 + 13 + 0  │
│  = 24  → Output[0,1] = 24               │
└─────────────────────────────────────────┘

Continue sliding until entire output is filled!
```

### Convolution with Real Numbers

```
┌──────────────────────────────────────────────────────────────────┐
│              Complete Convolution Example                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input (4×4):                       Filter (2×2):                │
│  ┌─────────────┐                   ┌───────────┐                │
│  │ 1  2  3  4  │                   │  1   0    │                │
│  │ 5  6  7  8  │        ★          │  0  -1    │                │
│  │ 9  10 11 12 │                   └───────────┘                │
│  │13 14 15 16 │                                                │
│  └─────────────┘                                                │
│                                                                   │
│  Output (3×3):                                                   │
│  ┌─────────────────┐                                            │
│  │ (1×1)+(2×0)+(5×0)+(6×-1) = 1-6 = -5     │  ?   ?  │          │
│  │ (2×1)+(3×0)+(6×0)+(7×-1) = 2-7 = -5    │  ?   ?  │          │
│  │ ...                                 →   │  ?   ?  │          │
│  └─────────────────┘                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Filters/Kernels

### Filter = Feature Detector

Setiap filter mendeteksi satu jenis feature:

```
┌────────────────────────────────────────────────────────────────────┐
│                        Types of Filters                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Vertical Edge Detector:        Horizontal Edge Detector:          │
│  ┌───────────┐                  ┌───────────┐                      │
│  │ 1  0 -1  │                  │  1  1  1  │                      │
│  │ 1  0 -1  │                  │  0  0  0  │                      │
│  │ 1  0 -1  │                  │ -1 -1 -1  │                      │
│  └───────────┘                  └───────────┘                      │
│  → Deteksi garis vertikal       → Deteksi garis horizontal        │
│                                                                     │
│  Diagonal Detector:              Sharpen Filter:                  │
│  ┌───────────┐                  ┌───────────┐                    │
│  │ 1  1  0   │                  │ 0 -1  0   │                     │
│  │ 1  0 -1   │                  │-1  5 -1   │                     │
│  │ 0 -1 -1   │                  │ 0 -1  0   │                     │
│  └───────────┘                  └───────────┘                     │
│  → Deteksi diagonal              → Sharpen image                  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Learned Filters

Di CNN, filter TIDAK人工设计 (human-designed). Mereka learned dari data!

```
┌────────────────────────────────────────────────────────────────────┐
│                    Learned Filters in CNN                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1 Filters (low-level):                                      │
│  ┌──────┬──────┬──────┬──────┐                                    │
│  │ ████ │ █░░█ │ █░░░ │ ████ │  → Learned automatically            │
│  │ █░░█ │ ██░█ │ ░█░░ │ █░░█ │     from training data              │
│  │ █░░█ │ █░░█ │ ░░██ │ █░░█ │                                    │
│  │ ░░░░ │ ░░░░ │ ░░░░ │ ░███ │                                    │
│  └──────┴──────┴──────┴──────┘                                    │
│  Edge detectors, gradient detectors, simple patterns               │
│                                                                     │
│  Layer 2 Filters (mid-level):                                       │
│  ┌──────┬──────┬──────┬──────┐                                    │
│  │ shape │ curve│ corner│ pattern│  → Combinations of            │
│  │ █████ │ █░█░ │ ███░░ │ █░██░ │    layer 1 features               │
│  │ █░░█ │ █░█░ │ █░░░░ │ ░████ │                                  │
│  │ ░████ │ █░█░ │ ░░███ │ ███░█ │                                 │
│  └──────┴──────┴──────┴──────┘                                    │
│                                                                     │
│  Layer 3 Filters (high-level):                                      │
│  ┌──────┬──────┬──────┬──────┐                                    │
│  │  eye  │ nose │ mouth│ face │   → Object parts!                  │
│  │ ░██░░ │ ████ │ ░█░█ │ ░█░█ │                                  │
│  │ ░██░░ │ █░░█ │ ████ │ ░██░ │                                 │
│  └──────┴──────┴──────┴──────┘                                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Feature Maps

### Apa itu Feature Map?

**Feature Map** = output dari convolution layer. Represents detected features.

```
┌────────────────────────────────────────────────────────────────┐
│                   Feature Maps                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Image                                                    │
│   ┌──────────────────┐                                          │
│   │                  │                                          │
│   │   ┌────────┐     │                                          │
│   │   │        │     │  ← Original image                        │
│   │   │        │     │                                          │
│   │   └────────┘     │                                          │
│   │                  │                                          │
│   └──────────────────┘                                          │
│            │                                                    │
│   Conv with 32 filters                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌────┬────┬────┬────┬─ ─ ─┐                                  │
│   │    │    │    │    │     │                                  │
│   │ FM │ FM │ FM │ FM │ ... │  ← 32 Feature Maps (28×28 each) │
│   │  1 │  2 │  3 │  4 │     │                                  │
│   │    │    │    │    │     │                                  │
│   ├────┼────┼────┼────┼─ ─ ─┤                                  │
│   │    │    │    │    │     │                                  │
│   │    │    │    │    │     │                                  │
│   └────┴────┴────┴────┴─ ─ ─┘                                  │
│                                                                 │
│   Each feature map = detected pattern by one filter              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Multiple Channels

```
Input:  RGB Image = 3 channels (Red, Green, Blue)

Conv Layer:
┌─────────────────────────────────────────┐
│  Conv with 32 filters                  │
│                                         │
│  Filter 1: 3×3×3 → 1 feature map       │
│              (R,G,B channels)           │
│                                         │
│  Filter 2: 3×3×3 → 1 feature map       │
│              ...                       │
│                                         │
│  Filter 32: 3×3×3 → 1 feature map      │
└─────────────────────────────────────────┘
         │
         ▼
Output: 32 feature maps (channels)
```

---

## Pooling

### Kenapa Pooling?

```
┌────────────────────────────────────────────────────────────────┐
│                        Why Pooling?                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (4×4):                 After Conv (still 4×4):          │
│  ┌─────────────┐             ┌─────────────┐                  │
│  │ 1  2  3  4  │             │ 9  8  7  6  │                  │
│  │ 5  6  7  8  │    →       │ 5  4  3  2  │                  │
│  │ 9  10 11 12 │             │ 1  2  3  4  │                  │
│  │13 14 15 16 │             │ 0  1  2  3  │                  │
│  └─────────────┘             └─────────────┘                  │
│                              (Same size!)                       │
│                                                                 │
│  Problem: Spatial redundancy, too many computations              │
│  Solution: Pooling (downsampling)                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Max Pooling

**Max Pooling** = take maximum value in each window.

```
┌────────────────────────────────────────────────────────────────┐
│                     Max Pooling 2×2                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (4×4):                 Window 2×2:                       │
│  ┌─────────────┐             ┌────────────────────────────┐   │
│  │ 1  2 │ 3  4  │             │  ┌──────────┐ ┌──────────┐ │   │
│  ├──────┼──────┤   Split     │  │ 1  2     │ │ 3  4     │ │   │
│  │ 5  6 │ 7  8  │    into    │  │ 5  6     │ │ 7  8     │ │   │
│  ├──────┼──────┤   2×2       │  └──────────┘ └──────────┘ │   │
│  │ 9  10│ 11 12 │   blocks   │  ┌──────────┐ ┌──────────┐ │   │
│  ├──────┼──────┤             │  │ 9  10    │ │ 11 12    │ │   │
│  │ 13 14│ 15 16 │             │  │ 13 14    │ │ 15 16    │ │   │
│  └──────┴──────┘             │  └──────────┘ └──────────┘ │   │
│                              └────────────────────────────┘   │
│                                    │                            │
│                              Max Pool                       │
│                                    │                          │
│                                    ▼                          │
│  Output (2×2):                 ┌───────────┐                 │
│  ┌─────────────┐               │ 6   │ 8   │  → max(1,2,5,6)=6 │
│  │  6   │  8   │               │─────┼─────│                  │
│  ├──────┼──────┤               │14   │ 16  │ → max(9,10,13,14)=14│
│  │ 14   │ 16   │               │6    │ 8   │                  │
│  └──────┴──────┘               │14   │ 16  │                  │
│                                └───────────┘                  │
│                                                                 │
│  Size reduced 2× (4×4 → 2×2)                                    │
│  Key features preserved (max values)                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Average Pooling

**Average Pooling** = take average of window.

```
Input window: [[1, 2], [3, 4]]
Average pooling: (1+2+3+4)/4 = 2.5

Used less common than max pooling
(Max is more discriminative)
```

---

## Conv2d di PyTorch

### Conv2d Layer

```python
import torch
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(in_channels=3,     # RGB image = 3 channels
                 out_channels=32,   # 32 filters
                 kernel_size=3,      # 3×3 filter
                 stride=1,           # slide 1 step
                 padding=1)          # add 1 pixel border

# Input: (batch, channels, height, width)
x = torch.randn(1, 3, 28, 28)  # 1 image, 3 channels, 28×28

# Forward pass
output = conv(x)
print(f"Output shape: {output.shape}")
# Conv with padding=1: 28×28 → 28×28 (same size)
# Conv without padding: 28×28 → 26×26 (smaller)
```

### MaxPool2d Layer

```python
import torch.nn as nn

# MaxPool2d(kernel_size, stride)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Input: (batch, channels, height, width)
x = torch.randn(1, 32, 28, 28)  # 32 feature maps, 28×28

# Output: halved size
output = pool(x)
print(f"Output shape: {output.shape}")
# (1, 32, 14, 14)  → 28×28 → 14×14 (2× reduction)
```

### Simple CNN for MNIST

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Conv Block 1: 1→32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28→14
            
            # Conv Block 2: 32→64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14→7
            
            # Conv Block 3: 64→128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 7→1×1
        )
        
        # Fully connected
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv_layers(x)     # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)   # flatten → (batch, 128)
        x = self.fc(x)               # → (batch, 10)
        return x
```

---

## CNN Architecture untuk Images

### LeNet-5 (1998): The First CNN

```
┌────────────────────────────────────────────────────────────────┐
│                      LeNet-5 Architecture                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (32×32)                                                  │
│      │                                                          │
│      ▼                                                          │
│  Conv1: 6 filters (5×5) → 6 feature maps (28×28)               │
│      │                                                          │
│      ▼                                                          │
│  AvgPool1: 2×2 → (14×14)                                       │
│      │                                                          │
│      ▼                                                          │
│  Conv2: 16 filters (5×5) → 16 feature maps (10×10)            │
│      │                                                          │
│      ▼                                                          │
│  AvgPool2: 2×2 → (5×5)                                          │
│      │                                                          │
│      ▼                                                          │
│  FC: 120 neurons                                                │
│      │                                                          │
│      ▼                                                          │
│  FC: 84 neurons                                                 │
│      │                                                          │
│      ▼                                                          │
│  Output: 10 classes (digits 0-9)                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Modern CNN (ResNet, VGG)

```
┌────────────────────────────────────────────────────────────────┐
│                    CNN Architecture Trends                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Early Era (LeNet, AlexNet):                                    │
│  - 5-8 layers                                                    │
│  - Small filters (3×3, 5×5)                                     │
│  - Few channels (64-256)                                        │
│                                                                 │
│  Deep Era (VGG, ResNet):                                        │
│  - 50-152 layers                                                │
│  - More channels (512-2048)                                     │
│  - Skip connections (ResNet)                                    │
│                                                                 │
│  Efficient Era (MobileNet, EfficientNet):                       │
│  - Depthwise separable convolutions                             │
│  - Compound scaling                                             │
│  - Optimized for mobile/edge                                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Kenapa CNN Bagus untuk Images?

### Keunggulan CNN

| Aspek | MLP | CNN |
|-------|-----|-----|
| **Parameters** | 100K+ | 1K-10M |
| **Spatial info** | Lost | Preserved |
| **Translation equivariant** | No | Yes |
| **Hierarchical features** | No | Yes |

### Parameter Comparison

```
MLP untuk 28×28 image:
  Input → Hidden (256) → Hidden (128) → Output (10)
  Params = 784×256 + 256 + 128×256 + 256 + 128×10 + 10
         = 200,704 + 256 + 32,768 + 256 + 1,280 + 10
         ≈ 235,000 parameters

CNN untuk 28×28 image:
  Conv(1→32, 3×3): 1×32×3×3 + 32 = 320
  Conv(32→64, 3×3): 32×64×3×3 + 64 = 18,496
  Conv(64→128, 3×3): 64×128×3×3 + 128 = 73,856
  FC: 128×10 + 10 = 1,290
  
  Total ≈ 94,000 parameters (60% less!)
```

### Hierarchical Feature Learning

```
┌────────────────────────────────────────────────────────────────┐
│              Feature Learning Hierarchy                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1 (Edges):                                               │
│  ┌──────────────────────────────────────────────┐              │
│  │ ═══════════ → ║║║║║ → ═══════════           │              │
│  │ ║║║║║║║║║║ → ═══════════ → ═║║║║║           │              │
│  │ ▓▓▓▓▓▓▓▓▓▓ → ║║║║║║ → ▓▓▓▓▓▓▓▓           │              │
│  └──────────────────────────────────────────────┘              │
│  → Horizontal, vertical, diagonal edges                        │
│                                                                 │
│  Layer 2 (Textures):                                           │
│  ┌──────────────────────────────────────────────┐              │
│  │ ░░▓▓▓▓▓▓░░ → ▓▓░░▓▓▓▓ → ░░▓▓▓▓▓▓░░           │              │
│  │ ░░▓▓▓▓▓▓░░ → ░░▓▓▓▓▓▓ → ░░▓▓▓▓▓▓░░           │              │
│  │ ░░▓▓▓▓▓▓░░ → ▓▓░░▓▓▓▓ → ░░▓▓▓▓▓▓░░           │              │
│  └──────────────────────────────────────────────┘              │
│  → Combinations of edges → textures, patterns                 │
│                                                                 │
│  Layer 3 (Parts):                                               │
│  ┌──────────────────────────────────────────────┐              │
│  │     ╔═══╗                                    │              │
│  │    ╔╝   ╚╗   →  Eye        →  Face           │              │
│  │    ║     ║                                    │              │
│  │     ╚═╦═╝                                    │              │
│  └──────────────────────────────────────────────┘              │
│  → Combinations of textures → object parts                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Latihan

### Latihan 1: Conv2d Forward Pass

```python
import torch
import torch.nn as nn

# Create Conv2d layer
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)

# Input: batch of 4 RGB images, 32×32
x = torch.randn(4, 3, 32, 32)

# Forward pass
output = conv(x)
print(f"Output shape: {output.shape}")
# Expected: (4, 8, 32, 32)
```

### Latihan 2: MaxPool2d

```python
import torch
import torch.nn as nn

# MaxPool2d
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Input: 4 feature maps, 16×16
x = torch.randn(4, 8, 16, 16)

# Output
output = pool(x)
print(f"Output shape: {output.shape}")
# Expected: (4, 8, 8, 8)
```

### Latihan 3: Build Simple CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define layers
        # Conv1: 1→16 channels, 3×3 kernel
        # Pool1: 2×2
        # Conv2: 16→32 channels, 3×3 kernel
        # Pool2: 2×2
        # FC: flatten → 10 classes
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)  # MNIST image
output = model(x)
print(f"Output shape: {output.shape}")  # Should be (1, 10)
```

### Latihan 4: Visualize Conv Filters

```python
import torch
import torchvision
import matplotlib.pyplot as plt

# Load pretrained ResNet (if available)
model = torchvision.models.resnet18(pretrained=True)

# Get first conv layer weights
first_conv = model.conv1
filters = first_conv.weight.data

print(f"Filters shape: {filters.shape}")
# (64, 3, 7, 7) = 64 filters, 3 channels, 7×7 kernel

# Visualize first 8 filters
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 8:
        # Get first channel of filter
        filter_img = filters[i, 0].numpy()
        ax.imshow(filter_img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i}')
plt.show()
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                    CNN Fundamentals                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Convolution: Sliding filter over input                      │
│     - Detects local patterns (edges, textures)                   │
│     - Shared weights (efficient)                                 │
│                                                                  │
│  2. Filters/Kernels: Feature detectors                          │
│     - Learned from data (not hand-designed)                      │
│     - Multiple filters = multiple feature maps                   │
│                                                                  │
│  3. Pooling: Downsampling                                       │
│     - Max pooling: preserve strongest activation                 │
│     - Reduces spatial size, computation                          │
│                                                                  │
│  4. Feature Hierarchy                                            │
│     - Low-level: edges, gradients                               │
│     - Mid-level: textures, patterns                             │
│     - High-level: object parts                                  │
│                                                                  │
│  5. Advantages over MLP                                         │
│     - Fewer parameters                                          │
│     - Preserve spatial information                              │
│     - Hierarchical feature learning                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [06-conv1d-for-landmarks.md](06-conv1d-for-landmarks.md) - Conv1d untuk Landmark

---

## Referensi

- CS231n Stanford: http://cs231n.stanford.edu/
- Convolution arithmetic: https://arxiv.org/abs/1603.07285
- 3Blue1Brown: "Deep learning" playlist (YouTube)
