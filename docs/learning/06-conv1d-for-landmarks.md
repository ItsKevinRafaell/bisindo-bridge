# Conv1d untuk Landmark Data

> **FILE KUNCI** - Ini penjelasan Conv1d yang khusus buat BISINDO. File ini explains kenapa kita pakai Conv1d, bukan Conv2d, dan gimana cara kerjanya dengan landmark data.

---

## Daftar Isi

1. [Conv1d vs Conv2d: Apa Bedanya?](#conv1d-vs-conv2d-apa-bedanya)
2. [Kenapa Conv1d untuk Landmark?](#kenapa-conv1d-untuk-landmark)
3. [Input Shape untuk BISINDO](#input-shape-untuk-bisindo)
4. [Conv1d Layer dalam PyTorch](#conv1d-layer-dalam-pytorch)
5. [Visualisasi Sliding Window](#visualisasi-sliding-window)
6. [Filters dan Feature Maps](#filters-dan-feature-maps)
7. [Arsitektur Conv1d BISINDO](#arsitektur-conv1d-bisindo)
8. [Pooling dalam 1D](#pooling-dalam-1d)
9. [Padding dan Stride](#padding-dan-stride)
10. [Perbandingan dengan ML](#perbandingan-dengan-ml)
11. [Latihan](#latihan)

---

## Conv1d vs Conv2d: Apa Bedanya?

### Dimensionality

```
┌────────────────────────────────────────────────────────────────┐
│                    Conv1d vs Conv2d                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Conv2d: 2 Dimensions (Height × Width)                         │
│  ┌─────────────────────────────────┐                           │
│  │    ┌───┐                       │                           │
│  │    │   │  ← 2D sliding window   │                           │
│  │    └───┘                       │                           │
│  │        ┌───┐                   │                           │
│  │        │   │  ← Slides 2D      │                           │
│  │        └───┘                   │                           │
│  │                                 │                           │
│  │  Input: (batch, channels, H, W) │                           │
│  │  Kernel: (out_channels, in_channels, KH, KW)               │
│  └─────────────────────────────────┘                           │
│                                                                 │
│  Conv1d: 1 Dimension (Length)                                  │
│  ┌─────────────────────────────────┐                           │
│  │  [●]─[●]─[●]─[●]─[●]─[●]─[●]  │                           │
│  │   └───┘                             │                           │
│  │    Filter slides 1D                  │                           │
│  │                                     │                           │
│  │  Input: (batch, channels, L)        │                           │
│  │  Kernel: (out_channels, in_channels, K)                      │
│  └─────────────────────────────────┘                           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Kapan Pakai yang Mana?

| Type | Use Case | Examples |
|------|----------|----------|
| **Conv2d** | 2D spatial data | Images, video frames |
| **Conv1d** | 1D sequential data | Time series, text, DNA, **landmarks** |

```
Conv2d examples:
┌──────────────────────────────────────┐
│  Image (H×W):                       │
│  ┌────────────────────────────────┐ │
│  │ ████████████████████████████████│ │
│  │ ██░░░░███░░░░░░░░░░░░░░░░░░░░░│ │
│  │ ██░░░░███░░░░░░░░░░░░░░░░░░░░░│ │
│  │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  └────────────────────────────────┘ │
│  Filter slides in 2D (H, W)         │
└──────────────────────────────────────┘

Conv1d examples:
┌──────────────────────────────────────┐
│  Time series:                       │
│  [1.2, 2.3, 3.1, 4.5, 3.2, ...]   │
│   └──Filter slides along time───────  │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Text (word embeddings):            │
│  [embed(word₁), embed(word₂), ...] │
│   └──Filter slides along sequence────│
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Landmarks (BISINDO):              │
│  [lm₀ₓ,lm₀ᵧ,lm₀ᵤ,lm₁ₓ,lm₁ᵧ,lm₁ᵤ...]│
│   └──Filter slides along sequence───┘
└──────────────────────────────────────┘
```

---

## Kenapa Conv1d untuk Landmark?

### Landmark = Sequence of Points

```
┌────────────────────────────────────────────────────────────────┐
│                    Hand Landmark Sequence                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tangan punya 21 landmark yang tersusun berurutan:             │
│                                                                 │
│  Position:     0     1     2     3     4     5     ...   20  │
│  Part:       Wrist  Thumb Thumb Thumb Thumb Index  ...  Pinky   │
│             ───┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬───          │
│                 │  │  │  │  │  │  │  │  │  │  │  │            │
│                 ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼           │
│              [x₀,y₀,z₀, x₁,y₁,z₁, x₂,y₂,z₂, x₃,y₃,z₃, ...]  │
│               └────────┬────────┘  └────────┬────────┘          │
│                   Landmark 0                Landmark 1          │
│                   (Wrist)                  (Thumb base)        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LANDMARK ADALAH SEQUENCE 1D!                          │   │
│  │                                                         │   │
│  │  Posisi 0 → 1 → 2 → 3 → 4 → 5 → ... → 20             │   │
│  │  Jari    Wrist Thumb Thumb Thumb Thumb Index ...       │   │
│  │                                                         │   │
│  │  Conv1d bisa "lihat" hubungan antar landmark            │   │
│  │  yang berdekatan dalam sequence!                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Spatial vs Sequential Relationships

```
MLP view (flatten):
┌────────────────────────────────────────┐
│  [x₀,y₀,z₀,x₁,y₁,z₁,...,x₂₀,y₂₀,z₂₀] │
│  └──┬──┘└──┬──┘     └─────┬─────┘     │
│    Semua   Semua         Semua         │
│    dianggap independent  connected    │
└────────────────────────────────────────┘

Conv1d view (sequential):
┌────────────────────────────────────────┐
│  [x₀,y₀,z₀]─[x₁,y₁,z₁]─[x₂,y₂,z₂]─...│
│      ↓            ↓            ↓       │
│   Filter      Filter      Filter      │
│   "Lihat      "Lihat      "Lihat       │
│    lokal"      lokal"      lokal"       │
└────────────────────────────────────────┘
```

### Kenapa Bukan Conv2d?

```
┌────────────────────────────────────────────────────────────────┐
│              Kenapa Conv1d, Bukan Conv2d?                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Conv2d butuh 2D structure:                                    │
│                                                                 │
│  Option 1: Reshape ke 2D image (21×3)                          │
│  ┌─────────────────────────────────────┐                      │
│  │  Landmark  │   x    │   y    │   z  │                      │
│  │      0     │  0.64  │  0.72  │ 0.00 │                      │
│  │      1     │  0.58  │  0.65  │ 0.00 │                      │
│  │      2     │  0.52  │  0.58  │ 0.00 │                      │
│  │     ...    │  ...   │  ...   │ ...  │                      │
│  │     20     │  0.51  │  0.83  │ 0.00 │                      │
│  └─────────────────────────────────────┘                      │
│  Masalah: Row = landmark ID, bukan spatial location!           │
│           Column x/y/z ga ada spatial meaning!                 │
│                                                                 │
│  Option 2: Conv1d (INI YANG KITA PAKAI)                        │
│  [x₀,y₀,z₀, x₁,y₁,z₁, x₂,y₂,z₂, ...]                         │
│  Filter menangkap sequential relationships antar landmark!      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Input Shape untuk BISINDO

### Shape Notation

```
┌────────────────────────────────────────────────────────────────┐
│                    PyTorch Tensor Shapes                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scalar:  torch.tensor(3.14)        → shape: ()                │
│  Vector:  torch.tensor([1,2,3])    → shape: (3,)              │
│  Matrix:  torch.tensor([[1,2],[3,4]]) → shape: (2,2)           │
│  3D:      ...                        → shape: (B,C,H,W)        │
│                                                                 │
│  Conv1d convention: (batch, channels, length)                  │
│  Conv2d convention: (batch, channels, height, width)            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Input Shapes untuk BISINDO

```python
import torch

# ========================================
# BISINDO Input Shapes
# ========================================

# Format 1: 1-Hand (63 features)
# 21 landmarks × 3 coordinates (x, y, z)
x_1hand = torch.randn(32, 1, 63)  # 32 samples, 1 channel, 63 features
print(f"1-Hand shape: {x_1hand.shape}")
# Output: torch.Size([32, 1, 63])

# Format 2: 2-Hand (84 features)
# Hand 1: 21 landmarks × 2 coordinates (x, y) = 42
# Hand 2: 21 landmarks × 2 coordinates (x, y) = 42
# Total = 84
x_2hand = torch.randn(32, 1, 84)  # 32 samples, 1 channel, 84 features
print(f"2-Hand shape: {x_2hand.shape}")
# Output: torch.Size([32, 1, 84])

# Format 3: 2-Hand (126 features)
# Hand 1: 21 × 3 = 63
# Hand 2: 21 × 3 = 63
# Total = 126
x_2hand_xyz = torch.randn(32, 1, 126)
print(f"2-Hand xyz shape: {x_2hand_xyz.shape}")
# Output: torch.Size([32, 1, 126])
```

### Data Layout dalam Memory

```
┌────────────────────────────────────────────────────────────────┐
│                    1-Hand Data Layout (63 values)              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CSV column: lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ...   │
│              ├───┘├───┘├───┘├───┘├───┘├───┘                     │
│              Landmark 0    Landmark 1      Landmark 20           │
│                                                                 │
│  Tensor layout (flattened):                                    │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬─ ─┬───┬───┬───┐         │
│  │x₀ │y₀ │z₀ │x₁ │y₁ │z₁ │x₂ │y₂ │z₂ │...│x₂₀│y₂₀│z₂₀│        │
│  └───┴───┴───┴───┴───┴───┴───┴───┴─ ─┴───┴───┴───┘          │
│     0   1   2   3   4   5   6   7   8      60  61  62          │
│                                                                 │
│  Position:  [0-2]=[landmark 0], [3-5]=[landmark 1], etc.    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    2-Hand Data Layout (84 values)                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Hand 1: 21 landmarks × 2 coords = 42 values] [Hand 2: 42 values]
│  ┌─────────────────────────────────────────────────┬───────────┐
│  │lm0_x,lm0_y,lm1_x,lm1_y,...,lm20_x,lm20_y│h2_x...,h2_y│
│  │        0-41 values                        │  42-83    │
│  └─────────────────────────────────────────────────┴───────────┘
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Conv1d Layer dalam PyTorch

### Basic Conv1d

```python
import torch
import torch.nn as nn

# Conv1d(in_channels, out_channels, kernel_size, stride, padding)
conv1d = nn.Conv1d(in_channels=1,      # 1 channel (landmark data)
                   out_channels=64,    # 64 filters = 64 feature maps
                   kernel_size=3,      # filter length = 3
                   stride=1,           # slide 1 step
                   padding=0)          # no padding

# Input: (batch, channels, length)
x = torch.randn(32, 1, 63)  # 32 samples, 1 channel, 63 features

# Forward pass
output = conv1d(x)

print(f"Input shape:  {x.shape}")      # torch.Size([32, 1, 63])
print(f"Output shape: {output.shape}")  # torch.Size([32, 64, ?])
# Output length = (63 - 3) / 1 + 1 = 61
```

### Parameter Count

```python
import torch.nn as nn

conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)

# Parameters per filter:
# in_channels × kernel_size + bias = 1 × 3 + 1 = 4

# Total parameters:
# filters × (in_channels × kernel_size + bias)
# = 64 × 4 = 256 parameters

print(f"Parameters: {sum(p.numel() for p in conv1d.parameters())}")
# Output: 256
```

### Multiple Conv1d Layers

```python
import torch.nn as nn

# Conv1d(in_channels, out_channels, kernel_size)
conv1 = nn.Conv1d(1, 64, 3, padding=1)   # 63 → 63
conv2 = nn.Conv1d(64, 128, 3, padding=1)  # 63 → 63
conv3 = nn.Conv1d(128, 64, 3, padding=1) # 63 → 63

# Test dengan batch
x = torch.randn(16, 1, 63)

x = conv1(x)   # (16, 1, 63) → (16, 64, 63)
x = torch.relu(x)

x = conv2(x)   # (16, 64, 63) → (16, 128, 63)
x = torch.relu(x)

x = conv3(x)   # (16, 128, 63) → (16, 64, 63)
x = torch.relu(x)

print(f"Final shape: {x.shape}")  # (16, 64, 63)
```

---

## Visualisasi Sliding Window

### Konsep Sliding Window

```
┌────────────────────────────────────────────────────────────────┐
│              Conv1d Sliding Window Visualization                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [x₀,y₀,z₀, x₁,y₁,z₁, x₂,y₂,z₂, x₃,y₃,z₃, ...]       │
│          └─────┘└─────┘└─────┘└─────┘                           │
│          Window 1  Window 2  Window 3  Window 4                │
│          (k=3)      (k=3)    (k=3)    (k=3)                   │
│                                                                 │
│  Filter (kernel_size=3):                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [w₀, w₁, w₂]  →  3 values per position                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Convolution

```
┌────────────────────────────────────────────────────────────────┐
│           Conv1d Step-by-Step (kernel_size=3, padding=0)        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (length=7):                                              │
│  Index:  [ 0  1  2  3  4  5  6  ]                            │
│  Values: [ 1  2  3  4  5  6  7  ]                            │
│                                                                 │
│  Kernel (size=3):                                              │
│  [ 0.5, -0.2, 0.1 ]                                           │
│                                                                 │
│  ─────────────────────────────────────────────────────────────│
│                                                                 │
│  Step 1: Position 0                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Input: [1, 2, 3]                                        │  │
│  │ Kernel:[0.5, -0.2, 0.1]                                 │  │
│  │                                                        │  │
│  │ Dot: 1×0.5 + 2×(-0.2) + 3×0.1                          │  │
│  │     = 0.5 - 0.4 + 0.3                                  │  │
│  │     = 0.4                                               │  │
│  │                                                        │  │
│  │ Output[0] = 0.4                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Step 2: Position 1 (slide right)                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Input: [2, 3, 4]                                        │  │
│  │ Kernel:[0.5, -0.2, 0.1]                                 │  │
│  │                                                        │  │
│  │ Dot: 2×0.5 + 3×(-0.2) + 4×0.1                          │  │
│  │     = 1.0 - 0.6 + 0.4                                  │  │
│  │     = 0.8                                               │  │
│  │                                                        │  │
│  │ Output[1] = 0.8                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Step 3: Position 2                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Input: [3, 4, 5]                                        │  │
│  │                                                        │  │
│  │ Dot: 3×0.5 + 4×(-0.2) + 5×0.1                          │  │
│  │     = 1.5 - 0.8 + 0.5                                  │  │
│  │     = 1.2                                               │  │
│  │                                                        │  │
│  │ Output[2] = 1.2                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ... Continue until position 4                                  │
│                                                                 │
│  ─────────────────────────────────────────────────────────────│
│                                                                 │
│  Final Output:                                                  │
│  Index:  [0    1    2    3    4  ]                            │
│  Values: [0.4, 0.8, 1.2, 1.6, 2.0]                            │
│                                                                 │
│  Input length: 7, Kernel: 3, Output length: 7-3+1 = 5        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Sliding Window dengan Landmark

```
┌────────────────────────────────────────────────────────────────┐
│           Conv1d pada Landmark Data (63 values)                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input sequence (63 values = 21 landmarks × 3 coords):         │
│  ┌───┬───┬───┬───┬───┬───┬─ ──┬───┬───┬───┐                 │
│  │x₀ │y₀ │z₀ │x₁ │y₁ │z₁ │... │x₂₀│y₂₀│z₂₀│                 │
│  └───┴───┴───┴───┴───┴───┴─ ──┴───┴───┴───┘                 │
│     │   │   │   │   │   │       │   │   │                     │
│     └───┴───┴───┴───┴───┴───────┴───┴───┘                     │
│        Landmark 0    Landmark 1         Landmark 20             │
│                                                                 │
│  Kernel_size=3 artinya filter melihat 3 consecutive values:    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │  Position: 0    1    2    3    4    ...         │    │  │
│  │  │  Values:    [x₀,y₀,z₀],[x₁,y₁,z₁],[x₂,y₂,z₂],...│    │  │
│  │  │               └──────┘ └──────┘ └──────┘          │    │  │
│  │  │                 Window 1   Window 2   Window 3     │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │                                                           │  │
│  │  Filter melihat: "3 consecutive coordinates"           │  │
│  │  Bisa deteksi: local patterns dalam 1 landmark!        │  │
│  │  - Apakah x,y,z punya hubungan tertentu?                   │  │
│  │  - Ada pattern geometris lokal?                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Kernel Size vs "View"

```
┌────────────────────────────────────────────────────────────────┐
│                 Kernel Size: Melihat Berapa Landmark?           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Kernel_size=3 (melihat 3 values = 1 landmark)                │
│  [x₀,y₀,z₀]─[x₁,y₁,z₁]─[x₂,y₂,z₂]                           │
│       └──Filter melihat 1 landmark──                           │
│  → Local geometry dalam 1 landmark                             │
│                                                                 │
│  Kernel_size=6 (melihat 6 values = 2 landmarks)              │
│  [x₀,y₀,z₀,x₁,y₁,z₁]─[x₂,y₂,z₂,x₃,y₃,z₃]                     │
│       └─────Filter melihat 2 landmark───────                  │
│  → Relationships antar 2 landmark                               │
│                                                                 │
│  Kernel_size=9 (melihat 9 values = 3 landmarks)               │
│  [x₀,y₀,z₀,x₁,y₁,z₁,x₂,y₂,z₂]─[x₃,y₃,z₃,x₄,y₄,z₄]           │
│       └───────Filter melihat 3 landmark───────────             │
│  → Finger-like patterns (thumb+index+middle)                   │
│                                                                 │
│  Kernel_size=21 (melihat 21 values = 7 landmarks)             │
│  → Whole finger segment                                        │
│                                                                 │
│  Kernel_size=42 (melihat 42 values = half hand)               │
│  → Half-hand patterns                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Filters dan Feature Maps

### Filter = Learned Pattern Detector

```python
import torch
import torch.nn as nn

# Create Conv1d layer
conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)

# Get first filter weights
first_filter = conv.weight[0]  # shape: (3,) for kernel_size=3
print(f"Filter shape: {first_filter.shape}")
print(f"Filter weights: {first_filter.data}")

# First 10 filters
for i in range(10):
    print(f"Filter {i}: {conv.weight[i].data}")
```

### Feature Map Visualization

```
┌────────────────────────────────────────────────────────────────┐
│                    Feature Maps in Conv1d                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (1 channel, 63 values):                                 │
│  [x₀,y₀,z₀,x₁,y₁,z₁,...,x₂₀,y₂₀,z₂₀]                         │
│                                                                 │
│              │                                                  │
│         Conv1d(64 filters)                                       │
│              │                                                  │
│              ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  Feature Map 1 (output channel 0):            │    │   │
│  │  │  [?, ?, ?, ?, ?, ?, ..., ?, ?]                 │    │   │
│  │  │   63 values - kernel detected pattern          │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  Feature Map 2 (output channel 1):            │    │   │
│  │  │  [?, ?, ?, ?, ?, ?, ..., ?, ?]                 │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  ... (64 feature maps total)                   │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  Feature Map 64 (output channel 63):          │    │   │
│  │  │  [?, ?, ?, ?, ?, ?, ..., ?, ?]                 │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Output: (batch, 64, 63) = 64 different feature maps          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Interpreting Feature Maps

```
┌────────────────────────────────────────────────────────────────┐
│               What Each Feature Map Detects?                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Filter 1: "Edge detector"                                    │
│  Input:  [0.5, 0.4, 0.6, 0.3, 0.2, 0.8, ...]                 │
│  Output: [███, ██░, █░░, ███, ░░░, ███, ...]                 │
│          ↑ high activation where values change                 │
│                                                                 │
│  Filter 2: "Peak detector"                                     │
│  Input:  [0.1, 0.2, 0.9, 0.3, 0.1, 0.8, ...]                 │
│  Output: [░░█, ░█░, ███, ░█░, ░░█, ███, ...]                 │
│          ↑ high where maximum in window                         │
│                                                                 │
│  Filter 3: "Valley detector"                                   │
│  Input:  [0.9, 0.2, 0.1, 0.8, 0.3, 0.1, ...]                 │
│  Output: [███, ░░░, ███, ░█░, ░░█, ███, ...]                 │
│          ↑ high where minimum in window                          │
│                                                                 │
│  ... (60 more filters learned automatically)                    │
│                                                                 │
│  Neural network learns sendiri filters apa yang useful           │
│  untuk classification!                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Arsitektur Conv1d BISINDO

### Complete BISINDO CNN Architecture

```python
import torch.nn as nn

class BISINDO_CNN(nn.Module):
    """
    CNN untuk BISINDO letter recognition
    Input: (batch, 1, 84) - 2-hand landmark data
    Output: (batch, 26) - 26 letters A-Z
    """
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        # ========================================
        # Convolutional Layers
        # ========================================
        self.conv_layers = nn.Sequential(
            # Block 1: Extract low-level patterns
            nn.Conv1d(1, 64, kernel_size=3, padding=1),   # 84 → 84
            nn.ReLU(),
            nn.MaxPool1d(2),                               # 84 → 42
            
            # Block 2: Extract mid-level patterns
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 42 → 42
            nn.ReLU(),
            nn.MaxPool1d(2),                               # 42 → 21
            
            # Block 3: Extract high-level patterns
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # 21 → 21
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)                        # 21 → 1
        )
        
        # ========================================
        # Fully Connected Layers
        # ========================================
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),       # Prevent overfitting
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, 84)
        x = self.conv_layers(x)   # (batch, 64, 1)
        x = self.fc_layers(x)      # (batch, 26)
        return x
```

### Shape Transformation

```
┌────────────────────────────────────────────────────────────────┐
│              BISINDO CNN: Shape Transformation                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                        │
│  (batch, 1, 84)                                                │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Conv1d(1→64, k=3, p=1):                                 │  │
│  │ (batch, 1, 84) → (batch, 64, 84)                        │  │
│  │                                                          │  │
│  │ Input: 84 values (21 landmarks × 2 hands × 2 coords)    │  │
│  │ 64 filters = 64 feature maps                             │  │
│  │ Each filter sees 3 consecutive values                   │  │
│  │ Padding=1 keeps length same (84)                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼ ReLU                                                   │
│       │                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ MaxPool1d(2):                                           │  │
│  │ (batch, 64, 84) → (batch, 64, 42)                        │  │
│  │ Halve length, keep all feature maps                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Conv1d(64→128, k=3, p=1):                              │  │
│  │ (batch, 64, 42) → (batch, 128, 42)                       │  │
│  │                                                          │  │
│  │ More filters = more complex patterns detected!            │  │
│  │ 128 different feature patterns                            │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼ ReLU → MaxPool1d(2): 42 → 21                        │
│       │                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Conv1d(128→64, k=3, p=1):                              │  │
│  │ (batch, 128, 21) → (batch, 64, 21)                       │  │
│  │                                                          │  │
│  │ Reduce back to 64 most important features                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼ ReLU                                                   │
│       │                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ AdaptiveAvgPool1d(1):                                   │  │
│  │ (batch, 64, 21) → (batch, 64, 1)                         │  │
│  │                                                          │  │
│  │ Collapse 21 positions → 1 (global average)               │  │
│  │ "Summary" of entire hand!                                │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼ Flatten                                                │
│       │                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Flatten: (batch, 64, 1) → (batch, 64)                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Linear(64→128):                                        │  │
│  │ (batch, 64) → (batch, 128)                              │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼ ReLU → Dropout(0.3)                                 │
│       │                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Linear(128→26):                                         │  │
│  │ (batch, 128) → (batch, 26)                              │  │
│  │                                                          │  │
│  │ Output: logits untuk 26 huruf!                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│       │                                                        │
│  Output: (batch, 26)                                           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Parameter Count

```python
import torch.nn as nn

class BISINDO_CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

model = BISINDO_CNN()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Expected parameter count breakdown:**

| Layer | Parameters |
|-------|------------|
| Conv1d(1→64, k=3) | 1×64×3 + 64 = 256 |
| Conv1d(64→128, k=3) | 64×128×3 + 128 = 24,576 |
| Conv1d(128→64, k=3) | 128×64×3 + 64 = 24,640 |
| Linear(64→128) | 64×128 + 128 = 8,320 |
| Linear(128→26) | 128×26 + 26 = 3,354 |
| **Total** | **~61,000** |

Compare dengan MLP:
- MLP(84→128→64→26) = 84×128 + 128 + 128×64 + 64 + 64×26 + 26 = **18,000+**

CNN lebih banyak parameters, tapi lebih efficient karena shared weights!

---

## Pooling dalam 1D

### MaxPool1d

```python
import torch
import torch.nn as nn

# MaxPool1d(kernel_size, stride)
pool = nn.MaxPool1d(kernel_size=2, stride=2)

# Input: (batch, channels, length)
x = torch.randn(4, 64, 84)  # 84 values

# Output: halved length
output = pool(x)
print(f"Output shape: {output.shape}")  # (4, 64, 42)
```

### Visualisasi MaxPool1d

```
┌────────────────────────────────────────────────────────────────┐
│                  MaxPool1d Visualization                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (12 values):                                            │
│  Index:  [0    1    2    3    4    5    6    7    8    9   10  11]
│  Values:[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
│          ├──────┤├──────┤├──────┤├──────┤├──────┤├──────┤      │
│          Window1 Window2 Window3 Window4 Window5 Window6       │
│           (k=2)   (k=2)   (k=2)   (k=2)   (k=2)   (k=2)      │
│                                                                 │
│  Max pooling: take MAXIMUM from each window                      │
│                                                                 │
│  Window 1: [1.0, 2.0] → max = 2.0                             │
│  Window 2: [3.0, 4.0] → max = 4.0                             │
│  Window 3: [5.0, 6.0] → max = 6.0                             │
│  Window 4: [7.0, 8.0] → max = 8.0                             │
│  Window 5: [9.0, 10.0] → max = 10.0                          │
│  Window 6: [11.0, 12.0] → max = 12.0                         │
│                                                                 │
│  Output (6 values):                                             │
│  Index:  [0      1      2      3      4      5    ]           │
│  Values:[2.0,   4.0,   6.0,   8.0,   10.0,   12.0]           │
│                                                                 │
│  Input length: 12, Pool size: 2, Output length: 6              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### AdaptiveAvgPool1d

```python
import torch
import torch.nn as nn

# AdaptiveAvgPool1d(output_size)
# Automatically calculates kernel/stride to produce desired output size
adaptive_pool = nn.AdaptiveAvgPool1d(1)

# Input: variable length
x = torch.randn(4, 64, 42)  # length could be 21, 42, 84, etc.

# Output: ALWAYS (batch, channels, 1)
output = adaptive_pool(x)
print(f"Output shape: {output.shape}")  # (4, 64, 1)
```

**Kenapa AdaptiveAvgPool1d(1)?**

```
┌────────────────────────────────────────────────────────────────┐
│           Kenapa Pool ke Size 1? (Global Average Pooling)       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Without GAP: you MUST know input size before defining model    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Example: Conv1d → 42 positions                          │   │
│  │ GAP: adaptive, works for ANY input length               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  GAP(1) means: "summarize entire sequence into ONE value"      │
│                                                                 │
│  Input:  [v₀, v₁, v₂, v₃, ..., v₂₀]                           │
│                ↓                                               │
│  GAP(1): [ avg(v₀:v₂₀) ]                                       │
│                                                                 │
│  Each of 64 channels → 1 average value                          │
│  64 channels = 64 "summarized" features                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Padding dan Stride

### Padding

```python
import torch
import torch.nn as nn

# No padding (default)
conv_no_pad = nn.Conv1d(1, 1, kernel_size=3)
x = torch.randn(1, 1, 7)  # length 7
output = conv_no_pad(x)
print(f"No padding: {output.shape[2]}")  # 7 - 3 + 1 = 5

# With padding=1
conv_pad = nn.Conv1d(1, 1, kernel_size=3, padding=1)
output = conv_pad(x)
print(f"With padding=1: {output.shape[2]}")  # 7 (same as input)
```

### Stride

```python
import torch
import torch.nn as nn

# Stride = how much to slide filter each step
conv_stride1 = nn.Conv1d(1, 1, kernel_size=3, stride=1)
conv_stride2 = nn.Conv1d(1, 1, kernel_size=3, stride=2)

x = torch.randn(1, 1, 13)

output1 = conv_stride1(x)
output2 = conv_stride2(x)

print(f"Stride=1: {output1.shape[2]}")  # 13 - 3 + 1 = 11
print(f"Stride=2: {output2.shape[2]}")  # floor((13-3)/2) + 1 = 6
```

### Complete Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                     Padding & Stride Effects                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input length: 7, Kernel size: 3                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Padding=0, Stride=1:                                    │   │
│  │ Input:  [■][■][■][■][■][■][■]                          │   │
│  │           └──┘    └──┘    └──┘    └──┘    └──┘         │   │
│  │ Output: [●][●][●][●][●]                               │   │
│  │ Length: 7 - 3 + 1 = 5                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Padding=1, Stride=1:                                    │   │
│  │ Input:  [0][■][■][■][■][■][■][■][0]  (7+2 padding)   │   │
│  │           └──┘    └──┘    └──┘    └──┘    └──┘         │   │
│  │ Output: [●][●][●][●][●][●][●]                        │   │
│  │ Length: (7+2) - 3 + 1 = 7                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Padding=0, Stride=2:                                    │   │
│  │ Input:  [■][■][■][■][■][■][■]                          │   │
│  │           └──┘       └──┘       └──┘                    │   │
│  │ Output: [●][●][●]                                      │   │
│  │ Length: floor((7-3)/2) + 1 = 3                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Perbandingan dengan ML

### CNN vs Random Forest

| Aspek | Random Forest | CNN |
|-------|-------------|-----|
| **Feature Learning** | Manual features | Automatic features |
| **Architecture** | Ensemble trees | Conv layers + FC |
| **Parameters** | ~18K | ~61K |
| **Training** | Fast (CPU) | Medium (GPU preferred) |
| **Inference Speed** | Fast | Fast |
| **Memory** | Large (100+ trees) | Small |
| **Interpretability** | Feature importance | Low (black box) |

### Kapan Pilih CNN vs RF?

```
┌────────────────────────────────────────────────────────────────┐
│                  Decision: CNN vs Random Forest                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pilih CNN kalau:                                               │
│  ✓ Dataset besar (10K+ samples)                                 │
│  ✓ Ada GPU                                                     │
│  ✓ Butuh best accuracy                                         │
│  ✓ Pattern sangat kompleks                                     │
│                                                                 │
│  Pilih RF kalau:                                                │
│  ✓ Dataset kecil (<10K)                                        │
│  ✓ CPU only                                                    │
│  ✓ Butuh interpretability (feature importance)                  │
│  ✓ Quick prototype                                             │
│                                                                 │
│  BISINDO:                                                       │
│  - Dataset: 26K+ samples ✓                                     │
│  - CNN accuracy: ~98%+                                         │
│  - RF accuracy: ~95%+                                         │
│  - Conclusion: CNN slightly better, but RF is competitive!       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Latihan

### Latihan 1: Conv1d Basics

```python
import torch
import torch.nn as nn

# Create Conv1d layer
conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)

# Input: batch of 4, 1 channel, 10 values
x = torch.randn(4, 1, 10)

# Forward pass
output = conv(x)

print(f"Input shape:  {x.shape}")    # (4, 1, 10)
print(f"Output shape: {output.shape}")  # (4, 8, 10)
```

### Latihan 2: Build BISINDO CNN

```python
import torch
import torch.nn as nn

class BISINDO_CNN(nn.Module):
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        # TODO: Implement
        # Conv1d layers
        # FC layers
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test
model = BISINDO_CNN()
x = torch.randn(8, 1, 84)
output = model(x)
print(f"Output shape: {output.shape}")  # (8, 26)
```

### Latihan 3: Visualize Feature Maps

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create model
model = BISINDO_CNN()

# Hook to capture intermediate outputs
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach())

# Register hook on first conv layer
model.conv_layers[0].register_forward_hook(hook_fn)

# Forward pass
x = torch.randn(1, 1, 84)
output = model(x)

# Plot first 8 feature maps
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < len(activations[0][0]):
        feature_map = activations[0][0, i].numpy()  # First sample, channel i
        ax.plot(feature_map)
        ax.set_title(f'Feature Map {i}')
        ax.grid(True)
plt.tight_layout()
plt.show()
```

### Latihan 4: Parameter Count

```python
import torch.nn as nn

# Calculate parameters for different configurations
def count_params(in_ch, out_ch, kernel):
    # weights + bias
    return in_ch * out_ch * kernel + out_ch

# Conv1d layers
p1 = count_params(1, 64, 3)
p2 = count_params(64, 128, 3)
p3 = count_params(128, 64, 3)

# FC layers
f1 = 64 * 128 + 128
f2 = 128 * 26 + 26

total = p1 + p2 + p3 + f1 + f2
print(f"Conv params: {p1 + p2 + p3:,}")
print(f"FC params: {f1 + f2:,}")
print(f"Total: {total:,}")
```

### Latihan 5: Compare Conv1d and Conv2d

```python
import torch
import torch.nn as nn

# Conv1d for 1D sequence
conv1d = nn.Conv1d(1, 8, kernel_size=3, padding=1)
x_1d = torch.randn(4, 1, 63)
out_1d = conv1d(x_1d)

# Conv2d for "image" representation (21x3)
conv2d = nn.Conv2d(1, 8, kernel_size=3, padding=1)
x_2d = x_1d.view(4, 1, 21, 3)  # Reshape to 2D
out_2d = conv2d(x_2d)

print(f"Conv1d output: {out_1d.shape}")  # (4, 8, 63)
print(f"Conv2d output: {out_2d.shape}")   # (4, 8, 21, 3)

# Conv1d is more natural for sequential landmark data!
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                    Conv1d untuk BISINDO                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Conv1d vs Conv2d                                            │
│     Conv1d: sliding filter along 1 dimension                     │
│     Conv2d: sliding filter along 2 dimensions                    │
│                                                                  │
│  2. Kenapa Conv1d untuk Landmark?                                │
│     Landmarks = sequential positions (0-20)                       │
│     Conv1d captures relationships antar consecutive landmarks    │
│                                                                  │
│  3. Input Shape                                                  │
│     1-hand:  (batch, 1, 63)  = 21 landmarks × 3 coords          │
│     2-hand:  (batch, 1, 84)  = 42 + 42 coords                  │
│                                                                  │
│  4. Kernel Size                                                 │
│     kernel_size=3: melihat 3 consecutive values                  │
│     Bisa detect local patterns (single landmark geometry)         │
│                                                                  │
│  5. Architecture BISINDO                                        │
│     Conv1d(1→64) → Pool(2) → Conv1d(64→128) → Pool(2)         │
│     → Conv1d(128→64) → AdaptiveAvgPool(1) → FC → Output(26)     │
│                                                                  │
│  6. Pooling                                                     │
│     MaxPool1d: downsampling, preserve strongest                 │
│     AdaptiveAvgPool1d: variable → fixed output size             │
│                                                                  │
│  7. Padding & Stride                                            │
│     padding=1: preserve length                                   │
│     stride: step size for sliding                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [07-bisindo-cnn-architecture.md](07-bisindo-cnn-architecture.md) - Arsitektur CNN BISINDO

---

## Referensi

- PyTorch Conv1d docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
- Conv1d for NLP: https://arxiv.org/abs/1508.06615
- 1D CNN for biosignal: https://arxiv.org/abs/1611.06420
