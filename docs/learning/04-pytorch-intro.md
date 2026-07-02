# PyTorch Basics

> Di file sebelumnya kita pakai NumPy. Sekarang kita移 ke PyTorch - library deep learning yang lebih powerful dengan automatic differentiation (calculus otomatis).

---

## Daftar Isi

1. [Kenapa PyTorch?](#kenapa-pytorch)
2. [Tensors: NumPy on Steroids](#tensors-numpy-on-steroids)
3. [Neural Network Modules](#neural-network-modules)
4. [Cross-Entropy Loss](#cross-entropy-loss)
5. [Optimizer](#optimizer)
6. [Training Loop](#training-loop)
7. [GPU Acceleration](#gpu-acceleration)
8. [PyTorch vs NumPy](#pytorch-vs-numpy)
9. [Latihan](#latihan)

---

## Kenapa PyTorch?

### Kekurangan NumPy untuk Deep Learning

```python
import numpy as np

# NumPy TIDAK bisa:
# 1. Compute gradients automatically
# 2. Run on GPU
# 3. Define complex layer architectures easily

# Misal, kita mau compute gradient di NumPy:
def loss_fn(weights, x, y):
    prediction = np.dot(x, weights)
    return np.mean((prediction - y) ** 2)

# Kamu harus hitung gradientnya MANUAL:
def gradient_fn(weights, x, y):
    prediction = np.dot(x, weights)
    grad = 2 * np.mean((prediction - y) * x, axis=0)
    return grad

# Ribet kan? apalagi untuk 100+ layers!
```

### Kelebihan PyTorch

```python
import torch

# PyTorch BISA:
# 1. Compute gradients automatically ✓
# 2. Run on GPU automatically ✓
# 3. Define layers easily with nn.Module ✓

x = torch.randn(100, 63)  # 100 samples, 63 features
linear = torch.nn.Linear(63, 128)
output = linear(x)

# Gradients dihitung otomatis!
output.sum().backward()  # PyTorch hitung gradients
print(linear.weight.grad)  # Gradient untuk weights
```

---

## Tensors: NumPy on Steroids

### Apa itu Tensor?

**Tensor** = generalization dari matrix:
- 0D tensor = scalar (1 number)
- 1D tensor = vector (array)
- 2D tensor = matrix (table)
- 3D+ tensor = "higher dimensional matrix"

### NumPy vs PyTorch

```python
import numpy as np
import torch

# NumPy Array
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy shape: {np_array.shape}")  # (2, 3)

# PyTorch Tensor
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Torch shape: {torch_tensor.shape}")  # torch.Size([2, 3])

# Convert NumPy → Tensor
np_array = np.random.randn(100, 63)
tensor = torch.from_numpy(np_array.astype(np.float32))

# Convert Tensor → NumPy
numpy_back = tensor.numpy()
```

### Shape Conventions

```python
# Single sample, 63 features
x = torch.randn(63)           # shape: (63,)

# Batch of 100 samples, 63 features
x = torch.randn(100, 63)       # shape: (100, 63)

# For CNN: (batch, channels, length)
x = torch.randn(100, 1, 63)    # shape: (100, 1, 63)
```

### Tensor Operations

```python
import torch

# Basic operations (sama kayak NumPy)
a = torch.randn(3, 4)
b = torch.randn(3, 4)

c = a + b              # Addition
c = torch.matmul(a, b.T)  # Matrix multiplication
c = a * 2              # Scalar multiplication
c = a.sum()            # Sum all elements
c = a.mean()           # Mean
c = a.max()            # Max
c = a.argmax()         # Index of max

# Reshape
x = torch.randn(100, 63)
x_reshaped = x.view(100, -1)  # -1 means "infer this dimension"
x_reshaped = x.reshape(100, 1, 63)  # Explicit reshape for CNN
```

### Requires Gradient

Ini **kunci** PyTorch:

```python
# Buat tensor yang akan di-optimize
weights = torch.randn(63, 128, requires_grad=True)
bias = torch.zeros(128, requires_grad=True)

# Forward pass
output = torch.matmul(input, weights) + bias
loss = output.sum()

# Backward pass (compute gradients)
loss.backward()

# gradients ada di .grad attribute
print(weights.grad)  # d(loss)/d(weights)
```

---

## Neural Network Modules

### nn.Module: Base Class

PyTorch menyediakan `nn.Module` untuk define neural networks:

```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # Wajib panggil parent constructor
        self.linear1 = nn.Linear(63, 128)
        self.linear2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 26)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.output(x)
        return x
```

### nn.Sequential: Quick Network

Untuk network sederhana, bisa pakai `nn.Sequential`:

```python
import torch.nn as nn

# Cara 1: Sequential (simple)
model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 26)
)

# Cara 2: Class (more flexible)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### Common Layers

```python
import torch.nn as nn

# Linear (fully connected)
nn.Linear(in_features, out_features)  # y = xW^T + b

# Activation functions
nn.ReLU()      # max(0, x)
nn.Sigmoid()    # 1/(1+e^-x)
nn.Tanh()       # (e^x - e^-x)/(e^x + e^-x)
nn.Softmax(dim)  # probabilities, specify dim

# Dropout (regularization)
nn.Dropout(p=0.5)  # 50% neurons di-"matikan" during training

# Normalization
nn.BatchNorm1d(num_features)  # normalize activations
```

---

## Cross-Entropy Loss

### Apa itu Cross-Entropy?

**Cross-Entropy Loss** = ukuran "ketidakpastian" antara prediction dan label.

```
Prediction: [0.1, 0.2, 0.7]  (70% yakin huruf C)
Actual:     [0, 0, 1]         ( sebenarnya huruf C)

Cross-Entropy = -log(0.7) = 0.357 (low loss = good)

Prediction: [0.1, 0.2, 0.3]  (ga yakin siapa)
Actual:     [0, 0, 1]         (tetap huruf C)

Cross-Entropy = -log(0.3) = 1.20 (high loss = bad)
```

### Di PyTorch

```python
import torch
import torch.nn as nn

# Define loss function
criterion = nn.CrossEntropyLoss()

# Forward pass
outputs = model(x_batch)  # shape: (batch_size, 26)
labels = torch.tensor([0, 3, 5, ...])  # indices, NOT one-hot!

# Compute loss
loss = criterion(outputs, labels)

print(f"Loss: {loss.item():.4f}")

# Atau pakai logits langsung (lebih numerically stable)
logits = model(x_batch)  # Pre-softmax outputs
loss = nn.functional.cross_entropy(logits, labels)
```

### Kenapa Pakai Cross-Entropy?

```python
# Cross-Entropy = Negative Log Likelihood
# Equivalent dengan: -log(predicted_probability_of_correct_class)

# Contoh: target class = 2
predicted_probs = torch.tensor([0.1, 0.2, 0.7])  # 70% for class 2

loss = -torch.log(predicted_probs[2])
print(loss)  # -log(0.7) = 0.357

# Kalau prediction 100% salah:
predicted_probs = torch.tensor([0.7, 0.2, 0.1])  # 70% for class 0, bukan 2
loss = -torch.log(predicted_probs[2])
print(loss)  # -log(0.1) = 2.302 (penalty besar!)
```

---

## Optimizer

### Apa itu Optimizer?

**Optimizer** = algorithm yang update weights berdasarkan gradients untuk minimize loss.

```
1. Forward pass: hitung loss
2. Backward pass: hitung gradients (dL/dW)
3. Update weights: W = W - learning_rate * gradients
```

### Adam: Adaptive Moment Estimation

**Adam** = optimizer paling populer. Good default choice.

```python
import torch.optim as optim

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()     # 1. Clear previous gradients
    outputs = model(x_batch)  # 2. Forward pass
    loss = criterion(outputs, y_batch)  # 3. Compute loss
    loss.backward()           # 4. Backward pass (compute gradients)
    optimizer.step()          # 5. Update weights
```

### Learning Rate

```python
# Learning rate terlalu besar: oscillates, tidak converge
optimizer = optim.Adam(model.parameters(), lr=1.0)  # ❌ Too big

# Learning rate terlalu kecil: lama converge
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # ❌ Too small

# Learning rate pas: converge smooth
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ✅ Good default
```

### Learning Rate Scheduler

Turunkan learning rate seiring waktu:

```python
# Option 1: Step LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Option 2: Reduce on Plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training loop
for epoch in range(100):
    train_loss = ...
    scheduler.step(train_loss)  # Otomatis turunkan LR kalau loss plateau
```

---

## Training Loop

### Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Prepare data
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # Labels as indices (0-25)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 2. Define model
model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 26)
)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set to training mode (important for dropout/batchnorm)
    
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == y_val_tensor).float().mean()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Acc: {accuracy:.4f}")
```

### Model Parameters

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

## GPU Acceleration

### Check GPU Availability

```python
import torch

# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# atau langsung
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Move Data to GPU

```python
import torch

# Create model on GPU
device = torch.device("cuda")
model = MyModel().to(device)

# Move data to GPU
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)

# Prediction on GPU
outputs = model(x_batch)
loss = criterion(outputs, y_batch)
```

### Complete GPU Training

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Model to GPU
model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 26)
).to(device)

# Training
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
```

---

## PyTorch vs NumPy

| Aspek | NumPy | PyTorch |
|-------|-------|---------|
| **Device** | CPU only | CPU + GPU |
| **Gradients** | Manual | Automatic |
| **Layers** | Manual matrix ops | Pre-built modules |
| **Training** | Write yourself | Built-in support |
| **Speed** | Fast (CPU) | Faster (GPU) |
| **Debugging** | Standard debugger | `torch.set_printoptions()` |

### When to Use Each

```python
# Use NumPy when:
# - Learning ML basics
# - Simple operations
# - Quick prototyping (no GPU needed)

# Use PyTorch when:
# - Deep learning (CNNs, RNNs, Transformers)
# - Need GPU acceleration
# - Production models
# - Complex architectures
```

---

## Latihan

### Latihan 1: Tensor Basics

```python
import torch

# Buat tensor 3x4 dengan random values
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Mean: {x.mean()}")

# Convert ke NumPy
np_x = x.numpy()
print(f"NumPy type: {type(np_x)}")
```

### Latihan 2: Linear Layer

```python
import torch
import torch.nn as nn

# Linear layer: 63 → 128
linear = nn.Linear(63, 128)

# Input: 1 sample
x = torch.randn(63)
output = linear(x)
print(f"Output shape: {output.shape}")  # (128,)

# Input: batch of 32
x = torch.randn(32, 63)
output = linear(x)
print(f"Output shape: {output.shape}")  # (32, 128)
```

### Latihan 3: Simple MLP

```python
import torch
import torch.nn as nn

# Define MLP
model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 26)
)

# Test forward pass
x = torch.randn(10, 63)  # 10 samples
output = model(x)
print(f"Output shape: {output.shape}")  # (10, 26)
```

### Latihan 4: Cross Entropy Loss

```python
import torch
import torch.nn as nn

# Create random logits
logits = torch.randn(5, 26)  # 5 samples, 26 classes
labels = torch.tensor([0, 4, 13, 25, 7])  # Ground truth indices

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
print(f"Loss: {loss.item():.4f}")

# Get predictions
_, preds = torch.max(logits, dim=1)
print(f"Predictions: {preds}")
print(f"Accuracy: {(preds == labels).float().mean():.2f}")
```

### Latihan 5: Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
model = nn.Sequential(
    nn.Linear(63, 128),
    nn.ReLU(),
    nn.Linear(128, 26)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
X = torch.randn(1000, 63)
y = torch.randint(0, 26, (1000,))

# Training
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete!")
```

### Latihan 6: GPU Training

```python
import torch
import torch.nn as nn

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model to GPU
model = nn.Linear(63, 26).to(device)

# Data to GPU
x = torch.randn(32, 63).to(device)
y = torch.randint(0, 26, (32,)).to(device)

# Forward pass on GPU
output = model(x)
print(f"Output device: {output.device}")
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                         PyTorch Basics                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Tensors                                                       │
│     GPU-accelerated arrays with automatic differentiation         │
│                                                                  │
│  2. nn.Module                                                    │
│     Base class untuk neural networks                              │
│     nn.Linear, nn.ReLU, nn.Sequential                           │
│                                                                  │
│  3. CrossEntropyLoss                                             │
│     Standard loss untuk multi-class classification               │
│                                                                  │
│  4. Adam Optimizer                                               │
│     lr=0.001, adaptive learning rate                            │
│                                                                  │
│  5. Training Loop                                               │
│     zero_grad → forward → loss → backward → step                 │
│                                                                  │
│  6. GPU                                                          │
│     .to(device) untuk accelerate                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [05-cnn-intro.md](05-cnn-intro.md) - CNN Introduction

---

## Referensi

- PyTorch Official Tutorial: https://pytorch.org/tutorials/
- PyTorch Documentation: https://pytorch.org/docs/
- DEEP LEARNING WITH PYTORCH: 60-Minute Blitz
