# Training & Hyperparameters

> Penjelasan lengkap tentang gimana cara training neural network: training loop, hyperparameters, dan tips untuk dapetin hasil terbaik.

---

## Daftar Isi

1. [Training Loop](#training-loop)
2. [Loss Function](#loss-function)
3. [Optimizer](#optimizer)
4. [Learning Rate](#learning-rate)
5. [Batch Size](#batch-size)
6. [Epochs](#epochs)
7. [Regularization](#regularization)
8. [Debugging Training](#debugging-training)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Common Issues](#common-issues)

---

## Training Loop

### What is Training Loop?

```
┌────────────────────────────────────────────────────────────────┐
│                    Training Loop Overview                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(num_epochs):                                │
│      │                                                         │
│      ├─► for batch in train_loader:                            │
│      │       │                                                 │
│      │       ├─► 1. Forward pass                              │
│      │       │       input → hidden → output                   │
│      │       │                                                 │
│      │       ├─► 2. Compute loss                              │
│      │       │       loss = criterion(output, target)          │
│      │       │                                                 │
│      │       ├─► 3. Backward pass                             │
│      │       │       loss.backward() → compute gradients        │
│      │       │                                                 │
│      │       └─► 4. Update weights                            │
│      │               optimizer.step() → adjust weights          │
│      │                                                         │
│      └─► Evaluate on validation set                            │
│              → Track progress, prevent overfitting               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Prepare data
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 2. Initialize model
model = BISINDO_CNN(input_dim=84, num_classes=26)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
num_epochs = 50

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        # Forward pass
        optimizer.zero_grad()           # Clear previous gradients
        outputs = model(batch_x)        # Prediksi
        loss = criterion(outputs, batch_y)  # Hitung loss
        
        # Backward pass
        loss.backward()                  # Compute gradients
        optimizer.step()                 # Update weights
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    
    # Validation phase (every 10 epochs)
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = val_outputs.argmax(1)
            val_acc = (val_preds == y_val_tensor).float().mean()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

print("Training complete!")
```

### Step-by-Step Explanation

```python
# Step 1: Clear gradients
optimizer.zero_grad()
# Kenapa perlu?
# - PyTorch ACCUMULATES gradients by default
# - Tanpa clear, gradients akan bertambah terus
# - Hasil: weights update terlalu besar → diverges!

# Step 2: Forward pass
outputs = model(batch_x)
# - Input: (batch_size, 84)
# - Output: (batch_size, 26) logits

# Step 3: Compute loss
loss = criterion(outputs, batch_y)
# - Prediksi vs ground truth
# - Loss tinggi = prediksi jauh dari benar

# Step 4: Backward pass
loss.backward()
# - PyTorch automatically computes gradients
# - d(loss)/d(weights) untuk setiap parameter
# - Tersimpan di .grad attribute

# Step 5: Update weights
optimizer.step()
# - Adjust weights berdasarkan gradients
# - "Gerak ke arah yang mengurangi loss"
```

---

## Loss Function

### CrossEntropyLoss untuk Classification

```python
import torch.nn as nn

# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()

# Input: raw logits (sebelum softmax)
# Target: class indices (0-25), NOT one-hot!
outputs = model(batch_x)  # shape: (batch, 26)
loss = criterion(outputs, batch_y)  # batch_y: (batch,) indices
```

### Alternative Losses

```python
# For binary classification (jika hanya 2 huruf)
criterion = nn.BCEWithLogitsLoss()
# Target: (batch, 1) one-hot atau float

# For multi-label (jika satu sample bisa banyak huruf)
criterion = nn.MultiLabelSoftMarginLoss()
# Target: (batch, num_classes) one-hot

# For regression (jika prediksi koordinat)
criterion = nn.MSELoss()
# Atau:
criterion = nn.L1Loss()
```

### Manual CrossEntropy

```python
def cross_entropy_manual(logits, targets):
    """Manual cross entropy untuk pemahaman."""
    # Step 1: Softmax → probabilities
    exp_logits = torch.exp(logits - logits.max())  # numerical stability
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    
    # Step 2: Negative log likelihood
    N = logits.size(0)
    log_probs = -torch.log(probs.gather(1, targets.view(N, 1)))
    loss = log_probs.mean()
    
    return loss

# Test
logits = torch.randn(4, 26)  # 4 samples, 26 classes
targets = torch.tensor([0, 4, 13, 25])  # class indices

manual = cross_entropy_manual(logits, targets)
official = nn.CrossEntropyLoss()(logits, targets)

print(f"Manual: {manual.item():.4f}")
print(f"Official: {official.item():.4f}")  # Should be same!
```

---

## Optimizer

### Adam Optimizer

**Adam (Adaptive Moment Estimation)** = kombinasi SGD + momentum + RMSprop.

```python
import torch.optim as optim

# Adam with default parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adam with custom parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # Momentum coefficients
    eps=1e-8,           # Numerical stability
    weight_decay=0      # L2 regularization
)
```

### Adam vs SGD

```python
# SGD with momentum (classic)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Adam (adaptive learning rate)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,  # Lower lr than SGD
    betas=(0.9, 0.999)
)

# AdamW (Adam with better regularization)
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

### When to Use Each

```
┌────────────────────────────────────────────────────────────────┐
│               Optimizer Comparison                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Adam:                                                         │
│  ✓ Best for: most cases, quick experiments                    │
│  ✓ Pros: adaptive lr, converges fast                          │
│  ✗ Cons: may generalize less than SGD                         │
│                                                                 │
│  SGD + Momentum:                                              │
│  ✓ Best for: final production training                        │
│  ✓ Pros: better generalization                               │
│  ✗ Cons: needs more tuning, slower convergence                │
│                                                                 │
│  AdamW:                                                       │
│  ✓ Best for: transformers, large models                     │
│  ✓ Pros: better weight decay                                  │
│  ✗ Cons: newer, less tested                                  │
│                                                                 │
│  Our BISINDO model:                                           │
│  → Adam is fine (model small, quick iteration)                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Learning Rate

### Apa itu Learning Rate?

```
Learning rate = "seberapa besar" kita update weights setiap step.

┌────────────────────────────────────────────────────────────────┐
│              Learning Rate Effect                                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LR TOO HIGH (e.g., 1.0):                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Loss                                                  │  │
│  │    │                                                   │  │
│  │    │         ╱╲    ╱╲                                 │  │
│  │    │       ╱    ╲╱    ╲                              │  │
│  │    │    ╱              ╲                             │  │
│  │    │──╱──────────────────╲────→ Epoch               │  │
│  │                                                        │  │
│  │  Oscillates! Never converges!                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  LR TOO LOW (e.g., 0.00001):                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Loss                                                  │  │
│  │    │                                                   │  │
│  │    │                                                   │  │
│  │    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                        │  │
│  │    │                                                   │  │
│  │    └───────────────────────────────────────────→     │  │
│  │         Takes forever to converge!                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  LR JUST RIGHT (e.g., 0.001):                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Loss                                                  │  │
│  │    │                                                   │  │
│  │    │                                                   │  │
│  │    │                                                  │  │
│  │    │ ╲                                                │  │
│  │    │  ╲                                               │  │
│  │    └───╲───────────────────────────────────────→     │  │
│  │         ╲ Converges nicely!                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Learning Rate Scheduling

```python
# Option 1: Step LR
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=20,    # Reduce every 20 epochs
    gamma=0.1         # Multiply lr by 0.1
)

# Option 2: Reduce on Plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Reduce when loss stops decreasing
    factor=0.1,       # Multiply by 0.1
    patience=5,       # Wait 5 epochs
    verbose=True
)

# Option 3: Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,         # Max epochs
    eta_min=1e-6      # Minimum lr
)

# Training loop with scheduler
for epoch in range(num_epochs):
    train()
    scheduler.step()  # Update lr
```

### Learning Rate Finder

```python
def find_lr(model, train_loader, criterion, optimizer, start_lr=1e-7, end_lr=1):
    """Find optimal learning rate."""
    
    losses = []
    lrs = []
    model.train()
    
    optimizer.param_groups[0]['lr'] = start_lr
    current_lr = start_lr
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(current_lr)
        
        current_lr *= 1.0001  # Gradually increase
        optimizer.param_groups[0]['lr'] = current_lr
        
        if current_lr > end_lr:
            break
    
    # Plot losses vs lrs
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    
    # Find optimal lr (where loss is decreasing fastest)
```

---

## Batch Size

### Apa itu Batch Size?

```
┌────────────────────────────────────────────────────────────────┐
│                    Batch Size Visualization                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset: 10,000 samples                                        │
│                                                                 │
│  Batch size 32:                                                 │
│  ┌────┬────┬────┬────┬────┐                                   │
│  │ 32 │ 32 │ 32 │ 32 │...│ → 313 batches per epoch          │
│  └────┴────┴────┴────┴────┘                                   │
│                                                                 │
│  Batch size 256:                                                │
│  ┌──────────┬──────────┬──────────┬──────────┐                │
│  │   256   │   256   │   256   │   ...   │ → 40 batches    │
│  └──────────┴──────────┴──────────┴──────────┘               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Batch Size Effects

| Batch Size | Pros | Cons | Best For |
|------------|------|------|----------|
| Small (16-32) | Better generalization, noisy gradient | Slow, unstable | Small datasets |
| Medium (64-256) | Balanced | - | Most cases |
| Large (512+) | Fast, stable | Memory issues, may generalize less | Large datasets, GPU |

### Batch Size untuk BISINDO

```python
# BISINDO training configuration
batch_size = 256  # Good default for most GPUs

# With GPU memory constraints
batch_size = 128  # If out of memory

# Small model or limited GPU
batch_size = 64   # Safer choice
```

### Gradient Accumulation

```python
# For large effective batch size with limited memory
effective_batch_size = 2048
actual_batch_size = 256
accumulation_steps = effective_batch_size // actual_batch_size

optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients
```

---

## Epochs

### Apa itu Epoch?

```
Epoch = 1 pass through entire training dataset.

Dataset: 10,000 samples
Batch size: 256
Batches per epoch: 10,000 / 256 = 40

Epoch 1: See all 10,000 samples once
Epoch 2: See all 10,000 samples again (2nd time)
...
Epoch 50: See all 10,000 samples (50th time)
```

### How Many Epochs?

```
┌────────────────────────────────────────────────────────────────┐
│                   Training Dynamics                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss                                                    Time   │
│    │                                                       │   │
│    │  Train loss:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                    │   │
│    │                      ╲                                  │   │
│    │                       ╲                                 │   │
│    │  Val loss:       ─────╲────────────                    │   │
│    │                             ╲                            │   │
│    │                              ╲ (starts overfitting!)  │   │
│    │                                                       │   │
│    │         │         │         │         │    Epoch     │   │
│    │         10        20        30        40             │   │
│    │                                                       │   │
│    │  → Optimal point: before val loss increases           │   │
│    │                                                       │   │
└────────────────────────────────────────────────────────────────┘
```

### Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping! No improvement for {self.patience} epochs")

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train()
    val_loss = validate()
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Stopped early!")
        break
```

### Save Best Model

```python
best_val_acc = 0.0

for epoch in range(num_epochs):
    train()
    val_acc = validate()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pt')
        print(f"Saved best model with acc: {val_acc:.4f}")
```

---

## Regularization

### Kenapa Butuh Regularization?

```
┌────────────────────────────────────────────────────────────────┐
│                 Overfitting vs Generalization                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Overfitting:                                                   │
│  - Model menghafal training data                                │
│  - Train acc: 100%, Val acc: 70%                              │
│  - "Kelemahan: tidak bisa generalize ke data baru"             │
│                                                                 │
│  Good generalization:                                          │
│  - Model belajar pattern, bukan menghafal                      │
│  - Train acc: 98%, Val acc: 96%                              │
│  - "Sempurna! Bisa handle data baru"                          │
│                                                                 │
│  Underfitting:                                                 │
│  - Model belum belajar cukup                                   │
│  - Train acc: 70%, Val acc: 68%                              │
│  - "Model terlalu sederhana"                                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Regularization Techniques

```python
# 1. Dropout
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Dropout(0.3),  # 30% neurons "off" during training
    nn.Linear(128, 26)
)

# 2. Weight Decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. BatchNorm (normalizes activations)
model = nn.Sequential(
    nn.Conv1d(1, 64, 3, padding=1),
    nn.BatchNorm1d(64),  # Normalize activations
    nn.ReLU(),
    ...
)

# 4. Data Augmentation (menambah variasi data)
# More diverse data = less overfitting
```

### Dropout Deep Dive

```python
import torch.nn as nn

# Dropout(p=0.5) during training:
# 50% neurons randomly set to 0
#
# Training:
# ┌─────────────────────────────────────────────────────────┐
# │ Input: [1.0, 2.0, 3.0, 4.0, 5.0]                    │
# │          ↓                                              │
# │ Dropout(0.5): [1.0, 0, 3.0, 0, 5.0] (random!)        │
# │          ↓                                              │
# │ Scale: [0.5, 0, 1.5, 0, 2.5] (× 1/(1-p))            │
# └─────────────────────────────────────────────────────────┘
#
# Testing:
# ┌─────────────────────────────────────────────────────────┐
# │ Dropout DISABLED: [1.0, 2.0, 3.0, 4.0, 5.0]           │
# │ (All neurons active, scaled by (1-p) automatically!)    │
# └─────────────────────────────────────────────────────────┘
```

---

## Debugging Training

### Common Issues dan Solutions

```
┌────────────────────────────────────────────────────────────────┐
│                  Training Debugging Guide                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Issue: Loss NaN                                               │
│  ├─ Cause: Learning rate too high                              │
│  ├─ Cause: Division by zero                                   │
│  ├─ Cause: Log of zero in loss                               │
│  └─ Fix: Reduce lr, add epsilon, check data                   │
│                                                                 │
│  Issue: Loss not decreasing                                    │
│  ├─ Cause: Learning rate too low                              │
│  ├─ Cause: Wrong model architecture                           │
│  ├─ Cause: Bug in loss computation                           │
│  └─ Fix: Increase lr, check model, verify labels              │
│                                                                 │
│  Issue: Training acc >> Val acc (overfitting)                 │
│  ├─ Cause: Model too complex                                   │
│  ├─ Cause: Not enough data                                     │
│  └─ Fix: Add dropout, augment data, early stopping              │
│                                                                 │
│  Issue: Val acc >> Train acc (data mismatch)                  │
│  ├─ Cause: Train/val distribution different                   │
│  └─ Fix: Check data pipeline, shuffling                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Debugging Checklist

```python
# 1. Check data shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Classes: {np.unique(y_train)}")

# 2. Check for NaN/Inf
assert not np.any(np.isnan(X_train)), "NaN in X_train!"
assert not np.any(np.isinf(X_train)), "Inf in X_train!"

# 3. Check model output
model.eval()
with torch.no_grad():
    out = model(X_train_tensor[:10])
    print(f"Output shape: {out.shape}")  # Should be (10, 26)
    print(f"Output range: [{out.min():.2f}, {out.max():.2f}]")

# 4. Check loss
criterion = nn.CrossEntropyLoss()
loss = criterion(out, y_train_tensor[:10])
print(f"Initial loss: {loss.item():.4f}")

# 5. Check gradients
model.train()
out = model(X_train_tensor[:10])
loss = criterion(out, y_train_tensor[:10])
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
        if param.grad.norm() > 10:
            print("  ⚠️  Gradient exploding!")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

```
┌────────────────────────────────────────────────────────────────┐
│               Hyperparameter Summary                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hyperparameter       │ Recommended │ BISINDO Value           │
│  ──────────────────────────────────────────────────────────── │
│  Learning rate        │ 0.001-0.01 │ 0.001 (Adam)            │
│  Batch size           │ 32-256     │ 256                     │
│  Epochs               │ 50-200     │ 50                      │
│  Dropout              │ 0.1-0.5    │ 0.3                     │
│  Weight decay         │ 1e-4-1e-2  │ 0 (Adam) or 1e-4 (SGD)  │
│  ──────────────────────────────────────────────────────────── │
│                                                                 │
│  Architecture-specific:                                          │
│  Conv channels       │ 32-256     │ 64, 128, 64              │
│  Kernel size        │ 3-7        │ 3                        │
│  FC hidden size      │ 64-512     │ 128                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Grid Search

```python
from itertools import product

# Define hyperparameter space
lr_values = [0.001, 0.0001]
batch_values = [128, 256]
dropout_values = [0.2, 0.3, 0.5]

best_acc = 0
best_params = {}

for lr, batch, dropout in product(lr_values, batch_values, dropout_values):
    # Create model
    model = BISINDO_CNN()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch)
    
    # Train
    train_model(model, train_loader, optimizer, epochs=30)
    
    # Evaluate
    acc = evaluate(model, val_loader)
    
    print(f"lr={lr}, batch={batch}, dropout={dropout} → acc={acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_params = {'lr': lr, 'batch': batch, 'dropout': dropout}

print(f"\nBest params: {best_params}")
print(f"Best acc: {best_acc:.4f}")
```

### Random Search

```python
import random

# Random search is often more efficient than grid search!
best_acc = 0
best_params = {}

for trial in range(20):
    lr = random.choice([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    batch = random.choice([64, 128, 256, 512])
    dropout = random.uniform(0.1, 0.5)
    
    # Train and evaluate...
    
    if acc > best_acc:
        best_acc = acc
        best_params = {'lr': lr, 'batch': batch, 'dropout': dropout}

print(f"Best params: {best_params}")
```

---

## Common Issues

### Issue 1: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
batch_size = 64  # or 32

# Solution 2: Clear cache
torch.cuda.empty_cache()

# Solution 3: Gradient checkpointing
# (trade compute for memory)
model.gradient_checkpointing_enable()

# Solution 4: Use smaller model
# (fewer channels)
```

### Issue 2: Model Not Converging

```python
# Check 1: Learning rate
lr = 0.001  # Good for Adam
# or
lr = 0.1    # Good for SGD

# Check 2: Data normalization
X_train = (X_train - X_train.mean()) / X_train.std()

# Check 3: Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

### Issue 3: Overfitting

```python
# Solution 1: More data
# (best solution)

# Solution 2: Data augmentation
# (rotate, scale, translate landmarks)

# Solution 3: Regularization
model = nn.Sequential(
    ...
    nn.Dropout(0.5),  # Increase dropout
)

# Solution 4: Early stopping
# (stop when val loss stops decreasing)

# Solution 5: Smaller model
# (reduce channel sizes)
```

---

## Latihan

### Latihan 1: Implement Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=50):
    """Complete training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                preds = outputs.argmax(1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {val_acc:.4f}")
```

### Latihan 2: Learning Rate Finder

```python
def plot_lr_finder(model, train_loader, start_lr=1e-7, end_lr=1, num_steps=100):
    """Plot loss vs learning rate."""
    import matplotlib.pyplot as plt
    
    optimizer = optim.SGD(model.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss()
    
    lrs = []
    losses = []
    multiplier = (end_lr / start_lr) ** (1 / num_steps)
    
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        if i >= num_steps:
            break
        
        optimizer.param_groups[0]['lr'] = start_lr * (multiplier ** i)
        lrs.append(optimizer.param_groups[0]['lr'])
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.show()
```

### Latihan 3: Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Usage
early_stop = EarlyStopping(patience=10)
for epoch in range(100):
    train()
    val_loss = validate()
    if early_stop.should_stop(val_loss):
        print(f"Stop at epoch {epoch+1}")
        break
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                   Training Summary                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Training Loop                                               │
│     zero_grad → forward → loss → backward → step                 │
│                                                                  │
│  2. Loss Function                                               │
│     CrossEntropyLoss untuk multi-class classification            │
│                                                                  │
│  3. Optimizer                                                  │
│     Adam: lr=0.001, adaptive learning rate                     │
│                                                                  │
│  4. Learning Rate                                               │
│     Too high → diverge, Too low → slow                         │
│     Scheduler: StepLR, ReduceLROnPlateau, CosineAnnealing      │
│                                                                  │
│  5. Batch Size                                                  │
│     256 default, adjust based on GPU memory                     │
│                                                                  │
│  6. Epochs                                                      │
│     50 typical, early stopping untuk prevent overfitting         │
│                                                                  │
│  7. Regularization                                              │
│     Dropout, weight decay, early stopping                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [09-evaluation.md](09-evaluation.md) - Evaluation Metrics

---

## Referensi

- PyTorch Optimizers: https://pytorch.org/docs/stable/optim.html
- Adam paper: https://arxiv.org/abs/1412.6980
- Cyclical Learning Rates: https://arxiv.org/abs/1506.01186
