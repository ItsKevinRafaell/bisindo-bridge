# Hands-on Exercises

> Latihan praktis untuk mengasah pemahaman CNN. Kerjakan dari atas ke bawah, pelan-pelan. Good luck!

---

## Daftar Isi

1. [Setup](#setup)
2. [Exercise 1: Load & Explore Data](#exercise-1-load--explore-data)
3. [Exercise 2: Implement Perceptron with NumPy](#exercise-2-implement-perceptron-with-numpy)
4. [Exercise 3: Build MLP with PyTorch](#exercise-3-build-mlp-with-pytorch)
5. [Exercise 4: Implement Conv1d Layer](#exercise-4-implement-conv1d-layer)
6. [Exercise 5: Build BISINDO CNN](#exercise-5-build-bisindo-cnn)
7. [Exercise 6: Training Loop](#exercise-6-training-loop)
8. [Exercise 7: Evaluation](#exercise-7-evaluation)
9. [Exercise 8: Debug Feature Maps](#exercise-8-debug-feature-maps)
10. [Bonus: Challenge Problems](#bonus-challenge-problems)

---

## Setup

### Install Dependencies

```bash
# PyTorch
pip install torch torchvision

# Scientific computing
pip install numpy pandas scikit-learn matplotlib seaborn

# Check installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Download BISINDO Dataset

```bash
# Dataset ada di:
ls dataset/landmarks_captured_v2.csv

# Atau gunakan script untuk download:
python -c "import pandas as pd; df = pd.read_csv('dataset/landmarks_captured_v2.csv'); print(df.shape)"
```

---

## Exercise 1: Load & Explore Data

### Tujuan
- Load dataset BISINDO landmarks
- Explore data structure
- Understand feature layout

### Tasks

```python
import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv('dataset/landmarks_captured_v2.csv')

print("Dataset info:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns[:10].tolist()} ... (total {len(df.columns)})")
print(f"  Letters: {sorted(df['letter'].unique())}")

# 2. Check class distribution
print("\nClass distribution:")
print(df['letter'].value_counts().sort_index())

# 3. Extract features
feature_cols = [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
X = df[feature_cols].values
y = df['letter'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 4. Check for NaN
print(f"\nNaN values: {np.isnan(X).sum()}")

# 5. Reshape sample to visualize hand
sample = X[0].reshape(21, 3)
print(f"\nFirst sample (reshaped to 21x3):")
print(sample[:5])  # First 5 landmarks
```

### Expected Output

```
Dataset info:
  Shape: (26192, 69)
  Columns: ['letter', 'image_path', 'split', ...]
  Letters: ['A', 'B', 'C', ...]

Feature matrix shape: (26192, 63)
Labels shape: (26192,)

NaN values: 0

First sample (reshaped to 21x3):
[[0.278 0.791 0.   ]
 [0.276 0.779 0.   ]
 [0.274 0.771 0.   ]
 [0.271 0.761 0.   ]
 [0.267 0.748 0.   ]]
```

### Bonus: Visualize Hand

```python
import matplotlib.pyplot as plt

def visualize_hand(landmarks_21x3, title="Hand"):
    """Visualize 21 landmarks as scatter plot."""
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(landmarks_21x3[:, 0], landmarks_21x3[:, 1], s=100, c='red')
    
    # Label each point
    for i in range(21):
        plt.annotate(str(i), (landmarks_21x3[i, 0], landmarks_21x3[i, 1]))
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    ]
    
    for start, end in connections:
        plt.plot([landmarks_21x3[start, 0], landmarks_21x3[end, 0]],
                 [landmarks_21x3[start, 1], landmarks_21x3[end, 1]], 'b-', lw=2)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Test
sample_reshaped = X[0].reshape(21, 3)
visualize_hand(sample_reshaped, f"Letter: {y[0]}")
```

---

## Exercise 2: Implement Perceptron with NumPy

### Tujuan
- Understand how a single neuron works
- Implement forward pass from scratch
- No libraries (except NumPy)

### Tasks

```python
import numpy as np

# 1. Implement Perceptron
class Perceptron:
    """Single perceptron (neuron)."""
    
    def __init__(self, input_size):
        # Initialize weights randomly
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
    
    def forward(self, x):
        """
        Forward pass: compute output.
        
        Args:
            x: Input vector (input_size,)
        
        Returns:
            Output scalar
        """
        # TODO: Implement
        # 1. Compute weighted sum: z = sum(x * weights) + bias
        # 2. Apply step function (output 1 if z > 0, else 0)
        pass
    
    def __repr__(self):
        return f"Perceptron(weights={self.weights.shape}, bias={self.bias:.4f})"


# 2. Test with simple data
perceptron = Perceptron(input_size=63)
x_sample = X[0]  # First landmark sample

output = perceptron.forward(x_sample)
print(f"Perceptron: {perceptron}")
print(f"Input shape: {x_sample.shape}")
print(f"Output: {output}")


# 3. Implement Multi-Layer version
class SimpleMLP:
    """MLP with 2 hidden layers (NumPy only)."""
    
    def __init__(self, input_size, hidden1, hidden2, output_size):
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0/hidden2)
        self.b3 = np.zeros(output_size)
    
    def relu(self, x):
        """ReLU activation: max(0, x)."""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax for multi-class output."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def forward(self, x):
        """
        Forward pass through 3 layers.
        
        Args:
            x: Input (batch, input_size) or (input_size,)
        
        Returns:
            Output probabilities (batch, output_size)
        """
        # Ensure 2D input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # TODO: Implement forward pass
        # Layer 1: x → h1 = ReLU(x @ W1 + b1)
        # Layer 2: h1 → h2 = ReLU(h1 @ W2 + b2)
        # Layer 3: h2 → logits = h2 @ W3 + b3
        # Return softmax(logits)
        pass
    
    def predict(self, x):
        """Return predicted class indices."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


# 4. Test MLP
mlp = SimpleMLP(input_size=63, hidden1=128, hidden2=64, output_size=26)
output = mlp.forward(x_sample)
prediction = mlp.predict(x_sample)

print(f"\nMLP output shape: {output.shape}")
print(f"MLP output (first 5): {output[0, :5]}")
print(f"Prediction: {prediction} ({chr(65 + prediction[0])})")
```

### Hints

```python
# Hint 1: Weighted sum
z = np.dot(x, self.weights) + self.bias  # scalar
z = x @ self.weights + self.bias          # same thing

# Hint 2: Layer forward
h1 = self.relu(x @ self.W1 + self.b1)    # (batch, hidden1)

# Hint 3: Flatten first if needed
x = x.flatten() if x.ndim > 1 else x
```

---

## Exercise 3: Build MLP with PyTorch

### Tujuan
- Translate NumPy code to PyTorch
- Learn PyTorch nn.Module
- Build working MLP model

### Tasks

```python
import torch
import torch.nn as nn

# 1. Implement MLP with PyTorch
class PyTorchMLP(nn.Module):
    """MLP using PyTorch nn.Module."""
    
    def __init__(self, input_dim=63, hidden_dim1=128, hidden_dim2=64, num_classes=26):
        super().__init__()
        
        # TODO: Define layers
        # Layer 1: Linear(63 → 128) + ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU()
        )
        
        # Layer 2: Linear(128 → 64) + ReLU
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        
        # Output layer: Linear(64 → 26)
        self.output = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x


# 2. Create model
model = PyTorchMLP()

# 3. Test forward pass
x_tensor = torch.FloatTensor(X[:32])  # Batch of 32
output = model(x_tensor)

print(f"Model: {model}")
print(f"Input shape: {x_tensor.shape}")
print(f"Output shape: {output.shape}")

# 4. Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

## Exercise 4: Implement Conv1d Layer

### Tujuan
- Understand Conv1d mechanics
- Implement sliding window operation
- Visualize filter effects

### Tasks

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Create Conv1d layer
conv1d = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)

# 2. Create sample input (batch=1, channel=1, length=63)
x = torch.randn(1, 1, 63)

# 3. Forward pass
output = conv1d(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output channels: {output.shape[1]} (should be 8)")


# 4. Manual convolution (for understanding)
def manual_conv1d(input_data, kernel, stride=1, padding=0):
    """
    Manual 1D convolution.
    
    Args:
        input_data: (length,)
        kernel: (kernel_size,)
        stride: step size
        padding: zero padding
    
    Returns:
        output: convolved signal
    """
    # Add padding
    if padding > 0:
        input_data = np.pad(input_data, (padding, padding), mode='constant')
    
    # Calculate output length
    output_len = (len(input_data) - len(kernel)) // stride + 1
    
    output = np.zeros(output_len)
    
    for i in range(output_len):
        # Extract window
        start = i * stride
        end = start + len(kernel)
        window = input_data[start:end]
        
        # Dot product
        output[i] = np.dot(window, kernel)
    
    return output

# Test with simple signal
signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
kernel = np.array([0.5, 0.5])  # Moving average
result = manual_conv1d(signal, kernel)

print(f"\nManual Conv1d test:")
print(f"Input: {signal}")
print(f"Kernel: {kernel}")
print(f"Output: {result}")


# 5. Visualize learned filters
fig, axes = plt.subplots(2, 4, figsize=(14, 6))

for i, ax in enumerate(axes.flat):
    if i < 8:
        # Get filter weights
        filter_weights = conv1d.weight[i, 0].detach().numpy()
        ax.plot(filter_weights)
        ax.set_title(f'Filter {i}')
        ax.grid(True)

plt.suptitle('Conv1d Learned Filters (first 8 of 64)')
plt.tight_layout()
plt.show()
```

---

## Exercise 5: Build BISINDO CNN

### Tujuan
- Build complete CNN architecture
- Understand channel progression
- Implement pooling layers

### Tasks

```python
import torch
import torch.nn as nn

# 1. Implement BISINDO CNN
class BISINDO_CNN(nn.Module):
    """CNN for BISINDO letter recognition."""
    
    def __init__(self, input_dim=84, num_classes=26):
        super().__init__()
        
        # Conv layers
        self.conv = nn.Sequential(
            # Block 1: 1 → 64 channels
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3: 128 → 64 channels, with GAP
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch, input_dim) e.g., (32, 84)
        
        Returns:
            Output logits (batch, num_classes)
        """
        # Add channel dimension: (batch, 84) → (batch, 1, 84)
        x = x.unsqueeze(1)
        
        # Conv feature extraction
        x = self.conv(x)
        
        # Classification
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        """Return predicted class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


# 2. Create and test model
model = BISINDO_CNN()

# Test with random input
batch_size = 32
x_batch = torch.randn(batch_size, 84)
output = model(x_batch)

print(f"Model: {model}")
print(f"Input shape: {x_batch.shape}")
print(f"Output shape: {output.shape}")

# Test predictions
predictions = model.predict(x_batch)
print(f"Predictions: {predictions[:10]}")
print(f"Predicted letters: {[chr(65 + p.item()) for p in predictions[:10]]}")


# 3. Trace shape transformation
def trace_model(model, x):
    """Print shape at each layer."""
    x = x.unsqueeze(1)
    print(f"Input: {x.shape}")
    
    # Conv1
    x = model.conv[0](x)
    print(f"After Conv1: {x.shape}")
    x = model.conv[1](x)
    print(f"After ReLU1: {x.shape}")
    x = model.conv[2](x)
    print(f"After Pool1: {x.shape}")
    
    # Conv2
    x = model.conv[3](x)
    print(f"After Conv2: {x.shape}")
    x = model.conv[4](x)
    print(f"After ReLU2: {x.shape}")
    x = model.conv[5](x)
    print(f"After Pool2: {x.shape}")
    
    # Conv3
    x = model.conv[6](x)
    print(f"After Conv3: {x.shape}")
    x = model.conv[7](x)
    print(f"After ReLU3: {x.shape}")
    x = model.conv[8](x)
    print(f"After GAP: {x.shape}")
    
    # FC
    x = x.view(x.size(0), -1)
    print(f"After Flatten: {x.shape}")
    x = model.fc[1](x)
    print(f"After FC1: {x.shape}")
    x = model.fc[2](x)
    print(f"After ReLU(FC): {x.shape}")
    x = model.fc[4](x)
    print(f"After Output: {x.shape}")

print("\n--- Shape Trace ---")
trace_model(model, torch.randn(1, 84))
```

---

## Exercise 6: Training Loop

### Tujuan
- Implement complete training pipeline
- Monitor training progress
- Save and load model

### Tasks

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor([ord(c) - ord('A') for c in y_train])  # A=0, B=1, ...
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor([ord(c) - ord('A') for c in y_test])

# Create dataloaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

print(f"Training samples: {len(X_train_t)}")
print(f"Test samples: {len(X_test_t)}")


# 2. Training function
def train_model(model, train_loader, val_loader=None, epochs=50, lr=0.001):
    """Train model and return history."""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            val_acc = val_correct / val_total
        else:
            val_acc = 0
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    return history


# 3. Create model and train
model = BISINDO_CNN()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Split training into train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_t.numpy(), y_train_t.numpy(), test_size=0.1, stratify=y_train_t.numpy()
)

train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

# Train!
print("\n--- Training ---")
history = train_model(model, train_loader, val_loader, epochs=50)


# 4. Save model
torch.save({
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
}, 'bisindo_cnn.pt')
print("\nModel saved to bisindo_cnn.pt")


# 5. Load model
checkpoint = torch.load('bisindo_cnn.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded from bisindo_cnn.pt")
```

---

## Exercise 7: Evaluation

### Tujuan
- Evaluate trained model
- Generate confusion matrix
- Identify weak letters

### Tasks

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load trained model (or use trained model from Exercise 6)
model.eval()

# 2. Make predictions
with torch.no_grad():
    outputs = model(X_test_t)
    _, y_pred = outputs.max(1)
    y_pred = y_pred.numpy()

# 3. Calculate accuracy
accuracy = accuracy_score(y_test_t.numpy(), y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# 4. Confusion matrix
cm = confusion_matrix(y_test_t.numpy(), y_pred, labels=range(26))

# 5. Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'BISINDO CNN Confusion Matrix (Accuracy: {accuracy*100:.2f}%)')
plt.tight_layout()
plt.show()


# 6. Find confused pairs
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)

confused_pairs = []
for i in range(26):
    for j in range(26):
        if i != j and cm_copy[i, j] > 0:
            confused_pairs.append({
                'actual': chr(65 + i),
                'predicted': chr(65 + j),
                'count': cm_copy[i, j]
            })

confused_pairs.sort(key=lambda x: x['count'], reverse=True)

print("\nTop Confused Pairs:")
print("-" * 40)
for pair in confused_pairs[:10]:
    print(f"  {pair['actual']} → {pair['predicted']}: {pair['count']} times")


# 7. Per-letter accuracy
letter_acc = cm.diagonal() / cm.sum(axis=1)
letter_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

print("\nPer-Letter Accuracy:")
print("-" * 40)
sorted_letters = sorted(zip(letter_names, letter_acc), key=lambda x: x[1])

for letter, acc in sorted_letters[:5]:
    print(f"  ⚠️  {letter}: {acc*100:.1f}% (need improvement)")

print("...")
for letter, acc in sorted_letters[-5:]:
    print(f"  ✓  {letter}: {acc*100:.1f}%")
```

---

## Exercise 8: Debug Feature Maps

### Tujuan
- Visualize what CNN learns
- Understand intermediate representations
- Debug model behavior

### Tasks

```python
import torch
import matplotlib.pyplot as plt

# 1. Register hooks to capture intermediate outputs
activations = {}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model = BISINDO_CNN()

# Register hooks
model.conv[0].register_forward_hook(get_activation('conv1'))
model.conv[3].register_forward_hook(get_activation('conv2'))
model.conv[6].register_forward_hook(get_activation('conv3'))

# 2. Forward pass with sample
sample = X_test_t[:1]  # First test sample
output = model(sample)

# 3. Visualize feature maps
def plot_feature_maps(activations, max_show=8):
    """Plot feature maps from intermediate layers."""
    
    fig, axes = plt.subplots(3, max_show, figsize=(16, 8))
    
    layer_names = ['conv1', 'conv2', 'conv3']
    layer_titles = ['After Conv1 (64 channels)', 
                    'After Conv2 (128 channels)',
                    'After Conv3 (64 channels)']
    
    for row, (name, title) in enumerate(zip(layer_names, layer_titles)):
        feat = activations[name][0]  # First sample
        
        for col in range(max_show):
            ax = axes[row, col]
            
            if col < feat.shape[0]:
                ax.plot(feat[col].numpy())
                ax.set_title(f'Channel {col}' if row == 0 else '')
            ax.axis('off')
        
        axes[row, 0].set_ylabel(title, fontsize=10, rotation=0, ha='right')
    
    plt.suptitle('Feature Maps at Different Layers')
    plt.tight_layout()
    plt.show()

plot_feature_maps(activations)


# 4. Analyze filter weights
fig, axes = plt.subplots(2, 4, figsize=(14, 6))

for i, ax in enumerate(axes.flat):
    if i < 4:
        # First layer filters
        weights = model.conv[0].weight[i, 0].detach().numpy()
        ax.plot(weights)
        ax.set_title(f'Conv1 Filter {i}')
        ax.grid(True)

plt.suptitle('Conv1d First Layer Filters (showing first 4)')
plt.tight_layout()
plt.show()


# 5. Analyze what different filters activate on
def analyze_filter_activation(model, sample, layer_idx=0, filter_idx=0):
    """See what input patterns activate a specific filter strongly."""
    
    hook_fn = lambda m, i, o: activations.__setitem__('target', o.detach())
    
    if layer_idx == 0:
        model.conv[0].register_forward_hook(hook_fn)
    elif layer_idx == 1:
        model.conv[3].register_forward_hook(hook_fn)
    else:
        model.conv[6].register_forward_hook(hook_fn)
    
    activations.clear()
    _ = model(sample)
    
    target = activations['target'][0, filter_idx]
    
    # Find positions with highest activation
    top_positions = torch.topk(target, 5).indices
    
    print(f"Filter {filter_idx} (layer {layer_idx}) strongest at positions: {top_positions.tolist()}")
    print(f"Activation values: {target[top_positions].tolist()}")
    
    return target

for i in range(3):
    analyze_filter_activation(model, sample, layer_idx=0, filter_idx=i)
```

---

## Bonus: Challenge Problems

### Challenge 1: Implement Data Augmentation

```python
def augment_landmarks(landmarks, noise_std=0.01, scale_range=(0.9, 1.1), rotation_deg=10):
    """
    Augment landmark data.
    
    Args:
        landmarks: array of shape (21, 3) or (63,)
        noise_std: std of Gaussian noise
        scale_range: min and max scale factor
        rotation_deg: max rotation in degrees
    
    Returns:
        Augmented landmarks
    """
    # TODO: Implement
    # 1. Add Gaussian noise
    # 2. Random scaling
    # 3. Random rotation (in 2D)
    pass

# Test
sample_reshaped = X[0].reshape(21, 3)
augmented = augment_landmarks(sample_reshaped)
```

### Challenge 2: Implement Learning Rate Finder

```python
def find_best_lr(model, train_loader, start_lr=1e-7, end_lr=1, num_iters=100):
    """
    Find optimal learning rate using "lr finder" technique.
    
    Returns:
        losses: list of losses
        lrs: list of learning rates tested
    """
    # TODO: Implement
    # 1. Start with very small lr
    # 2. Exponentially increase lr each iteration
    # 3. Record loss
    # 4. Plot and find optimal lr
    pass
```

### Challenge 3: Build Ensemble Model

```python
def ensemble_predict(models, x):
    """
    Predict using ensemble of models.
    
    Args:
        models: list of trained models
        x: input tensor
    
    Returns:
        final prediction (majority vote)
    """
    # TODO: Implement
    # 1. Get predictions from all models
    # 2. Average probabilities OR majority vote
    # 3. Return final prediction
    pass
```

### Challenge 4: Implement Temporal Smoothing

```python
def temporal_smooth(predictions_history, window_size=5):
    """
    Smooth predictions over time using sliding window.
    
    Args:
        predictions_history: list of (probabilities, count) tuples
        window_size: size of smoothing window
    
    Returns:
        smoothed predictions
    """
    # TODO: Implement
    # Hint: Use rolling average of probabilities
    pass
```

---

## Solutions (Partial)

### Solution for Exercise 2: Perceptron

```python
# Solution for SimpleMLP.forward()
def forward(self, x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Layer 1
    h1 = self.relu(x @ self.W1 + self.b1)
    
    # Layer 2
    h2 = self.relu(h1 @ self.W2 + self.b2)
    
    # Output
    logits = h2 @ self.W3 + self.b3
    
    return self.softmax(logits)
```

### Solution for Exercise 3: PyTorch MLP

```python
# Already implemented! Just verify it works.
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim=63, hidden_dim1=128, hidden_dim2=64, num_classes=26):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU())
        self.output = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                      Exercise Summary                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Completed:                                                      │
│  ✓ Exercise 1: Load & Explore Data                             │
│  ✓ Exercise 2: NumPy Perceptron                                │
│  ✓ Exercise 3: PyTorch MLP                                     │
│  ✓ Exercise 4: Conv1d Layer                                     │
│  ✓ Exercise 5: BISINDO CNN                                     │
│  ✓ Exercise 6: Training Loop                                     │
│  ✓ Exercise 7: Evaluation                                       │
│  ✓ Exercise 8: Feature Map Debugging                           │
│                                                                  │
│  Next Steps:                                                     │
│  → Run training on full dataset                                 │
│  → Experiment with different architectures                      │
│  → Try different hyperparameters                                │
│  → Deploy to webcam inference                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Referensi

- BISINDO Bridge project: `/home/kevin/bisindo-bridge`
- PyTorch tutorials: https://pytorch.org/tutorials/
- MNIST with PyTorch: Great beginner exercise
