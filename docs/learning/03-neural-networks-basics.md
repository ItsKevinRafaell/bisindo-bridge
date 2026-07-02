# Neural Networks Basics: Dari Perceptron ke MLP

> Sekarang kita masuk ke neural networks. Penjelasan dari nol - mulai dari satu neuron sampai multi-layer network. Semua pakai NumPy, tanpa library ML.

---

## Daftar Isi

1. [Perceptron: Satu Neuron](#perceptron-satu-neuron)
2. [Activation Functions](#activation-functions)
3. [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
4. [Forward Pass](#forward-pass)
5. [Implementasi NumPy (Tanpa PyTorch!)](#implementasi-numpy-tanpa-pytorch)
6. [Kenapa Layer越多 Semakin Bagus?](#kenapa-layer-lebih-semakin-bagus)
7. [Latihan](#latihan)

---

## Perceptron: Satu Neuron

### Apa itu Perceptron?

**Perceptron** = model sederhana yang terima input, kasih output.

```
┌─────────────────────────────────────────────────────┐
│                  Perceptron                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│    x₀ ── w₀ ─┐                                      │
│    x₁ ── w₁ ─┤                                      │
│    x₂ ── w₂ ─┤──→ Σ(wᵢxᵢ) + b = z ──→ f(z) = y    │
│    ...     ─┤                                      │
│    x₆₂ ──w₆₂┘                                      │
│                                                      │
│    x = input features (63 values)                    │
│    w = weights (learned parameters)                  │
│    b = bias (learned parameter)                      │
│    f = activation function                           │
│    y = output                                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Langkah demi Langkah

```python
import numpy as np

# Input: 3 features (simplified example)
x = np.array([0.5, 0.3, 0.8])

# Weights: learned parameters
w = np.array([0.2, -0.4, 0.6])

# Bias: learned parameter
b = 0.1

# Step 1: Weighted sum
z = np.dot(w, x) + b
# = (0.2×0.5) + (-0.4×0.3) + (0.6×0.8) + 0.1
# = 0.1 - 0.12 + 0.48 + 0.1
# = 0.56

print(f"Weighted sum z = {z:.4f}")
# Output: z = 0.5600
```

---

## Activation Functions

### Kenapa Butuh Activation?

Tanpa activation: **linear transformation only**
```
x₁ = w₁x + b₁
x₂ = w₂(x₁) + b₂ = w₂(w₁x + b₁) + b₂ = (w₂w₁)x + (w₂b₁ + b₂)
```

Dengan cascade linear, semua collapse jadi **satu linear function**. Tidak bisa capture complex patterns!

### ReLU: Rectified Linear Unit

Yang **paling populer** di deep learning modern.

```python
def relu(x):
    """f(x) = max(0, x)"""
    return np.maximum(0, x)

# Test
x = np.array([-2, -1, 0, 1, 2])
print(relu(x))
# Output: [0, 0, 0, 1, 2]
```

**Visual:**
```
relu(x)
  │
2 │          ╱
  │         ╱
1 │        ╱
  │───────╱──────────── x
0 │──────┼─────────────
  │      │ 0
-1│      
  │
  └────────────────────
     -2  -1   0   1   2
```

**Kenapa ReLU bagus?**
- Simple: cepat compute
- Non-linear: bisa learn complex patterns
- Sparse activation: some neurons "die" (output 0) → regularization effect

### Softmax: Untuk Multi-Class

Digunakan di **output layer** untuk classification.

```python
def softmax(x):
    """Convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

# Test dengan 3 classes
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Probabilities: {probs}")
# Output: [0.72, 0.26, 0.02]
print(f"Sum: {probs.sum()}")
# Output: 1.0 (valid probability distribution)
```

---

## Multi-Layer Perceptron (MLP)

### Kenapa Butuh Multiple Layers?

```
1 Layer (Linear):                       3 Layers (Non-Linear):
                                       
y = w₁x + b₁                           h₁ = f(w₁x + b₁)
                                        h₂ = f(w₂h₁ + b₂)
                                        y  = f(w₃h₂ + b₃)
Terlalu sederhana!                      Bisa capture patterns kompleks!
```

### MLP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MLP Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INPUT          HIDDEN 1        HIDDEN 2        OUTPUT         │
│   (63)            (128)            (64)            (26)         │
│                                                                  │
│   ┌───┐        ┌───┐           ┌───┐           ┌───┐           │
│   │x₀ │        │   │           │   │           │   │           │
│   ├───┤        │ h₀ │           │   │           │ y₀ │ ← A     │
│   │x₁ │──────▶│ h₁ │──────────▶│   │──────────▶│ y₁ │ ← B     │
│   ├───┤        │... │           │... │           │... │         │
│   │... │       │h₁₂₇│           │ h₆₃│           │y₂₅│ ← Z     │
│   ├───┤        │   │           │   │           │   │           │
│   │x₆₂│        └───┘           └───┘           └───┘           │
│   └───┘                                                          │
│                                                                  │
│   weights:    W₁ (63×128)    W₂ (128×64)    W₃ (64×26)          │
│   biases:    b₁ (128)       b₂ (64)         b₃ (26)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Dimensions

```python
# Input layer
input_dim = 63    # 21 landmarks × 3 coordinates

# Hidden layer 1
hidden1_dim = 128
W1 = np.random.randn(input_dim, hidden1_dim)   # shape: (63, 128)
b1 = np.random.randn(hidden1_dim)                # shape: (128,)

# Hidden layer 2
hidden2_dim = 64
W2 = np.random.randn(hidden1_dim, hidden2_dim)  # shape: (128, 64)
b2 = np.random.randn(hidden2_dim)               # shape: (64,)

# Output layer
output_dim = 26  # 26 letters A-Z
W3 = np.random.randn(hidden2_dim, output_dim)   # shape: (64, 26)
b3 = np.random.randn(output_dim)                 # shape: (26,)
```

---

## Forward Pass

### Forward Pass itu Apa?

**Forward pass** = proses propagate input through network untuk dapetin output.

```
Input (63) ──▶ Hidden1 (128) ──▶ Hidden2 (64) ──▶ Output (26)
                     │               │
                   ReLU            ReLU
                     │               │
                  weights         weights
```

### Step-by-Step Forward Pass

```python
def forward_pass(x, W1, b1, W2, b2, W3, b3):
    """
    x: input (63,)
    Returns: output logits (26,)
    """
    
    # Layer 1: Input → Hidden1
    z1 = np.dot(x, W1) + b1     # (63,) @ (63, 128) → (128,)
    h1 = relu(z1)                 # (128,)
    
    # Layer 2: Hidden1 → Hidden2
    z2 = np.dot(h1, W2) + b2      # (128,) @ (128, 64) → (64,)
    h2 = relu(z2)                 # (64,)
    
    # Layer 3: Hidden2 → Output
    z3 = np.dot(h2, W3) + b3      # (64,) @ (64, 26) → (26,)
    
    return z3  # Return logits (akan diproses softmax di akhir)


# Contoh penggunaan
x_sample = X_train[0]  # One sample, shape: (63,)
output_logits = forward_pass(x_sample, W1, b1, W2, b2, W3, b3)
print(f"Output logits: {output_logits}")
# Output: [2.3, -0.5, 1.2, ...]  (26 values)

# Convert ke probabilities
output_probs = softmax(output_logits)
print(f"Predicted class: {np.argmax(output_probs)}")  # Index dari probabilitas tertinggi
```

### Visualisasi Forward Pass

```
Input: [0.64, 0.72, 0.0, ..., 0.58]  (63 values)
          │
          ▼
┌─────────────────────────────────────┐
│  Linear: np.dot(x, W1) + b1        │
│  Shape: (63,) @ (63, 128) → (128,) │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  ReLU: max(0, z1)                  │
│  Shape: (128,)                      │
│  Effect: zeroes out negatives       │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Linear: np.dot(h1, W2) + b2       │
│  Shape: (128,) @ (128, 64) → (64,) │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  ReLU: max(0, z2)                  │
│  Shape: (64,)                      │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Linear: np.dot(h2, W3) + b3       │
│  Shape: (64,) @ (64, 26) → (26,)  │
└─────────────────────────────────────┘
          │
          ▼
Output: [2.3, -0.5, 1.2, ..., 0.8]  (26 logits)
          │
          ▼
┌─────────────────────────────────────┐
│  Softmax → probabilities           │
│  [0.15, 0.01, 0.08, ..., 0.04]     │
└─────────────────────────────────────┘
          │
          ▼
Prediksi: "A" (argmax = 0)
```

---

## Implementasi NumPy (Tanpa PyTorch!)

### Complete MLP dari Nol

```python
import numpy as np

class SimpleMLP:
    """MLP 2 hidden layers, NumPy only"""
    
    def __init__(self, input_dim=63, hidden1=128, hidden2=64, output_dim=26):
        # Xavier initialization (good for deep networks)
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0/hidden2)
        self.b3 = np.zeros(output_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def forward(self, X):
        """Forward pass untuk multiple samples"""
        # X shape: (n_samples, input_dim)
        
        # Layer 1
        z1 = np.dot(X, self.W1) + self.b1
        h1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(h1, self.W2) + self.b2
        h2 = self.relu(z2)
        
        # Layer 3 (output)
        z3 = np.dot(h2, self.W3) + self.b3
        output = self.softmax(z3)
        
        return output
    
    def predict(self, X):
        """Return predicted classes"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# Usage
mlp = SimpleMLP(input_dim=63, hidden1=128, hidden2=64, output_dim=26)
predictions = mlp.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
```

### Total Parameters

```python
# Hitung total parameters
params = {
    'W1': 63 * 128,
    'b1': 128,
    'W2': 128 * 64,
    'b2': 64,
    'W3': 64 * 26,
    'b3': 26,
}

total = sum(params.values())
print(f"Total parameters: {total:,}")
# Output: 63*128 + 128 + 128*64 + 64 + 64*26 + 26
#        = 8064 + 128 + 8192 + 64 + 1664 + 26 = 18,138
```

---

## Kenapa Layer越多 Semakin Bagus?

### Hierarchy of Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Hierarchy in Deep Networks                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1 (Low-level):           Layer 2 (Mid-level):                │
│  ┌─────────────────────────┐    ┌─────────────────────────┐         │
│  │ • Edge detectors        │    │ • Finger shapes        │         │
│  │ • Simple gradients      │──▶ │ • Angles between       │         │
│  │ • Position changes      │    │   landmarks            │         │
│  └─────────────────────────┘    └───────────┬─────────────┘         │
│                                               │                      │
│  Layer 3 (High-level):         Layer 4 (Output):                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │ • Hand configuration    │──▶ │ • Letter predictions    │        │
│  │ • Finger relationships  │    │ • Confidence scores      │        │
│  │ • Global shape          │    │                         │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Concrete Example: BISINDO Letter "V"

```
Input: 21 landmark coordinates
        │
        ▼
┌───────────────────┐
│ Layer 1: Detect   │  → "Ada edge di sini"
│ simple patterns   │  → "Gradien naik/turun"
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Layer 2: Combine  │  → "Index & middle extended"
│ into finger shapes│  → "Thumb tucked"
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Layer 3: Hand     │  → "Two fingers up"
│ configuration     │  → "Angle between fingers ~20°"
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Output: Letter    │  → "V" with 95% confidence
└───────────────────┘
```

---

## Latihan

### Latihan 1: Dot Product

```python
import numpy as np

# Dua vektor
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 4, 6, 8, 10])

# Hitung dot product
result = np.dot(a, b)
print(f"Dot product: {result}")  # Should be 110
```

### Latihan 2: Perceptron

```python
import numpy as np

def perceptron(x, w, b):
    """Single perceptron"""
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0  # Step function

# Test
x = np.array([1, 0, 1])
w = np.array([0.5, -0.3, 0.8])
b = -0.5

output = perceptron(x, w, b)
print(f"Output: {output}")  # ?
```

### Latihan 3: ReLU

```python
import numpy as np

def relu(x):
    # Implement ReLU
    return ...

# Test
x = np.array([-2, -1, 0, 1, 2])
print(relu(x))  # Should be [0, 0, 0, 1, 2]
```

### Latihan 4: Forward Pass (3 layers)

```python
import numpy as np

# Initialize random weights
np.random.seed(42)
W1 = np.random.randn(63, 128) * 0.01
b1 = np.zeros(128)
W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)
W3 = np.random.randn(64, 26) * 0.01
b3 = np.zeros(26)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Forward pass untuk 1 sample
x = np.random.randn(63)  # Random input
h1 = relu(np.dot(x, W1) + b1)
h2 = relu(np.dot(h1, W2) + b2)
logits = np.dot(h2, W3) + b3
probs = softmax(logits)

print(f"Output probabilities shape: {probs.shape}")
print(f"Sum of probs: {probs.sum():.4f}")
print(f"Predicted class: {np.argmax(probs)}")
```

### Latihan 5: MLP Class

```python
import numpy as np

class NumPyMLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        # TODO: Initialize weights and biases
        # Hints: Use Xavier initialization
        pass
    
    def forward(self, X):
        # TODO: Implement forward pass
        pass
    
    def predict(self, X):
        # TODO: Return predicted classes
        pass

# Test
mlp = NumPyMLP(63, [128, 64], 26)
print(f"MLP created successfully!")
```

---

## Ringkasan

```
┌─────────────────────────────────────────────────────────────────┐
│                      Neural Network Basics                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Perceptron                                                  │
│     weighted_sum + activation → single output                   │
│                                                                  │
│  2. Activation Functions                                        │
│     ReLU: max(0, x) - most popular                              │
│     Softmax: probabilities for multi-class                      │
│                                                                  │
│  3. MLP (Multi-Layer Perceptron)                                │
│     input → hidden → hidden → output                            │
│     Non-linear activations enable complex patterns              │
│                                                                  │
│  4. Forward Pass                                                │
│     Input → Layer 1 → Layer 2 → ... → Output                   │
│     Pure matrix multiplications + activations                  │
│                                                                  │
│  5. Parameters                                                  │
│     Weights: W (input_dim × output_dim)                         │
│     Biases: b (output_dim,)                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next:** [04-pytorch-intro.md](04-pytorch-intro.md) - PyTorch Basics

---

## Referensi

- 3Blue1Brown: "But what is a neural network?" (YouTube)
- Michael Nielsen: "Neural Networks and Deep Learning"
- NumPy documentation: https://numpy.org/doc/
