# CNN Learning Path - BISINDO Bridge

> Materi belajar CNN terstruktur untuk proyek BISINDO Bridge. Mulai dari nol sampe bisa bikin CNN sendiri.

---

## Cara Baca

**Urutan membaca (dari atas ke bawah):**

```
1. 01-prerequisites.md     ← NumPy, reshape, math dasar
2. 02-what-is-ml.md       ← ML concepts tanpa jargon
3. 03-neural-networks-basics.md ← Perceptron → MLP, pure NumPy
4. 04-pytorch-intro.md    ← PyTorch basics, nn.Module
5. 05-cnn-intro.md        ← Conv2d intuition (gambar)
6. 06-conv1d-for-landmarks.md ← Conv1d untuk landmark (FILE KUNCI!)
7. 07-bisindo-cnn-architecture.md ← CNN kita: layer-by-layer
8. 08-training-and-hyperparams.md ← Training loop, hyperparameters
9. 09-evaluation.md       ← Accuracy, F1, confusion matrix
10. 10-exercises.md        ← Hands-on exercises
```

**Estimasi waktu:** 10-15 jam untuk pemula total

---

## Ringkasan Tiap File

### Week 1: Foundations (Files 1-3)

| File | Topik | Waktu |
|------|-------|-------|
| 01-prerequisites | NumPy, reshape, dot product | 1-2 jam |
| 02-what-is-ml | ML concepts, supervised learning | 1-2 jam |
| 03-neural-networks | Perceptron, MLP, forward pass | 2-3 jam |

### Week 2: Deep Learning (Files 4-6)

| File | Topik | Waktu |
|------|-------|-------|
| 04-pytorch-intro | PyTorch, tensors, training loop | 2-3 jam |
| 05-cnn-intro | Conv2d, filters, pooling | 2-3 jam |
| 06-conv1d-for-landmarks | **Conv1d untuk BISINDO** | 3-4 jam |

### Week 3: Project (Files 7-10)

| File | Topik | Waktu |
|------|-------|-------|
| 07-bisindo-cnn-architecture | CNN BISINDO kita | 2-3 jam |
| 08-training-hyperparams | Training, hyperparameters | 2-3 jam |
| 09-evaluation | Metrics, evaluation | 1-2 jam |
| 10-exercises | Hands-on coding | 3-4 jam |

---

## File Kunci

### 06-conv1d-for-landmarks.md ⭐
**FILE PALING PENTING!**

Ini explains:
- Kenapa pakai Conv1d, bukan Conv2d
- Gimana sliding window bekerja di landmark data
- Kenapa landmark = sequence, bukan image

### 07-bisindo-cnn-architecture.md
CNN architecture kita:
```python
Conv1d(1→64) → Pool → Conv1d(64→128) → Pool → Conv1d(128→64) → GAP
    → FC(64→128) → FC(128→26)
```

### 10-exercises.md
Latihan hands-on, dari load data sampe training CNN.

---

## Quick Reference

### CNN Architecture BISINDO

```
Input: (batch, 84) - 2-hand landmark data
       │
       ▼ unsqueeze(1)
       (batch, 1, 84)
       │
       ▼ Conv1d(1→64, k=3, p=1)
       (batch, 64, 84)
       │
       ▼ ReLU + MaxPool(2)
       (batch, 64, 42)
       │
       ▼ Conv1d(64→128, k=3, p=1)
       (batch, 128, 42)
       │
       ▼ ReLU + MaxPool(2)
       (batch, 128, 21)
       │
       ▼ Conv1d(128→64, k=3, p=1)
       (batch, 64, 21)
       │
       ▼ ReLU + AdaptiveAvgPool(1)
       (batch, 64, 1)
       │
       ▼ Flatten
       (batch, 64)
       │
       ▼ FC(64→128)
       (batch, 128)
       │
       ▼ ReLU + Dropout(0.3)
       (batch, 128)
       │
       ▼ FC(128→26)
       (batch, 26) - logits
```

### Key Formulas

```python
# Conv1d output length
output_length = (input_length - kernel_size + 2*padding) / stride + 1

# MaxPool1d output length
output_length = input_length / stride

# Parameter count
Conv1d_params = in_channels × out_channels × kernel_size + out_channels

# Cross-Entropy Loss
Loss = -log(predicted_probability_of_correct_class)
```

---

## Sumber Tambahan

### Video
- 3Blue1Brown "Neural Networks" playlist (YouTube)
- CS231n Stanford (cs231n.stanford.edu)

### Kursus
- Fast.ai "Practical Deep Learning for Coders"
- Andrew Ng ML Course (Coursera)

### Dokumentasi
- PyTorch: pytorch.org/docs/
- NumPy: numpy.org/doc/

---

## Pertanyaan?

Kalau ada yang bingung, langsung tanya aja. Bisa juga buka issue di repo atau tanyain di meeting.

---

## Progress Checklist

- [ ] 01-prerequisites.md
- [ ] 02-what-is-ml.md  
- [ ] 03-neural-networks-basics.md
- [ ] 04-pytorch-intro.md
- [ ] 05-cnn-intro.md
- [ ] 06-conv1d-for-landmarks.md ⭐
- [ ] 07-bisindo-cnn-architecture.md
- [ ] 08-training-and-hyperparams.md
- [ ] 09-evaluation.md
- [ ] 10-exercises.md
