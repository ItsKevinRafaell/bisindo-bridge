# Cara Training Model BISINDO di Google Colab

## 📋 Panduan Step-by-Step

### Step 1: Buka Google Colab

1. Buka browser → https://colab.research.google.com
2. Login dengan akun Google kamu
3. Klik **"New Notebook"** (notebook baru)

### Step 2: Aktifkan GPU (PENTING!)

1. Di menu atas, klik **Runtime** → **Change runtime type**
2. Di bagian **Hardware accelerator**, pilih **T4 GPU**
3. Klik **Save**
4. Akan muncul warning "Runtime will be restarted" → klik **Yes**

**Cara cek GPU sudah aktif:**
```python
!nvidia-smi
```
Kalau muncul info GPU (Tesla T4), berarti GPU aktif ✅

### Step 3: Copy-Paste Script Training

**Opsi A: Copy semua sekaligus (RECOMMENDED)**

1. Buka file `colab_training.py` yang ada di laptop kamu
2. Copy SEMUA isi file
3. Paste ke cell pertama di Colab
4. Tekan **Shift + Enter** untuk run

**Opsi B: Upload file langsung**

1. Di Colab, klik icon folder di sidebar kiri (Files)
2. Klik icon upload (📤)
3. Upload file `colab_training.py`
4. Buat cell baru, ketik:
   ```python
   %run colab_training.py
   ```
5. Tekan **Shift + Enter**

### Step 4: Tunggu Training Selesai

Script akan otomatis:
1. ✅ Install dependencies (~1 menit)
2. ✅ Download dataset dari GitHub (~2 menit)
3. ✅ Load & preprocess 520 images (~1 menit)
4. ✅ Setup data augmentation
5. ✅ Build MobileNetV2 model
6. ✅ Train Phase 1 - Feature Extraction (~10 menit)
7. ✅ Train Phase 2 - Fine-tuning (~5 menit)
8. ✅ Evaluate model
9. ✅ Save model & artifacts
10. ✅ Download model otomatis

**Total waktu: ~20-25 menit**

### Step 5: Download Model

Setelah training selesai, script akan otomatis download file:
- `bisindo_model_artifacts.zip`

Isi zip:
- `bisindo_mobilenetv2_final.keras` (model utama)
- `label_map.json` (mapping huruf A-Z)
- `results_summary.json` (accuracy, loss, dll)
- Training plots (accuracy, loss, confusion matrix)

### Step 6: Pindahkan Model ke Laptop

1. Cari file `bisindo_model_artifacts.zip` di folder Downloads laptop kamu
2. Extract zip tersebut
3. Pindahkan semua file ke:
   ```
   /home/kevin/bisindo-bridge/models/
   ```

**Cara cepat (terminal):**
```bash
cd ~/Downloads
unzip bisindo_model_artifacts.zip
mv bisindo_mobilenetv2_final.keras /home/kevin/bisindo-bridge/models/
mv label_map.json /home/kevin/bisindo-bridge/models/
mv results_summary.json /home/kevin/bisindo-bridge/models/
```

### Step 7: Verifikasi Model

Buka terminal, cek model sudah ada:
```bash
ls -lh /home/kevin/bisindo-bridge/models/
```

Harusnya ada:
- `bisindo_mobilenetv2_final.keras` (~30 MB)
- `label_map.json`
- `results_summary.json`

---

## 🎯 Cara Baca Hasil Training

### Akurasi Model

Di akhir script, akan muncul:
```
📊 Final Test Accuracy: 0.9231 (92.31%)
📊 Final Test Loss: 0.2341
```

**Interpretasi:**
- **> 90%** = Bagus banget ✅
- **80-90%** = Bagus ✅
- **70-80%** = Lumayan, bisa di-improve
- **< 70%** = Perlu perbaikan (augmentasi, more data, dll)

### Confusion Matrix

File `confusion_matrix.png` menunjukkan:
- **Diagonal** (kiri atas ke kanan bawah) = prediksi benar
- **Off-diagonal** = prediksi salah
- Warna makin gelap = makin banyak sample

**Contoh baca:**
- Baris "A", kolom "A" = 83 (benar prediksi A)
- Baris "A", kolom "B" = 0 (tidak ada yang salah prediksi ke B)

### Class yang Susah

Kalau ada class yang accuracy-nya rendah, cek di classification report:
```
              precision    recall  f1-score
           A       0.95      0.98      0.96
           B       0.97      0.95      0.96
           ...
```

Class dengan **f1-score rendah** = susah diprediksi, mungkin perlu lebih banyak data.

---

## 🚨 Troubleshooting

### Problem 1: "No GPU detected"

**Solusi:**
1. Runtime → Change runtime type → T4 GPU
2. Restart runtime
3. Run cell `!nvidia-smi` untuk cek

### Problem 2: Out of Memory (OOM)

**Solusi:**
1. Reduce batch size di script (ubah 32 → 16)
2. Restart runtime
3. Factory reset runtime (Runtime → Factory reset runtime)

### Problem 3: Training lambat banget (> 1 jam)

**Penyebab:** GPU tidak aktif, pakai CPU

**Solusi:**
1. Pastikan GPU aktif (Step 2)
2. Kalau GPU tetap lambat, coba pakai Colab Pro (paid)

### Problem 4: Download tidak otomatis

**Solusi:**
1. Cari file manual di sidebar kiri Colab (icon folder)
2. Klik kanan file `bisindo_model_artifacts.zip`
3. Pilih **Download**

### Problem 5: Accuracy rendah (< 80%)

**Solusi:**
1. Tambah epochs (ubah 15 → 20 untuk Phase 1)
2. Tambah augmentasi (rotation_range=30, zoom_range=0.3)
3. Unfreeze lebih banyak layers (ubah 30 → 50 untuk Phase 2)

---

## 📊 Expected Results

Berdasarkan dataset BISINDO (520 images, 26 classes):

| Metric | Expected Range |
|--------|----------------|
| Test Accuracy | 85-95% |
| Test Loss | 0.1-0.4 |
| Training Time | 15-25 menit (GPU) |
| Model Size | ~30 MB |

**Kalau hasil kamu:**
- ✅ **> 90%** = Langsung pakai, bagus banget
- ⚠️ **85-90%** = Bisa dipakai, tapi bisa di-improve
- ❌ **< 85%** = Perlu tuning (lihat Troubleshooting #5)

---

## 🎓 Tips untuk Improve Accuracy

### 1. Data Augmentation Lebih Agresif
```python
train_datagen = ImageDataGenerator(
    rotation_range=30,      # dari 20
    width_shift_range=0.3,  # dari 0.2
    height_shift_range=0.3, # dari 0.2
    shear_range=0.3,        # dari 0.2
    zoom_range=0.3,         # dari 0.2
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # dari [0.8, 1.2]
    fill_mode='nearest'
)
```

### 2. Unfreeze Lebih Banyak Layers
```python
# Ubah dari 30 → 50
for layer in base_model.layers[:-50]:
    layer.trainable = False
```

### 3. Train Lebih Lama
```python
# Phase 1: 15 → 20 epochs
history_phase1 = model.fit(..., epochs=20, ...)

# Phase 2: 10 → 15 epochs
history_phase2 = model.fit(..., epochs=15, ...)
```

### 4. Learning Rate Tuning
```python
# Phase 1: coba 5e-4 (lebih kecil)
optimizer=keras.optimizers.Adam(learning_rate=5e-4)

# Phase 2: coba 5e-6 (lebih kecil)
optimizer=keras.optimizers.Adam(learning_rate=5e-6)
```

---

## ✅ Checklist Setelah Training

- [ ] GPU sudah aktif (cek `!nvidia-smi`)
- [ ] Script sudah run sampai selesai
- [ ] Test accuracy > 85%
- [ ] File `bisindo_model_artifacts.zip` sudah download
- [ ] File sudah extract ke `/home/kevin/bisindo-bridge/models/`
- [ ] File `bisindo_mobilenetv2_final.keras` ada di folder models
- [ ] File `label_map.json` ada di folder models

---

## 🚀 Next Step: Build Streamlit App

Setelah model ready, next step:
1. Build Streamlit app untuk webcam detection
2. Integrate Text-to-Speech (TTS)
3. Test real-time prediction
4. Deploy app

**Lanjut ke:** `docs/STREAMLIT_APP_GUIDE.md` (akan dibuat setelah training selesai)

---

## 📞 Butuh Bantuan?

Kalau stuck di step manapun:
1. Cek error message di Colab
2. Screenshot error-nya
3. Tanyakan ke Claude dengan error message lengkap

**Contoh pertanyaan bagus:**
- "Error OOM di epoch 5, solusinya gimana?"
- "Accuracy cuma 75%, gimana cara improve?"
- "Download tidak otomatis, cara manual gimana?"

**Contoh pertanyaan kurang bagus:**
- "Error, kenapa?" (terlalu umum)
- "Gagal, tolong" (tidak ada detail)

---

**Last Updated:** 2026-06-15  
**Status:** Ready to use
