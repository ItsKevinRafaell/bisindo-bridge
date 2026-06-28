# Team Guide - BISINDO Bridge

## Project Structure

```
bisindo-bridge/
├── train/                      # TRAINING scripts
├── models/                     # OUTPUT models
├── eval/                       # EVALUATION
├── web/                        # FRONTEND (fungsional)
├── meeting/                    # SERVER (fungsional)
└── docs/                       # DOCUMENTATION
```

---

## ML Team (2 orang)

### Checkpoints

- [ ] **Checkpoint 1** - Setup & Baseline
  - Clone repo, install deps
  - Run baseline script
  - Catat accuracy baseline

- [ ] **Checkpoint 2** - Experiment 1
  - Experiment dengan parameter berbeda
  - Bandingin hasil vs baseline
  - Catat accuracy baru

- [ ] **Checkpoint 3** - Experiment 2
  - Experiment lagi dengan setting berbeda
  - Pilih best configuration
  - Catat final accuracy

- [ ] **Checkpoint 4** - Final
  - Final model + metrics
  - Report ke proposal team

### Report Format
```
ML Team Report:
- Algoritma: [nama]
- Best accuracy: [xx.xx%]
- Best hyperparameters: [detail]
- Training time: [xx menit]
- Notes: [opsional]
```

---

## DL Team (2 orang)

### Checkpoints

- [ ] **Checkpoint 1** - Setup & Baseline
  - Clone repo, install TensorFlow
  - Run baseline script
  - Catat accuracy baseline

- [ ] **Checkpoint 2** - Experiment 1
  - Experiment dengan arsitektur berbeda
  - Bandingin hasil vs baseline
  - Catat accuracy baru

- [ ] **Checkpoint 3** - Experiment 2
  - Experiment lagi dengan setting berbeda
  - Pilih best configuration
  - Catat final accuracy

- [ ] **Checkpoint 4** - Final
  - Final model + metrics
  - Report ke proposal team

### Report Format
```
DL Team Report:
- Arsitektur: [nama]
- Best accuracy: [xx.xx%]
- Best configuration: [detail]
- Training time: [xx menit]
- Notes: [opsional]
```

---

## Proposal Team (1 orang)

### Checkpoints

- [ ] **Checkpoint 1** - Outline
  - Baca PROPOSAL_OUTLINE.md
  - Tulis draft outline

- [ ] **Checkpoint 2** - Pendahuluan
  - Tulis latar belakang
  - Tulis rumusan masalah

- [ ] **Checkpoint 3** - Metodologi
  - Tulis deskripsi ML approach
  - Tulis deskripsi DL approach

- [ ] **Checkpoint 4** - Results
  - Isi dengan data dari ML & DL team
  - Bandingkan hasil

- [ ] **Checkpoint 5** - Final
  - Selesai write-up
  - Review & finalize

---

## Evaluasi (Semua Tim)

```bash
python eval/compare.py
```

Output: `models/comparison_report.md`

---

## Communication

Weekly sync via WhatsApp. Format report:

```
[NAMA] - [TIM]
Progress: [deskripsi singkat]
Accuracy: [xx.xx%]
Blocker: [ada/tidak]
Next: [plan下周]
```