# Changelog Jurnal — Apa yang Diubah dari LaTeX Asli

> File asli: `~/Downloads/AIJurnal.pdf` (8 halaman, IEEEtran conference)
> File baru: `docs/jurnal-data/main.tex` (revised, ~10-12 halaman setelah compile)
> Figures: `docs/jurnal-data/figures/` (4 PNG verified)

---

## Ringkasan Perubahan

| Area | Sebelum | Sesudah | Alasan |
|------|---------|---------|--------|
| Judul | "Real-Time BISINDO Recognition Using MediaPipe Hand Landmark" | "BISINDO Bridge: A Lightweight Dual-Hand 1D-CNN System for Real-Time Indonesian Sign Language Recognition in the Browser" | Emphasize dual-hand + lightweight + browser (contribution) |
| Abstract | 84 fitur tanpa breakdown, meeting = future | 84 = 21×2×2 breakdown, 3-stage normalization, 8 variants dengan trade-off 99.12% vs 98.45%, meeting = implemented + stabilized | Fix factual + highlight trade-off |
| Contributions | 4 generik | 4 spesifik dengan angka (138K/26K/208K, 3-stage norm, 8 variants, ONNX 241KB + meeting) | Concrete numbers |
| Normalization | Cuma translasi (1 step) | 3 step: translate + scale by hand size + StandardScaler | Yang bener ada 3 step |
| Feature | 84 muncul tiba-tiba | 84 = 21×2×2 dual-hand, 63 = 21×3×1 single-hand, breakdown + comparison | Clear |
| CNN Architecture | "Several layers" vague | Tabel 2 arch: v1 pool=1 (64 flat) dan v2 pool=8 (512 flat), filters, kernel, FC | Reviewer pasti minta detail |
| Training | "Adam + Cross-Entropy" doang | Adam lr 0.001, batch 256, 50 epochs, split 85/15 seed 42, dropout 0.3, StandardScaler | Reproducibility |
| Table I | 7 model tanpa "what changed" | Table I: 8 model dengan features+hands + Table detail what changed per model + comparison chart | Biar gak tanda tanya |
| Champion vs Deployed | "cnn_balanced digunakan pada web app & meeting" ❌ SALAH | Champion = balanced 99.12% (1-hand, metrics-only), Deployed = 2hand 98.45% (84-feat, ONNX [1,84]) — verified | Critical fix: balanced 63-feat gak match deployed 84-feat |
| Confusion Matrix | Placeholder `figures/confusion_matrix.png` belum ada gambar | 2 figures: cnn (96.04% re-eval 20K test) + 2hand (98.45% re-eval 3.9K test) | Verified real, bukan ilustrasi |
| Per-Class | Angka training-time gak bisa dibuktiin | Re-evaluated via sklearn classification_report, verified | Data verified |
| Improvement 3.08% | "Karena balancing" only | "Balancing + data quantity (138K→208K)" | Apple-to-apple fix |
| ONNX / Browser | Gak ada section sama sekali | Section 3.8 baru: PyTorch→ONNX opset 12, 241KB, onnxruntime-web 1.17.1, MediaPipe 0.10.35 float16, flow lengkap, verified diff <0.000004, privacy | Contribution signifikan |
| Meeting bug | Gak ada | Section 4.5 baru: 3 bugs (no TURN, no retry, race join) + fix (TURN publik, ICE-restart, proactive join) | Sudah ada di peer-manager.js, ini story bagus |
| Limitation | Gak ada | Section 4.6 baru: dataset bias 71%, single split, re-eval 2/8 only, quick tunnel URL churn, public TURN, alphabet only | Jujur = dipercaya reviewer |
| Related Work | Kadang bilang "BISINDO research limited" tapi cite 4 paper | Consistent: "dominated by sequence-based, single-frame underexplored" | Fix contradict |

---

## File yang Harus Disertakan Saat Submit / Compile

```
docs/jurnal-data/
├── main.tex                              ← file utama (compile ini)
├── figures/
│   ├── confusion_matrix.png              ← Fig.1 main (2Hand 98.45%, 130KB) - alias dari confusion_matrix_2hand.png
│   ├── confusion_matrix_2hand.png        ← 2Hand confusion (130KB)
│   ├── confusion_matrix_cnn.png          ← Baseline confusion (126KB, 96.04%)
│   ├── model_comparison_chart.png        ← 8 model bar chart (132KB)
│   └── cnn_2hand_per_class_chart.png     ← Per-class Precision/Recall/F1 (82KB)
├── cnn_2hand_confusion_matrix.png        ← Alias, same as figures/confusion_matrix.png (138KB)
├── cnn_2hand_per_class.csv               ← Table II verified (26 letters)
├── cnn_2hand_per_class_eval.json         ← Per-class JSON
├── project_info.json                     ← All metadata
└── references.bib                        ← (HARUS BUAT - lihat note di bawah)
```

**PENTING**: `references.bib` belum ada — file asli pakai `\bibliography{references}`. Ada 2 opsi:
1. Kirim `references.bib` asli dari teman (file .bib yang dipakai compile PDF awal)
2. Atau ganti `\bibliography{references}` dengan manual `\begin{thebibliography}` inline (copy dari PDF halaman 7-8, 16 references)

---

## Cara Compile PDF

pdflatex tidak ada di VPS/lokal ini (Fedora tanpa texlive). Di laptop atau Overleaf:

```bash
# Lokal (kalau ada texlive)
cd docs/jurnal-data
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Atau upload ke Overleaf:
# - Upload main.tex + folder figures/ + references.bib
# - Compile
```

Atau pakai Docker:

```bash
docker run --rm -v $(pwd):/work -w /work texlive/texlive:latest \
  sh -c "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
```

---

## Yang Masih Need Action dari Tim

1. **references.bib**: file referensi belum ada — minta dari teman yang bikin PDF awal (file .bib) atau convert manual dari References halaman 7-8
2. **Review**: baca main.tex dari atas sampai bawah, cek apakah ada yang mau diubah
3. **Judul**: kalau mau keep judul lama "Real-Time BISINDO Recognition Using MediaPipe Hand Landmark" juga boleh — tapi judul baru lebih menggambarkan contribution
4. **Figure placement**: LaTeX figure placement [ht] mungkin perlu adjust setelah compile (kadang figure lompat halaman)
5. **Page limit**: kalau conference ada limit (misal 6-8 halaman), perlu potong — yang bisa dipotong: 1 confusion matrix aja (pilih 2hand), per-class chart optional

---

## Critical Fix yang Wajib (Kalau Gak Fix, Factual Error)

1. **Section 4 Conclusion**: "cnn_balanced digunakan pada web app & meeting" → HARUS jadi "cnn_2hand deployed (84-feat), balanced champion accuracy (63-feat)"
2. **Table I**: Tambah kolom what changed — jangan 7 baris angka doang tanpa penjelasan
3. **Normalization**: 1 step → 3 step
4. **CNN arch**: "Several layers" → tabel dengan filters + kernel + FC
5. **Confusion matrix**: placeholder → PNG verified

---

## v2 Rewrite (2026-07-22) — 2-Hand + Product Focus

| Area | Sebelum (v1) | Sesudah (v2) |
|------|--------------|--------------|
| Title | Dual-Hand … in the Browser | Dual-Hand 1D-CNN … for Real-Time Indonesian Sign Language **Meetings** (T1) |
| 1-hand | Full 8-model table + dual CM + arch v2 | Minimal: 3-row ablation + 1 sentence |
| System Overview | Tidak ada | Section baru + `system_flow.png` |
| Implementation | ONNX di Method; meeting di Results | Section **Implementation** penuh: ONNX, sentence builder, meeting arch, reliability, deploy |
| Confusion matrix | 2 figs (2hand + baseline) | **Hanya** 2hand di body |
| Per-class table | 7 huruf truncated | **26 huruf** full dari CSV |
| Params | “~30K” | **61,274** exact |
| ONNX size | 241KB claim | 247,229 bytes documented |
| 2-hand letters | “16 of 26” | **17 of 26** (A B D F G H J K M N P Q S T W X Y) from code — ⚠️ STALE: paper final pakai **15/26** (M, N, Q digolongkan single-handed karena ambigu di kamera 2D front-view; lihat justifikasi di main.tex) |
| Latency/FPS | Tidak ada / klaim samar | Explicit PLACEHOLDER (no fake numbers) |
| Figures baru | — | `system_flow.png`, `meeting_arch.png` |

