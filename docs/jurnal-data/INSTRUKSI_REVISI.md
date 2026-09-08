# Revisi Jurnal v2 — 2-Hand + Product Focus

> Baca file ini. ~~Target: IEEE Conference (IEEEtran), 6–10 halaman~~ → **SUPERSEDED: final = Springer LNCS (llncs.cls), keputusan Kevin 2026-09-07.** Sisa konten IEEE di file ini cuma histori.
> File utama: `main.tex`. Figures: `figures/`.

---

## ⚡ STATUS AUDIT 2026-09-07 (baca dulu — dokumen ini SUDAH BASI di 4 poin)

Paper sekarang **LNCS 12 halaman (llncs.cls), BUKAN IEEEtran** — venue berubah di sesi revisi berikutnya. Checklist bawah sudah diaudit silang ke `main.tex` versi sekarang:

| Item | Status di paper sekarang |
|---|---|
| Title T1 | ❌ BASI — judul evolusi jadi: *BISINDO Bridge: Lightweight Dual-Hand Landmark Recognition for In-Browser Sign Language Communication in Video Meetings* |
| Abstract 98.45% + ONNX + WebRTC + dual-hand 84 | ✅ semua ada |
| "Hanya 1 CM (2hand)" | ❌ BASI — **0 CM raster** di body; diganti Fig 6 (TikZ top-6 misclassifikasi) + Table 3 per-class. Desain lebih clean, keputusan belakangan |
| Per-class 26 huruf | ✅ Table 3 |
| Ablation 3 baris | ✅ persis 3 |
| Implementation + System Overview | ✅ ada |
| PLACEHOLDER | ⚠️ 3 komentar `% PLACEHOLDER:` tersisa tapi semua terjawab di teks: named-tunnel = "planned", user-study = "not claimed" (keputusan permanen: skip kfold/user-study), scalability = Table 5 real (N=2/6) |
| grep 138K/208K/Arch-v2 | ✅ nol kemunculan di body |
| Latency | ✅ terukur: 0.53 ms mean ONNX, 18.7 FPS pipeline |
| Upload Overleaf | 👉 aksi manual |

### 4 divergensi notes vs paper (keputusan belakangan, BUKAN kelalaian)

1. **Judul** — T1 di tabel bawah ≠ judul sekarang (lihat atas).
2. **Confusion matrix** — instruksi "1 CM raster" diganti TikZ Fig 6 + Table 3. Raster CM tidak ada sama sekali.
3. **17 vs 15 huruf dua-tangan** — CHANGES.md (v2) bilang 17/26; paper sekarang bilang **15/26** dengan justifikasi eksplisit (M, N, Q ambigu di kamera 2D front-view; digolongkan single-handed). Paper self-consistent; angka 17 di CHANGES.md = STALE.
4. **Venue/format** — notes: IEEE 6–10 hal. Paper: LNCS 12 hal. ✅ **DIPUTUSKAN Kevin 2026-09-07: LNCS dipertahankan.** Bukan lagi item terbuka — header "Target: IEEE" di file ini ga berlaku lagi.

---

## Keputusan desain (2026-07-22)

| Pilihan | Value |
|---------|-------|
| Venue | IEEE Conference (bukan journal panjang) |
| 1-hand di body | **Minimal** — 1 paragraf + 1 baris ablation table |
| Fokus | Dual-hand 84-feat + ONNX browser + WebRTC meeting |
| Title | T1: *BISINDO Bridge: Dual-Hand 1D-CNN Recognition with Browser Inference for Real-Time Indonesian Sign Language Meetings* |
| Deployed model | `cnn_2hand` 98.45% (bukan balanced 99.12%) |
| Params | 61,274 (exact) |
| ONNX size | 247,229 bytes (~241 KB) |

---

## Outline `main.tex` (v2)

1. Abstract — 2-hand lead, product, 1-hand 1 kalimat
2. Introduction — 4 contributions product-first
3. Related Work — SLR / MediaPipe / 1D-CNN / gap
4. **System Overview** — Fig. `system_flow.png`
5. Methodology — dataset 26K dual-hand first; arch v1 only
6. **Implementation** — ONNX, sentence builder, meeting arch, reliability, deploy
7. Experiments — CM 2hand only, full 26-letter table, ablation 3 baris, latency PLACEHOLDER
8. Discussion + Limitations
9. Conclusion
10. Bibliography (manual thebibliography)

---

## Rule 1-hand (WAJIB)

**Boleh:**
- 1 kalimat di abstract
- 1 paragraf di dataset / discussion
- Table ablation 3 baris (Baseline 96.04 / Balanced 99.12 / **2Hand 98.45**)

**Dilarang di body:**
- Table 8 model full
- Confusion matrix baseline 63-feat
- Arch v2 (flatten 512) detail table
- Narasi panjang 138K→208K

---

## PLACEHOLDER yang harus diisi (jangan invent angka)

Cari di `main.tex`: `% PLACEHOLDER:`

| ID | Apa | Cara isi |
|----|-----|----------|
| `latency-browser` | mean/P95 ONNX ms Chrome laptop | instrument meeting.js / test.html |
| `fps-mediapipe` | full pipeline FPS 640×480 | same |
| `phone-benchmark` | Android Chrome (opsional) | device test |
| `product-smoke` | N peers, reconnect rate | log session internal |
| `user-study` | formal N-user | atau biarkan “internal only” |
| `hardware-train` | CPU/GPU training | catatan mesin |
| `named-tunnel` | stable URL | Cloudflare named tunnel |
| `contrib-ethics` | consent wording | 1 kalimat formal |
| `kfold` / `cross-signer` | eval tambahan | retrain / re-split |
| `bib-cleanup` | author/year/DOI | perbaiki 3–4 bibitem lemah |
| `cite:browser-slr` | ref browser ONNX/WebRTC | optional |

---

## Figures

| File | Dipakai di |
|------|------------|
| `figures/system_flow.png` | Fig. system overview |
| `figures/meeting_arch.png` | Fig. meeting architecture |
| `figures/confusion_matrix_2hand.png` | **Satu-satunya** CM di body |
| `figures/cnn_2hand_per_class_chart.png` | Per-class chart |
| `figures/confusion_matrix_cnn.png` | **Tidak** di body (opsional appendix) |
| `figures/model_comparison_chart.png` | **Tidak** di body |

---

## Checklist compile / submit (di-audit 2026-09-07 — lihat STATUS AUDIT di atas)

- [x] ~~Title T1 final~~ → BASI: judul sekarang *BISINDO Bridge: Lightweight Dual-Hand Landmark Recognition for In-Browser Sign Language Communication in Video Meetings*
- [x] Abstract sebut 98.45%, ONNX, WebRTC, dual-hand 84
- [x] ~~Hanya 1 confusion matrix (2hand)~~ → BASI: 0 CM raster; diganti TikZ Fig 6 (top-6 misclass) + Table 3 per-class
- [x] Per-class 26 huruf (Table 3 di tex, full)
- [x] Ablation table 3 baris saja
- [x] Section Implementation + System Overview ada
- [x] Semua angka hilang = PLACEHOLDER terlihat → 3 komentar tersisa, semuanya sudah terjawab di teks (planned / not claimed / Table 5 real)
- [x] `grep 138,471\|208,000\|Arch1\|v2 architecture` = nol di body
- [x] ~~Upload ke Overleaf: `main.tex` + `figures/*`~~ → tersedia `overleaf-upload.zip` di folder (aksi manual tinggal upload)
- [x] Isi PLACEHOLDER latency sebelum camera-ready → 0.53 ms mean ONNX, 18.7 FPS pipeline (terukur)
- [x] ~~Konfirmasi template~~ → **DIPUTUSKAN 2026-09-07 (Kevin): TETAP LNCS.** IEEE-era notes di file ini resmi superseded. Paper final = llncs.cls 12 hal

---

## Compile

```bash
# Overleaf: upload main.tex + figures/
# atau Docker:
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  sh -c "pdflatex main.tex && pdflatex main.tex"
```

Tidak ada `references.bib` — bibliography inline di `main.tex`.
