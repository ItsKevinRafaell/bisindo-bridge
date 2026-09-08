# BRIEF GAMBAR MANUAL — BISINDO Bridge (jurnal)

10 gambar diganti placeholder di paper (kotak putus-putus bertuliskan
"TEMPORARY PLACEHOLDER"). Tugas: bikin 10 gambar di bawah ini, taruh di folder
`figures/manual/` dengan nama file yang disebut, lalu agent tinggal swap
(\manualfig -> includegraphics, caption & nomor gambar SUDAH FIX, tidak berubah).

Format umum:
- PNG, background putih/polisih, minimal 1600 px lebar (rasio bebas, ikut spek).
- Warna bebas asal konsisten antar-gambar; teks HARUS terbaca saat dicetak
  (font minimal ~9pt setelah di-scale ke lebar 17 cm).
- Bahasa label: Inggris (ikuti paper).

---

## 1. fig-system.png — Pipeline end-to-end (hal. 2)
Gambar flowchart 2 baris (bentuk ular/S):
- Baris 1 (kiri→kanan): `Webcam frame` → `MediaPipe Hands (2 x 21 landmarks)` →
  `Flatten (84)` → `Normalize (3 stages)` → `1D-CNN ONNX (61,274 params, 241 KB)`
- Baris 2 (kanan→kiri): `Softmax` → `Sentence builder` → `Local TTS` →
  `Socket.IO` → `Peers (full-mesh WebRTC)`
- Ujung baris 1 nyambung turun ke baris 2 (di sisi kanan).
- SEMUA kotak baris 1 + Softmax/Sentence builder/TTS dibungkus kotak putus-putus
  berlabel: `on-device — no video leaves the browser`.
- Panah: solid, simpel, SATU arah, jangan menyilang.

## 2. fig-landmarks.png — 21 landmark di 2 tangan asli (hal. 4)
- Sumber paling gampang: screenshot dari aplikasi BISINDO Bridge saat preview
  webcam, posisi tangan nandatanganin huruf DUA TANGAN (misal huruf B).
  Kalau aplikasi belum nampilin titik landmark: buka MediaPipe Studio
  (https://developers.google.com/mediapipe) di browser, pilih Hand Landmarker,
  live webcam, screenshot dua tangan.
- Crop ketat, 2 tangan jelas, overlay skeleton terlihat (21 titik per tangan).
- Tambahin (boleh di Canva) panah kecil: label `21 landmarks` di satu tangan.

## 3. fig-normgeo.png — 3 tahap normalisasi (hal. 5)
- 3 panel berjejer dari frame tangan YANG SAMA:
  1. `Raw` — dua tangan di posisi/ukuran beda (screenshot apa adanya)
  2. `Wrist translated` — kedua pergelangan di titik origin (bisa di-edit:
     geser screenshot supaya pergelangan rata tengah)
  3. `Size scaled` — dua tangan jadi ukuran sama (crop/zoom samain)
- Satu tangan diberi tint teal, satu biru (boleh overlay transparan).
- Caption kecil di bawah: `Stage 3: per-axis z-score (not shown)`.

## 4. fig-twolane.png — Offline vs Online (hal. 7)
- Dua baris flowchart:
  - Baris atas, judul `OFFLINE — once, at training time` (warna biru):
    `dual-hand dataset (26,192)` → `3-stage normalization (fit scaler)` →
    `train 1D-CNN (61,274 params, 152.8 s CPU)` → `export ONNX (241 KB)`
  - Baris bawah, judul `ONLINE — every frame, in-browser` (warna hijau):
    `MediaPipe Hands` → `normalize (saved scaler)` →
    `ONNX Runtime Web (0.53 ms)` → `letter + sentence builder`
- SATU panah putus-putus merah dari `export ONNX` (baris atas) turun ke
  `ONNX Runtime Web` (baris bawah), label: `same weights + scaler`.

## 5. fig-privacy.png — Data flow & privasi (hal. 8)
- Kiri: kotak putus-putus besar berlabel `your browser` berisi
  `camera` → `MediaPipe` → `ONNX 1D-CNN`. Di bawah kotak: teks kecil
  `84 floats stay on-device`.
- Kanan atas: kotak `Socket.IO server (signaling + text only)`.
- Kanan bawah: kotak `peer browsers (full-mesh WebRTC)`.
- Panah 1 (putus-putus): browser → server, label `letters, sentences (text events)`.
- Panah 2: browser → peers, label `audio / video (peer-to-peer)` — garis ini
  HARUS memutar MELEWATI/BAWAH server (jangan lewat kotak server) supaya
  kelihatan video ga pernah sentuh server.

## 6. fig-meeting.png — Arsitektur meeting full-mesh (hal. 9)
- 4 lingkaran `P1 P2 P3 P4` menyusun persegi.
- GARIS SOLID merah: antar SEMUA pasangan peer (6 garis, dua arah) = media P2P.
- Kotak `Flask-Socket.IO (signaling only, no media relay)` di kanan;
  garis PUTUS-PUTUS dari tiap peer ke server = signaling (SDP/ICE).
- Kotak kecil `Google STUN (NAT traversal)` dan `TURN (best-effort)` —
  garis putus-putus dari peer ke STUN/TURN.
- Legenda kiri bawah: solid merah = `media (P2P, full-mesh, O(n^2))`,
  putus-putus = `signaling (SDP/ICE)`.
- TIPS: garis media antar peer bisa agak melengkung (curve) supaya ga numpuk
  di tengah; yang penting 6 garis kebaca semua.

## 7. fig-features.png — Anatomi vektor 84-D (hal. 6)
- Satu strip panjang horizontal, dibagi 84 sel kecil (bikin di Canva/Excel,
  JANGAN gambar 84 kotak satu-satu — bikin blok besar lalu garis tick tipis).
- Blok kiri (teal): `hand 1 — slots 0–41`. Blok kanan (biru):
  `hand 2 — slots 42–83`.
- Garis putus-putus MERAH vertikal di antara slot 41 dan 42, label kecil:
  `block boundary`.
- Satu jendela width-3 di dalam hand 1 di-highlight (kotak merah tipis di atas
  3 sel), label: `width-3 kernel: joint + 2 neighbours (same hand)`.
- Jendela kernel yang memotong garis batas juga di-highlight, label:
  `the only kernel that spans both hands`.
- INGAT: strip-nya PANJANG dan RENDAH (rasio ~6:1), teks label DI ATAS/DI BAWAH
  strip, bukan di dalam sel.

## 8. fig-cnn-arch.png — Arsitektur 1D-CNN (hal. 7)
- SATU baris lurus kotak berjejer (jangan dua baris, jangan panah muter):
  `Input 1×84` → `Conv1 64, k=3, ReLU, Pool 2 (→64×42)` →
  `Conv2 128, k=3, ReLU, Pool 2 (→128×21)` → `Conv3 64, k=3, ReLU, AAP(1) (→64×1)`
  → `Flatten 64` → `FC1 64→128, ReLU, Drop 0.3` → `FC2 128→26` → `Softmax A–Z`
- Di ATAS deretan kotak conv, gambar "penggaris" bar yang makin pendek:
  84 → 42 → 21 → 1 (label: `spatial length`).
- Warna: conv teal, FC oranye, softmax merah (atau bebas, konsisten aja).
- Total parameter 61,274 — udah ada di caption, ga perlu digambar.

## 9. fig-context.png — Sistem dipakai beneran (hal. 3, System Overview)
- Foto ATAU mock screenshot: 2 orang meeting, masing-masing di depan laptop.
  Satu orang NANDATANGANIN huruf ke webcam; di layar temennya kelihatan huruf
  muncul di panel sentence.
- Kalau foto: lighting ruaman normal, wajah boleh diblur. Kalau mock: screenshot
  call 2 orang (preview webcam nampilin tangan) + panel huruf kelihatan.
- Rasio landscape (~3:2), jangan terlalu tinggi.

## 10. fig-samples.png — Contoh frame dataset (hal. 4-5, Methodology-Dataset)
- Grid 3 kolom x 2 baris, 6 frame ASLI dari dataset, huruf A-F.
- Tiap tile: crop konsisten (tangan ketengah), badge huruf kecil di pojok.
- Sumber: frame dari folder dataset / preview aplikasi / screenshot MediaPipe
  Studio dengan tangan dua-duanya kelihatan. Ukuran tile sama semua.

---

Setelah semua PNG ada di `figures/manual/`, bilang ke agent: "gambar manual udah
masuk, swap 10 placeholder" — sisanya (swap, recompile, audit, zip) otomatis.
Catatan: gambar latency, frame rate, dan confusion matrix TIDAK termasuk tugas
manual — itu udah dibikinin chart matplotlib bersih dari data asli.
