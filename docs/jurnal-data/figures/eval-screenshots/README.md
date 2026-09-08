# Bukti Visual Evaluasi Sistem Live (BISINDO Bridge)

Sesi uji: room `room_egl4zl`, quick tunnel Cloudflare ke laptop (bukan VPS).
Peak peserta: **6 orang tersambung**. Sesi ini menangkap beberapa bug nyata
pada arsitektur full-mesh WebRTC — jadi bukti utama keterbatasan skalabilitas
yang dibahas di paper (bukan hasil ideal yang dikarang).

Angka utama diambil dari console browser; screenshot ini bukti pendukung.

---

## 01-lobby-join.jpeg — Halaman lobby / pre-join
Landing "BISINDO Meeting → Siap Meeting?" dengan preview kamera, form nama +
kode room, kontrol mic/kamera, tombol "Gabung Sekarang". Kondisi normal, tanpa bug.

## 02-normal-4tiles.jpeg — Kondisi NORMAL (baseline)
Grid 2x2, 4 tile ter-render penuh, semua video tampil, indikator active-speaker
(border biru) berfungsi. Ini baseline pembanding "semuanya jalan".
Catatan: sebagian video blur / angle rendah = kualitas kamera peserta, BUKAN bug app.

## 03-letter-detection.jpeg — Fitur pengenalan huruf isyarat
Panel BISINDO menampilkan "Sedang Membangun: XGANFENJGU", prediksi huruf "U",
tombol saran kata (U / AKU), kontrol Space/Enter/Speak/Clear, status "Model ready".
BUG yang tertangkap: **confidence tampil "1017%"** (melebihi 100%) — bug formatting
persentase pada UI prediksi (nilai kemungkinan tidak dinormalisasi / salah kali 100).
Rangkaian huruf "XGANFENJGU" = akumulasi deteksi mentah yang belum membentuk kata,
menunjukkan huruf per huruf ter-commit tapi belum ada koreksi kata/spellcheck.

## 04-mesh-6joined-4shown.jpeg — Bukti KUNCI: 6 join, hanya 4 tampil
Counter peserta di header = **6**. Panel People mendaftarkan 6 nama, TAPI grid
video cuma menampilkan **4 orang** (Kevin, jomsing, nona, jouxing). Dua peserta
lain (A311A dan satu lagi) tidak muncul feed videonya di sisi ini.
=> Reception ASIMETRIS: jumlah orang di room != jumlah video yang benar-benar
diterima tiap klien. Ini gejala langsung breakdown full-mesh O(n^2).
TEMUAN KEAMANAN (bonus): salah satu username diisi string
`SELECT * FROM users WHERE user` — percobaan SQL-injection lewat field nama.
Perlu input sanitization / validasi username (future work keamanan).

## 05-black-tile-bug.jpeg — Tile hitam & salah framing
Dari 5 tile yang muncul: tile **A311A HITAM TOTAL** (video tidak ter-render /
stream gagal masuk walau peserta ada di room), dan tile **jouxing salah framing**
(cuma langit-langit yang kelihatan). Slot ke-6 kosong. Warna border tile beda-beda
(status/speaker). Ini contoh "muka ga muncul / layar ngebug" yang dimaksud.

## 06-failed-join-blank.jpeg — Device GAGAL tersambung
Device lain (user "hehe") secara teknis SUDAH masuk room (nama muncul, URL tunnel
`salad-delete-crimes-squad.trycloudflare.com/room/room_egl4zl`), TAPI layar utama
**hitam total, tidak ada peserta lain yang muncul** — koneksi media WebRTC gagal
terbentuk. Chrome juga menampilkan "didn't shut down correctly" (indikasi crash
sebelumnya). => Kasus "gagal join penuh": masuk room OK, negosiasi media/ICE gagal.

---

## Ringkasan bug yang terdokumentasi (untuk paper)
1. Reception asimetris: 6 tersambung, tiap klien cuma terima 4 stream (SS 04).
2. Tile hitam / stream gagal render walau peserta hadir (SS 05, A311A).
3. Salah framing kamera peserta (SS 05, jouxing) — kualitas capture, bukan sistem.
4. Gagal join penuh: masuk room tapi media WebRTC tidak terbentuk, layar blank (SS 06).
5. Freeze/stuck stream di tengah sesi (dilaporkan user; gejala sejenis SS 05).
6. Bug UI confidence "1017%" pada prediksi huruf (SS 03) — formatting persen.
7. Keamanan: field username menerima payload SQL-injection tanpa sanitasi (SS 04).

Semua ini KONSISTEN dengan argumen paper: arsitektur full-mesh peer-to-peer
tidak scale melewati ~4-6 peserta pada jaringan campuran (WiFi + seluler);
jumlah koneksi O(n^2) menyebabkan stream drop, join gagal, dan freeze.
Motivasi jelas untuk future work: pindah ke SFU/relay (mis. mediasoup/LiveKit).
