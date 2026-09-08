# BISINDO Bridge — Panduan Ambil Data Eval

Status placeholder di `main.tex`:
- [DONE] #1 Classifier-only latency (ONNX) — sudah diukur otomatis, sudah masuk jurnal.
- [MANUAL] #2 Browser end-to-end FPS — instrumen sudah ditanam, kamu jalanin di Chrome.
- [MANUAL] #3 Mesh scalability (N=2..5) — butuh beberapa device/orang.
- [MANUAL] #4 Broadcast latency (emit -> render) — butuh 2 device, clock sinkron.

Instrumen kode sudah ditanam di:
- `meeting/static/js/meeting.js`  (FPS bench + broadcast latency collector)
- `meeting/app.py`                (server meneruskan `t_sent`)

Cara start server lokal + quick tunnel (SUDAH OTOMATIS di app.py):
    cd meeting && BISINDO_TUNNEL=1 BISINDO_PORT=4500 python app.py
    # Lokal:  http://localhost:4500
    # Publik: URL https://xxxx.trycloudflare.com muncul di log (buat share ke device lain)
    # Tunnel forward ke laptop ini (port 4500). VPS sudah tidak dipakai.

--------------------------------------------------------------------
## CEKLIS SCREENSHOT (ambil pas testing, buat lampiran/bukti jurnal)
Angka utama TETAP dari console (lebih akurat). SS = bukti pendukung.

[ ] SS-1  Overlay deteksi jalan: tangan + landmark + huruf terprediksi muncul.
          (bukti sistem berfungsi — 1x cukup)
[ ] SS-2  Console `[fps-bench]` 3 baris hasil FPS (#2). Bukti angka real.
[ ] SS-3  Tampilan meeting saat N=3 orang: berapa kotak video ke-render.  (#3)
[ ] SS-4  Tampilan meeting saat N=4 dan/atau N=5 (kalau sempat kumpulin orang). (#3)
[ ] SS-5  Kalau ada peer yang video-nya freeze/hitam: SS itu — bukti degradasi mesh. (#3)
[ ] SS-6  Console `[bcast-lat]` hasil broadcast latency (#4).
[ ] SS-7  (opsional) Log server yang nampilin URL trycloudflare + status ok.

Simpan semua SS ke: docs/jurnal-data/figures/eval-screenshots/
Kasih nama jelas: fps.png, mesh-n3.png, mesh-n4.png, freeze.png, bcast.png dst.

--------------------------------------------------------------------
## #1 Classifier latency (SUDAH SELESAI)
Skrip: /tmp/bench_onnx.py  (regenerate kalau perlu)
Hasil: mean 0.53 ms / median 0.31 ms / P95 1.21 ms (~1870 inf/s, CPU 14-thread).
Sudah tertulis di main.tex bagian sebelum System Evaluation.

--------------------------------------------------------------------
## #2 Browser end-to-end FPS  (kamu jalanin, ~1 menit)
Butuh: 1 device + Chrome + webcam. TIDAK perlu orang lain.

Langkah:
1. Jalankan meeting server, buka di Chrome, join room, nyalakan kamera.
2. Pastikan deteksi tangan jalan (overlay landmark muncul).
3. Buka DevTools (F12) -> Console.
4. Ketik:  startFpsBench(100)   lalu Enter.
5. Tahan tangan di depan kamera ~5-10 detik sampai 100 frame terkumpul.
6. Console otomatis print 3 baris `[fps-bench]`. COPY 3 baris itu, kasih ke agent.

Yang gw butuh: per-frame ms (mean/median/P95) dan FPS (mean/median/P95).
Catat juga: device + CPU/GPU + resolusi kamera (default 640x480).

--------------------------------------------------------------------
## #3 Mesh scalability  (butuh 2-5 device/orang)
Butuh: N device join room yang sama (N=2,3,4,5). Bisa campur laptop+HP,
atau beberapa tab di device beda (jangan 1 device banyak tab — bias).

Untuk tiap N, tiap observer catat:
  - N (jumlah partisipan di room)
  - Observer (nama/device)
  - Streams received: dari N-1 peer, berapa video yang benar-benar tampil
  - Frame drops? (y/n) — video patah-patah / freeze
  - Device / uplink (mis. "Laptop / WiFi 50Mbps", "HP / 4G")

Isi ke tabel `tab:scalability` di main.tex (baris N=2..5).
Screenshot OPSIONAL sebagai bukti; yang wajib angkanya.
Tips: kalau susah kumpulin 5 orang, minimal N=2 dan N=3 real, N=4/5 boleh
ditandai "not tested" dan dibahas sebagai future work — jangan mengarang data.

--------------------------------------------------------------------
## #4 Broadcast latency emit->render  (butuh 2 device)
Butuh: 2 device join room sama. IDEAL: clock kedua device sinkron (NTP / same LAN),
karena latency dihitung Date.now() penerima minus t_sent pengirim.

Langkah:
1. Device A dan B join room sama, kamera nyala.
2. Di Device B (penerima), buka DevTools Console — biarkan terbuka.
3. Di Device A, peragakan huruf sampai beberapa huruf ter-commit
   (tiap commit ngirim event ke B). Lakukan ~15-30 huruf.
4. Di Device B console, ketik:  reportBcastLat()  lalu Enter.
5. COPY baris `[bcast-lat]` (n, mean, median, P95, min, max), kasih ke agent.

PENTING soal clock: kalau 2 device tidak sinkron NTP, angka absolut bisa bias
(bahkan negatif). Paling bersih: 2 tab di 1 device yang sama (clock identik) untuk
baseline pipeline+socket, ATAU pastikan kedua device NTP-synced. Laporkan metode
yang dipakai supaya bisa ditulis jujur di jurnal.

--------------------------------------------------------------------
## Setelah dapat angka
Kasih semua angka ke agent; agent akan isi:
- #2 -> paragraf "End-to-end latency and frame rate" + placeholder e2e-fps
- #3 -> tabel tab:scalability
- #4 -> placeholder broadcast-latency
Kalau ada yang tidak sempat diukur, agent akan reframe jadi "not evaluated / future work"
secara jujur, bukan mengarang.
