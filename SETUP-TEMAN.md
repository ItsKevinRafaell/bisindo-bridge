# Setup dari Nol — buat teman kelompok

Tujuan dokumen ini: supaya kamu bisa **menjalankan project di laptop sendiri**
untuk ambil screenshot / evidence, tanpa perlu bertanya-tanya.

---

## 0. Prasyarat (sekali saja)

1. **Python 3.10 atau 3.11** — cek di terminal: `python --version`
   (download: https://www.python.org/downloads/ — centang "Add to PATH" di Windows)
2. **Git** — cek: `git --version` (download: https://git-scm.com)
3. **Browser Chrome atau Edge** (deteksi tangan pakai MediaPipe yang jalan di browser)
4. **Webcam** (laptop bawaan cukup)

## 1. Clone repo

```bash
git clone https://github.com/ItsKevinRafaell/bisindo-bridge.git
cd bisindo-bridge
```

Repo ini publik, jadi tidak perlu akun/izin apa pun untuk clone.

## 2. Virtual environment + install dependencies

```bash
python -m venv .venv
```

Aktifkan:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`
- **Mac/Linux:** `source .venv/bin/activate`

Lalu install:

```bash
pip install -r requirements.txt
```

Kalau ada error `ModuleNotFoundError: No module named 'flask'`, jalankan lagi
`pip install -r requirements.txt` dari folder root (pastikan venv aktif — ada
`(.venv)` di depan prompt).

## 3. Jalankan aplikasi

```bash
cd meeting
python app.py
```

Tunggu sampai muncul log server, lalu buka di browser:

```
http://localhost:4500
```

> Pakai `http://localhost:4500` persis seperti itu (bukan IP lain) supaya izin
> kamera tidak diblokir browser.

Saat diminta izin kamera → **Allow**. Kamera dinyalakan berarti pipeline
MediaPipe + model ONNX sudah jalan 100% di browser kamu (tidak mengirim video
ke mana pun).

## 4. Yang bisa di-screenshot untuk evidence

| Bukti | Cara |
|---|---|
| Meeting + recognition realtime | Buat room → izinkan kamera → lakukan gestur huruf → panel prediksi tampil |
| Full-mesh 2 peserta | Buka `http://localhost:4500` di **2 tab**, join room yang sama |
| Gestur dua-tangan vs satu-tangan | Deployed model = 2 tangan (84 fitur); gestur 1 tangan tetap terbaca (blok tangan kedua zero-padded) |
| Statistik server | Buka `http://localhost:4500/api/stats` dan `/api/health` |
| Backspace/delete gesture | Peragakan gestur delete → huruf terakhir terhapus |

Checklist evidence lengkap (SS-1 … SS-n) ada di
`docs/jurnal-data/EVAL_PROTOCOL.md`.

## 5. Catatan penting

- **Dataset 2.2 GB TIDAK ada di repo** (kegedean untuk GitHub). Untuk running
  demo + screenshot tidak perlu dataset sama sekali. Kalau mau retrain model,
  minta dataset ke Kevin.
- Model ONNX untuk browser sudah ada di dalam repo (`web/models/`), jadi
  recognition langsung jalan setelah clone.
- Tunnel ngrok / TURN / Cloudflare **tidak perlu** untuk demo lokal — WebRTC
  otomatis fallback ke STUN untuk koneksi localhost/2-tab.
- Semua yang dijalankan murni lokal di laptopmu; tidak ada data yang dikirim
  ke server publik.

## 6. Troubleshooting

| Masalah | Solusi |
|---|---|
| `python` tidak dikenali | Coba `py` (Windows) atau `python3` (Mac/Linux) |
| `ModuleNotFoundError: flask` | `pip install flask flask-socketio flask-cors` (venv harus aktif) |
| Kamera hitam / tidak minta izin | Pastikan URL `http://localhost:4500`, tutup aplikasi lain yang pegang kamera (Zoom/Meet) |
| Port 4500 terpakai | Matikan proses lain, atau jalankan `set PORT=4501` lalu `python app.py` (Windows) / `PORT=4501 python app.py` (Mac/Linux) |
| Prediksi tidak stabil | Tangan harus masuk frame, pencahayaan cukup, jarak ±50–70 cm dari webcam |

Kalau mentok, screenshot error-nya dan kirim ke grup.
