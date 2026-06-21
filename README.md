# BISINDO Bridge

Two-way Bahasa Isyarat Indonesia (BISINDO) translator. Crowd-sourced 2-hand
gesture capture → CSV → trained model → browser test page.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    VERCEL (static)                           │
│  /           — landing page (Test / Meeting / Contribute)    │
│  /test       — test model in your browser (offline)          │
│  /capture    — record samples → POST to laptop              │
│  /models/    — TF.js model + scaler + labels                 │
└──────────────────────────────────────────────────────────────┘
            ▲ HTTPS
            │
┌──────────────────────────────────────────────────────────────┐
│              LAPTOP (Flask-SocketIO)                         │
│  meeting/app.py   — meeting rooms, video relay, /api/sample  │
│  dataset/landmarks_captured_v2.csv   — 67-col, 2-hand only   │
│  models/landmark_classifier.pkl     — RF for server predict  │
└──────────────────────────────────────────────────────────────┘
            ▲
            │ Cloudflare named tunnel (stable URL)
            │
       user browser / phone
```

## Quick start

### 1. Run the laptop server
```bash
./start.sh           # installs deps, starts Flask-SocketIO on :5000
```
The server prints its local URL. Test: `curl http://localhost:5000/api/health`.

### 2. Run the test page (no server needed)
```bash
cd web && python3 -m http.server 8000
# open http://localhost:8000/test.html
```
Or deploy the `web/` folder to Vercel.

### 3. Train a model
```bash
python3 train_landmark_model.py
```
Reads `dataset/landmarks_captured_v2.csv`, writes:
- `models/landmark_classifier.pkl` — sklearn RF (laptop server)
- `models/landmark_classifier_scaler.pkl`, `*_labels.pkl`, `*_metadata.json`
- `models/report.md` — per-letter accuracy, confusion matrix
- `models/mlp_model.h5` — Keras MLP
- `web/models/model.json` + shards + `labels.json` + `scaler.json` (TF.js)

Requires: `pip install tensorflow tensorflowjs` for the TF.js export.

## Data schema (67-col CSV)

```
letter, image_path, split, num_hands, contributor, lm0_x, lm0_y, lm0_z, ..., lm20_z
```

- 2-hand only. Hand 2 is **not** stored — the model pads it with zeros at
  inference (left vs right handedness is not in BISINDO alphabet).
- `contributor` is the user's name (or "legacy" / "kaggle" for migrated data).
- `num_hands` is always `2` in the new schema.

To migrate old CSVs (130-col mixed format): `python3 tools/migrate_csv.py`.

## Data targets

| Goal | Per-letter | Total | Expected acc |
|------|-----------|-------|--------------|
| Minimum demo | 500 | 13k | 88-92% |
| Good demo | 2,000 | 52k | 93-96% |
| Production | 5,000 | 130k | 95-97% |
| Overkill (project default) | 50,000 | 1.3M | 98%+ |

Multi-user > single-user. With 50k/letter, get 5+ contributors per letter
or the model overfits to one person's hand shape.

## Capture sources (all write to the same CSV)

1. **Web capture** (`web/capture.html`): phone camera, MediaPipe HandLandmarker,
   POSTs JSON to laptop. Easiest for distributed users.
2. **Laptop /train page** (`meeting/templates/train.html`): same as web but via
   Socket.IO on the local server.
3. **Desktop CLI** (`collect_fast.py`): webcam + OpenCV window. Single user, fast.
4. **Auto-capture** (`src/auto_capture_new.py`): continuous capture via OpenCV.

## Stable public URL (Cloudflare + DuckDNS)

The capture page needs a stable HTTPS URL to reach the laptop. Setup:

1. Create free Cloudflare account: https://dash.cloudflare.com/sign-up
2. Create free DuckDNS account: https://www.duckdns.org/, claim `bisindo-bridge`
3. Install `cloudflared`: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
4. ```bash
   cloudflared tunnel login
   cloudflared tunnel create bisindo
   ```
5. Write `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: bisindo
   credentials-file: /home/kevin/.cloudflared/<UUID>.json
   ingress:
     - hostname: bisindo-bridge.duckdns.org
       service: http://localhost:5000
     - service: http_status:404
   ```
6. ```bash
   cloudflared tunnel route dns bisindo bisindo-bridge.duckdns.org
   ./start-tunnel.sh
   ```
7. Update `web/config.js`:
   ```js
   export const LAPTOP_API_URL = "https://bisindo-bridge.duckdns.org";
   ```
8. Re-deploy to Vercel: `vercel --prod`.

If your public IP changes (most ISPs do periodically), point DuckDNS at the
new IP — `cloudflared` does this automatically if you use the Cloudflare
DDNS updater, or you can do it manually at https://www.duckdns.org/.

## Deploy to Vercel

```bash
npm i -g vercel
vercel login
vercel --prod
```

The `vercel.json` already routes `/`, `/test`, `/capture`, `/models/*`. The
TF.js model needs to be in `web/models/` (run the training script first).

## File map

```
bisindo-bridge/
├── meeting/                  # Flask-SocketIO server (laptop)
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/            # legacy /train + / meeting
│   └── static/mediapipe/     # empty (see README.md)
├── web/                      # Vercel-deployable static frontend
│   ├── index.html            # landing
│   ├── test.html             # browser TF.js inference
│   ├── capture.html          # phone capture → laptop
│   ├── config.js             # LAPTOP_API_URL constant
│   └── models/               # model.json, labels.json, scaler.json (after training)
├── src/                      # Python helpers
│   ├── landmark_classifier.py  # inference (laptop server)
│   ├── augment_landmarks.py    # data augmentation
│   ├── auto_capture_new.py     # OpenCV capture
│   ├── gesture_detector.py
│   ├── landmark_extractor.py
│   ├── landmark_extractor_v2.py
│   ├── landmark_trainer.py
│   ├── tts_engine.py
│   ├── speech_processor.py
│   └── gesture_guide_db.py
├── dataset/
│   ├── landmarks_captured_v2.csv   # unified 67-col, 2-hand
│   ├── landmarks.csv
│   ├── landmarks_metadata.json
│   └── readme.md
├── models/                   # output of training
│   ├── landmark_classifier.pkl
│   ├── landmark_classifier_scaler.pkl
│   ├── landmark_classifier_labels.pkl
│   ├── landmark_classifier_metadata.json
│   ├── mlp_model.h5
│   └── report.md
├── tools/
│   └── migrate_csv.py        # one-shot 130-col → 67-col
├── train_landmark_model.py
├── collect_fast.py           # desktop CLI capture
├── start.sh
├── start-tunnel.sh
├── vercel.json
└── README.md
```

## Troubleshooting

**"Classifier load failed"** on server start — run `train_landmark_model.py` first.

**Capture page says "laptop offline"** — the laptop server isn't reachable at
`LAPTOP_API_URL`. Check the URL, run `./start-tunnel.sh`, verify the tunnel.

**TF.js test page says "Model load failed"** — the TF.js files aren't in
`web/models/`. Re-run training, or check Vercel deployment included them.

**Predictions are wrong** — not enough data per letter. Target is 50,000
per letter with multiple contributors. Run `models/report.md` to see
per-letter accuracy.