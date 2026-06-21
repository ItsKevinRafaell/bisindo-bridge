# MediaPipe assets

This directory is **intentionally empty** in the cleaned-up codebase.

Both `meeting/templates/train.html` and `meeting/templates/index.html` now use the
new `@mediapipe/tasks-vision` HandLandmarker API, loaded from the jsDelivr CDN.
No local wasm/tflite/task files are needed.

The legacy `hands.js` / `*.wasm` / `hands.binarypb` files were removed in the
v2 cleanup because the legacy `Hands` solution is deprecated and conflicts with
the new `HandLandmarker` runtime.

If you need to load MediaPipe locally (e.g., offline kiosk), download the
runtime + hand_landmarker.task into this directory and update the JS to use
local paths.