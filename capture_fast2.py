#!/usr/bin/env python3
"""Ultra fast capture - x,y only (84 features, 2 hands)."""
import os, time, csv, argparse
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

CSV_FILE = "dataset/landmarks_2hands.csv"
# 84 cols: lm0_x,lm0_y,...lm20_x,lm20_y + h2_lm0_x,h2_lm0_y,...h2_lm20_x,h2_lm20_y
CSV_HEADER = ["letter","path","split","num_hands","contributor"]
CSV_HEADER += [f"lm{i}_{c}" for i in range(21) for c in ("x","y")]
CSV_HEADER += [f"h2_lm{i}_{c}" for i in range(21) for c in ("x","y")]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("letter", help="Letter (e.g. D)")
    p.add_argument("--rate", type=float, default=0.05, help="Sec between captures (default: 0.05 = 20/sec)")
    p.add_argument("--count", type=int, default=1000, help="Target (default: 1000)")
    args = p.parse_args()

    letter = args.letter.upper()

    # MediaPipe
    base = python.BaseOptions(model_asset_path=os.path.expanduser("~/.cache/mediapipe/models/hand_landmarker.task"))
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(opts)

    # CSV
    f = open(CSV_FILE, "a", newline="")
    writer = csv.writer(f)
    if os.path.getsize(CSV_FILE) == 0:
        writer.writerow(CSV_HEADER)

    print(f"Letter: {letter} | Target: {args.count} | Rate: {1/args.rate:.0f}/sec")
    print("Show hand. Q = quit\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    last_capture = 0
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

    while count < args.count:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        now = time.time()

        h, w = frame.shape[:2]

        # Draw
        if result.hand_landmarks:
            for hi, hand in enumerate(result.hand_landmarks):
                pts = [(int(p.x*w), int(p.y*h)) for p in hand]
                color = (0,255,0) if hi == 0 else (255,0,0)
                for s, e in conns:
                    cv2.line(frame, pts[s], pts[e], color, 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0,0,255), -1)

        # Capture
        if result.hand_landmarks and (now - last_capture) > args.rate:
            # Hand 1
            hand1 = result.hand_landmarks[0]
            lm1 = [[p.x, p.y] for p in hand1]

            # Hand 2 (if exists, else zeros)
            if len(result.hand_landmarks) >= 2:
                hand2 = result.hand_landmarks[1]
                lm2 = [[p.x, p.y] for p in hand2]
            else:
                lm2 = [[0.0, 0.0]] * 21

            row = [letter, f"{letter}_{count}_{int(time.time())}", "train", len(result.hand_landmarks), "capture"]
            row += [v for p in lm1 for v in p]  # Hand 1: 42 values
            row += [v for p in lm2 for v in p]  # Hand 2: 42 values
            writer.writerow(row)
            count += 1
            last_capture = now

        # UI
        cv2.putText(frame, f"{letter}: {count}/{args.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            cv2.putText(frame, "2 HANDS!", (200,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print(f"\nDone: {count} samples saved to {CSV_FILE}")

if __name__ == "__main__":
    main()