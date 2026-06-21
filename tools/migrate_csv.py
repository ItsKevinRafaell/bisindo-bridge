#!/usr/bin/env python3
"""
Migrate BISINDO landmark CSVs to unified 67-column schema.

Source files:
  - dataset/landmarks_captured.csv  (130 cols: 1+2 hands mixed, NO contributor)
  - dataset/landmarks_augmented.csv (66 cols: 1 hand only, all 26 letters)

Target schema (67 cols, 2-hand only):
  letter, image_path, split, num_hands, contributor, lm0_x..lm20_z (63 cols)

Rules:
  - Drop all rows where num_hands != 2 (1-hand rows are noise for 2-hand model).
  - From captured.csv: drop the 63 trailing _h2_* columns; insert "contributor"="legacy".
  - From augmented.csv: pad 63 zeros for hand 2, num_hands=2, contributor="kaggle".
  - Output: dataset/landmarks_captured_v2.csv
"""

import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "dataset")
SRC_CAPTURED = os.path.join(DATASET, "landmarks_captured.csv")
SRC_AUGMENTED = os.path.join(DATASET, "landmarks_augmented.csv")
OUT = os.path.join(DATASET, "landmarks_captured_v2.csv")
BACKUP = os.path.join(DATASET, "landmarks_captured_v2_backup.csv")

# Target header (67 cols)
HEADER = ["letter", "image_path", "split", "num_hands", "contributor"]
HEADER += [f"lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]


def normalize_row(letter, image_path, split, num_hands, contributor, hand1, hand2):
    if num_hands != 2:
        return None
    if len(hand1) != 63 or len(hand2) != 63:
        return None
    row = {
        "letter": letter,
        "image_path": image_path,
        "split": split,
        "num_hands": 2,
        "contributor": contributor,
    }
    for i in range(21):
        row[f"lm{i}_x"] = hand1[i * 3]
        row[f"lm{i}_y"] = hand1[i * 3 + 1]
        row[f"lm{i}_z"] = hand1[i * 3 + 2]
    return row


def migrate_captured(path, contributor_tag):
    """Read captured.csv (130-col, 1+2 hand mix) → yield normalized rows."""
    count = 0
    skipped = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expect: letter, image_path, split, num_hands, lm0_x..lm20_z (63), lm0_h2_x..lm20_h2_z (63)
        if len(header) != 130:
            print(f"  WARNING: {path} has {len(header)} cols, expected 130")
        for row in reader:
            if len(row) < 130:
                skipped += 1
                continue
            try:
                letter = row[0]
                image_path = row[1]
                split = row[2]
                num_hands = int(row[3])
            except (ValueError, IndexError):
                skipped += 1
                continue
            hand1 = row[4:67]
            hand2 = row[67:130]
            try:
                hand1 = [float(v) for v in hand1]
                hand2 = [float(v) for v in hand2]
            except ValueError:
                skipped += 1
                continue
            norm = normalize_row(letter, image_path, split, num_hands, contributor_tag, hand1, hand2)
            if norm is None:
                skipped += 1
                continue
            count += 1
            yield norm
    print(f"  captured: kept {count}, skipped {skipped}")


def migrate_augmented(path, contributor_tag):
    """Read augmented.csv (66-col, 1 hand only) → yield 2-hand rows with zero-padded hand 2."""
    count = 0
    skipped = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 66:
            print(f"  WARNING: {path} has {len(header)} cols, expected 66")
        for idx, row in enumerate(reader):
            if len(row) < 66:
                skipped += 1
                continue
            try:
                letter = row[0]
                image_path = row[1]
                split = row[2]
                hand1 = [float(v) for v in row[3:66]]
            except (ValueError, IndexError):
                skipped += 1
                continue
            hand2 = [0.0] * 63
            norm = normalize_row(letter, image_path, split, 2, contributor_tag, hand1, hand2)
            if norm is None:
                skipped += 1
                continue
            count += 1
            yield norm
    print(f"  augmented: kept {count}, skipped {skipped}")


def main():
    if not os.path.exists(SRC_CAPTURED):
        print(f"ERROR: {SRC_CAPTURED} missing")
        sys.exit(1)

    if os.path.exists(OUT):
        os.replace(OUT, BACKUP)
        print(f"  backed up previous {OUT} → {BACKUP}")

    counts = {}
    contributors = {}
    written = 0

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()

        print("Migrating landmarks_captured.csv (2-hand rows only)…")
        for row in migrate_captured(SRC_CAPTURED, "legacy"):
            writer.writerow(row)
            counts[row["letter"]] = counts.get(row["letter"], 0) + 1
            contributors[row["contributor"]] = contributors.get(row["contributor"], 0) + 1
            written += 1

        if os.path.exists(SRC_AUGMENTED):
            print("Migrating landmarks_augmented.csv (pad hand 2 with zeros)…")
            for row in migrate_augmented(SRC_AUGMENTED, "kaggle"):
                writer.writerow(row)
                counts[row["letter"]] = counts.get(row["letter"], 0) + 1
                contributors[row["contributor"]] = contributors.get(row["contributor"], 0) + 1
                written += 1

    print(f"\nWrote {written} rows → {OUT}")
    print(f"Unique letters: {len(counts)}")
    for letter in sorted(counts):
        print(f"  {letter}: {counts[letter]}")
    print(f"\nContributors:")
    for c, n in sorted(contributors.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
