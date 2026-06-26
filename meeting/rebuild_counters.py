#!/usr/bin/env python3
"""
Rebuild counters.json from CSV actual data.
Run this after any data issues or manual CSV edits.

Usage: python rebuild_counters.py
"""
import csv
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset')
CSV_PATH = os.path.join(DATA_DIR, 'landmarks_captured_v2.csv')
COUNTERS_PATH = os.path.join(DATA_DIR, 'counters.json')

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def rebuild():
    counters = defaultdict(int)
    total = 0

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            letter = row.get('letter', '').upper()
            if letter in LETTERS:
                counters[letter] += 1
                total += 1

    # Fill missing letters with 0
    for letter in LETTERS:
        counters.setdefault(letter, 0)

    result = dict(counters)

    with open(COUNTERS_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Counters rebuilt: {total} samples")
    print(f"📁 Saved to: {COUNTERS_PATH}")
    for letter in LETTERS:
        print(f"  {letter}: {result[letter]}")

if __name__ == '__main__':
    rebuild()
