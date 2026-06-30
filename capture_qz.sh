#!/bin/bash
# Batch capture Q-Z (10 huruf × 1000 samples)
for letter in Q R S T U V W X Y Z; do
  echo "=== Collecting $letter ==="
  python3 capture_fast2.py $letter --count 1000
  echo ""
done
echo "=== Done! ==="
