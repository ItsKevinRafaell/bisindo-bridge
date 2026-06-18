#!/usr/bin/env python3
"""
Batch Capture - Kumpulin data semua huruf A-Z
Usage: python capture_all.py
"""

import subprocess
import time

def capture_letter(letter, count=50):
    """Capture samples untuk satu huruf"""
    print(f"\n{'='*50}")
    print(f"CAPTURING LETTER: {letter}")
    print(f"{'='*50}")
    print(f"1. Jalankan terminal baru")
    print(f"2. Ketik: python src/auto_capture.py --letter {letter} --count {count}")
    print(f"3. Tahan posisi tangan ~0.5 detik per capture")
    print(f"4. Tekan 'q' kalau sudah {count} samples")
    print(f"5. Tekan ENTER di terminal ini kalau sudah selesai")
    input(f"\n>>> Tekan ENTER kalau sudah selesai capture {letter}...")

def main():
    print("="*50)
    print("BATCH CAPTURE - Semua huruf BISINDO A-Z")
    print("="*50)
    print("""
Cara pakai:
1. Jalankan script ini
2. Untuk setiap huruf:
   - Terminal baru akan terbuka
   - Jalankan: python src/auto_capture.py --letter X --count 50
   - Capture 50 samples
   - Tekan 'q' untuk simpan
3. Ulangi untuk semua huruf
""")

    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    print(f"\nHuruf yang perlu di-capture: {', '.join(letters)}")
    print(f"Target: 50 samples per huruf = 1300 total")

    response = input("\nMau mulai capture semua huruf? (y/n): ")
    if response.lower() != 'y':
        print("Dibatalkan.")
        return

    for letter in letters:
        capture_letter(letter, count=50)

    print("\n" + "="*50)
    print("SEMUA CAPTURE SELESAI!")
    print("="*50)
    print("\nNext: Re-train model dengan data baru:")
    print("  python train.py")


if __name__ == '__main__':
    main()
