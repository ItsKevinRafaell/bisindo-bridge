# CNN Lessons Learned - BISINDO Bridge

## Experimental Results

### 1. Augmentation Impact

| Approach | Test Accuracy | Webcam Performance |
|----------|--------------|-------------------|
| Original data only | ~98% | Best |
| Aggressive augmentation (6-8x) | ~100% | Worse |
| Simple augmentation (2x) | ~95% | Worse than original |

**Conclusion:** Augmentation degraded real-world performance.

---

## Why Augmentation Failed

### Problem 1: Synthetic vs Real
- Original data: real samples from actual users
- Augmented: mathematically generated variations

### Problem 2: Model Confusion
- Augmentation creates unrealistic hand positions
- Model learns synthetic patterns
- Webcam test uses real patterns
- → Mismatch → worse performance

### Problem 3: MediaPipe Limitation
- MediaPipe detects 1 hand when 2 hands overlap
- "2-hand" data in CSV is actually 1-hand patterns
- Augmenting 1-hand patterns doesn't help 2-hand recognition

---

## MediaPipe 2-Hand Detection

### When MediaPipe Detects 2 Hands
- Both hands visible in frame
- Hands separated (not overlapping)
- Good lighting
- Clear background

### When MediaPipe Detects 1 Hand
- Hands overlapping/covering each other
- One hand outside frame
- Poor lighting
- Fast movement

### Implication
- BISINDO 2-hand letters require specific hand positions
- MediaPipe may not detect both hands in natural signing
- This is a MediaPipe limitation, not a data problem

---

## What Worked

### Best Approach
1. Use original clean data
2. Train without aggressive augmentation
3. Simple augmentation (2x rotate ±10°) may help slightly
4. Focus on real-world testing, not just validation accuracy

### Model Selection
- CNN 1D with simple architecture works well
- 98%+ validation accuracy achievable
- Real webcam accuracy depends on:
  - Hand position
  - Lighting
  - Camera angle
  - MediaPipe detection quality

---

## Recommendations for Future Work

### Data Collection
1. Collect more diverse hand sizes/positions
2. Ensure both hands visible for 2-hand letters
3. Multiple contributors for variety

### Model Improvement
1. Keep augmentation simple (rotate ±10°, scale ±5%)
2. Don't over-augment
3. Test on real webcam, not just validation

### Alternative Approaches
1. Use 2 cameras for full hand tracking
2. Custom hand detection model (future work)
3. Accept MediaPipe limitations

---

## Key Takeaways

1. **Original data > augmented data** for real-world use
2. **Validation accuracy ≠ webcam accuracy**
3. **MediaPipe has fundamental limits** for 2-hand detection
4. **Simple is better** than complex augmentation
5. **Test on target platform** (webcam) not just metrics

---

## Timeline

- June 2026: Initial experiments with augmentation
- Finding: Augmentation degraded performance
- Decision: Use original data for final model