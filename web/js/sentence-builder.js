/**
 * SentenceBuilder - Assembles letter predictions into words and sentences.
 * JavaScript mirror of the Python SentenceBuilder.
 * Used by web/test.html and meeting/static/js/meeting.js.
 */
class SentenceBuilder {
    constructor(options = {}) {
        // Timing thresholds (milliseconds)
        this.stabilityThreshold = options.stabilityMs || 500;
        this.spaceNoHandMs = options.spaceNoHandMs || 2000;
        this.enterNoHandMs = options.enterNoHandMs || 4000;

        // State
        this.currentLetter = null;
        this.letterStartTime = 0;
        this.letterStable = false;
        this.currentWord = '';
        this.words = [];
        this.sentences = [];
        this.lastHandTime = Date.now();
        this.lastCommittedLetter = '';

        // Callbacks
        this.onLetterCommit = options.onLetterCommit || (() => {});
        this.onWordComplete = options.onWordComplete || (() => {});
        this.onSentenceComplete = options.onSentenceComplete || (() => {});
        this.onGesture = options.onGesture || (() => {});
    }

    /**
     * Called every frame with prediction result.
     * @param {Object|null} prediction - {letter, confidence} or null
     * @param {boolean} handDetected - whether a hand is detected
     * @param {string|null} gestureState - "fist" | "palm" | "signing" | null
     */
    feed(prediction, handDetected, gestureState) {
        const now = Date.now();

        if (!handDetected) {
            // Timer-based space/enter
            const noHandDuration = now - this.lastHandTime;
            if (noHandDuration >= this.enterNoHandMs && this.currentWord) {
                this.completeSentence();
            } else if (noHandDuration >= this.spaceNoHandMs && this.currentWord) {
                this.insertSpace();
            }
            return;
        }

        this.lastHandTime = now;

        // Gesture overrides
        if (gestureState === 'fist') {
            this.insertSpace();
            this.onGesture('space');
            return;
        }
        if (gestureState === 'palm') {
            this.completeSentence();
            this.onGesture('enter');
            return;
        }

        // Letter stability check
        if (prediction && prediction.confidence > 0.5) {
            const letter = prediction.letter;
            if (letter === this.currentLetter) {
                // Same letter - check stability
                if (!this.letterStable && (now - this.letterStartTime >= this.stabilityThreshold)) {
                    this.letterStable = true;
                    this.commitLetter(letter);
                }
            } else {
                // New letter - restart timer
                this.currentLetter = letter;
                this.letterStartTime = now;
                this.letterStable = false;
            }
        }
    }

    /**
     * Commit a letter to the current word (with deduplication).
     * @param {string} letter
     */
    commitLetter(letter) {
        if (letter === this.lastCommittedLetter) return; // dedup
        this.currentWord += letter;
        this.lastCommittedLetter = letter;
        this.onLetterCommit(letter, this.currentWord);
    }

    /** Insert a space, finalizing the current word. */
    insertSpace() {
        if (!this.currentWord) return;
        this.words.push(this.currentWord);
        this.onWordComplete(this.currentWord, this.words);
        this.currentWord = '';
        this.lastCommittedLetter = '';
        this.currentLetter = null;
    }

    /** Complete the current sentence from accumulated words. */
    completeSentence() {
        if (this.currentWord) this.insertSpace();
        if (this.words.length === 0) return;
        const sentence = this.words.join(' ');
        this.sentences.push(sentence);
        this.onSentenceComplete(sentence, this.sentences);
        this.words = [];
    }

    // Manual triggers (for UI buttons)

    manualSpace() {
        this.insertSpace();
    }

    manualEnter() {
        this.completeSentence();
    }

    clearAll() {
        this.currentWord = '';
        this.words = [];
        this.lastCommittedLetter = '';
        this.currentLetter = null;
    }

    /** @returns {Object} snapshot of current builder state */
    getState() {
        return {
            currentWord: this.currentWord,
            words: [...this.words],
            sentences: [...this.sentences],
            buildingSentence: [...this.words, this.currentWord].filter(Boolean).join(' '),
        };
    }
}

// Export for use in different contexts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SentenceBuilder };
} else {
    window.SentenceBuilder = SentenceBuilder;
}
