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
        this.gestureHoldMs = options.gestureHoldMs || 2000;

        // State
        this.currentLetter = null;
        this.letterStartTime = 0;
        this.letterStable = false;
        this.currentWord = '';
        this.words = [];
        this.sentences = [];
        this.lastHandTime = Date.now();
        this.lastCommittedLetter = '';

        // Gesture tracking
        this.currentGesture = null;
        this.gestureStartTime = 0;
        this.gestureTriggered = false;

        // Callbacks
        this.onLetterCommit = options.onLetterCommit || (() => {});
        this.onWordComplete = options.onWordComplete || (() => {});
        this.onSentenceComplete = options.onSentenceComplete || (() => {});
        this.onGesture = options.onGesture || (() => {});
        this.onBackspace = options.onBackspace || (() => {});
        this.onDeleteWord = options.onDeleteWord || (() => {});
    }

    /**
     * Called every frame with prediction result.
     * @param {Object|null} prediction - {letter, confidence} or null
     * @param {boolean} handDetected - whether a hand is detected
     * @param {string|null} gestureState - "fist" | "palm" | "signing" | null
     */
    feed(prediction, handDetected, gestureState) {
        const now = Date.now();

        console.log(`[SentenceBuilder] feed: hand=${handDetected}, gesture=${gestureState}, currentGesture=${this.currentGesture}, word="${this.currentWord}"`);

        if (!handDetected) {
            // Timer-based space/enter
            const noHandDuration = now - this.lastHandTime;
            if (noHandDuration >= this.enterNoHandMs && this.currentWord) {
                this.completeSentence();
            } else if (noHandDuration >= this.spaceNoHandMs && this.currentWord) {
                this.insertSpace();
            }
            // Reset gesture tracking when no hand
            this.currentGesture = null;
            this.gestureTriggered = false;
            return;
        }

        this.lastHandTime = now;

        // Track gesture hold time
        if (gestureState === 'fist' || gestureState === 'palm') {
            if (gestureState !== this.currentGesture) {
                // New gesture started
                console.log(`[SentenceBuilder] New gesture started: ${gestureState}`);
                this.currentGesture = gestureState;
                this.gestureStartTime = now;
                this.gestureTriggered = false;
            } else {
                // Same gesture continuing
                const gestureDuration = now - this.gestureStartTime;
                console.log(`[SentenceBuilder] Gesture hold: ${gestureState} for ${gestureDuration}ms (need ${this.gestureHoldMs}ms), triggered=${this.gestureTriggered}`);

                // Trigger action after hold time
                if (gestureDuration >= this.gestureHoldMs && !this.gestureTriggered) {
                    this.gestureTriggered = true;
                    console.log(`[SentenceBuilder] Gesture triggered! Duration: ${gestureDuration}ms`);

                    if (gestureState === 'fist') {
                        // Fist hold = backspace (delete last letter)
                        console.log(`[SentenceBuilder] Calling backspace(), currentWord="${this.currentWord}"`);
                        this.backspace();
                        this.onGesture('backspace');
                        this.onBackspace(this.currentWord);
                        console.log(`[SentenceBuilder] After backspace: currentWord="${this.currentWord}"`);
                    } else if (gestureState === 'palm') {
                        // Palm hold = delete word
                        console.log(`[SentenceBuilder] Calling deleteWord(), currentWord="${this.currentWord}"`);
                        this.deleteWord();
                        this.onGesture('delete_word');
                        this.onDeleteWord(this.currentWord);
                        console.log(`[SentenceBuilder] After deleteWord: currentWord="${this.currentWord}"`);
                    }
                }
            }
            return;
        }

        // Reset gesture tracking if not fist/palm
        if (gestureState !== this.currentGesture) {
            this.currentGesture = null;
            this.gestureTriggered = false;
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

    /** Backspace - delete the last letter from current word. */
    backspace() {
        if (this.currentWord.length > 0) {
            this.currentWord = this.currentWord.slice(0, -1);
            this.lastCommittedLetter = this.currentWord.length > 0
                ? this.currentWord[this.currentWord.length - 1]
                : '';
            this.onBackspace(this.currentWord);
        }
    }

    /** Delete the entire current word. */
    deleteWord() {
        if (this.currentWord.length > 0) {
            const deletedWord = this.currentWord;
            this.currentWord = '';
            this.lastCommittedLetter = '';
            this.currentLetter = null;
            this.onDeleteWord(deletedWord);
        }
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
