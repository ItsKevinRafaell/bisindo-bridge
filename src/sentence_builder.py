"""SentenceBuilder - Assembles individual letter predictions into words and sentences."""

import time
from typing import Callable, Optional


class SentenceBuilder:
    """Converts a stream of letter predictions into words and sentences."""

    def __init__(
        self,
        stability_ms: int = 500,
        space_no_hand_ms: int = 2000,
        enter_no_hand_ms: int = 4000,
    ):
        """Initialize timing thresholds and state.

        Args:
            stability_ms: Time a letter must hold before committing.
            space_no_hand_ms: No-hand duration to insert a space.
            enter_no_hand_ms: No-hand duration to complete a sentence.
        """
        # Timing thresholds
        self.stability_threshold = stability_ms
        self.space_no_hand_ms = space_no_hand_ms
        self.enter_no_hand_ms = enter_no_hand_ms

        # State
        self.current_letter: Optional[str] = None
        self.letter_start_time: float = 0
        self.letter_stable: bool = False
        self.current_word: str = ''
        self.words: list[str] = []
        self.sentences: list[str] = []
        self.last_hand_time: float = time.time()
        self.last_committed_letter: str = ''

        # Optional callbacks
        self.on_letter_commit: Optional[Callable] = None
        self.on_word_complete: Optional[Callable] = None
        self.on_sentence_complete: Optional[Callable] = None
        self.on_gesture: Optional[Callable] = None

    def feed(
        self,
        prediction: Optional[dict],
        hand_detected: bool,
        gesture_state: Optional[str] = None,
    ) -> None:
        """Called every frame with a prediction result.

        Args:
            prediction: Dict with 'letter' and 'confidence', or None.
            hand_detected: Whether a hand is detected in the frame.
            gesture_state: One of "fist", "palm", "signing", "none", or None.
        """
        now = time.time()

        if not hand_detected:
            no_hand_duration = (now - self.last_hand_time) * 1000  # ms
            if no_hand_duration >= self.enter_no_hand_ms and self.current_word:
                self.complete_sentence()
            elif no_hand_duration >= self.space_no_hand_ms and self.current_word:
                self.insert_space()
            return

        self.last_hand_time = now

        # Gesture overrides
        if gesture_state == 'fist':
            self.insert_space()
            if self.on_gesture:
                self.on_gesture('space')
            return
        if gesture_state == 'palm':
            self.complete_sentence()
            if self.on_gesture:
                self.on_gesture('enter')
            return

        # Letter stability check
        if prediction and prediction.get('confidence', 0) > 0.5:
            letter = prediction['letter']
            if letter == self.current_letter:
                if not self.letter_stable:
                    elapsed = (now - self.letter_start_time) * 1000
                    if elapsed >= self.stability_threshold:
                        self.letter_stable = True
                        self._commit_letter(letter)
            else:
                self.current_letter = letter
                self.letter_start_time = now
                self.letter_stable = False

    def _commit_letter(self, letter: str) -> None:
        if letter == self.last_committed_letter:
            return  # dedup
        self.current_word += letter
        self.last_committed_letter = letter
        if self.on_letter_commit:
            self.on_letter_commit(letter, self.current_word)

    def insert_space(self) -> None:
        """Finish the current word and start a new one."""
        if not self.current_word:
            return
        self.words.append(self.current_word)
        if self.on_word_complete:
            self.on_word_complete(self.current_word, self.words)
        self.current_word = ''
        self.last_committed_letter = ''
        self.current_letter = None

    def complete_sentence(self) -> None:
        """Finish the current sentence from accumulated words."""
        if self.current_word:
            self.insert_space()
        if not self.words:
            return
        sentence = ' '.join(self.words)
        self.sentences.append(sentence)
        if self.on_sentence_complete:
            self.on_sentence_complete(sentence, self.sentences)
        self.words = []

    def manual_space(self) -> None:
        """Manual space trigger (for UI buttons)."""
        self.insert_space()

    def manual_enter(self) -> None:
        """Manual enter trigger (for UI buttons)."""
        self.complete_sentence()

    def clear_all(self) -> None:
        """Reset all state."""
        self.current_word = ''
        self.words = []
        self.last_committed_letter = ''
        self.current_letter = None

    def get_state(self) -> dict:
        """Return current builder state as a dict."""
        return {
            'current_word': self.current_word,
            'words': list(self.words),
            'sentences': list(self.sentences),
            'building_sentence': (
                ' '.join([*self.words, self.current_word])
                if self.current_word
                else ' '.join(self.words)
            ),
        }
