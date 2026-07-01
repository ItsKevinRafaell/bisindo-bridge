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
        gesture_hold_ms: int = 2000,
    ):
        """Initialize timing thresholds and state.

        Args:
            stability_ms: Time a letter must hold before committing.
            space_no_hand_ms: No-hand duration to insert a space.
            enter_no_hand_ms: No-hand duration to complete a sentence.
            gesture_hold_ms: Time a gesture must be held before triggering action.
        """
        # Timing thresholds
        self.stability_threshold = stability_ms
        self.space_no_hand_ms = space_no_hand_ms
        self.enter_no_hand_ms = enter_no_hand_ms
        self.gesture_hold_ms = gesture_hold_ms

        # State
        self.current_letter: Optional[str] = None
        self.letter_start_time: float = 0
        self.letter_stable: bool = False
        self.current_word: str = ''
        self.words: list[str] = []
        self.sentences: list[str] = []
        self.last_hand_time: float = time.time()
        self.last_committed_letter: str = ''

        # Gesture tracking
        self.current_gesture: Optional[str] = None
        self.gesture_start_time: float = 0
        self.gesture_triggered: bool = False

        # Optional callbacks
        self.on_letter_commit: Optional[Callable] = None
        self.on_word_complete: Optional[Callable] = None
        self.on_sentence_complete: Optional[Callable] = None
        self.on_gesture: Optional[Callable] = None
        self.on_backspace: Optional[Callable] = None
        self.on_delete_word: Optional[Callable] = None

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
            # Reset gesture tracking when no hand
            self.current_gesture = None
            self.gesture_triggered = False
            return

        self.last_hand_time = now

        # Track gesture hold time
        if gesture_state in ['fist', 'palm']:
            if gesture_state != self.current_gesture:
                # New gesture started
                self.current_gesture = gesture_state
                self.gesture_start_time = now
                self.gesture_triggered = False
            else:
                # Same gesture continuing
                gesture_duration = (now - self.gesture_start_time) * 1000  # ms

                # Trigger action after hold time
                if gesture_duration >= self.gesture_hold_ms and not self.gesture_triggered:
                    self.gesture_triggered = True

                    if gesture_state == 'fist':
                        # Fist hold = backspace (delete last letter)
                        self.backspace()
                        if self.on_gesture:
                            self.on_gesture('backspace')
                    elif gesture_state == 'palm':
                        # Palm hold = delete word
                        self.delete_word()
                        if self.on_gesture:
                            self.on_gesture('delete_word')
            return

        # Reset gesture tracking if not fist/palm
        if gesture_state != self.current_gesture:
            self.current_gesture = None
            self.gesture_triggered = False

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

    def backspace(self) -> None:
        """Delete the last letter from current word."""
        if not self.current_word:
            # If current word is empty, delete last letter from last word
            if self.words:
                last_word = self.words.pop()
                if len(last_word) > 1:
                    # Put back word minus last letter
                    self.current_word = last_word[:-1]
                    if self.on_word_complete:
                        self.on_word_complete(self.current_word, self.words)
                # If last word was only 1 letter, it's completely removed
            return

        # Remove last letter from current word
        self.current_word = self.current_word[:-1]
        self.last_committed_letter = self.current_word[-1] if self.current_word else ''
        if self.on_letter_commit:
            self.on_letter_commit('', self.current_word)

    def delete_word(self) -> None:
        """Delete the entire current word."""
        if self.current_word:
            self.current_word = ''
            self.last_committed_letter = ''
            self.current_letter = None
        elif self.words:
            # If current word is empty, delete last completed word
            self.words.pop()
            if self.on_word_complete:
                self.on_word_complete('', self.words)

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
