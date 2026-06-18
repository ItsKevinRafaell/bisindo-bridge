"""
BISINDO Text-to-Speech Engine
==============================
Converts recognized text to Indonesian speech audio.
"""

import os
import io
import tempfile
import time
import hashlib


class TTSEngine:
    """Text-to-Speech engine using gTTS (Google Text-to-Speech)."""

    def __init__(self, language='id', cache_dir=None):
        """
        Initialize TTS engine.

        Args:
            language: Language code (default: 'id' for Indonesian)
            cache_dir: Directory to cache audio files (None = temp directory)
        """
        self.language = language
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache = {}  # In-memory cache: text -> audio bytes

    def _get_cache_key(self, text):
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def text_to_audio(self, text):
        """
        Convert text to audio.

        Args:
            text: Text to convert to speech

        Returns:
            audio_bytes: MP3 audio data as bytes, or None if failed
        """
        if not text or not text.strip():
            return None

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.language, slow=False)

            # Save to bytes
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()

            # Cache it
            self.cache[cache_key] = audio_bytes

            return audio_bytes

        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return None

    def save_audio(self, text, filepath):
        """
        Save text as audio file.

        Args:
            text: Text to convert
            filepath: Output file path (.mp3)

        Returns:
            success: True if saved successfully
        """
        audio_bytes = self.text_to_audio(text)
        if audio_bytes:
            try:
                with open(filepath, 'wb') as f:
                    f.write(audio_bytes)
                return True
            except Exception as e:
                print(f"❌ Save audio error: {e}")
                return False
        return False

    def clear_cache(self):
        """Clear audio cache."""
        self.cache.clear()
