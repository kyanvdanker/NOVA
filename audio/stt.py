"""
NOVA Speech-to-Text
Uses faster-whisper for fast, accurate transcription.
Works on laptop and Raspberry Pi.
"""

import numpy as np
import io
import wave
import time
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


log = logging.getLogger("nova.stt")


class STT:
    """
    Speech-to-Text using faster-whisper.
    Lazy loads model to save startup time.
    """

    def __init__(self):
        self._model = None
        self._model_name = config.WHISPER_MODEL

    def _ensure_model(self):
        """Lazy load Whisper model."""
        if self._model is not None:
            return

        log.info(f"Loading Whisper model '{self._model_name}'...")
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_name,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )
            log.info("Whisper model loaded")
        except ImportError:
            log.warning("faster-whisper not installed, trying openai-whisper...")
            self._load_openai_whisper()

    def _load_openai_whisper(self):
        """Fallback to openai-whisper."""
        try:
            import whisper
            self._model = whisper.load_model(self._model_name)
            self._use_openai_api = True
            log.info("openai-whisper loaded")
        except ImportError:
            raise RuntimeError(
                "Neither faster-whisper nor openai-whisper is installed.\n"
                "Run: pip install faster-whisper"
            )

    def transcribe_raw(self, pcm_data: bytes) -> str:
        """
        Transcribe raw PCM audio bytes.
        Returns transcribed text string.
        """
        self._ensure_model()

        # Convert PCM to float32 numpy array
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.time()
        text = self._transcribe_array(audio)
        elapsed = time.time() - t0

        # Clean up transcription
        text = text.strip()

        log.info(f"STT: '{text}' ({elapsed:.2f}s)")
        return text

    def transcribe_file(self, wav_path: Path) -> str:
        """Transcribe a WAV file."""
        self._ensure_model()

        # Load WAV into numpy
        with wave.open(str(wav_path), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return self._transcribe_array(audio).strip()

    def _transcribe_array(self, audio: np.ndarray) -> str:
        """Internal: run transcription on float32 array."""
        if hasattr(self, '_use_openai_api') and self._use_openai_api:
            # openai-whisper API
            result = self._model.transcribe(audio, language=config.WHISPER_LANGUAGE)
            return result.get("text", "")
        else:
            # faster-whisper API
            segments, info = self._model.transcribe(
                audio,
                language=config.WHISPER_LANGUAGE,
                vad_filter=True,  # additional VAD pass
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                },
                beam_size=5,
                best_of=5,
                temperature=0.0,
                word_timestamps=False,
                condition_on_previous_text=True,
            )
            return " ".join(seg.text for seg in segments)

    def is_directed_at_nova(self, text: str) -> bool:
        """
        Quick heuristic: is this utterance likely directed at Nova?
        Used in ambient/context mode without wake word.
        """
        text_lower = text.lower().strip()

        # Direct name mentions
        for ww in config.WAKE_WORDS:
            if ww in text_lower:
                return True

        # Question patterns that suggest AI interaction
        question_starters = [
            "what", "who", "when", "where", "why", "how",
            "can you", "could you", "please", "tell me",
            "show me", "open", "close", "find", "search",
            "remind", "create", "make", "set", "turn",
            "play", "stop", "pause", "volume",
        ]
        for starter in question_starters:
            if text_lower.startswith(starter):
                return True

        # Too short = probably ambient noise / filler words
        word_count = len(text_lower.split())
        if word_count <= 2:
            return False

        # Anything with a verb + object pattern probably works
        return word_count >= 4