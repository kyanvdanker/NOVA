"""
NOVA Text-to-Speech
Primary: piper-tts (fast neural TTS, works offline on Pi)
Fallback: espeak, then system TTS
"""

import subprocess
import threading
import tempfile
import time
import os
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.tts")


class TTS:
    """
    Text-to-speech engine with:
    - Interruption support (stop mid-sentence)
    - Sentence chunking (start speaking sooner)
    - Multiple backend support
    """

    def __init__(self, on_speaking_start=None, on_speaking_end=None):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._interrupt_event = threading.Event()
        self._speaking = False

        self.on_speaking_start = on_speaking_start
        self.on_speaking_end = on_speaking_end

        self._engine = self._detect_engine()
        log.info(f"TTS engine: {self._engine}")

    def _detect_engine(self) -> str:
        """Auto-detect best available TTS engine."""
        if config.TTS_ENGINE == "piper":
            if self._check_piper():
                return "piper"
        if self._check_espeak():
            return "espeak"
        return "system"

    def _check_piper(self) -> bool:
        try:
            result = subprocess.run(["piper", "--help"],
                                    capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try python piper
            try:
                import piper
                return True
            except ImportError:
                pass
        return False

    def _check_espeak(self) -> bool:
        try:
            subprocess.run(["espeak", "--version"], capture_output=True, timeout=3)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str, blocking: bool = True):
        """
        Speak text. If blocking=True, wait until done.
        Interruptible by calling interrupt().
        """
        if not text or not text.strip():
            return

        # Clean text for speaking
        text = self._clean_text(text)

        self._interrupt_event.clear()
        self._speaking = True
        if self.on_speaking_start:
            self.on_speaking_start()

        if blocking:
            self._speak_blocking(text)
        else:
            t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
            t.start()

    def speak_async(self, text: str):
        """Non-blocking speak."""
        self.speak(text, blocking=False)

    def _speak_blocking(self, text: str):
        """Actually speak, handling chunked output for low latency."""
        try:
            sentences = self._split_sentences(text)
            for sentence in sentences:
                if self._interrupt_event.is_set():
                    log.info("TTS interrupted")
                    break
                if sentence.strip():
                    self._speak_sentence(sentence.strip())
        finally:
            self._speaking = False
            if self.on_speaking_end:
                self.on_speaking_end()

    def _speak_sentence(self, text: str):
        """Speak a single sentence."""
        if self._engine == "piper":
            self._speak_piper(text)
        elif self._engine == "espeak":
            self._speak_espeak(text)
        else:
            self._speak_system(text)

    def _speak_piper(self, text: str):
        """Use piper-tts."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            # Generate audio
            proc = subprocess.Popen(
                [
                    "piper",
                    "--model", config.PIPER_MODEL,
                    "--output_file", wav_path,
                    "--length_scale", str(1.0 / config.PIPER_SPEED),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.communicate(input=text.encode(), timeout=30)

            # Play audio
            if not self._interrupt_event.is_set():
                self._play_wav(wav_path)
        except Exception as e:
            log.warning(f"Piper TTS failed: {e}, falling back to espeak")
            self._speak_espeak(text)
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    def _speak_espeak(self, text: str):
        """Use espeak as fallback."""
        try:
            cmd = [
                "espeak",
                "-v", "en-us",
                "-s", "160",   # speed (words/min)
                "-p", "45",    # pitch
                "-a", "100",   # amplitude
                text,
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            while self._proc.poll() is None:
                if self._interrupt_event.is_set():
                    self._proc.terminate()
                    return
                time.sleep(0.05)
        except Exception as e:
            log.warning(f"espeak failed: {e}")
            self._speak_system(text)

    def _speak_system(self, text: str):
        """Last resort: OS-level TTS."""
        import platform
        system = platform.system()
        try:
            if system == "Darwin":
                self._proc = subprocess.Popen(["say", text])
            elif system == "Linux":
                self._proc = subprocess.Popen(["spd-say", text])
            elif system == "Windows":
                self._proc = subprocess.Popen(
                    ["powershell", "-c",
                     f'Add-Type -AssemblyName System.Speech; '
                     f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")']
                )
            if self._proc:
                while self._proc.poll() is None:
                    if self._interrupt_event.is_set():
                        self._proc.terminate()
                        return
                    time.sleep(0.05)
        except Exception as e:
            log.error(f"System TTS failed: {e}")

    def _play_wav(self, wav_path: str):
        """Play a WAV file, stoppable."""
        try:
            import sounddevice as sd
            import soundfile as sf
            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            while sd.get_stream().active:
                if self._interrupt_event.is_set():
                    sd.stop()
                    return
                time.sleep(0.05)
            sd.wait()
        except ImportError:
            # Fallback: aplay / afplay / paplay
            import platform
            system = platform.system()
            if system == "Linux":
                cmd = ["aplay", wav_path]
            elif system == "Darwin":
                cmd = ["afplay", wav_path]
            else:
                cmd = ["powershell", "-c", f'(New-Object Media.SoundPlayer "{wav_path}").PlaySync()']

            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            while self._proc.poll() is None:
                if self._interrupt_event.is_set():
                    self._proc.terminate()
                    return
                time.sleep(0.05)

    def interrupt(self):
        """Immediately stop speaking."""
        self._interrupt_event.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        log.debug("TTS interrupted")

    def _clean_text(self, text: str) -> str:
        """Remove markdown and clean for speech."""
        import re
        # Remove markdown formatting
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)       # bold/italic
        text = re.sub(r'#{1,6}\s+', '', text)               # headers
        text = re.sub(r'`+([^`]+)`+', r'\1', text)         # code
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links
        text = re.sub(r'^[-*•]\s+', '', text, flags=re.MULTILINE)  # bullets
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
        text = re.sub(r'\n+', ' ', text)                    # newlines
        text = re.sub(r'\s+', ' ', text)                    # extra spaces
        return text.strip()

    def _split_sentences(self, text: str):
        """
        Split text into sentences for streaming playback.
        This reduces latency — first sentence plays while rest is processed.
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Merge very short sentences
        merged = []
        current = ""
        for s in sentences:
            if len(current) + len(s) < 80:
                current += " " + s if current else s
            else:
                if current:
                    merged.append(current)
                current = s
        if current:
            merged.append(current)
        return merged if merged else [text]