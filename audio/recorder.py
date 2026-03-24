"""
NOVA Audio Recorder
- Records from microphone with Voice Activity Detection (VAD)
- Stops automatically when user stops talking
- Detects if user is interrupting Nova's speech
"""

import pyaudio
import wave
import numpy as np
import threading
import time
import io
from pathlib import Path
from collections import deque
from typing import Optional, Callable
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger("nova.recorder")


class AudioRecorder:
    """
    Continuous audio recorder with smart VAD.
    Handles:
      - Pre-roll buffer (captures audio just before speech starts)
      - Energy-based speech detection
      - Silence-based end-of-speech detection
      - Interruption detection (user speaking over Nova)
    """

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._recording = False
        self._monitoring = False

        # Pre-roll ring buffer: keeps last N seconds before speech
        pre_roll_frames = int(config.VAD_PRE_ROLL_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE)
        self._pre_roll = deque(maxlen=pre_roll_frames)

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[bytes], None]] = None
        self.on_interrupt: Optional[Callable] = None

        # State
        self._audio_frames = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0

        silence_frames_needed = config.VAD_SILENCE_THRESH_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE
        self._silence_threshold_frames = int(silence_frames_needed)

        min_speech_frames = config.VAD_MIN_SPEECH_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE
        self._min_speech_frames = int(min_speech_frames)

        # Noise floor adaptation
        self._noise_floor = config.VAD_ENERGY_THRESHOLD
        self._noise_samples = deque(maxlen=50)  # ~3 seconds of noise history

        # Is Nova currently speaking? (set externally by TTS)
        self.nova_speaking = False

    def _rms(self, data: bytes) -> float:
        """Calculate RMS energy of audio chunk."""
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _is_speech(self, data: bytes, context_mode: bool = False) -> bool:
        """Determine if chunk contains speech based on energy."""
        rms = self._rms(data)
        threshold = config.CONTEXT_ENERGY_GATE if context_mode else self._noise_floor
        # Adaptive threshold: 1.8x noise floor
        dynamic_threshold = self._noise_floor * 1.8
        effective_threshold = max(threshold, dynamic_threshold)
        return rms > effective_threshold

    def _update_noise_floor(self, data: bytes):
        """Adapt noise floor to ambient conditions."""
        rms = self._rms(data)
        if rms > 0:
            self._noise_samples.append(rms)
        if len(self._noise_samples) >= 10:
            # Use 30th percentile as noise floor estimate
            sorted_samples = sorted(self._noise_samples)
            percentile_idx = max(0, int(len(sorted_samples) * 0.3))
            self._noise_floor = max(config.VAD_ENERGY_THRESHOLD * 0.5,
                                    sorted_samples[percentile_idx])

    def start_monitoring(self):
        """Start continuous audio monitoring in background thread."""
        self._monitoring = True
        self._stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.CHUNK_SIZE,
        )
        log.info("Audio monitoring started")

    def stop_monitoring(self):
        """Stop audio monitoring."""
        self._monitoring = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def read_chunk(self) -> Optional[bytes]:
        """Read one chunk from stream."""
        if self._stream and self._monitoring:
            try:
                return self._stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
            except Exception as e:
                log.warning(f"Audio read error: {e}")
                return None
        return None

    def record_utterance(self, context_mode: bool = False, timeout: float = 30.0) -> Optional[bytes]:
        """
        Record a single user utterance.
        Starts when speech is detected, ends when silence threshold is reached.
        Returns raw PCM bytes, or None if nothing captured.
        """
        log.info(f"Listening for utterance (context_mode={context_mode})...")
        self._audio_frames = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0

        start_time = time.time()

        # Add pre-roll buffer to capture speech onset
        self._audio_frames.extend(self._pre_roll)

        while time.time() - start_time < timeout:
            chunk = self.read_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue

            # Update noise floor when not in speech
            if not self._in_speech:
                self._update_noise_floor(chunk)
                self._pre_roll.append(chunk)

            is_speech = self._is_speech(chunk, context_mode)

            if is_speech:
                if not self._in_speech:
                    # Speech started
                    self._in_speech = True
                    self._silence_frames = 0
                    # Include pre-roll
                    self._audio_frames = list(self._pre_roll)
                    if self.on_speech_start:
                        self.on_speech_start()
                    log.debug("Speech started")

                    # Interrupt Nova if she's speaking
                    if self.nova_speaking and self.on_interrupt:
                        log.info("User interrupted Nova")
                        self.on_interrupt()

                self._audio_frames.append(chunk)
                self._speech_frames += 1
                self._silence_frames = 0

            elif self._in_speech:
                # Silence during speech
                self._audio_frames.append(chunk)
                self._silence_frames += 1

                if self._silence_frames >= self._silence_threshold_frames:
                    # End of utterance
                    log.debug(f"Utterance ended ({self._speech_frames} speech frames)")
                    break

        if self._speech_frames < self._min_speech_frames:
            log.debug("Too short, discarding")
            return None

        return b"".join(self._audio_frames)

    def save_to_wav(self, pcm_data: bytes, path: Path) -> Path:
        """Save PCM bytes to WAV file."""
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return path

    def pcm_to_wav_bytes(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM to WAV bytes (in memory)."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def get_energy_level(self) -> float:
        """Get current mic energy (for visualisation)."""
        chunk = self.read_chunk()
        if chunk:
            return self._rms(chunk)
        return 0.0

    def cleanup(self):
        self.stop_monitoring()
        self.pa.terminate()


class WakeWordListener:
    """
    Listens continuously for the wake word.
    Primary: openwakeword (if available)
    Fallback: whisper + keyword match in transcript
    """

    def __init__(self, recorder: AudioRecorder, stt=None):
        self.recorder = recorder
        self.stt = stt  # STT engine (faster-whisper)
        self._active = False
        self._oww = None
        self._load_openwakeword()

    def _load_openwakeword(self):
        """Try to load openwakeword."""
        if not config.USE_OPENWAKEWORD:
            return
        try:
            from openwakeword.model import Model
            # Try loading custom nova model, fall back to built-ins
            try:
                self._oww = Model(wakeword_models=[config.OPENWAKEWORD_MODEL],
                                  inference_framework="onnx")
                log.info("Loaded openwakeword model: " + config.OPENWAKEWORD_MODEL)
            except Exception:
                # Use bundled models
                self._oww = Model(inference_framework="onnx")
                log.info("Loaded openwakeword bundled models")
        except ImportError:
            log.info("openwakeword not installed, using whisper fallback")
        except Exception as e:
            log.warning(f"openwakeword load failed: {e}")

    def wait_for_wake_word(self) -> bool:
        """
        Block until wake word is detected.
        Returns True when wake word heard, False if stopped.
        """
        self._active = True
        log.info(f"Waiting for wake word: {config.WAKE_WORDS}")

        if self._oww:
            return self._oww_listen()
        else:
            return self._whisper_listen()

    def _oww_listen(self) -> bool:
        """Use openwakeword for wake detection (very fast, low CPU)."""
        import numpy as np
        chunk_samples = int(config.SAMPLE_RATE * config.OPENWAKEWORD_CHUNK_MS / 1000)

        while self._active:
            chunk = self.recorder.read_chunk()
            if chunk is None:
                continue

            audio_array = np.frombuffer(chunk, dtype=np.int16)
            prediction = self._oww.predict(audio_array)

            for model_name, score in prediction.items():
                if score >= config.OPENWAKEWORD_THRESHOLD:
                    log.info(f"Wake word detected (score={score:.2f})")
                    return True

        return False

    def _whisper_listen(self) -> bool:
        """
        Fallback: record short clips, transcribe, check for wake word.
        Less latency-efficient but works without openwakeword.
        """
        from nova.audio.stt import STT
        if self.stt is None:
            self.stt = STT()

        buffer = []
        buffer_duration = 0
        clip_duration = 2.0  # check every 2 seconds

        while self._active:
            chunk = self.recorder.read_chunk()
            if chunk is None:
                continue

            buffer.append(chunk)
            buffer_duration += config.CHUNK_SIZE / config.SAMPLE_RATE

            if buffer_duration >= clip_duration:
                pcm = b"".join(buffer)
                buffer = []
                buffer_duration = 0

                # Only transcribe if there's speech energy
                rms = self.recorder._rms(pcm)
                if rms < config.VAD_ENERGY_THRESHOLD * 0.5:
                    continue

                text = self.stt.transcribe_raw(pcm).lower().strip()
                if text:
                    log.debug(f"Wake listen: '{text}'")
                    for ww in config.WAKE_WORDS:
                        if ww in text:
                            log.info(f"Wake word '{ww}' detected in: '{text}'")
                            return True

        return False

    def stop(self):
        self._active = False