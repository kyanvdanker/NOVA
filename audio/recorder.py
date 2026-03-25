"""
NOVA Audio Recorder v2
Key fixes over v1:
  1. Echo gate: VAD threshold raised while TTS plays + tail period → no more
     Nova hearing her own voice and responding to it
  2. Pre-roll buffer: captures audio before speech energy threshold hit
  3. Proper noise floor adaptation

Wake word listener now also uses the event bus.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.recorder")


class AudioRecorder:
    """
    Microphone recorder with smart VAD and echo gating.
    """

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._monitoring = False

        # Pre-roll ring buffer
        pre_roll_frames = int(config.VAD_PRE_ROLL_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE)
        self._pre_roll = deque(maxlen=pre_roll_frames)

        # ── Echo Gate ──────────────────────────────────────────────────────────
        # When Nova speaks, the microphone picks up her voice.
        # We raise the VAD threshold massively while TTS is playing and for
        # VAD_SPEAKING_GATE_TAIL seconds afterwards.
        self.nova_speaking = False          # set by TTS callbacks
        self._speaking_end_time = 0.0       # when TTS last stopped

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[bytes], None]] = None
        self.on_interrupt: Optional[Callable] = None

        # Noise floor adaptation
        self._noise_floor = config.VAD_ENERGY_THRESHOLD
        self._noise_samples: deque = deque(maxlen=60)

    def _rms(self, data: bytes) -> float:
        """Root mean square energy of audio chunk."""
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _effective_threshold(self, context_mode: bool = False) -> float:
        """
        Returns the current effective VAD threshold.
        Raised significantly while Nova is speaking or just finished.
        """
        now = time.time()
        in_tail = (now - self._speaking_end_time) < config.VAD_SPEAKING_GATE_TAIL

        if self.nova_speaking or in_tail:
            return config.VAD_SPEAKING_GATE     # high gate — blocks echo

        # Adaptive threshold: 1.8× noise floor, but at least the config minimum
        adaptive = self._noise_floor * 1.8
        base = config.CONTEXT_ENERGY_GATE if context_mode else config.VAD_ENERGY_THRESHOLD
        return max(base, adaptive)

    def _is_speech(self, data: bytes, context_mode: bool = False) -> bool:
        rms = self._rms(data)
        return rms > self._effective_threshold(context_mode)

    def _update_noise_floor(self, data: bytes):
        """Continuously adapt to ambient noise."""
        rms = self._rms(data)
        if 10 < rms < config.VAD_ENERGY_THRESHOLD * 2:  # plausible ambient range
            self._noise_samples.append(rms)
        if len(self._noise_samples) >= 15:
            sorted_s = sorted(self._noise_samples)
            self._noise_floor = max(
                config.VAD_ENERGY_THRESHOLD * 0.4,
                sorted_s[int(len(sorted_s) * 0.3)]
            )

    def notify_speaking_ended(self):
        """TTS calls this when playback finishes."""
        self.nova_speaking = False
        self._speaking_end_time = time.time()

    def start_monitoring(self):
        """Open microphone stream."""
        self._stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.CHUNK_SIZE,
        )
        self._monitoring = True
        log.info("Audio monitoring started (echo gate active)")

    def stop_monitoring(self):
        self._monitoring = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def read_chunk(self) -> Optional[bytes]:
        if self._stream and self._monitoring:
            try:
                return self._stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
            except Exception as e:
                log.warning(f"Audio read error: {e}")
                return None
        return None

    def record_utterance(self, context_mode: bool = False,
                         timeout: float = 30.0) -> Optional[bytes]:
        """
        Record one user utterance with VAD.
        Starts when speech detected, ends after silence threshold.
        Returns raw PCM or None if too short.
        """
        log.debug(f"Waiting for utterance (context={context_mode})")
        audio_frames = []
        in_speech = False
        silence_frames = 0
        speech_frames = 0

        silence_threshold_frames = int(
            config.VAD_SILENCE_THRESH_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE
        )
        min_speech_frames = int(
            config.VAD_MIN_SPEECH_SEC * config.SAMPLE_RATE / config.CHUNK_SIZE
        )

        start_time = time.time()

        while time.time() - start_time < timeout:
            chunk = self.read_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue

            if not in_speech:
                self._update_noise_floor(chunk)
                self._pre_roll.append(chunk)

            is_speech = self._is_speech(chunk, context_mode)

            if is_speech:
                if not in_speech:
                    in_speech = True
                    silence_frames = 0
                    audio_frames = list(self._pre_roll)  # include pre-roll
                    if self.on_speech_start:
                        self.on_speech_start()

                    # Interrupt Nova if she's speaking
                    if self.nova_speaking and self.on_interrupt:
                        log.info("User interrupting Nova")
                        self.on_interrupt()

                audio_frames.append(chunk)
                speech_frames += 1
                silence_frames = 0

            elif in_speech:
                audio_frames.append(chunk)
                silence_frames += 1
                if silence_frames >= silence_threshold_frames:
                    break  # end of utterance

        if speech_frames < min_speech_frames:
            return None

        return b"".join(audio_frames)

    def save_to_wav(self, pcm_data: bytes, path: Path) -> Path:
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return path

    def pcm_to_wav_bytes(self, pcm_data: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def get_energy_level(self) -> float:
        chunk = self.read_chunk()
        return self._rms(chunk) if chunk else 0.0

    def cleanup(self):
        self.stop_monitoring()
        self.pa.terminate()


import pvporcupine
from pvrecorder import PvRecorder   # optional, or keep using your recorder
import logging

log = logging.getLogger("nova.recorder")


class WakeWordListener:
    """
    Simple and reliable wake word listener using Porcupine.
    Just put in any word/phrase you want.
    """

    def __init__(self, recorder, stt=None):
        self.recorder = recorder
        self.stt = stt
        self._active = False
        self.porcupine = None
        self._load_porcupine()

    def _load_porcupine(self):
        try:
            # Option A: Use built-in keywords (no key needed for some)
            # self.porcupine = pvporcupine.create(keywords=["porcupine", "bumblebee"])

            # Option B: Use your custom wake word (RECOMMENDED)
            self.porcupine = pvporcupine.create(
                access_key="U1syWtYhRiNTSQ/c5o/2Z1DVuWGA6BqvDrhgET+3EnU/rERTvs2ZAQ==",   # ← Put your Picovoice key here
                keyword_paths=[r"wakewords/hey-nova_en_windows_v4_0_0.ppn"]   # path to your .ppn file
            )

            log.info(f" Porcupine loaded successfully. Wake word: Hey Nova")

        except Exception as e:
            log.error(f"Failed to load Porcupine: {e}")
            self.porcupine = None

    def wait_for_wake_word(self) -> bool:
        """Main method called from main.py"""
        if self.porcupine:
            return self._porcupine_listener()
        else:
            log.warning("Porcupine not loaded → using Whisper fallback")
            return self._whisper_listen()   # we'll add this next if needed

    def _porcupine_listener(self) -> bool:
        """Listen using Porcupine - handles 1024 chunk size safely."""
        log.info("Listening for wake word with Porcupine...")   # removed emoji to avoid issues

        try:
            while self._active:
                chunk = self.recorder.read_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue

                pcm = list(np.frombuffer(chunk, dtype=np.int16))

                # Porcupine requires exactly 512 samples per call
                if len(pcm) == 1024:
                    # Split into two frames
                    if self.porcupine.process(pcm[:512]) >= 0 or self.porcupine.process(pcm[512:]) >= 0:
                        log.info(" Wake word detected by Porcupine!")
                        return True
                else:
                    # Fallback for other sizes
                    if self.porcupine.process(pcm) >= 0:
                        log.info(" Wake word detected by Porcupine!")
                        return True

        finally:
            if self.porcupine:
                self.porcupine.delete()

        return False
    
    def stop(self):
        self._active = False
        if self.porcupine:
            try:
                self.porcupine.delete()
            except:
                pass
        log.info("Wake word listener stopped")