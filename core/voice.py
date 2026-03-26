"""
Voice Module — Wake word (Porcupine), STT (faster-whisper), TTS (edge-tts / pyttsx3)
Designed for low latency on embedded hardware (Raspberry Pi / x86).
"""
import asyncio
import io
import os
import queue
import struct
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config.settings import (
    PORCUPINE_ACCESS_KEY, WAKE_WORD,
    AUDIO_SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RECORD_SECONDS,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_LANGUAGE,
    TTS_ENGINE, TTS_VOICE, TTS_RATE, TTS_VOLUME,
    ASSISTANT_NAME
)


class VoiceEngine:
    def __init__(self):
        self.porcupine = None
        self.whisper_model = None
        self.audio_stream = None
        self.pa = None
        self._speaking = False
        self._running = False
        self._wake_callback: Optional[Callable] = None
        self._tts_queue = asyncio.Queue()
        self._audio_lock = threading.Lock()

    def initialize(self):
        """Load all voice models."""
        print("  🎤 Loading Whisper STT...")
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type="int8"  # fast on CPU
            )
            print(f"  ✅ Whisper '{WHISPER_MODEL}' loaded")
        except ImportError:
            print("  ⚠️  faster-whisper not installed. Trying openai-whisper...")
            import whisper
            self.whisper_model = whisper.load_model(WHISPER_MODEL)

        print("  🔊 Loading TTS engine...")
        if TTS_ENGINE == "pyttsx3":
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 175)
            voices = self._tts_engine.getProperty("voices")
            for v in voices:
                if "male" in v.name.lower() or "guy" in v.name.lower():
                    self._tts_engine.setProperty("voice", v.id)
                    break
        print("  ✅ TTS ready")

        from pathlib import Path

        print("  👂 Loading Porcupine wake word...")
        try:
            import pvporcupine
            if PORCUPINE_ACCESS_KEY == "YOUR_KEY_HERE":
                print("  ⚠️  No Porcupine key set — wake word disabled. Use text mode.")
                self.porcupine = None
            else:
                if WAKE_WORD.lower().endswith(".ppn"):
                    candidate = Path(WAKE_WORD)
                    if not candidate.is_absolute():
                        candidate = Path(__file__).parent.parent / candidate
                    if not candidate.exists():
                        raise FileNotFoundError(f"Wake word file not found: {candidate}")
                    print(f"  ℹ️  Using custom PPn file: {candidate}")
                    self.porcupine = pvporcupine.create(
                        access_key=PORCUPINE_ACCESS_KEY,
                        keyword_paths=[str(candidate)]
                    )
                else:
                    print(f"  ℹ️  Using built-in keyword: {WAKE_WORD}")
                    self.porcupine = pvporcupine.create(
                        access_key=PORCUPINE_ACCESS_KEY,
                        keywords=[WAKE_WORD]
                    )
                print(f"  ✅ Wake word '{WAKE_WORD}' active")
        except Exception as e:
            print(f"  ⚠️  Porcupine failed ({e}). Wake word disabled.")
            self.porcupine = None

        # PyAudio
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            print("  ✅ Audio system ready")
        except Exception as e:
            print(f"  ⚠️  PyAudio failed: {e}")

    async def speak(self, text: str):
        """Convert text to speech and play it."""
        self._speaking = True
        try:
            if TTS_ENGINE == "edge-tts":
                await self._speak_edge(text)
            elif TTS_ENGINE == "pyttsx3":
                await asyncio.get_event_loop().run_in_executor(None, self._speak_pyttsx3, text)
            else:
                await self._speak_edge(text)
        finally:
            self._speaking = False

    async def _speak_edge(self, text: str):
        try:
            import edge_tts
            import pygame

            communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return

            # Play via pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            sound = pygame.mixer.Sound(io.BytesIO(audio_data))
            sound.play()
            # Wait for playback
            duration = sound.get_length()
            await asyncio.sleep(duration + 0.1)

        except ImportError:
            # Fallback: save to file and play
            try:
                import edge_tts
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_path = f.name
                communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
                await communicate.save(tmp_path)
                os.system(f"mpg123 -q {tmp_path} 2>/dev/null || play {tmp_path} 2>/dev/null || aplay {tmp_path} 2>/dev/null")
                os.unlink(tmp_path)
            except Exception as e:
                print(f"TTS error: {e}")
                self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()

    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text using Whisper."""
        if not self.whisper_model:
            return ""
        try:
            # Write to temp wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_data)

            # Transcribe
            try:
                # faster-whisper API
                segments, _ = self.whisper_model.transcribe(
                    tmp_path,
                    language=WHISPER_LANGUAGE,
                    vad_filter=True
                )
                text = " ".join(s.text for s in segments).strip()
            except AttributeError:
                # openai-whisper API
                result = self.whisper_model.transcribe(tmp_path, language=WHISPER_LANGUAGE)
                text = result["text"].strip()

            os.unlink(tmp_path)
            return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def record_until_silence(self) -> Optional[bytes]:
        """Record audio until silence is detected. Returns raw PCM bytes."""
        if not self.pa:
            return None

        import pyaudio
        chunk_size = 1024
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=chunk_size
        )

        frames = []
        silence_frames = 0
        silence_limit = int(SILENCE_DURATION * AUDIO_SAMPLE_RATE / chunk_size)
        max_frames = int(MAX_RECORD_SECONDS * AUDIO_SAMPLE_RATE / chunk_size)
        recording_started = False

        try:
            for _ in range(max_frames):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)

                # Check amplitude
                audio_array = np.frombuffer(data, dtype=np.int16)
                amplitude = np.abs(audio_array).mean()

                if amplitude > SILENCE_THRESHOLD:
                    recording_started = True
                    silence_frames = 0
                elif recording_started:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break
        finally:
            stream.stop_stream()
            stream.close()

        if not recording_started:
            return None

        return b"".join(frames)

    def start_wake_word_listener(self, callback: Callable):
        """Start background thread listening for wake word."""
        self._wake_callback = callback
        self._running = True

        if not self.porcupine or not self.pa:
            print("  ℹ️  Wake word listener disabled — use text input mode")
            return

        thread = threading.Thread(target=self._wake_word_loop, daemon=True)
        thread.start()
        print(f"  👂 Listening for '{WAKE_WORD}'...")

    def _wake_word_loop(self):
        import pyaudio
        try:
            stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            while self._running:
                if self._speaking:
                    time.sleep(0.1)
                    continue
                pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                if not pcm:
                    continue
                pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                try:
                    keyword_index = self.porcupine.process(pcm_unpacked)
                except Exception as e:
                    print(f"Wake word processing error: {e}")
                    continue

                if keyword_index >= 0:
                    print(f"\n  🟢 Wake word detected! index={keyword_index}")
                    if self._wake_callback:
                        self._wake_callback()
        except Exception as e:
            print(f"Wake word loop error: {e}")
        finally:
            try:
                stream.close()
            except:
                pass

    def stop(self):
        self._running = False
        if self.porcupine:
            self.porcupine.delete()
        if self.pa:
            self.pa.terminate()