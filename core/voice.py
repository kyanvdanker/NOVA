"""
Voice Module — Wake word (Porcupine), STT (faster-whisper), TTS (edge-tts / pyttsx3)
Designed for low latency. Graceful fallbacks at every level.
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
    ASSISTANT_NAME, BASE_DIR
)


def _resolve_wake_word_path(wake_word: str) -> Optional[Path]:
    """Resolve wake word path — tries multiple locations."""
    if not wake_word.lower().endswith(".ppn"):
        return None  # Built-in keyword, no file needed

    candidates = [
        Path(wake_word),
        BASE_DIR / wake_word,
        BASE_DIR / "wakewords" / Path(wake_word).name,
        Path.home() / ".nova" / "wakewords" / Path(wake_word).name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


class VoiceEngine:
    def __init__(self):
        self.porcupine = None
        self.whisper_model = None
        self.pa = None
        self._speaking = False
        self._running = False
        self._wake_callback: Optional[Callable] = None
        self._wake_word_available = False

    def initialize(self):
        self._init_stt()
        self._init_tts()
        self._init_wake_word()
        self._init_audio()

    def _init_stt(self):
        print("  🎤 Loading Whisper STT...")
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
            print(f"  ✅ Whisper '{WHISPER_MODEL}' loaded (faster-whisper)")
        except ImportError:
            try:
                import whisper
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                print(f"  ✅ Whisper '{WHISPER_MODEL}' loaded (openai-whisper)")
            except ImportError:
                print("  ⚠️  No Whisper found. Voice input disabled. Install: pip install faster-whisper")
                self.whisper_model = None

    def _init_tts(self):
        print("  🔊 Initialising TTS engine...")
        if TTS_ENGINE == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                voices = engine.getProperty("voices")
                for v in voices:
                    if any(x in v.name.lower() for x in ("male", "guy", "david", "mark")):
                        engine.setProperty("voice", v.id)
                        break
                engine.stop()
                print("  ✅ pyttsx3 TTS ready")
            except Exception as e:
                print(f"  ⚠️  pyttsx3 failed: {e}")
        else:
            # edge-tts — checked lazily at speak time
            print("  ✅ edge-tts TTS ready (lazy init)")

    def _init_wake_word(self):
        print("  👂 Loading wake word engine...")

        # Check if key is configured
        if not PORCUPINE_ACCESS_KEY:
            print("  ℹ️  No Porcupine key set (PORCUPINE_ACCESS_KEY). Wake word disabled.")
            print("      Get a free key at: https://console.picovoice.ai/")
            return

        try:
            import pvporcupine

            if WAKE_WORD.lower().endswith(".ppn"):
                # Custom wake word file
                resolved = _resolve_wake_word_path(WAKE_WORD)
                if resolved is None:
                    print(f"  ⚠️  Wake word file not found: '{WAKE_WORD}'")
                    print(f"      Searched: {BASE_DIR / WAKE_WORD}")
                    print(f"      Falling back to built-in 'jarvis' keyword.")
                    try:
                        self.porcupine = pvporcupine.create(
                            access_key=PORCUPINE_ACCESS_KEY,
                            keywords=["jarvis"]
                        )
                        print("  ✅ Wake word 'jarvis' active (fallback)")
                        self._wake_word_available = True
                    except Exception as fe:
                        print(f"  ⚠️  Fallback wake word also failed: {fe}")
                else:
                    self.porcupine = pvporcupine.create(
                        access_key=PORCUPINE_ACCESS_KEY,
                        keyword_paths=[str(resolved)]
                    )
                    print(f"  ✅ Custom wake word loaded: {resolved.name}")
                    self._wake_word_available = True
            else:
                # Built-in keyword
                keyword = WAKE_WORD.lower().strip()
                # Validate it's a real built-in keyword
                valid_keywords = [
                    "alexa", "americano", "blueberry", "bumblebee", "computer",
                    "grapefruit", "grasshopper", "hey google", "hey siri", "jarvis",
                    "ok google", "picovoice", "porcupine", "terminator"
                ]
                if keyword not in valid_keywords:
                    print(f"  ⚠️  '{keyword}' is not a built-in keyword.")
                    print(f"      Valid keywords: {valid_keywords}")
                    keyword = "jarvis"
                    print(f"      Using 'jarvis' instead.")
                self.porcupine = pvporcupine.create(
                    access_key=PORCUPINE_ACCESS_KEY,
                    keywords=[keyword]
                )
                print(f"  ✅ Wake word '{keyword}' active")
                self._wake_word_available = True

        except ImportError:
            print("  ⚠️  pvporcupine not installed. Wake word disabled.")
            print("      Install: pip install pvporcupine")
        except Exception as e:
            print(f"  ⚠️  Porcupine init failed: {e}")
            print("      Wake word disabled — text mode still works.")

    def _init_audio(self):
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            print("  ✅ PyAudio ready")
        except ImportError:
            print("  ⚠️  PyAudio not installed. Microphone disabled.")
            print("      Install: pip install pyaudio")
        except Exception as e:
            print(f"  ⚠️  PyAudio init failed: {e}")

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

            communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return

            # Try pygame first
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                sound = pygame.mixer.Sound(io.BytesIO(audio_data))
                sound.play()
                await asyncio.sleep(sound.get_length() + 0.1)
                return
            except ImportError:
                pass

            # Fallback: save to file and play via system
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
                f.write(audio_data)
            try:
                import platform
                sys_name = platform.system()
                if sys_name == "Windows":
                    os.system(f'start /wait "" "{tmp_path}"')
                elif sys_name == "Darwin":
                    os.system(f"afplay '{tmp_path}'")
                else:
                    os.system(f"mpg123 -q '{tmp_path}' 2>/dev/null || "
                              f"ffplay -nodisp -autoexit '{tmp_path}' 2>/dev/null || "
                              f"play '{tmp_path}' 2>/dev/null")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except ImportError:
            print("  ⚠️  edge-tts not installed. Run: pip install edge-tts")
            self._speak_pyttsx3(text)
        except Exception as e:
            print(f"  TTS error: {e}")
            try:
                self._speak_pyttsx3(text)
            except:
                pass

    def _speak_pyttsx3(self, text: str):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"  pyttsx3 error: {e}")

    def transcribe(self, audio_data: bytes) -> str:
        if not self.whisper_model:
            return ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_data)

            try:
                segments, _ = self.whisper_model.transcribe(
                    tmp_path, language=WHISPER_LANGUAGE, vad_filter=True
                )
                text = " ".join(s.text for s in segments).strip()
            except AttributeError:
                result = self.whisper_model.transcribe(tmp_path, language=WHISPER_LANGUAGE)
                text = result["text"].strip()

            try:
                os.unlink(tmp_path)
            except:
                pass
            return text
        except Exception as e:
            print(f"  Transcription error: {e}")
            return ""

    def record_until_silence(self) -> Optional[bytes]:
        if not self.pa:
            return None

        try:
            import pyaudio
        except ImportError:
            return None

        chunk_size = 1024
        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=chunk_size
            )
        except Exception as e:
            print(f"  Audio stream error: {e}")
            return None

        frames = []
        silence_frames = 0
        silence_limit = int(SILENCE_DURATION * AUDIO_SAMPLE_RATE / chunk_size)
        max_frames = int(MAX_RECORD_SECONDS * AUDIO_SAMPLE_RATE / chunk_size)
        recording_started = False

        try:
            for _ in range(max_frames):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
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

        return b"".join(frames) if recording_started else None

    def start_wake_word_listener(self, callback: Callable):
        self._wake_callback = callback
        self._running = True

        if not self._wake_word_available or not self.porcupine or not self.pa:
            print("  ℹ️  Wake word listener not available — use text or press Enter.")
            return

        thread = threading.Thread(target=self._wake_word_loop, daemon=True)
        thread.start()
        print(f"  👂 Listening for wake word...")

    def _wake_word_loop(self):
        try:
            import pyaudio
        except ImportError:
            return

        stream = None
        try:
            stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            print("  🟢 Wake word listener active")
            while self._running:
                if self._speaking:
                    time.sleep(0.05)
                    continue

                try:
                    pcm = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.1)
                    continue

                if not pcm:
                    continue

                try:
                    pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    keyword_index = self.porcupine.process(pcm_unpacked)
                    if keyword_index >= 0:
                        print(f"\n  🟢 Wake word detected!")
                        if self._wake_callback:
                            self._wake_callback()
                except Exception as e:
                    # Don't crash the loop on a single bad frame
                    pass

        except Exception as e:
            print(f"  Wake word loop error: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def stop(self):
        self._running = False
        if self.porcupine:
            try:
                self.porcupine.delete()
            except:
                pass
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass