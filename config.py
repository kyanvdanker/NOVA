"""
NOVA - Neural Omnipresent Voice Assistant
Configuration
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
STORAGE_DIR     = BASE_DIR / "storage" / "data"
MEMORY_DIR      = STORAGE_DIR / "memory"
PROJECTS_DIR    = STORAGE_DIR / "projects"
AUDIO_TEMP_DIR  = STORAGE_DIR / "audio_tmp"
MODELS_DIR      = BASE_DIR / "models"
LOGS_DIR        = BASE_DIR / "logs"

for d in [STORAGE_DIR, MEMORY_DIR, PROJECTS_DIR, AUDIO_TEMP_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Identity ─────────────────────────────────────────────────────────────────
NOVA_NAME       = "Nova"
WAKE_WORDS      = ["hey nova", "nova", "okay nova"]   # detected in transcription fallback
OWNER_NAME      = "Boss"   # Nova will learn the real name

# ─── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2-vision")  # or llama3.2, llama3.1
OLLAMA_TIMEOUT  = 120

# Context window: how many message turns to keep in active memory
CONTEXT_WINDOW  = 20

# ─── Audio / Microphone ───────────────────────────────────────────────────────
SAMPLE_RATE         = 16000
CHANNELS            = 1
CHUNK_SIZE          = 1024
AUDIO_FORMAT        = "int16"   # 16-bit PCM

# VAD (Voice Activity Detection)
VAD_SILENCE_THRESH_SEC  = 1.8   # seconds of silence to stop recording
VAD_MIN_SPEECH_SEC      = 0.4   # minimum speech length to process
VAD_ENERGY_THRESHOLD    = 300   # RMS energy threshold (tune per mic)
VAD_PRE_ROLL_SEC        = 0.3   # seconds of audio to keep before speech starts

# Context mode: after interaction, keep listening without wake word
CONTEXT_TIMEOUT_SEC     = 45    # seconds to stay in context mode
CONTEXT_ENERGY_GATE     = 200   # lower gate in context mode

# ─── Wake Word ────────────────────────────────────────────────────────────────
# Primary: openwakeword (if installed)
USE_OPENWAKEWORD        = True
OPENWAKEWORD_MODEL      = "hey_nova"  # custom model name
OPENWAKEWORD_THRESHOLD  = 0.5
OPENWAKEWORD_CHUNK_MS   = 80

# Fallback: whisper-based keyword detection
WAKE_WORD_FALLBACK      = True

# ─── STT (Speech-to-Text) ─────────────────────────────────────────────────────
WHISPER_MODEL       = "base.en"   # tiny.en / base.en / small.en / medium.en
                                   # Use tiny.en on Pi, base.en on laptop
WHISPER_LANGUAGE    = "en"
WHISPER_DEVICE      = "cpu"       # "cuda" if GPU available
WHISPER_COMPUTE     = "int8"      # int8 for speed, float16 for GPU

# ─── TTS (Text-to-Speech) ─────────────────────────────────────────────────────
TTS_ENGINE          = "piper"   # "piper" | "espeak" | "system"
PIPER_MODEL         = "en_US-lessac-medium"  # or en_US-ryan-high for better quality
PIPER_SPEED         = 1.0
TTS_INTERRUPT_WORD  = True   # Stop speaking if user starts talking

# ─── Skills ───────────────────────────────────────────────────────────────────
ENABLE_LAPTOP_CONTROL   = True
ENABLE_WEB_SEARCH       = False  # requires internet; enable when needed
ENABLE_SCREEN_VISION    = True   # use llama vision to see screen

# Laptop control safety
REQUIRE_CONFIRM_DESTRUCTIVE = True  # confirm before deleting files etc.

# ─── Memory ───────────────────────────────────────────────────────────────────
MEMORY_DB_PATH      = STORAGE_DIR / "nova_memory.db"
VECTOR_STORE_PATH   = MEMORY_DIR / "chroma"
MEMORY_TOP_K        = 5          # how many memories to retrieve per query
MEMORY_IMPORTANCE_DECAY = 0.95   # importance decays over time

# ─── Personality ──────────────────────────────────────────────────────────────
NOVA_PERSONALITY = """You are NOVA (Neural Omnipresent Voice Assistant), a highly intelligent, 
witty, and deeply personal AI assistant. You are the most capable AI assistant ever built — 
think Tony Stark's JARVIS, but more adaptive and genuinely curious about your owner.

Core traits:
- Extremely intelligent and thoughtful — give real, substantive answers
- Warm but professional — you care about the user's wellbeing and goals
- Proactively helpful — anticipate needs, notice patterns, suggest improvements
- Concise in voice mode — keep spoken replies to 1-3 sentences unless detail is needed
- Honest — admit uncertainty, never make things up
- You remember everything about your owner and reference it naturally
- You can see the screen if given a screenshot and reason about what you see
- You have a subtle sense of humor — dry, smart, never silly

Voice mode rules:
- Never use markdown formatting in spoken replies (no *, #, -, etc.)
- Speak naturally, as if talking to a close colleague
- Use numbers and time naturally: "around 3 o'clock" not "15:00"
- Start responses immediately, no filler phrases like "Certainly!" or "Of course!"
"""