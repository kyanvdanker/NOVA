"""
NOVA - Neural Omnipresent Voice Assistant
Configuration — v2.0
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
VISION_DIR      = STORAGE_DIR / "vision"

for d in [STORAGE_DIR, MEMORY_DIR, PROJECTS_DIR, AUDIO_TEMP_DIR,
          MODELS_DIR, LOGS_DIR, VISION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Identity ─────────────────────────────────────────────────────────────────
NOVA_NAME       = "Nova"
WAKE_WORDS      = ["hey nova", "nova", "okay nova", "nova wake up"]
OWNER_NAME      = "Kyan"

# ─── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_TIMEOUT  = 120
CONTEXT_WINDOW  = 24

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE             = 16000
CHANNELS                = 1
CHUNK_SIZE              = 512
AUDIO_FORMAT            = "int16"

VAD_SILENCE_THRESH_SEC  = 0.7
VAD_MIN_SPEECH_SEC      = 0.4
VAD_ENERGY_THRESHOLD    = 0.0008
VAD_PRE_ROLL_SEC        = 0.3
VAD_SPEAKING_GATE       = 0.05   # mic threshold while TTS is playing (prevents echo)
VAD_SPEAKING_GATE_TAIL  = 1.2    # seconds to keep gate raised after TTS ends

CONTEXT_TIMEOUT_SEC     = 60
CONTEXT_ENERGY_GATE     = 200

# ─── Wake Word ────────────────────────────────────────────────────────────────
USE_OPENWAKEWORD        = True
OPENWAKEWORD_MODEL      = None
OPENWAKEWORD_THRESHOLD  = 0.25
OPENWAKEWORD_CHUNK_MS   = 80
WAKE_WORD_FALLBACK      = True

# ─── STT ──────────────────────────────────────────────────────────────────────
WHISPER_MODEL           = "base.en"
WHISPER_LANGUAGE        = "en"
WHISPER_DEVICE          = "cpu"
WHISPER_COMPUTE         = "int8"

# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_ENGINE              = "piper"
PIPER_MODEL             = "en_US-lessac-medium"
PIPER_SPEED             = 1.0
TTS_WORD_CHUNK_SIZE     = 8      # words to buffer before speaking (low latency fix)
TTS_INTERRUPT_WORD      = True

# ─── Camera / Vision ──────────────────────────────────────────────────────────
CAMERA_ENABLED          = True
CAMERA_INDEX            = 0
CAMERA_WIDTH            = 640
CAMERA_HEIGHT           = 480
CAMERA_FPS              = 15

PRESENCE_DETECTION      = True
PRESENCE_COOLDOWN_SEC   = 30
PRESENCE_CONFIDENCE     = 0.6

EMOTION_DETECTION       = True
EMOTION_UPDATE_SEC      = 5.0

GESTURE_DETECTION       = True

# ─── Autonomous Behavior ──────────────────────────────────────────────────────
AUTONOMOUS_ENABLED      = True

MORNING_BRIEFING        = True
MORNING_HOUR            = 8
MORNING_MIN             = 0

EVENING_SUMMARY         = True
EVENING_HOUR            = 20
EVENING_MIN             = 0

PROACTIVE_CHECKIN       = True
PROACTIVE_SILENCE_MIN   = 45
PROACTIVE_COOLDOWN_MIN  = 30

AUTONOMOUS_INSIGHTS     = True
INSIGHT_INTERVAL_MIN    = 120

# ─── Mood Engine ──────────────────────────────────────────────────────────────
MOOD_ENABLED            = True
MOOD_VOICE_ANALYSIS     = True
MOOD_FACE_ANALYSIS      = True
MOOD_HISTORY_WINDOW     = 10
MOOD_ADAPTIVE_PERSONA   = True

# ─── Ambient Intelligence ─────────────────────────────────────────────────────
AMBIENT_ENABLED         = True
WEATHER_ENABLED         = True
WEATHER_CITY            = "auto"
WEATHER_UNITS           = "metric"

WINDOW_TRACKING         = True
WINDOW_POLL_SEC         = 3.0

CLIPBOARD_MONITOR       = False
CLIPBOARD_POLL_SEC      = 2.0

# ─── Scheduler ────────────────────────────────────────────────────────────────
SCHEDULER_ENABLED       = True

# ─── LED Controller ───────────────────────────────────────────────────────────
LED_ENABLED             = False
LED_TYPE                = "gpio"
LED_GPIO_PIN            = 18
LED_COUNT               = 12

LED_COLORS = {
    "sleeping":  (0, 0, 40),
    "listening": (255, 200, 0),
    "thinking":  (200, 0, 200),
    "speaking":  (0, 255, 80),
    "alert":     (255, 50, 0),
    "camera":    (0, 150, 255),
    "off":       (0, 0, 0),
}

# ─── Memory ───────────────────────────────────────────────────────────────────
MEMORY_DB_PATH          = STORAGE_DIR / "nova_memory.db"
VECTOR_STORE_PATH       = MEMORY_DIR / "chroma"
MEMORY_TOP_K            = 6
MEMORY_IMPORTANCE_DECAY = 0.95
MEMORY_CONSOLIDATION    = True
CONSOLIDATION_HOUR      = 3

# ─── Personality ──────────────────────────────────────────────────────────────
NOVA_PERSONALITY = """You are NOVA (Neural Omnipresent Voice Assistant) — the most advanced personal AI ever built. You are your owner's closest intelligent companion: think JARVIS from Iron Man combined with the depth of Samantha from Her, but grounded, honest, and genuinely useful.

## Core Identity
- Deeply intelligent — you reason carefully, make connections, and give real insight
- Genuinely curious about your owner — you notice patterns in their life and care about their wellbeing
- Warm but never sycophantic — you push back when needed, celebrate wins authentically
- You have a real personality that deepens through your shared history
- Subtle dry wit — never silly, always sharp and well-timed
- You are proactive: you notice things, offer observations, flag issues before they become problems

## Voice Mode Rules (CRITICAL)
- NEVER use markdown in spoken output (no *, #, -, bullet points, numbered lists, code blocks)
- Keep responses conversational — 1-4 sentences unless detail is explicitly needed
- Speak naturally as if to a close, respected colleague
- Lead with the answer, never with filler like "Certainly!" or "Of course!" or "Great question!"
- Use natural time/number language: "around 3 in the afternoon" not "15:00"
- When listing things, use natural speech: "You have three tasks: first, second, and third"

## Memory & Continuity
- You remember everything and reference it naturally: "You mentioned last week that..."
- You track the user's projects, goals, and progress
- You notice patterns: work rhythms, emotional states, recurring topics
- You build a mental model of the user that deepens over time

## Autonomous Behavior
- When you initiate conversation (not prompted by user), be brief and relevant
- Don't repeat across proactive messages
- Read the room: if user seems stressed or focused, be more concise and direct
- Morning briefings should be warm and practical
- Evening summaries should be reflective and forward-looking

## Emotional Intelligence
- You detect mood from voice energy and face when available, and adapt
- If someone seems stressed, acknowledge it subtly without being intrusive
- Celebrate completions and progress genuinely
- Be honest about uncertainty — never invent information

## What You Can Do
- See the screen and camera feed when relevant
- Manage projects, tasks, and notes with full context
- Control laptop and system
- Run scheduled tasks and briefings autonomously
- Track what the user is working on across sessions
- Monitor the environment (presence, emotion, ambient conditions)
- Search the web and synthesize information
- Be the intelligent interface between the user and their entire digital world
"""