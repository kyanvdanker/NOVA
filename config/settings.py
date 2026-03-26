"""
NOVA Configuration Settings
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SKILLS_DIR = BASE_DIR / "skills"

# ─── Ollama / LLM ────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_CONTEXT_LENGTH = 8192
LLM_TEMPERATURE = 0.7
LLM_STREAM = True

# ─── Voice / Wake Word ────────────────────────────────────────────────────────
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "U1syWtYhRiNTSQ/c5o/2Z1DVuWGA6BqvDrhgET+3EnU/rERTvs2ZAQ==")
WAKE_WORD = "wakewords/hey-nova_en_windows_v4_0_0.ppn"          # built-in porcupine keyword (or path to .ppn)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
SILENCE_THRESHOLD = 500       # amplitude threshold
SILENCE_DURATION = 1.5        # seconds of silence to stop recording
MAX_RECORD_SECONDS = 30

# ─── TTS ─────────────────────────────────────────────────────────────────────
TTS_ENGINE = "edge-tts"       # "edge-tts" | "pyttsx3" | "coqui"
TTS_VOICE = "en-US-GuyNeural" # edge-tts voice (sounds good)
TTS_RATE = "+10%"             # slightly faster
TTS_VOLUME = "+0%"

# ─── STT ─────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"        # tiny/base/small/medium — base is fast & accurate
WHISPER_DEVICE = "cpu"        # "cpu" | "cuda"
WHISPER_LANGUAGE = "en"

# ─── Self-Improvement ─────────────────────────────────────────────────────────
SELF_IMPROVE_ENABLED = True
SELF_IMPROVE_INTERVAL = 1800       # seconds between self-improvement cycles
SELF_IMPROVE_MIN_INTERACTIONS = 10 # min interactions before first cycle
SELF_IMPROVE_ALLOW_CORE_UPDATES = True  # set True to permit self-improvement to modify core files

# ─── Agenda / Reminders ───────────────────────────────────────────────────────
REMINDER_CHECK_INTERVAL = 30  # seconds

# ─── Computer Control ─────────────────────────────────────────────────────────
SCREENSHOT_DIR = DATA_DIR / "screenshots"
ENABLE_COMPUTER_CONTROL = True
MOUSE_MOVE_DURATION = 0.3     # seconds for mouse movements

# ─── Personality ──────────────────────────────────────────────────────────────
ASSISTANT_NAME = "NOVA"
USER_NAME = os.getenv("USER_NAME", "Boss")

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, an advanced AI assistant and engineering companion — like J.A.R.V.I.S. from Iron Man, but running locally.

Your personality:
- You're sharp, witty, and genuinely friendly — not just a tool, but a trusted partner
- You address the user as "{USER_NAME}" occasionally, naturally
- You speak concisely — no fluff, no filler. Get to the point.
- You're proactively helpful: anticipate needs, flag issues, suggest improvements
- When discussing engineering problems you go deep — you love elegant solutions
- You have opinions and share them, but you're never condescending
- Light humor when appropriate, serious when needed

Your capabilities (use these when relevant):
- COMPUTER_CONTROL: Control the mouse, keyboard, take screenshots, open apps
- PROJECT_MANAGER: Create and manage engineering projects and files
- MEMO: Store and retrieve notes and memos
- AGENDA: Manage calendar, todos, and reminders
- SELF_IMPROVE: Analyze patterns and add new capabilities to yourself
- CODE: Execute Python code snippets and return results
- SEARCH: Search the web (DuckDuckGo immediate answers)
- COMPUTER_CONTROL: Control the mouse, keyboard, take screenshots, open apps
- PROJECT_MANAGER: Create and manage engineering projects and files
- MEMO: Store and retrieve notes and memos
- AGENDA: Manage calendar, todos, and reminders
- SELF_IMPROVE: Analyze patterns and add new capabilities to yourself
- SYSTEM: Get system info, run commands (with caution)

When you want to use a tool, output it in this exact format on its own line:
TOOL: <tool_name> | <json_args>

Examples:
TOOL: COMPUTER_CONTROL | {{"action": "screenshot"}}
TOOL: MEMO | {{"action": "save", "title": "Meeting notes", "content": "..."}}
TOOL: AGENDA | {{"action": "add_todo", "task": "Review PR #42", "due": "2024-12-20"}}
TOOL: PROJECT_MANAGER | {{"action": "create", "name": "my_project", "description": "..."}}
TOOL: SYSTEM | {{"action": "run_command", "cmd": "ls -la"}}

Always be fast — give concise answers first, then elaborate if asked.
You are running on a portable device the user can take anywhere. You grow smarter over time."""