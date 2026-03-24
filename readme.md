# NOVA — Neural Omnipresent Voice Assistant

```
  ███╗   ██╗ ██████╗ ██╗   ██╗  █████╗
  ████╗  ██║██╔═══██╗██║   ██║ ██╔══██╗
  ██╔██╗ ██║██║   ██║██║   ██║ ███████║
  ██║╚██╗██║██║   ██║╚██╗ ██╔╝ ██╔══██║
  ██║ ╚████║╚██████╔╝ ╚████╔╝  ██║  ██║
  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝   ╚═╝  ╚═╝
```

> *"Hey Nova"* — Your personal JARVIS. Runs locally. Learns you. Controls your world.

---

## What is NOVA?

NOVA is a fully local, deeply personal AI assistant built around **Llama 3.2 Vision** running via Ollama. It lives on your laptop today, in a portable cube tomorrow, and a self-driving robot eventually.

- 🎙️ **Wake word activated** — say "Hey Nova" anytime
- 🧠 **Learns about you** — remembers your name, preferences, schedule, interests
- 👂 **Smart listening** — stops recording when you stop talking, knows when you're done
- 🗣️ **Interruptible** — talk over Nova at any time, she'll stop and listen
- 📋 **Project management** — full project, task, and memo system
- 💻 **Laptop control** — opens apps, manages files, reads your screen
- 👁️ **Vision** — can see and describe your screen using Llama 3.2 Vision
- 🔒 **100% local** — nothing leaves your machine

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- A microphone

### 2. Install & Setup
```bash
git clone <your-repo> nova-assistant
cd nova-assistant
chmod +x setup.sh
./setup.sh
```

### 3. Pull the AI model
```bash
ollama pull llama3.2-vision
# Or for faster/lighter:
ollama pull llama3.2
```

### 4. Run NOVA
```bash
./run_nova.sh
# Or manually:
source venv/bin/activate
python3 nova/main.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        NOVA                             │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │  Audio   │   │  Brain   │   │     Skills       │   │
│  │          │   │          │   │                  │   │
│  │  Wake    │   │  Llama   │   │  Projects &      │   │
│  │  Word    │──▶│  3.2     │──▶│  Memos           │   │
│  │          │   │  Vision  │   │                  │   │
│  │  VAD     │   │          │   │  Laptop          │   │
│  │  Record  │   │  Memory  │   │  Control         │   │
│  │          │   │  (SQL +  │   │                  │   │
│  │  STT     │   │  Vector) │   │  Intent          │   │
│  │  (Whisper│   │          │   │  Router          │   │
│  │          │   │          │   │                  │   │
│  │  TTS     │   │          │   │  Vision          │   │
│  │  (Piper) │   │          │   │  (Screenshots)   │   │
│  └──────────┘   └──────────┘   └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Wake Word | openwakeword + Whisper fallback | "Hey Nova" detection |
| STT | faster-whisper (base.en) | Speech → Text |
| LLM | Llama 3.2 Vision via Ollama | Intelligence & reasoning |
| TTS | Piper TTS → espeak fallback | Text → Speech |
| Memory | SQLite + ChromaDB | Learn & remember |
| Projects | SQLite | Tasks, memos, projects |
| Laptop | pyautogui, subprocess, mss | System control |

---

## Voice Commands

### General
```
Hey Nova                    → Wake Nova up
Nova, what time is it?      → Current time/date
Nova, what can you do?      → Help
```

### Projects
```
Create project Website Redesign
Open project Website Redesign
List my projects
```

### Tasks & Memos
```
Add task Review the wireframes
Note down: Meeting with client on Friday
Show my tasks
Mark task done: Review wireframes
```

### Laptop Control
```
Open Firefox
Open VS Code
Close Slack
Find file report.pdf
Open file ~/Documents/notes.txt
Set volume to 50
What's my CPU usage?
```

### Screen / Vision
```
What's on my screen?
Describe what I'm working on
Take a screenshot
```

### Memory
```
My name is [name]
I prefer dark mode
I work as a software engineer
What do you know about me?
```

---

## Configuration

Edit `nova/config.py` to customize:

```python
# Change wake words
WAKE_WORDS = ["hey nova", "nova", "okay nova"]

# Change AI model
OLLAMA_MODEL = "llama3.2-vision"   # or "llama3.1", "llama3.2"

# Whisper model size (smaller = faster, less accurate)
WHISPER_MODEL = "base.en"   # tiny.en / base.en / small.en / medium.en

# Context mode: how long Nova stays active without wake word
CONTEXT_TIMEOUT_SEC = 45

# How long silence before recording stops
VAD_SILENCE_THRESH_SEC = 1.8
```

---

## Raspberry Pi Deployment

### Recommended Pi: Raspberry Pi 4 (4GB+) or Pi 5

```bash
# On the Pi
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev espeak

# Install Ollama for ARM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2   # Use non-vision model for speed on Pi

# Run setup
./setup.sh
```

### Pi Performance Tips

- Use `WHISPER_MODEL = "tiny.en"` in config.py for faster STT
- Use `OLLAMA_MODEL = "llama3.2"` instead of vision model
- Reduce `CONTEXT_WINDOW = 10` to save RAM
- Use USB microphone for better quality
- Add a USB speaker or 3.5mm speaker

### Pi as Always-On Server
```bash
# Install as systemd service
sudo cp nova.service /etc/systemd/system/
sudo systemctl enable nova
sudo systemctl start nova
```

---

## The Cube (Phase 2)

The cube form factor will add:
- Raspberry Pi 5 + SSD
- USB microphone array (better far-field pickup)
- Mini speaker
- LED ring for visual status (sleeping/listening/thinking/speaking)
- Optional: small display for text output
- Battery pack for portability

Status LED colors:
- 🔵 Blue = Sleeping (wake word mode)
- 🟡 Yellow = Listening
- 🔴 Red = Thinking
- 🟢 Green = Speaking

---

## The Robot (Phase 3)

Future expansion:
- Mobile base (wheels + motors)
- Follow-me mode using camera
- Physical interaction
- Room-to-room awareness
- Sensor array (temperature, light, proximity)

---

## Privacy

NOVA is **100% local**:
- No cloud APIs
- No data sent anywhere
- Ollama runs locally
- Whisper runs locally
- All memory stored in local SQLite/ChromaDB

The only internet traffic is during setup (downloading models).

---

## Troubleshooting

**"Cannot connect to Ollama"**
```bash
ollama serve   # Start Ollama
ollama pull llama3.2-vision   # Pull the model
```

**"Wake word not detected"**
- Check microphone is working: `arecord -l`
- Lower `VAD_ENERGY_THRESHOLD` in config.py
- Switch to Whisper fallback: `USE_OPENWAKEWORD = False`

**"Speech not transcribed"**
- Check microphone level
- Try `WHISPER_MODEL = "small.en"` for better accuracy
- Run `python3 -c "import faster_whisper; print('OK')"`

**"TTS not working"**
- Install espeak: `sudo apt-get install espeak`
- Or on Mac: TTS falls back to `say` command automatically

**Slow responses on Pi**
- Use `llama3.2` instead of `llama3.2-vision`
- Set `WHISPER_MODEL = "tiny.en"`
- Consider Pi 5 or Pi 4 8GB

---

## Extending NOVA

### Add a new skill
1. Create `nova/skills/my_skill.py`
2. Add intent patterns to `nova/skills/intent_router.py`
3. Add handler method to `IntentRouter`

### Train a custom wake word
```bash
# Using openWakeWord
pip install openwakeword
python3 -c "
from openwakeword import train
# Record 'Hey Nova' samples and train
"
```

### Add to NOVA's personality
Edit `NOVA_PERSONALITY` in `nova/config.py`.

---

## License

MIT — do whatever you want with it.

---

*Built for the future. Runs today.*#   N O V A  
 