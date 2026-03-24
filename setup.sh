#!/bin/bash
# NOVA Setup Script
# Installs all dependencies for laptop (Linux/macOS) and Raspberry Pi

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BOLD}"
echo "  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ "
echo "  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗"
echo "  ██╔██╗ ██║██║   ██║██║   ██║███████║"
echo "  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║"
echo "  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║"
echo "  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝"
echo -e "  Neural Omnipresent Voice Assistant${NC}"
echo ""
echo -e "${BOLD}Setup Script${NC}"
echo "─────────────────────────────────────────"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)
IS_PI=false
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "armv7l" ]]; then
    IS_PI=true
fi

echo -e "Platform: ${GREEN}$OS $ARCH${NC}"
if $IS_PI; then
    echo -e "Mode: ${GREEN}Raspberry Pi${NC}"
else
    echo -e "Mode: ${GREEN}Laptop/Desktop${NC}"
fi
echo ""

# ─── Python check ─────────────────────────────────────────────────────────────
echo -e "${BOLD}Checking Python...${NC}"
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo -e "Python $PYTHON_VERSION ${GREEN}✓${NC}"
else
    echo -e "${RED}Python 3.10+ required. You have $PYTHON_VERSION${NC}"
    exit 1
fi

# ─── Virtual environment ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi
source venv/bin/activate

# ─── System dependencies ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installing system dependencies...${NC}"

if [[ "$OS" == "Linux" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y \
        portaudio19-dev \
        libportaudio2 \
        python3-dev \
        espeak \
        espeak-ng \
        alsa-utils \
        xclip \
        libffi-dev \
        libssl-dev \
        scrot \
        notify-send \
        2>/dev/null || true
    echo -e "${GREEN}System packages installed${NC}"
elif [[ "$OS" == "Darwin" ]]; then
    if command -v brew &> /dev/null; then
        brew install portaudio espeak 2>/dev/null || true
        echo -e "${GREEN}Homebrew packages installed${NC}"
    else
        echo -e "${YELLOW}Homebrew not found. Install portaudio manually if needed.${NC}"
    fi
fi

# ─── Python packages ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installing Python packages...${NC}"
pip install --upgrade pip -q

if $IS_PI; then
    # Raspberry Pi optimized install
    echo "Installing Pi-optimized packages..."
    pip install -q pyaudio numpy sounddevice soundfile
    pip install -q "faster-whisper>=0.10" --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -q requests psutil pyperclip python-dateutil Pillow mss
    # Lighter chromadb for Pi
    pip install -q chromadb
    # openwakeword with onnxruntime (not GPU)
    pip install -q openwakeword onnxruntime
else
    # Laptop full install
    pip install -q -r requirements.txt
fi
echo -e "${GREEN}Python packages installed${NC}"

# ─── Ollama ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Checking Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}Ollama is installed${NC}"
    echo "Pulling Llama 3.2 Vision model (this may take a while)..."
    ollama pull llama3.2-vision || ollama pull llama3.2 || echo -e "${YELLOW}Could not pull model automatically. Run: ollama pull llama3.2-vision${NC}"
else
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    if [[ "$OS" == "Linux" ]] || [[ "$OS" == "Darwin" ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
        echo -e "${GREEN}Ollama installed${NC}"
        echo "Starting Ollama service..."
        ollama serve &
        sleep 3
        ollama pull llama3.2-vision || ollama pull llama3.2
    else
        echo -e "${RED}Please install Ollama manually from https://ollama.ai${NC}"
        echo "Then run: ollama pull llama3.2-vision"
    fi
fi

# ─── Piper TTS ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setting up Piper TTS...${NC}"
if command -v piper &> /dev/null; then
    echo -e "${GREEN}Piper TTS already installed${NC}"
else
    echo "Installing Piper TTS..."
    pip install -q piper-tts 2>/dev/null || true

    if ! command -v piper &> /dev/null; then
        echo -e "${YELLOW}Piper binary not found. Trying manual install...${NC}"
        PIPER_VERSION="1.2.0"
        if $IS_PI; then
            PIPER_ARCH="aarch64"
        else
            PIPER_ARCH="x86_64"
        fi
        PIPER_URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"
        mkdir -p ~/.local/bin
        curl -fsSL "$PIPER_URL" -o /tmp/piper.tar.gz 2>/dev/null
        tar -xzf /tmp/piper.tar.gz -C ~/.local/bin/ 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"

        if command -v piper &> /dev/null; then
            echo -e "${GREEN}Piper installed${NC}"
        else
            echo -e "${YELLOW}Using espeak as TTS fallback${NC}"
        fi
    fi

    # Download default voice model
    VOICE_DIR="$HOME/.local/share/piper/voices"
    mkdir -p "$VOICE_DIR"
    VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    echo "Downloading voice model..."
    curl -fsSL "$VOICE_URL" -o "$VOICE_DIR/en_US-lessac-medium.onnx" 2>/dev/null || true
    curl -fsSL "$CONFIG_URL" -o "$VOICE_DIR/en_US-lessac-medium.onnx.json" 2>/dev/null || true
fi

# ─── openwakeword models ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setting up wake word detection...${NC}"
python3 -c "
import sys
try:
    import openwakeword
    openwakeword.utils.download_models()
    print('openwakeword models downloaded')
except Exception as e:
    print(f'Wake word setup: {e}')
    print('Using Whisper fallback for wake word detection')
" 2>/dev/null || echo -e "${YELLOW}Wake word setup incomplete — using Whisper fallback${NC}"

# ─── Create run script ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Creating run script...${NC}"
cat > run_nova.sh << 'EOF'
#!/bin/bash
# Run NOVA
cd "$(dirname "$0")"
source venv/bin/activate

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

echo "Starting NOVA..."
python3 nova/main.py
EOF
chmod +x run_nova.sh
echo -e "${GREEN}Created run_nova.sh${NC}"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────"
echo -e "${GREEN}${BOLD}NOVA setup complete!${NC}"
echo ""
echo -e "To start NOVA, run: ${BOLD}./run_nova.sh${NC}"
echo -e "Or: ${BOLD}source venv/bin/activate && python3 nova/main.py${NC}"
echo ""
echo -e "Make sure Ollama is running: ${BOLD}ollama serve${NC}"
echo -e "And the model is pulled: ${BOLD}ollama pull llama3.2-vision${NC}"
echo ""
echo -e "${BOLD}Wake word:${NC} 'Hey Nova' or 'Nova'"
echo -e "${BOLD}Say:${NC} 'Nova, what can you do?' to get started"
echo "─────────────────────────────────────────"