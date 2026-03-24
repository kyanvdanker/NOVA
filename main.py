"""
NOVA - Neural Omnipresent Voice Assistant
Main Orchestrator

Modes:
  SLEEPING   → waiting for wake word
  LISTENING  → recording user utterance
  THINKING   → processing with LLM
  SPEAKING   → TTS output
  CONTEXT    → active after recent interaction (no wake word needed)
"""

import sys
import time
import threading
import logging
import signal
from pathlib import Path
from enum import Enum, auto
from typing import Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "nova.log"),
    ]
)
log = logging.getLogger("nova.main")


class State(Enum):
    SLEEPING = auto()    # Waiting for wake word
    CONTEXT  = auto()    # Active context, no wake word needed
    LISTENING = auto()   # Recording user speech
    THINKING = auto()    # Processing with LLM
    SPEAKING = auto()    # Playing TTS output


class Nova:
    """Main NOVA orchestrator."""

    def __init__(self):
        log.info("=" * 60)
        log.info("  NOVA - Neural Omnipresent Voice Assistant  ")
        log.info("  Say 'Hey Nova' to wake me up              ")
        log.info("=" * 60)

        self.state = State.SLEEPING
        self._running = False
        self._context_timer: Optional[threading.Timer] = None

        # Initialize components
        self._init_components()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _init_components(self):
        """Initialize all subsystems."""
        log.info("Initializing NOVA subsystems...")

        # Audio
        from audio.recorder import AudioRecorder, WakeWordListener
        from audio.stt import STT
        from audio.tts import TTS

        self.recorder = AudioRecorder()
        self.stt = STT()
        self.tts = TTS(
            on_speaking_start=self._on_speaking_start,
            on_speaking_end=self._on_speaking_end
        )
        self.wake_listener = WakeWordListener(self.recorder, self.stt)

        # Brain
        from brain.memory import MemoryManager
        self.memory = MemoryManager()

        from brain.llm import LLM
        self.llm = LLM(memory_manager=self.memory)

        # Skills
        from skills.laptop_control import LaptopControl
        self.laptop = LaptopControl()
        self.laptop.set_confirm_callback(self._confirm_action)

        import skills.projects as projects_module
        self.projects = projects_module

        from skills.intent_router import IntentRouter
        self.router = IntentRouter(
            llm=self.llm,
            memory=self.memory,
            laptop_control=self.laptop,
            projects_module=self.projects,
        )

        # Wire up interruption
        self.recorder.on_interrupt = self.tts.interrupt
        self.recorder.on_speech_start = self._on_speech_start

        log.info("All subsystems initialized")

    def run(self):
        """Main run loop."""
        self._running = True

        # Start audio monitoring
        self.recorder.start_monitoring()

        # Greet user
        profile = self.memory.get_profile()
        name = profile.get("name", "")
        greeting = f"Nova online. {f'Hello {name}.' if name else 'Hello.'} Say hey Nova to get started."
        self.tts.speak_async(greeting)

        log.info("Nova is running")

        try:
            while self._running:
                if self.state == State.SLEEPING:
                    self._sleep_loop()
                elif self.state == State.CONTEXT:
                    self._context_loop()
                elif self.state == State.LISTENING:
                    self._listen_and_process()
                time.sleep(0.01)

        except Exception as e:
            log.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self._cleanup()

    # ─── State Loops ──────────────────────────────────────────────────────────

    def _sleep_loop(self):
        """Wait for wake word. This blocks until wake word heard."""
        log.info("Sleeping... waiting for wake word")
        detected = self.wake_listener.wait_for_wake_word()
        if detected and self._running:
            self._wake_up()

    def _context_loop(self):
        """
        Active context mode: any speech is treated as directed at Nova.
        After CONTEXT_TIMEOUT_SEC of silence, go back to sleeping.
        """
        log.debug("In context mode, listening for speech...")
        audio = self.recorder.record_utterance(
            context_mode=True,
            timeout=config.CONTEXT_TIMEOUT_SEC
        )

        if audio:
            self.state = State.LISTENING
            self._process_audio(audio)
        else:
            # Timeout — go back to sleep
            log.info("Context timeout, going to sleep")
            self.state = State.SLEEPING

    def _listen_and_process(self):
        """Record a full utterance after wake word."""
        audio = self.recorder.record_utterance(context_mode=False, timeout=15.0)
        if audio:
            self._process_audio(audio)
        else:
            self.state = State.CONTEXT  # Stay active if nothing said

    # ─── Core Processing ──────────────────────────────────────────────────────

    def _process_audio(self, audio: bytes):
        """STT → intent routing → LLM → TTS pipeline."""
        self.state = State.THINKING

        # Transcribe
        text = self.stt.transcribe_raw(audio).strip()
        if not text:
            log.debug("Empty transcription, ignoring")
            self.state = State.CONTEXT
            return

        log.info(f"User: {text}")

        # Try skill routing first (fast, no LLM needed)
        response = self.router.route(text)

        if response is None:
            # Check if screen analysis needed
            if any(w in text.lower() for w in ["screen", "on my computer", "what am i working on"]):
                screenshot = self.laptop.take_screenshot()
                if screenshot:
                    response_stream = self.llm.chat(text, image_base64=screenshot)
                else:
                    response_stream = self.llm.chat(text)
            else:
                response_stream = self.llm.chat(text)

            # Speak streaming response
            self.state = State.SPEAKING
            self._stream_speak(response_stream)
        else:
            # Direct skill response
            self.state = State.SPEAKING
            self.tts.speak(response)
            log.info(f"Nova: {response}")

        # Stay in context mode after interaction
        self._reset_context_timer()
        self.state = State.CONTEXT

    def _stream_speak(self, text_stream):
        """
        Stream LLM output to TTS with sentence chunking.
        Speaks first sentence while rest is still generating.
        """
        buffer = ""
        full_response = ""
        sentence_enders = {'.', '!', '?', '\n'}

        for chunk in text_stream:
            if not self._running:
                break
            buffer += chunk
            full_response += chunk

            # Look for complete sentences to speak
            while any(e in buffer for e in sentence_enders):
                # Find the earliest sentence end
                end_idx = len(buffer)
                for e in sentence_enders:
                    idx = buffer.find(e)
                    if idx != -1 and idx < end_idx:
                        end_idx = idx + 1

                sentence = buffer[:end_idx].strip()
                buffer = buffer[end_idx:].strip()

                if sentence:
                    self.tts.speak(sentence)

        # Speak any remaining text
        if buffer.strip():
            self.tts.speak(buffer.strip())

        if full_response:
            log.info(f"Nova: {full_response[:200]}{'...' if len(full_response) > 200 else ''}")

    # ─── State Management ─────────────────────────────────────────────────────

    def _wake_up(self):
        """Wake up Nova — play confirmation sound and enter listening."""
        log.info("Waking up!")
        # Quick audio feedback
        self.tts.speak("Yes?")
        # Cancel any context timer
        self._cancel_context_timer()
        self.state = State.LISTENING

    def _reset_context_timer(self):
        """Reset the context mode timeout."""
        self._cancel_context_timer()
        self._context_timer = threading.Timer(
            config.CONTEXT_TIMEOUT_SEC,
            self._context_timeout
        )
        self._context_timer.daemon = True
        self._context_timer.start()

    def _cancel_context_timer(self):
        if self._context_timer:
            self._context_timer.cancel()
            self._context_timer = None

    def _context_timeout(self):
        """Called when context mode expires."""
        if self.state == State.CONTEXT:
            log.info("Context expired, sleeping")
            self.state = State.SLEEPING

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def _on_speaking_start(self):
        self.recorder.nova_speaking = True
        self.state = State.SPEAKING

    def _on_speaking_end(self):
        self.recorder.nova_speaking = False

    def _on_speech_start(self):
        """User started speaking (used for interruption detection)."""
        if self.state == State.SPEAKING:
            log.info("User interrupted — stopping speech")
            self.tts.interrupt()

    def _confirm_action(self, action: str) -> bool:
        """
        Ask user to confirm a potentially dangerous action.
        Uses TTS + STT for voice confirmation.
        """
        self.tts.speak(f"Are you sure you want to {action}? Say yes to confirm.")

        audio = self.recorder.record_utterance(timeout=10.0)
        if audio:
            text = self.stt.transcribe_raw(audio).lower()
            confirmed = any(w in text for w in ["yes", "confirm", "do it", "go ahead"])
            if confirmed:
                self.tts.speak("Confirmed.")
            else:
                self.tts.speak("Cancelled.")
            return confirmed
        return False

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self, signum=None, frame=None):
        log.info("Shutting down Nova...")
        self._running = False
        self.wake_listener.stop()
        self.tts.speak("Shutting down. Goodbye.")
        time.sleep(2)  # Let TTS finish

    def _cleanup(self):
        self._cancel_context_timer()
        self.recorder.cleanup()
        log.info("Nova shutdown complete")


def main():
    """Entry point."""
    nova = Nova()
    nova.run()


if __name__ == "__main__":
    main()