"""
NOVA — Neural Omnipresent Voice Assistant
Main Orchestrator v2

New in v2:
  - Camera presence detection (auto-wakes Nova)
  - Autonomous proactive behavior
  - Mood engine integration
  - Ambient intelligence (weather, active window)
  - LED status ring
  - Event bus architecture
  - Tool calling from LLM
  - Scheduler with reminders
  - Echo gate fix (no more self-triggering)
  - Word-level TTS streaming (no more response delays)
"""

import sys
import time
import threading
import logging
import signal
from pathlib import Path
from enum import Enum, auto
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import config

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "nova.log"),
    ]
)
log = logging.getLogger("nova.main")

# Fix Windows console Unicode/emoji issues
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class State(Enum):
    SLEEPING  = auto()
    CONTEXT   = auto()
    LISTENING = auto()
    THINKING  = auto()
    SPEAKING  = auto()


class Nova:
    """
    NOVA main orchestrator.
    Wires together all subsystems through the event bus.
    """

    def __init__(self):
        log.info("=" * 60)
        log.info("  N O V A  —  Neural Omnipresent Voice Assistant  v2.0")
        log.info("=" * 60)

        self.state = State.SLEEPING
        self._running = False
        self._context_timer: Optional[threading.Timer] = None

        self._init_all()

    def _init_all(self):
        """Initialize all subsystems in dependency order."""
        log.info("Booting subsystems...")

        # ── Event Bus ─────────────────────────────────────────────────────────
        from utils.event_bus import bus, Event
        self.bus = bus
        self.Event = Event
        self._wire_bus_handlers()

        # ── LED ───────────────────────────────────────────────────────────────
        from utils.led import LEDController
        self.led = LEDController()
        self.led.set_state("sleeping")

        # ── Audio ─────────────────────────────────────────────────────────────
        from audio.recorder import AudioRecorder, WakeWordListener
        from audio.stt import STT
        from audio.tts import TTS

        self.recorder = AudioRecorder()
        self.stt = STT()
        self.tts = TTS(
            on_speaking_start=self._on_speaking_start,
            on_speaking_end=self._on_speaking_end,
        )
        self.wake_listener = WakeWordListener(self.recorder, self.stt)

        # ── Memory ────────────────────────────────────────────────────────────
        from brain.memory import MemoryManager
        self.memory = MemoryManager()

        # ── Ambient Intelligence ──────────────────────────────────────────────
        from skills.ambient import AmbientIntelligence
        self.ambient = AmbientIntelligence()
        self.ambient.on_window_change = self._on_window_change

        # ── Mood Engine ───────────────────────────────────────────────────────
        from brain.mood import MoodEngine
        self.mood = MoodEngine()

        # ── LLM ───────────────────────────────────────────────────────────────
        from brain.llm import LLM
        self.llm = LLM(
            memory_manager=self.memory,
            mood_engine=self.mood,
            ambient=self.ambient,
        )
        self._register_tool_handlers()

        # ── Laptop Control ────────────────────────────────────────────────────
        from skills.laptop_control import LaptopControl
        self.laptop = LaptopControl()
        self.laptop.set_confirm_callback(self._voice_confirm)

        # ── Projects ──────────────────────────────────────────────────────────
        import skills.projects as projects_module
        self.projects = projects_module

        # ── Intent Router ─────────────────────────────────────────────────────
        from skills.intent_router import IntentRouter
        self.router = IntentRouter(
            llm=self.llm,
            memory=self.memory,
            laptop_control=self.laptop,
            projects_module=self.projects,
        )

        # ── Camera / Vision ───────────────────────────────────────────────────
        from vision.camera import CameraVision
        self.camera = CameraVision()
        self.camera.on_presence_change = self._on_presence_change
        self.camera.on_emotion_change  = self._on_emotion_change
        self.camera.on_gesture         = self._on_gesture

        # ── Scheduler ─────────────────────────────────────────────────────────
        from utils.scheduler import Scheduler
        self.scheduler = Scheduler()
        self.scheduler.set_fired_callback(self._on_scheduled_job)

        # ── Autonomous Engine ─────────────────────────────────────────────────
        from brain.autonomous import AutonomousEngine
        self.autonomous = AutonomousEngine(
            memory_manager=self.memory,
            llm=self.llm,
            projects_module=self.projects,
            ambient_module=self.ambient,
            mood_engine=self.mood,
        )
        self.autonomous.on_message = self._speak_autonomous

        # ── Wire recorder callbacks ───────────────────────────────────────────
        self.recorder.on_speech_start = self._on_speech_start
        self.recorder.on_interrupt    = self.tts.interrupt

        log.info("All subsystems initialized")

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ─── Event Bus Handlers ───────────────────────────────────────────────────

    def _wire_bus_handlers(self):
        """Subscribe all inter-system events."""
        E = self.Event
        self.bus.subscribe(E.STATE_CHANGED, self._on_state_changed)
        self.bus.subscribe(E.NOVA_INTERRUPTED, lambda p: self.tts.interrupt())

    def _on_state_changed(self, payload):
        state_name = payload.data
        if hasattr(self, "led"):
            state_to_led = {
                "sleeping":  "sleeping",
                "listening": "listening",
                "thinking":  "thinking",
                "speaking":  "speaking",
                "context":   "sleeping",
            }
            self.led.set_state(state_to_led.get(state_name, "sleeping"))

    def _set_state(self, new_state: State):
        self.state = new_state
        self.bus.publish(self.Event.STATE_CHANGED, new_state.name.lower())

    # ─── Run Loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._running = True

        # Start all background services
        self.recorder.start_monitoring()
        self.ambient.start()
        self.camera.start()
        self.scheduler.start()
        self.autonomous.start()

        # Greet
        profile = self.memory.get_profile()
        name = profile.get("name", "")
        greeting = self._generate_startup_greeting(name)
        self.tts.speak_async(greeting)

        log.info("Nova is running. Say 'Hey Nova' to wake me up.")

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
            log.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _generate_startup_greeting(self, name: str) -> str:
        """Generate a natural startup greeting."""
        import datetime
        hour = datetime.datetime.now().hour
        if hour < 12:
            period = "morning"
        elif hour < 17:
            period = "afternoon"
        else:
            period = "evening"

        base = f"Good {period}{', ' + name if name else ''}."

        # Check for pending reminders
        try:
            reminders = self.scheduler.list_reminders()
            if reminders:
                next_r = reminders[0]
                base += f" You have a reminder coming up: {next_r['message']}."
        except Exception:
            pass

        base += f" Say 'Hey Nova' when you need me."
        return base

    # ─── State Loops ──────────────────────────────────────────────────────────

    def _sleep_loop(self):
        """Wait for wake word - non-blocking version"""
        self._set_state(State.SLEEPING)
        log.info("Waiting for wake word... (Say 'Hey Nova')")

        try:
            # This call is blocking, but we catch exceptions properly
            detected = self.wake_listener.wait_for_wake_word()
            
            if detected and self._running:
                self._wake_up()

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received during wake word listening")
            self._running = False
        except Exception as e:
            log.error(f"Error in wake word detection: {e}")
            # Small delay to prevent 100% CPU on repeated errors
            time.sleep(0.5)

    def _context_loop(self):
        """Active context: listen for any speech without wake word."""
        audio = self.recorder.record_utterance(
            context_mode=True,
            timeout=config.CONTEXT_TIMEOUT_SEC
        )
        if audio:
            self._set_state(State.LISTENING)
            self._process_audio(audio)
        else:
            # Silence timeout
            self._set_state(State.SLEEPING)

    def _listen_and_process(self):
        """Record full utterance after wake word."""
        audio = self.recorder.record_utterance(context_mode=False, timeout=15.0)
        if audio:
            self._process_audio(audio)
        else:
            self._set_state(State.CONTEXT)

    # ─── Core Processing Pipeline ─────────────────────────────────────────────

    def _process_audio(self, audio: bytes):
        """Full pipeline: STT → mood → intent → LLM/skill → TTS."""
        self._set_state(State.THINKING)

        # Transcribe
        text = self.stt.transcribe_raw(audio).strip()
        if not text:
            self._set_state(State.CONTEXT)
            return

        log.info(f"[User] {text}")
        self.bus.publish(self.Event.UTTERANCE_READY, text)
        self.autonomous.update_last_interaction()

        # Mood analysis (non-blocking — updates mood engine for next response)
        if config.MOOD_ENABLED and config.MOOD_VOICE_ANALYSIS:
            threading.Thread(
                target=self._analyze_mood_from_voice,
                args=(audio, text),
                daemon=True
            ).start()

        # Try skill routing first (no LLM needed for known commands)
        response = self.router.route(text)

        if response is None:
            # Check if screen vision needed
            needs_screen = any(w in text.lower() for w in
                               ["screen", "on my computer", "what am i working", "describe what"])
            needs_camera = any(w in text.lower() for w in
                               ["see me", "look at me", "what do i look like", "camera", "am i"])

            screen_b64 = None
            camera_b64 = None

            if needs_screen:
                screen_b64 = self.laptop.take_screenshot()
            if needs_camera and self.camera.is_available:
                camera_b64 = self.camera.capture_base64()

            self._set_state(State.SPEAKING)
            stream = self.llm.chat(
                text,
                image_base64=screen_b64,
                camera_frame_b64=camera_b64,
            )
            self._stream_to_tts(stream)
        else:
            log.info(f"[Nova/skill] {response}")
            self._set_state(State.SPEAKING)
            self.tts.speak(response)

        self._reset_context_timer()
        self._set_state(State.CONTEXT)

    def _stream_to_tts(self, text_stream):
        """
        Feed LLM word stream into TTS with intelligent chunking.
        
        Fixes:
        1. Word-level emission from LLM means no more big accumulation delays
        2. We group words into ~8-word chunks before calling speak()
           for natural rhythm without long waits
        3. Sentence boundaries get a natural pause
        """
        word_buffer = []
        full_response = ""
        CHUNK_SIZE = config.TTS_WORD_CHUNK_SIZE

        sentence_enders = {'.', '!', '?'}

        for token in text_stream:
            if not self._running:
                break
            if not token.strip():
                continue

            full_response += token
            # Split token into words (each token is ~1-2 words from LLM)
            words = token.split()
            word_buffer.extend(words)

            # Check if we should flush
            should_flush = False

            # Flush on sentence end
            last_word = word_buffer[-1] if word_buffer else ""
            if last_word and last_word[-1] in sentence_enders:
                should_flush = True

            # Flush on chunk size
            if len(word_buffer) >= CHUNK_SIZE:
                should_flush = True

            if should_flush and word_buffer:
                chunk_text = " ".join(word_buffer)
                word_buffer = []
                if not self.tts.is_interrupted:
                    self.tts.speak(chunk_text)

        # Flush remainder
        if word_buffer:
            self.tts.speak(" ".join(word_buffer))

        if full_response:
            log.info(f"[Nova] {full_response[:200]}{'...' if len(full_response) > 200 else ''}")
            # Store the assistant's response to memory explicitly if not done by LLM
            # (already handled in llm.chat but kept here for safety)

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def _wake_up(self):
        log.info("Wake word detected — activating")
        self.led.set_state("listening")
        self.tts.speak("Yes?")  # minimal, natural acknowledgement
        self._cancel_context_timer()
        self._set_state(State.LISTENING)

    def _on_speaking_start(self):
        self.recorder.nova_speaking = True
        self.led.set_state("speaking")
        self.bus.publish(self.Event.NOVA_SPEAKING_START)

    def _on_speaking_end(self):
        self.recorder.notify_speaking_ended()  # starts the echo gate tail timer
        self.bus.publish(self.Event.NOVA_SPEAKING_END)

    def _on_speech_start(self):
        """Microphone picked up speech start."""
        if self.state == State.SPEAKING:
            log.info("User is speaking — interrupting Nova")
            self.tts.interrupt()
            self.bus.publish(self.Event.NOVA_INTERRUPTED)
        self.led.set_state("listening")

    def _on_presence_change(self, event):
        """Camera detected someone arriving or leaving."""
        if event.arrived:
            log.info("Presence detected")
            self.bus.publish(self.Event.PERSON_ARRIVED, event)
            # Wake Nova from sleep if sleeping
            if self.state == State.SLEEPING:
                self._cancel_context_timer()
                self._set_state(State.CONTEXT)
                self.autonomous.on_person_arrived(event.face_count)
        else:
            log.info("Person left")
            self.bus.publish(self.Event.PERSON_LEFT, event)
            self._set_state(State.SLEEPING)

    def _on_emotion_change(self, emotion: str, confidence: float):
        """Camera detected emotion change."""
        reading = self.mood.analyze_face(emotion, confidence)
        self.bus.publish(self.Event.EMOTION_CHANGED, {"emotion": emotion, "confidence": confidence})
        # Check if mood warrants autonomous response
        if config.AUTONOMOUS_ENABLED:
            self.autonomous.on_mood_detected(self.mood.current_mood_name, self.mood.confidence)

    def _on_gesture(self, gesture_event):
        """Camera detected a gesture."""
        self.bus.publish(self.Event.GESTURE_DETECTED, gesture_event)
        g = gesture_event.gesture

        if g == "raise_hand":
            # Raised hand = attention gesture, same as wake word
            log.info("Gesture: raised hand → waking up")
            self._wake_up()

        elif g == "thumbs_up":
            # Thumbs up = confirm / yes
            log.info("Gesture: thumbs up")
            if self.state == State.SLEEPING:
                self.tts.speak_async("Thumbs up received.")

        elif g == "wave":
            # Wave = dismiss / goodbye
            if self.state not in [State.SLEEPING]:
                self.tts.speak_async("Understood.")
                self._set_state(State.SLEEPING)

    def _on_window_change(self, window_info: dict):
        """Active window changed — update context."""
        self.bus.publish(self.Event.WINDOW_CHANGED, window_info)
        # Could trigger smart context loading for project files

    def _analyze_mood_from_voice(self, audio: bytes, transcript: str):
        """Run mood analysis in background."""
        try:
            reading = self.mood.analyze_voice(audio, transcript)
            if reading:
                self.bus.publish(self.Event.MOOD_CHANGED,
                                 {"mood": reading.mood.value, "confidence": reading.confidence})
        except Exception as e:
            log.debug(f"Mood analysis error: {e}")

    def _on_scheduled_job(self, job):
        """A scheduled job fired."""
        payload = job.payload
        job_type = payload.get("type", "reminder")

        if job_type == "reminder":
            msg = payload.get("message", "Reminder.")
            self.bus.publish(self.Event.REMINDER_FIRED, msg)
            self._speak_autonomous(f"Reminder: {msg}")

        elif job_type == "morning_briefing":
            self.autonomous._queue_morning_briefing()

        elif job_type == "evening_summary":
            self.autonomous._queue_evening_summary()

        elif job_type == "memory_consolidation":
            threading.Thread(target=self._consolidate_memories, daemon=True).start()

    def _speak_autonomous(self, text: str):
        """
        Nova speaks autonomously (not in response to user).
        Only speaks if Nova isn't already busy.
        """
        if not text or not text.strip():
            return

        # Don't interrupt user interactions
        if self.state in [State.LISTENING, State.THINKING]:
            log.debug(f"Skipping autonomous message (busy): {text[:50]}")
            return

        log.info(f"[Nova/autonomous] {text[:100]}")
        self.led.animate(config.LED_COLORS.get("alert", (255, 100, 0)), "pulse", 1.0)
        self._set_state(State.SPEAKING)
        self.tts.speak(text)
        self._reset_context_timer()
        self._set_state(State.CONTEXT)

    # ─── Tool Handlers ────────────────────────────────────────────────────────

    def _register_tool_handlers(self):
        """Register LLM tool call handlers."""
        self.llm.tool_handlers = {
            "open_application":  self._tool_open_app,
            "create_project":    self._tool_create_project,
            "add_task":          self._tool_add_task,
            "remember_fact":     self._tool_remember_fact,
            "set_reminder":      self._tool_set_reminder,
            "search_web":        self._tool_search_web,
        }

    def _tool_open_app(self, app_name: str, **kwargs) -> str:
        success, msg = self.laptop.open_app(app_name)
        return msg

    def _tool_create_project(self, name: str, description: str = "", **kwargs) -> str:
        project = self.projects.create_project(name, description)
        self.router.set_active_project(project["id"], project["name"])
        self.bus.publish(self.Event.PROJECT_CREATED, project)
        return f"Created project '{name}'."

    def _tool_add_task(self, content: str, project_name: str = None,
                       task_type: str = "task", **kwargs) -> str:
        project_id = None
        if project_name:
            p = self.projects.get_project_by_name(project_name)
            if p:
                project_id = p["id"]
        self.projects.add_memo(content, project_id=project_id, memo_type=task_type)
        self.bus.publish(self.Event.TASK_ADDED, {"content": content})
        return f"Added {task_type}: {content[:50]}"

    def _tool_remember_fact(self, category: str, fact: str, **kwargs) -> str:
        self.memory.add_fact(category, fact, source="stated")
        self.bus.publish(self.Event.MEMORY_STORED, {"category": category, "fact": fact})
        return ""  # Silent — Nova already said it in her response

    def _tool_set_reminder(self, message: str, when: str, **kwargs) -> str:
        result = self.scheduler.add_reminder(message, when)
        if result:
            self.bus.publish(self.Event.REMINDER_SET, result)
            return f"Reminder set for {result['when_str']}."
        return "I couldn't parse that time. Try 'in 30 minutes' or 'tomorrow at 9am'."

    def _tool_search_web(self, query: str, **kwargs) -> str:
        """Basic web search via DuckDuckGo (no API key needed)."""
        try:
            import requests
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            resp = requests.get("https://api.duckduckgo.com/",
                                params=params, timeout=8)
            data = resp.json()
            abstract = data.get("AbstractText", "")
            if abstract:
                return f"Web search result: {abstract[:300]}"
            # Try instant answer
            answer = data.get("Answer", "")
            if answer:
                return f"Answer: {answer}"
            return "Couldn't find a clear answer online."
        except Exception as e:
            return f"Web search failed: {e}"

    # ─── Voice Confirmation ───────────────────────────────────────────────────

    def _voice_confirm(self, action: str) -> bool:
        """Ask user to confirm action by voice."""
        self.tts.speak(f"Are you sure you want to {action}? Say yes to confirm.")
        audio = self.recorder.record_utterance(timeout=8.0)
        if audio:
            text = self.stt.transcribe_raw(audio).lower()
            if any(w in text for w in ["yes", "confirm", "do it", "go ahead", "yep", "yeah"]):
                self.tts.speak("Done.")
                return True
        self.tts.speak("Cancelled.")
        return False

    # ─── Memory Consolidation ─────────────────────────────────────────────────

    def _consolidate_memories(self):
        """Overnight: Nova re-reads recent memories and extracts key facts."""
        log.info("Memory consolidation running...")
        try:
            stats = self.memory.get_stats()
            if stats["episodes"] < 10:
                return

            # Ask LLM to extract key facts from recent episodes
            recent = self.memory.retrieve("important facts about user", k=10)
            if not recent:
                return

            memories_text = "\n".join(recent)
            prompt = f"""Review these recent conversation excerpts and extract the 3-5 most important 
facts, preferences, or patterns about the user. Return as a JSON list of objects with 
"category" and "fact" keys. Only extract clearly stated facts.

Conversations:
{memories_text}

JSON only, no other text."""

            response = self.llm.generate_autonomous(prompt)
            if response:
                import json, re
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                    for f in facts[:5]:
                        self.memory.add_fact(
                            f.get("category", "personal"),
                            f.get("fact", ""),
                            confidence=0.7,
                            source="consolidated"
                        )
                    log.info(f"Memory consolidation: extracted {len(facts)} facts")
        except Exception as e:
            log.error(f"Memory consolidation error: {e}")

    # ─── Context Timer ────────────────────────────────────────────────────────

    def _reset_context_timer(self):
        self._cancel_context_timer()
        self._context_timer = threading.Timer(
            config.CONTEXT_TIMEOUT_SEC,
            lambda: self._set_state(State.SLEEPING) if self.state == State.CONTEXT else None
        )
        self._context_timer.daemon = True
        self._context_timer.start()

    def _cancel_context_timer(self):
        if self._context_timer:
            self._context_timer.cancel()
            self._context_timer = None

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    def _handle_signal(self, signum=None, frame=None):
        log.info("Shutdown signal received")
        self._running = False
        self.wake_listener.stop()
        self.tts.speak("Shutting down. See you next time.")
        time.sleep(2.5)

    def _cleanup(self):
        self._cancel_context_timer()
        if hasattr(self, "recorder"):
            self.recorder.cleanup()
        if hasattr(self, "camera"):
            self.camera.stop()
        if hasattr(self, "ambient"):
            self.ambient.stop()
        if hasattr(self, "autonomous"):
            self.autonomous.stop()
        if hasattr(self, "scheduler"):
            self.scheduler.stop()
        if hasattr(self, "led"):
            self.led.off()
        log.info("Nova shutdown complete.")


# ─── TTS patch for interrupt tracking ─────────────────────────────────────────
# Monkey-patch TTS to expose interrupt status for _stream_to_tts
def _patch_tts(tts_instance):
    tts_instance.is_interrupted = False
    original_interrupt = tts_instance.interrupt
    def patched_interrupt():
        tts_instance.is_interrupted = True
        original_interrupt()
    tts_instance.interrupt = patched_interrupt

    original_speak = tts_instance.speak
    def patched_speak(text, blocking=True):
        tts_instance.is_interrupted = False
        return original_speak(text, blocking)
    tts_instance.speak = patched_speak


def main():
    nova = Nova()
    # Patch TTS interrupt tracking
    _patch_tts(nova.tts)
    nova.run()


if __name__ == "__main__":
    main()