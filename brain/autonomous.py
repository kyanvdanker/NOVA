"""
NOVA Autonomous Behavior Engine
Nova's proactive intelligence — she initiates conversations, notices patterns,
gives briefings, checks in on you, and acts on her own initiative.

Think: the difference between a tool you have to ask vs. a colleague who
notices things and speaks up.
"""

import time
import logging
import threading
import random
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Dict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.autonomous")


class TriggerType(Enum):
    MORNING_BRIEFING    = "morning_briefing"
    EVENING_SUMMARY     = "evening_summary"
    PRESENCE_ARRIVAL    = "presence_arrival"
    PROACTIVE_CHECKIN   = "proactive_checkin"
    REMINDER            = "reminder"
    INSIGHT             = "insight"
    MOOD_RESPONSE       = "mood_response"
    MEMORY_SURFACE      = "memory_surface"
    MILESTONE           = "milestone"
    WEATHER_ALERT       = "weather_alert"
    TASK_NUDGE          = "task_nudge"
    FOCUS_BREAK         = "focus_break"


@dataclass
class AutonomousMessage:
    trigger: TriggerType
    prompt: str             # What to send to the LLM to generate the message
    priority: int           # 1=low, 2=normal, 3=high, 4=urgent
    context: dict           # Extra context for the LLM
    can_skip: bool = True   # If Nova is busy, can this be skipped?
    spoken: bool = False


class AutonomousEngine:
    """
    Manages all of Nova's proactive, self-initiated behaviors.
    Runs a scheduler loop in a background thread.
    """

    def __init__(self, memory_manager, llm, projects_module,
                 ambient_module=None, mood_engine=None):
        self.memory = memory_manager
        self.llm = llm
        self.projects = projects_module
        self.ambient = ambient_module
        self.mood = mood_engine

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: List[AutonomousMessage] = []
        self._queue_lock = threading.Lock()

        # State tracking
        self._last_morning_briefing = 0.0
        self._last_evening_summary = 0.0
        self._last_insight = 0.0
        self._last_proactive_checkin = 0.0
        self._last_user_interaction = time.time()
        self._last_presence_greeting = 0.0
        self._session_start = time.time()
        self._focus_start: Optional[float] = None

        # Callback: when a message is ready to speak
        self.on_message: Optional[Callable[[str], None]] = None

        log.info("Autonomous engine initialized")

    def update_last_interaction(self):
        """Call this every time the user says something."""
        self._last_user_interaction = time.time()
        self._focus_start = None  # reset focus tracking

    def start(self):
        """Start autonomous engine in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Autonomous engine started")

    def stop(self):
        self._running = False

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _run_loop(self):
        """Main scheduler loop — checks triggers every 30 seconds."""
        while self._running:
            try:
                self._check_all_triggers()
                self._process_queue()
            except Exception as e:
                log.error(f"Autonomous engine error: {e}", exc_info=True)
            time.sleep(30)

    def _check_all_triggers(self):
        """Check all autonomous trigger conditions."""
        now = time.time()
        dt = datetime.now()

        # Morning briefing
        if (config.MORNING_BRIEFING
                and dt.hour == config.MORNING_HOUR
                and dt.minute < 5
                and now - self._last_morning_briefing > 3600 * 20):
            self._queue_morning_briefing()
            self._last_morning_briefing = now

        # Evening summary
        if (config.EVENING_SUMMARY
                and dt.hour == config.EVENING_HOUR
                and dt.minute < 5
                and now - self._last_evening_summary > 3600 * 20):
            self._queue_evening_summary()
            self._last_evening_summary = now

        # Proactive check-in (if present and silent)
        if config.PROACTIVE_CHECKIN:
            silence_sec = now - self._last_user_interaction
            cooldown_sec = config.PROACTIVE_COOLDOWN_MIN * 60
            silence_threshold = config.PROACTIVE_SILENCE_MIN * 60
            if (silence_sec > silence_threshold
                    and now - self._last_proactive_checkin > cooldown_sec):
                self._queue_proactive_checkin(silence_sec)
                self._last_proactive_checkin = now

        # Insight injection
        if (config.AUTONOMOUS_INSIGHTS
                and now - self._last_insight > config.INSIGHT_INTERVAL_MIN * 60
                and now - self._last_user_interaction < 3600):  # only if recently active
            self._queue_insight()
            self._last_insight = now

        # Task nudge (if there are overdue tasks)
        if dt.hour in [9, 14] and dt.minute < 5:
            self._maybe_queue_task_nudge()

        # Focus break reminder (if been working for 90+ minutes)
        if self._focus_start is None:
            self._focus_start = self._last_user_interaction
        focus_duration = now - (self._focus_start or now)
        if focus_duration > 90 * 60 and now - self._last_proactive_checkin > 30 * 60:
            self._queue_focus_break(focus_duration)
            self._last_proactive_checkin = now

    def _process_queue(self):
        """Process the highest-priority queued message."""
        with self._queue_lock:
            if not self._queue:
                return
            # Sort by priority
            self._queue.sort(key=lambda m: -m.priority)
            msg = self._queue.pop(0)

        if self.on_message:
            text = self._generate_message(msg)
            if text:
                log.info(f"Autonomous [{msg.trigger.value}]: {text[:80]}...")
                self.on_message(text)

    def _generate_message(self, msg: AutonomousMessage) -> Optional[str]:
        """Generate the actual text for an autonomous message using LLM."""
        try:
            # Build context
            profile = self.memory.get_user_profile_text()
            mood_ctx = ""
            if self.mood:
                mood_ctx = self.mood.get_mood_context_str()

            system_addition = f"""
{config.NOVA_PERSONALITY}

## Autonomous Message Context
You are generating a PROACTIVE message — you are initiating this, not responding.
Keep it brief (1-3 sentences max). Natural, warm, never robotic.
Type: {msg.trigger.value}
{mood_ctx}
Owner profile: {profile or 'not yet known'}
"""
            # Temporarily augment the system prompt
            old_prompt = self.llm._system_prompt
            self.llm._system_prompt = system_addition

            result = self.llm.chat_blocking(msg.prompt)

            self.llm._system_prompt = old_prompt
            return result.strip()
        except Exception as e:
            log.error(f"Message generation failed: {e}")
            return None

    # ─── Trigger Builders ─────────────────────────────────────────────────────

    def _queue_morning_briefing(self):
        """Queue a morning briefing."""
        # Gather context
        projects = []
        tasks = []
        try:
            projects = self.projects.list_projects("active")[:3]
            tasks = self.projects.get_memos(memo_type="task", status="open")[:5]
        except Exception:
            pass

        weather_str = ""
        if self.ambient:
            weather = self.ambient.get_weather_summary()
            if weather:
                weather_str = f"Weather today: {weather}."

        project_names = ", ".join(p["name"] for p in projects) if projects else "none"
        task_count = len(tasks)
        dt = datetime.now()
        greeting = "Good morning" if dt.hour < 12 else "Good afternoon"

        prompt = f"""Generate a warm, brief morning briefing for your owner.
Today is {dt.strftime('%A, %B %d')}.
{weather_str}
Active projects: {project_names or 'none tracked'}.
Open tasks: {task_count}.
Keep it to 2-3 sentences. Start with '{greeting}'. 
Be warm, practical, and forward-looking. Don't list everything — just give the vibe and one priority."""

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.MORNING_BRIEFING,
            prompt=prompt,
            priority=3,
            context={"projects": projects, "tasks": tasks},
        ))

    def _queue_evening_summary(self):
        """Queue an evening summary."""
        stats = self.memory.get_stats()
        interactions_today = self._estimate_today_interactions()

        prompt = f"""Generate a brief, warm evening summary for your owner.
Today: {datetime.now().strftime('%A, %B %d')}.
Interactions today: approximately {interactions_today}.
Memory stats: {stats.get('episodes', 0)} total memories.
Keep it to 2-3 sentences. Reflect gently on the day and look ahead to tomorrow.
End with something quietly encouraging. Don't be saccharine."""

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.EVENING_SUMMARY,
            prompt=prompt,
            priority=2,
            context={"stats": stats},
        ))

    def _queue_proactive_checkin(self, silence_sec: float):
        """Queue a proactive check-in after long silence."""
        silence_min = int(silence_sec / 60)
        mood_str = self.mood.current_mood_name if self.mood else "neutral"

        prompts = [
            f"You haven't heard from your owner in {silence_min} minutes. They appear to be present. Generate a brief, natural check-in. Not intrusive. Maybe just offer a presence. 1 sentence.",
            f"Your owner has been quiet for {silence_min} minutes (mood: {mood_str}). Check in gently. Could offer help, or just acknowledge you're here. Very brief.",
            f"After {silence_min} minutes of silence, gently check in with your owner. Ask if they need anything or just note you're available. 1 short sentence.",
        ]

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.PROACTIVE_CHECKIN,
            prompt=random.choice(prompts),
            priority=1,
            context={"silence_min": silence_min, "mood": mood_str},
        ))

    def _queue_insight(self):
        """Queue an interesting observation or insight."""
        # Get recent memories to find something to riff on
        recent = self.memory.retrieve("recent interesting topic", k=3)
        profile = self.memory.get_user_profile_text()
        facts = self.memory.get_facts()[:3]
        fact_texts = "; ".join(f["fact"] for f in facts[:2]) if facts else ""

        prompts = [
            f"""Based on what you know about your owner ({profile or 'not much yet'}), 
share one brief interesting observation, connection, or thought they might appreciate.
Could be about their work, an interesting fact related to what they've discussed, or a pattern you've noticed.
1-2 sentences max. Natural, intelligent, never forced.""",

            f"""You've noticed something interesting. Share it.
Owner context: {profile or 'unknown'}
Recent topics: {fact_texts or 'general conversation'}
Make it insightful and relevant. Not a tip or suggestion — an observation or interesting angle.
Keep it to 1-2 natural sentences.""",
        ]

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.INSIGHT,
            prompt=random.choice(prompts),
            priority=1,
            context={},
            can_skip=True,
        ))

    def _maybe_queue_task_nudge(self):
        """Nudge about overdue or pending tasks at key times."""
        try:
            tasks = self.projects.get_memos(memo_type="task", status="open")
            if not tasks:
                return

            high_priority = [t for t in tasks if t.get("priority") in ("high", "urgent")]
            target_tasks = high_priority[:2] if high_priority else tasks[:2]
            task_list = "; ".join(t["content"][:50] for t in target_tasks)

            prompt = f"""Briefly mention {len(target_tasks)} pending task(s) to your owner.
Tasks: {task_list}
Keep it to 1 sentence. Not naggy — just a friendly reminder woven naturally into conversation.
Don't say "reminder" or "don't forget" — be more natural."""

            self._enqueue(AutonomousMessage(
                trigger=TriggerType.TASK_NUDGE,
                prompt=prompt,
                priority=2,
                context={"tasks": target_tasks},
                can_skip=True,
            ))
        except Exception as e:
            log.debug(f"Task nudge error: {e}")

    def _queue_focus_break(self, duration_sec: float):
        """Suggest a break after extended focus time."""
        hours = duration_sec / 3600
        prompt = f"""Your owner has been working for about {hours:.1f} hours. 
Gently suggest a short break. 1 sentence. Natural, caring, not bossy.
Don't say "you should" — suggest it as an observation."""

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.FOCUS_BREAK,
            prompt=prompt,
            priority=2,
            context={"duration_hours": round(hours, 1)},
        ))

    def on_person_arrived(self, face_count: int = 1):
        """Called when camera detects someone arriving at desk."""
        now = time.time()
        if now - self._last_presence_greeting < config.PRESENCE_COOLDOWN_SEC:
            return
        self._last_presence_greeting = now

        dt = datetime.now()
        profile = self.memory.get_user_profile_text()
        name = self.memory.get_profile().get("name", "")
        hour = dt.hour

        if hour < 10:
            time_of_day = "morning"
        elif hour < 13:
            time_of_day = "mid-morning"
        elif hour < 17:
            time_of_day = "afternoon"
        elif hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "late evening"

        prompt = f"""Someone has sat down at the computer. It's {time_of_day} on {dt.strftime('%A')}.
{'Owner name: ' + name if name else 'Owner name: unknown'}
Generate a brief, natural arrival greeting. 1-2 sentences.
Could acknowledge the time of day, mention something relevant, or just be present.
Don't say "Welcome back" — be more creative and personal."""

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.PRESENCE_ARRIVAL,
            prompt=prompt,
            priority=3,
            context={"time_of_day": time_of_day, "name": name},
        ))

    def on_mood_detected(self, mood_name: str, confidence: float):
        """React to significant mood detection."""
        if confidence < 0.65:
            return
        if mood_name not in ("stressed", "frustrated"):
            return

        now = time.time()
        if now - self._last_proactive_checkin < 600:  # 10 min cooldown
            return
        self._last_proactive_checkin = now

        prompt = f"""You've detected your owner seems {mood_name} right now.
Respond with a brief, natural acknowledgment. 1 sentence.
Don't be clinical or say "I detect you are stressed."
Be human: you just noticed something and want them to know you're here.
Offer help subtly without being overbearing."""

        self._enqueue(AutonomousMessage(
            trigger=TriggerType.MOOD_RESPONSE,
            prompt=prompt,
            priority=2,
            context={"mood": mood_name},
        ))

    def queue_reminder(self, text: str, priority: int = 3):
        """Queue a specific reminder (set by user or scheduler)."""
        prompt = f"Deliver this reminder naturally: '{text}'. 1-2 sentences."
        self._enqueue(AutonomousMessage(
            trigger=TriggerType.REMINDER,
            prompt=prompt,
            priority=priority,
            context={"reminder": text},
            can_skip=False,
        ))

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _enqueue(self, msg: AutonomousMessage):
        with self._queue_lock:
            # Don't duplicate same trigger type
            if not any(m.trigger == msg.trigger for m in self._queue):
                self._queue.append(msg)
                log.debug(f"Queued: {msg.trigger.value} (priority={msg.priority})")

    def _estimate_today_interactions(self) -> int:
        """Estimate number of interactions today from memory."""
        try:
            import sqlite3
            import time as t
            today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
            conn = __import__('sqlite3').connect(str(config.MEMORY_DB_PATH))
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM episodes WHERE timestamp > ?", (today_start,))
            count = c.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0