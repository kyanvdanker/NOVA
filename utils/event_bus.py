"""
NOVA Event Bus
Central nervous system — all subsystems communicate through this.
Decouples components and enables complex multi-system reactions.

Events flow like:
  camera detects face → PRESENCE_ARRIVED event
    → autonomous engine queues greeting
    → LED goes from sleeping to context color
    → main loop wakes up

  voice detected → SPEECH_START event
    → LED switches to listening
    → TTS interrupts if speaking
    → autonomous engine pauses
"""

import threading
import time
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger("nova.bus")


class Event(Enum):
    # System lifecycle
    NOVA_STARTED        = "nova_started"
    NOVA_STOPPING       = "nova_stopping"

    # Audio pipeline
    WAKE_WORD_DETECTED  = "wake_word_detected"
    SPEECH_START        = "speech_start"
    SPEECH_END          = "speech_end"
    UTTERANCE_READY     = "utterance_ready"     # transcribed text ready
    NOVA_SPEAKING_START = "nova_speaking_start"
    NOVA_SPEAKING_END   = "nova_speaking_end"
    NOVA_INTERRUPTED    = "nova_interrupted"

    # Intelligence
    INTENT_CLASSIFIED   = "intent_classified"
    LLM_RESPONSE_START  = "llm_response_start"
    LLM_RESPONSE_END    = "llm_response_end"

    # Camera / Presence
    PERSON_ARRIVED      = "person_arrived"
    PERSON_LEFT         = "person_left"
    EMOTION_CHANGED     = "emotion_changed"
    GESTURE_DETECTED    = "gesture_detected"
    ATTENTION_LOST      = "attention_lost"
    ATTENTION_GAINED    = "attention_gained"

    # Mood
    MOOD_CHANGED        = "mood_changed"

    # Autonomous
    AUTONOMOUS_MESSAGE  = "autonomous_message"
    REMINDER_FIRED      = "reminder_fired"
    BRIEFING_READY      = "briefing_ready"

    # Skills
    PROJECT_CREATED     = "project_created"
    TASK_ADDED          = "task_added"
    TASK_COMPLETED      = "task_completed"
    REMINDER_SET        = "reminder_set"

    # Ambient
    WEATHER_UPDATED     = "weather_updated"
    WINDOW_CHANGED      = "window_changed"
    CLIPBOARD_CHANGED   = "clipboard_changed"

    # State changes
    STATE_CHANGED       = "state_changed"

    # Memory
    MEMORY_STORED       = "memory_stored"
    PROFILE_UPDATED     = "profile_updated"


@dataclass
class EventPayload:
    event: Event
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


class EventBus:
    """
    Thread-safe publish/subscribe event bus.
    All NOVA subsystems communicate through this.
    """

    def __init__(self):
        self._subscribers: Dict[Event, List[Callable]] = {}
        self._wildcard_subscribers: List[Callable] = []
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=200)
        self._async_queue: deque = deque()
        self._async_thread = threading.Thread(target=self._async_dispatcher, daemon=True)
        self._async_thread.start()

    def subscribe(self, event: Event, callback: Callable,
                  async_dispatch: bool = True):
        """
        Subscribe to an event.
        callback(payload: EventPayload) will be called.
        async_dispatch: if True, callbacks run in background thread.
        """
        with self._lock:
            if event not in self._subscribers:
                self._subscribers[event] = []
            self._subscribers[event].append((callback, async_dispatch))

    def subscribe_all(self, callback: Callable):
        """Subscribe to ALL events (for logging, debugging)."""
        with self._lock:
            self._wildcard_subscribers.append(callback)

    def publish(self, event: Event, data: Any = None, source: str = "unknown"):
        """Publish an event to all subscribers."""
        payload = EventPayload(event=event, data=data, source=source)
        self._history.append(payload)

        with self._lock:
            subs = list(self._subscribers.get(event, []))
            wildcards = list(self._wildcard_subscribers)

        # Dispatch to subscribers
        for callback, async_dispatch in subs:
            if async_dispatch:
                self._async_queue.append((callback, payload))
            else:
                self._safe_call(callback, payload)

        # Wildcard subscribers always async
        for callback in wildcards:
            self._async_queue.append((callback, payload))

    def _async_dispatcher(self):
        """Background thread for async event dispatch."""
        while True:
            while self._async_queue:
                callback, payload = self._async_queue.popleft()
                self._safe_call(callback, payload)
            time.sleep(0.005)

    def _safe_call(self, callback: Callable, payload: EventPayload):
        try:
            callback(payload)
        except Exception as e:
            log.error(f"Event handler error ({payload.event.value}): {e}", exc_info=True)

    def get_history(self, event: Optional[Event] = None,
                    last_n: int = 20) -> List[EventPayload]:
        """Get recent event history."""
        history = list(self._history)
        if event:
            history = [p for p in history if p.event == event]
        return history[-last_n:]

    def last_event(self, event: Event) -> Optional[EventPayload]:
        """Get the most recent occurrence of an event."""
        for payload in reversed(list(self._history)):
            if payload.event == event:
                return payload
        return None

    def time_since(self, event: Event) -> float:
        """Seconds since the last occurrence of an event. inf if never."""
        last = self.last_event(event)
        if last is None:
            return float("inf")
        return time.time() - last.timestamp


# Global singleton event bus
bus = EventBus()