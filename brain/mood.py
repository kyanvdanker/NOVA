"""
NOVA Mood Engine
Tracks the user's emotional and cognitive state from:
- Voice: energy, speech rate, pauses, pitch variation
- Face: DeepFace emotion analysis
- Patterns: time of day, recent interaction history
- Context: what they're working on

Adapts Nova's persona: more gentle when stressed,
more upbeat when energized, quieter when focused.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.mood")


class Mood(Enum):
    FOCUSED    = "focused"      # deep work, minimal distractions
    ENERGIZED  = "energized"    # upbeat, productive, fast-talking
    NEUTRAL    = "neutral"      # baseline
    TIRED      = "tired"        # slow speech, low energy
    STRESSED   = "stressed"     # fast + high pitch + short bursts
    FRUSTRATED = "frustrated"   # detected from voice frustration + context
    HAPPY      = "happy"        # positive tone + face expression


@dataclass
class MoodReading:
    mood: Mood
    confidence: float
    timestamp: float
    source: str   # "voice", "face", "pattern", "combined"
    details: dict = field(default_factory=dict)


@dataclass
class VoiceFeatures:
    rms_energy: float = 0.0
    speech_rate_wps: float = 0.0     # words per second
    pause_ratio: float = 0.0         # fraction of time silent
    pitch_variance: float = 0.0
    duration_sec: float = 0.0


class MoodEngine:
    """
    Multi-source mood tracker that adapts Nova's behavior.
    """

    def __init__(self):
        self._history: deque = deque(maxlen=config.MOOD_HISTORY_WINDOW * 3)
        self._current_mood = Mood.NEUTRAL
        self._current_confidence = 0.5
        self._last_update = time.time()

        # Baselines (learned over time)
        self._energy_baseline = 400.0
        self._rate_baseline = 2.5     # words/sec
        self._energy_samples: deque = deque(maxlen=100)

        log.info("Mood engine initialized")

    # ─── Input Methods ────────────────────────────────────────────────────────

    def analyze_voice(self, pcm_data: bytes, transcript: str) -> Optional[MoodReading]:
        """
        Analyze voice audio + transcript for mood signals.
        Returns a MoodReading.
        """
        features = self._extract_voice_features(pcm_data, transcript)
        mood, confidence = self._voice_to_mood(features)

        self._energy_samples.append(features.rms_energy)
        self._update_baseline()

        reading = MoodReading(
            mood=mood,
            confidence=confidence,
            timestamp=time.time(),
            source="voice",
            details={
                "energy": round(features.rms_energy, 1),
                "rate": round(features.speech_rate_wps, 2),
                "duration": round(features.duration_sec, 1),
            }
        )
        self._add_reading(reading)
        return reading

    def analyze_face(self, emotion: str, confidence: float) -> Optional[MoodReading]:
        """Ingest face emotion from camera system."""
        face_to_mood = {
            "happy":     (Mood.HAPPY,      0.8),
            "surprise":  (Mood.ENERGIZED,  0.6),
            "neutral":   (Mood.NEUTRAL,    0.7),
            "sad":       (Mood.TIRED,      0.6),
            "angry":     (Mood.FRUSTRATED, 0.7),
            "fear":      (Mood.STRESSED,   0.7),
            "disgust":   (Mood.FRUSTRATED, 0.6),
        }
        mood_conf = face_to_mood.get(emotion.lower(), (Mood.NEUTRAL, 0.4))

        reading = MoodReading(
            mood=mood_conf[0],
            confidence=mood_conf[1] * confidence,
            timestamp=time.time(),
            source="face",
            details={"face_emotion": emotion, "face_confidence": confidence}
        )
        self._add_reading(reading)
        return reading

    # ─── Analysis ─────────────────────────────────────────────────────────────

    def _extract_voice_features(self, pcm_data: bytes, transcript: str) -> VoiceFeatures:
        """Extract acoustic features from raw PCM audio."""
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        duration = len(audio) / config.SAMPLE_RATE

        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Speech rate
        word_count = len(transcript.split()) if transcript else 0
        rate = word_count / max(duration, 0.1)

        # Pause ratio (frames below threshold = silence)
        frame_size = 1024
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        silence_frames = sum(1 for f in frames if len(f) == frame_size and
                             np.sqrt(np.mean(f**2)) < config.VAD_ENERGY_THRESHOLD * 0.5)
        pause_ratio = silence_frames / max(len(frames), 1)

        # Pitch variance (using zero-crossing rate as proxy)
        zcr_frames = []
        for frame in frames:
            if len(frame) == frame_size:
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
                zcr_frames.append(zcr)
        pitch_var = float(np.std(zcr_frames)) if zcr_frames else 0.0

        return VoiceFeatures(
            rms_energy=rms,
            speech_rate_wps=rate,
            pause_ratio=pause_ratio,
            pitch_variance=pitch_var,
            duration_sec=duration,
        )

    def _voice_to_mood(self, features: VoiceFeatures) -> tuple:
        """Map voice features to mood classification."""
        energy_ratio = features.rms_energy / max(self._energy_baseline, 1)
        rate = features.speech_rate_wps

        # Stressed: fast + high energy + low pauses
        if rate > 3.5 and energy_ratio > 1.4 and features.pause_ratio < 0.3:
            return Mood.STRESSED, 0.7

        # Energized: fast + normal/high energy
        if rate > 3.0 and energy_ratio > 1.1:
            return Mood.ENERGIZED, 0.65

        # Tired: slow + low energy + many pauses
        if rate < 1.5 and energy_ratio < 0.7 and features.pause_ratio > 0.5:
            return Mood.TIRED, 0.65

        # Focused: slow deliberate speech (medium-low rate, normal energy)
        if 1.5 <= rate <= 2.5 and 0.8 <= energy_ratio <= 1.2 and features.pause_ratio < 0.4:
            return Mood.FOCUSED, 0.55

        # Happy: high energy, high pitch variance
        if energy_ratio > 1.2 and features.pitch_variance > 0.05:
            return Mood.HAPPY, 0.55

        return Mood.NEUTRAL, 0.5

    def _update_baseline(self):
        """Adapt energy baseline to user's typical voice."""
        if len(self._energy_samples) >= 20:
            samples = list(self._energy_samples)
            # Use 40th percentile as baseline (typical neutral speech)
            self._energy_baseline = float(np.percentile(samples, 40))

    def _add_reading(self, reading: MoodReading):
        """Add reading to history and recompute current mood."""
        self._history.append(reading)
        self._recompute_current()

    def _recompute_current(self):
        """Weighted vote across recent readings."""
        if not self._history:
            return

        now = time.time()
        votes: Dict[Mood, float] = {}

        for reading in self._history:
            age = now - reading.timestamp
            # Decay older readings
            weight = reading.confidence * np.exp(-age / 120)  # 2-min half-life
            # Source weights
            source_w = {"voice": 1.0, "face": 0.8, "pattern": 0.5, "combined": 1.2}
            weight *= source_w.get(reading.source, 0.5)

            votes[reading.mood] = votes.get(reading.mood, 0) + weight

        if votes:
            best_mood = max(votes, key=votes.get)
            total = sum(votes.values())
            self._current_mood = best_mood
            self._current_confidence = votes[best_mood] / max(total, 0.001)

    # ─── Accessors ────────────────────────────────────────────────────────────

    @property
    def current_mood(self) -> Mood:
        return self._current_mood

    @property
    def current_mood_name(self) -> str:
        return self._current_mood.value

    @property
    def confidence(self) -> float:
        return self._current_confidence

    def get_nova_adaptation(self) -> dict:
        """
        Get instructions for how Nova should adapt to current mood.
        Returns dict of behavioral adjustments.
        """
        adaptations = {
            Mood.FOCUSED: {
                "verbosity": "minimal",
                "tone": "precise and efficient",
                "proactive": False,
                "speech_speed_multiplier": 1.05,
                "hint": "User is in deep focus. Be very brief. Skip pleasantries.",
            },
            Mood.ENERGIZED: {
                "verbosity": "normal",
                "tone": "match their energy, be upbeat",
                "proactive": True,
                "speech_speed_multiplier": 1.1,
                "hint": "User is energized. Match their pace. Be enthusiastic.",
            },
            Mood.TIRED: {
                "verbosity": "concise",
                "tone": "gentle and warm",
                "proactive": False,
                "speech_speed_multiplier": 0.95,
                "hint": "User seems tired. Be gentle, clear, and supportive.",
            },
            Mood.STRESSED: {
                "verbosity": "minimal",
                "tone": "calm and grounding",
                "proactive": False,
                "speech_speed_multiplier": 0.92,
                "hint": "User appears stressed. Stay calm. Be direct. Offer help proactively.",
            },
            Mood.FRUSTRATED: {
                "verbosity": "empathetic first",
                "tone": "understanding, solution-focused",
                "proactive": False,
                "speech_speed_multiplier": 0.95,
                "hint": "User seems frustrated. Acknowledge it briefly, then focus on solutions.",
            },
            Mood.HAPPY: {
                "verbosity": "normal",
                "tone": "warm and engaged",
                "proactive": True,
                "speech_speed_multiplier": 1.05,
                "hint": "User is in a good mood. Be warm and engaged.",
            },
            Mood.NEUTRAL: {
                "verbosity": "normal",
                "tone": "professional and friendly",
                "proactive": True,
                "speech_speed_multiplier": 1.0,
                "hint": "",
            },
        }
        return adaptations.get(self._current_mood, adaptations[Mood.NEUTRAL])

    def get_mood_context_str(self) -> str:
        """Get mood context for LLM system prompt injection."""
        adaptation = self.get_nova_adaptation()
        hint = adaptation.get("hint", "")
        if not hint:
            return ""
        return f"\n## Current User Mood: {self._current_mood.value.title()}\n{hint}"

    def get_trend(self) -> str:
        """Describe mood trend over session."""
        if len(self._history) < 4:
            return "neutral"
        recent = list(self._history)[-4:]
        moods = [r.mood for r in recent]
        stressed_count = moods.count(Mood.STRESSED) + moods.count(Mood.FRUSTRATED)
        positive_count = moods.count(Mood.HAPPY) + moods.count(Mood.ENERGIZED)
        if stressed_count >= 2:
            return "declining"
        if positive_count >= 2:
            return "improving"
        return "stable"