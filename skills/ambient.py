"""
NOVA Ambient Intelligence
Monitors the environment and digital context:
- Weather (real-time, via Open-Meteo — free, no API key)
- Active window tracking (what app/file is in focus)
- Clipboard monitoring (optional)
- Time-of-day context
- System health monitoring
- Network status
"""

import time
import subprocess
import platform
import threading
import logging
import json
from typing import Optional, Dict, Callable
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.ambient")

SYSTEM = platform.system()


class AmbientIntelligence:
    """
    Passive environment monitor. Feeds context into Nova's brain.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State
        self._weather: Optional[dict] = None
        self._weather_last_fetch = 0.0
        self._active_window: Optional[dict] = None
        self._clipboard_last: str = ""
        self._network_ok = True

        # Callbacks
        self.on_window_change: Optional[Callable[[dict], None]] = None
        self.on_clipboard_change: Optional[Callable[[str], None]] = None

        # Cache
        self._location: Optional[dict] = None

        log.info("Ambient intelligence initialized")

    def start(self):
        """Start ambient monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Initial weather fetch in background
        threading.Thread(target=self._fetch_weather, daemon=True).start()
        log.info("Ambient monitoring started")

    def stop(self):
        self._running = False

    def _run_loop(self):
        window_counter = 0
        clipboard_counter = 0

        while self._running:
            # Window tracking
            if config.WINDOW_TRACKING:
                window_counter += 1
                if window_counter >= int(config.WINDOW_POLL_SEC / 0.5):
                    self._update_active_window()
                    window_counter = 0

            # Clipboard monitoring
            if config.CLIPBOARD_MONITOR:
                clipboard_counter += 1
                if clipboard_counter >= int(config.CLIPBOARD_POLL_SEC / 0.5):
                    self._check_clipboard()
                    clipboard_counter = 0

            # Weather refresh every 30 minutes
            if time.time() - self._weather_last_fetch > 1800:
                threading.Thread(target=self._fetch_weather, daemon=True).start()

            time.sleep(0.5)

    # ─── Weather ──────────────────────────────────────────────────────────────

    def _fetch_weather(self):
        """Fetch weather from Open-Meteo (free, no API key needed)."""
        try:
            import requests

            # Get location from IP if auto
            if not self._location:
                self._location = self._get_location()

            if not self._location:
                return

            lat = self._location.get("lat", 52.3)
            lon = self._location.get("lon", 4.9)
            city = self._location.get("city", "")

            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,"
                f"apparent_temperature,weather_code,wind_speed_10m"
                f"&hourly=temperature_2m,precipitation_probability"
                f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
                f"precipitation_sum,sunrise,sunset"
                f"&forecast_days=2"
                f"&timezone=auto"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                daily = data.get("daily", {})

                self._weather = {
                    "city": city,
                    "temp_c": current.get("temperature_2m"),
                    "feels_like_c": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_kmh": current.get("wind_speed_10m"),
                    "code": current.get("weather_code"),
                    "description": self._weather_code_to_text(current.get("weather_code", 0)),
                    "sunrise": daily.get("sunrise", [""])[0] if daily.get("sunrise") else "",
                    "sunset": daily.get("sunset", [""])[0] if daily.get("sunset") else "",
                    "today_max": daily.get("temperature_2m_max", [None])[0],
                    "today_min": daily.get("temperature_2m_min", [None])[0],
                    "rain_chance": self._get_rain_chance(data),
                    "fetched_at": time.time(),
                }
                self._weather_last_fetch = time.time()
                log.info(f"Weather: {self._weather['description']}, {self._weather['temp_c']}°C")
        except Exception as e:
            log.debug(f"Weather fetch failed: {e}")

    def _get_location(self) -> Optional[dict]:
        """Get approximate location from IP."""
        try:
            import requests
            resp = requests.get("https://ipapi.co/json/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "lat": data.get("latitude"),
                    "lon": data.get("longitude"),
                    "city": data.get("city", ""),
                    "country": data.get("country_name", ""),
                }
        except Exception:
            pass

        # Hardcoded fallback (Netherlands based on user location)
        return {"lat": 51.9, "lon": 5.2, "city": "Gelderland", "country": "Netherlands"}

    def _weather_code_to_text(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "fog", 48: "icy fog",
            51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
            61: "light rain", 63: "moderate rain", 65: "heavy rain",
            71: "light snow", 73: "moderate snow", 75: "heavy snow",
            80: "light showers", 81: "moderate showers", 82: "heavy showers",
            95: "thunderstorm", 96: "thunderstorm with hail",
        }
        return codes.get(code, "unknown conditions")

    def _get_rain_chance(self, data: dict) -> Optional[int]:
        """Extract highest rain probability for next 8 hours."""
        try:
            probs = data.get("hourly", {}).get("precipitation_probability", [])[:8]
            return max(probs) if probs else None
        except Exception:
            return None

    def get_weather_summary(self) -> Optional[str]:
        """Get a one-line weather summary."""
        if not self._weather:
            return None
        w = self._weather
        summary = f"{w['description']}, {w['temp_c']}°C"
        if w.get("rain_chance") and w["rain_chance"] > 40:
            summary += f", {w['rain_chance']}% chance of rain"
        if w.get("city"):
            summary = f"{w['city']}: {summary}"
        return summary

    def get_weather_dict(self) -> Optional[dict]:
        return self._weather

    # ─── Active Window ────────────────────────────────────────────────────────

    def _update_active_window(self):
        """Get the currently focused window title and app."""
        try:
            if SYSTEM == "Linux":
                info = self._get_active_window_linux()
            elif SYSTEM == "Darwin":
                info = self._get_active_window_mac()
            elif SYSTEM == "Windows":
                info = self._get_active_window_windows()
            else:
                return

            if info and info != self._active_window:
                prev = self._active_window
                self._active_window = info
                if prev and self.on_window_change:
                    threading.Thread(
                        target=self.on_window_change,
                        args=(info,),
                        daemon=True
                    ).start()
        except Exception:
            pass

    def _get_active_window_linux(self) -> Optional[dict]:
        """Get active window on Linux using xdotool."""
        try:
            wid = subprocess.check_output(
                ["xdotool", "getactivewindow"], timeout=2
            ).decode().strip()
            name = subprocess.check_output(
                ["xdotool", "getwindowname", wid], timeout=2
            ).decode().strip()
            pid = subprocess.check_output(
                ["xdotool", "getwindowpid", wid], timeout=2
            ).decode().strip()
            app = subprocess.check_output(
                ["ps", "-p", pid, "-o", "comm="], timeout=2
            ).decode().strip() if pid else ""
            return {"title": name, "app": app, "pid": pid}
        except Exception:
            pass

        # Fallback: wmctrl
        try:
            out = subprocess.check_output(["wmctrl", "-a", ":ACTIVE:"],
                                          timeout=2).decode()
            return {"title": out.strip()[:100], "app": "", "pid": ""}
        except Exception:
            return None

    def _get_active_window_mac(self) -> Optional[dict]:
        script = '''
tell application "System Events"
    set frontApp to name of first process whose frontmost is true
    set frontWindow to ""
    try
        set frontWindow to name of front window of (first process whose frontmost is true)
    end try
    return frontApp & "|" & frontWindow
end tell'''
        try:
            out = subprocess.check_output(
                ["osascript", "-e", script], timeout=3
            ).decode().strip()
            parts = out.split("|", 1)
            return {"app": parts[0], "title": parts[1] if len(parts) > 1 else "", "pid": ""}
        except Exception:
            return None

    def _get_active_window_windows(self) -> Optional[dict]:
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return {"title": buf.value, "app": "", "pid": ""}
        except Exception:
            return None

    @property
    def active_window(self) -> Optional[dict]:
        return self._active_window

    def get_active_window_context(self) -> str:
        """Get a string describing what the user is working on."""
        if not self._active_window:
            return ""
        w = self._active_window
        title = w.get("title", "")
        app = w.get("app", "")
        if title and app:
            return f"Currently using: {app} — {title[:60]}"
        elif title:
            return f"Current window: {title[:60]}"
        return ""

    # ─── Clipboard ────────────────────────────────────────────────────────────

    def _check_clipboard(self):
        """Monitor clipboard for changes."""
        try:
            import pyperclip
            current = pyperclip.paste()
            if current and current != self._clipboard_last and len(current) < 2000:
                self._clipboard_last = current
                if self.on_clipboard_change:
                    threading.Thread(
                        target=self.on_clipboard_change,
                        args=(current,),
                        daemon=True
                    ).start()
        except Exception:
            pass

    # ─── Time Context ─────────────────────────────────────────────────────────

    def get_time_context(self) -> dict:
        """Rich time context for LLM."""
        now = datetime.now()
        hour = now.hour

        if 5 <= hour < 9:
            period = "early morning"
            energy = "gentle"
        elif 9 <= hour < 12:
            period = "morning"
            energy = "focused"
        elif 12 <= hour < 14:
            period = "midday"
            energy = "transition"
        elif 14 <= hour < 17:
            period = "afternoon"
            energy = "sustained"
        elif 17 <= hour < 20:
            period = "early evening"
            energy = "winding down"
        elif 20 <= hour < 23:
            period = "evening"
            energy = "relaxed"
        else:
            period = "late night"
            energy = "quiet"

        return {
            "datetime_str": now.strftime("%A, %B %d %Y, %I:%M %p"),
            "period": period,
            "energy": energy,
            "is_weekend": now.weekday() >= 5,
            "day_name": now.strftime("%A"),
        }

    def build_context_string(self) -> str:
        """Build a rich context string for LLM system prompt."""
        parts = []
        tc = self.get_time_context()
        parts.append(f"Time: {tc['datetime_str']} ({tc['period']})")
        w = self.get_weather_summary()
        if w:
            parts.append(f"Weather: {w}")
        win = self.get_active_window_context()
        if win:
            parts.append(win)
        return "\n".join(parts)