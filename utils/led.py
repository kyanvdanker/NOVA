"""
NOVA LED Controller
Controls the LED ring in the cube housing.
- Raspberry Pi: GPIO via rpi_ws281x (NeoPixel)
- USB: serial protocol for standalone LED rings
- Fallback: terminal ANSI color indicator
Supports breathing animations, color cycling, pulse effects.
"""

import time
import threading
import logging
import math
from typing import Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.led")

Color = Tuple[int, int, int]  # RGB 0-255


class LEDController:
    """LED ring controller with animation support."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._strip = None

        # Animation state
        self._current_mode = "off"
        self._target_color: Color = (0, 0, 0)
        self._animation_speed = 1.0

        if config.LED_ENABLED:
            self._init_hardware()

    def _init_hardware(self):
        """Initialize LED hardware."""
        if config.LED_TYPE == "gpio":
            self._init_neopixel()
        elif config.LED_TYPE == "usb":
            self._init_usb()
        else:
            log.info("LED type 'none' — terminal mode only")

    def _init_neopixel(self):
        """Initialize NeoPixel strip via rpi_ws281x."""
        try:
            from rpi_ws281x import PixelStrip, Color as WsColor
            self._strip = PixelStrip(
                config.LED_COUNT,
                config.LED_GPIO_PIN,
                800000,  # LED signal frequency
                10,      # DMA channel
                False,   # Invert signal
                255,     # Brightness
                0,       # Channel
            )
            self._strip.begin()
            self._WsColor = WsColor
            log.info(f"NeoPixel LED ring initialized ({config.LED_COUNT} LEDs)")
        except ImportError:
            log.info("rpi_ws281x not available (not on Pi?)")
        except Exception as e:
            log.warning(f"LED init failed: {e}")

    def _init_usb(self):
        """Initialize USB LED ring via serial."""
        try:
            import serial
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                if "Arduino" in port.description or "CH340" in port.description:
                    self._serial = serial.Serial(port.device, 115200, timeout=1)
                    log.info(f"USB LED ring connected on {port.device}")
                    return
        except ImportError:
            pass

    # ─── State Methods ────────────────────────────────────────────────────────

    def set_state(self, state: str):
        """Set LED state matching Nova's operational state."""
        colors = config.LED_COLORS
        color = colors.get(state, colors.get("off", (0, 0, 0)))

        animations = {
            "sleeping":  ("breathe", 0.4),
            "listening": ("pulse", 1.2),
            "thinking":  ("spin", 1.5),
            "speaking":  ("wave", 1.0),
            "alert":     ("flash", 2.0),
            "camera":    ("breathe", 0.6),
        }
        anim, speed = animations.get(state, ("solid", 1.0))
        self.animate(color, anim, speed)

    def animate(self, color: Color, mode: str = "solid", speed: float = 1.0):
        """Start an animation."""
        with self._lock:
            self._target_color = color
            self._current_mode = mode
            self._animation_speed = speed

        if not self._running and config.LED_ENABLED:
            self._running = True
            self._thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._thread.start()

        # Terminal fallback
        self._terminal_indicator(mode, color)

    def off(self):
        """Turn LEDs off."""
        with self._lock:
            self._current_mode = "off"
            self._target_color = (0, 0, 0)
        self._set_all((0, 0, 0))

    # ─── Animation Loop ───────────────────────────────────────────────────────

    def _animation_loop(self):
        """Render animation frames."""
        t = 0.0
        while self._running:
            with self._lock:
                mode = self._current_mode
                color = self._target_color
                speed = self._animation_speed

            if mode == "off":
                self._set_all((0, 0, 0))
                time.sleep(0.1)
                continue

            try:
                if mode == "solid":
                    self._set_all(color)
                    time.sleep(0.1)

                elif mode == "breathe":
                    # Sinusoidal brightness breathing
                    brightness = (math.sin(t * speed * math.pi) + 1) / 2
                    c = tuple(int(v * brightness) for v in color)
                    self._set_all(c)
                    t += 0.05
                    time.sleep(0.05)

                elif mode == "pulse":
                    # Fast pulse
                    brightness = max(0, math.sin(t * speed * math.pi * 2))
                    c = tuple(int(v * brightness) for v in color)
                    self._set_all(c)
                    t += 0.04
                    time.sleep(0.04)

                elif mode == "spin":
                    # Rotating dot around the ring
                    if self._strip:
                        n = config.LED_COUNT
                        pos = int(t * speed * n) % n
                        for i in range(n):
                            dist = min(abs(i - pos), n - abs(i - pos))
                            brightness = max(0, 1 - dist / 3.0)
                            c = tuple(int(v * brightness) for v in color)
                            self._set_pixel(i, c)
                        self._strip.show()
                    t += 0.02
                    time.sleep(0.02)

                elif mode == "wave":
                    # Wave ripple
                    if self._strip:
                        n = config.LED_COUNT
                        for i in range(n):
                            phase = (i / n) * 2 * math.pi
                            brightness = (math.sin(t * speed * math.pi * 2 + phase) + 1) / 2
                            c = tuple(int(v * brightness) for v in color)
                            self._set_pixel(i, c)
                        self._strip.show()
                    t += 0.03
                    time.sleep(0.03)

                elif mode == "flash":
                    on = int(t * speed * 2) % 2 == 0
                    self._set_all(color if on else (0, 0, 0))
                    t += 0.1
                    time.sleep(0.1)

            except Exception as e:
                log.debug(f"LED animation error: {e}")
                time.sleep(0.1)

    def _set_all(self, color: Color):
        """Set all LEDs to a color."""
        if self._strip:
            try:
                c = self._WsColor(*color)
                for i in range(config.LED_COUNT):
                    self._strip.setPixelColor(i, c)
                self._strip.show()
            except Exception:
                pass

    def _set_pixel(self, index: int, color: Color):
        """Set a single pixel."""
        if self._strip:
            try:
                self._strip.setPixelColor(index, self._WsColor(*color))
            except Exception:
                pass

    def _terminal_indicator(self, mode: str, color: Color):
        """Show a colored indicator in terminal when no hardware available."""
        if config.LED_ENABLED:
            return  # Hardware handles it

        state_icons = {
            "sleeping":  "💤",
            "listening": "👂",
            "thinking":  "🧠",
            "speaking":  "🔊",
            "alert":     "⚠️",
            "camera":    "👁️",
            "off":       "⚫",
            "solid":     "●",
            "breathe":   "◌",
            "pulse":     "●",
        }
        icon = state_icons.get(mode, "●")
        r, g, b = color
        # ANSI 256 color approximation
        print(f"\r[NOVA] {icon} ", end="", flush=True)

    def stop(self):
        self._running = False
        self.off()