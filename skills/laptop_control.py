"""
NOVA Laptop Control
Gives Nova the ability to:
- Open/close applications
- Manage files and folders
- Take screenshots (for vision)
- Control system settings
- Run shell commands (with safety checks)
- Search the filesystem
- Clipboard management
"""

import subprocess
import platform
import shutil
import os
import glob
import time
import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.laptop")

SYSTEM = platform.system()  # 'Linux', 'Darwin', 'Windows'

# Apps that are SAFE to open/close without confirmation
SAFE_APPS = {
    # Browsers
    "browser", "firefox", "chrome", "chromium", "safari", "edge",
    # Productivity
    "terminal", "calculator", "calendar", "clock", "notes",
    "gedit", "kate", "mousepad", "text editor",
    # Media
    "music", "vlc", "spotify", "rhythmbox",
    # Dev tools
    "code", "vscode", "sublime", "atom", "cursor",
    # Communication
    "slack", "discord", "telegram",
    # Files
    "files", "nautilus", "finder", "explorer",
}

# Commands that REQUIRE confirmation before execution
DANGEROUS_PATTERNS = ["rm -rf", "rmdir", "format", "mkfs", "dd if=", "> /dev/",
                       "DROP TABLE", "DELETE FROM", "sudo rm"]


class LaptopControl:
    """Controls the laptop/desktop system."""

    def __init__(self):
        self._confirm_callback = None  # Set by main to ask user for confirmation

    def set_confirm_callback(self, callback):
        """Set callback for dangerous operations. callback(action) -> bool"""
        self._confirm_callback = callback

    # ─── Applications ─────────────────────────────────────────────────────────

    def open_app(self, app_name: str) -> Tuple[bool, str]:
        """Open an application by name."""
        app_lower = app_name.lower().strip()
        log.info(f"Opening app: {app_name}")

        try:
            if SYSTEM == "Linux":
                result = self._open_app_linux(app_lower)
            elif SYSTEM == "Darwin":
                result = self._open_app_mac(app_lower)
            elif SYSTEM == "Windows":
                result = self._open_app_windows(app_lower)
            else:
                return False, f"Unsupported OS: {SYSTEM}"

            if result:
                return True, f"Opening {app_name}"
            return False, f"Couldn't find {app_name}"
        except Exception as e:
            log.error(f"Open app error: {e}")
            return False, str(e)

    def _open_app_linux(self, app: str) -> bool:
        # Try direct command
        candidates = [
            app,
            app.replace(" ", "-"),
            app.replace(" ", ""),
            # Common aliases
            {"browser": "xdg-open https://", "files": "nautilus",
             "terminal": "gnome-terminal", "text editor": "gedit",
             "code": "code", "vscode": "code"}.get(app, app)
        ]
        for cmd in candidates:
            try:
                subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                continue
        # Try xdg-open
        try:
            subprocess.Popen(["xdg-open", app], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _open_app_mac(self, app: str) -> bool:
        # Map common names to macOS apps
        app_map = {
            "browser": "Safari",
            "chrome": "Google Chrome",
            "firefox": "Firefox",
            "terminal": "Terminal",
            "code": "Visual Studio Code",
            "vscode": "Visual Studio Code",
            "files": "Finder",
            "music": "Music",
            "calculator": "Calculator",
            "notes": "Notes",
        }
        app_name = app_map.get(app, app.title())
        try:
            subprocess.Popen(["open", "-a", app_name],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            # Try open directly
            try:
                subprocess.Popen(["open", app], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return True
            except Exception:
                return False

    def _open_app_windows(self, app: str) -> bool:
        app_map = {
            "browser": "start chrome",
            "chrome": "start chrome",
            "firefox": "start firefox",
            "terminal": "start cmd",
            "notepad": "start notepad",
            "calculator": "start calc",
            "files": "start explorer",
        }
        cmd = app_map.get(app, f"start {app}")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def close_app(self, app_name: str) -> Tuple[bool, str]:
        """Close/kill an application."""
        try:
            if SYSTEM == "Linux":
                subprocess.run(["pkill", "-f", app_name], capture_output=True)
            elif SYSTEM == "Darwin":
                subprocess.run(["pkill", "-f", app_name], capture_output=True)
            elif SYSTEM == "Windows":
                subprocess.run(["taskkill", "/F", "/IM", f"{app_name}.exe"],
                               capture_output=True)
            return True, f"Closed {app_name}"
        except Exception as e:
            return False, str(e)

    # ─── File Operations ──────────────────────────────────────────────────────

    def open_file(self, path: str) -> Tuple[bool, str]:
        """Open a file with its default application."""
        p = Path(path).expanduser()
        if not p.exists():
            # Try to find it
            found = self.find_file(p.name)
            if found:
                p = Path(found[0])
            else:
                return False, f"File not found: {path}"

        try:
            if SYSTEM == "Linux":
                subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            elif SYSTEM == "Darwin":
                subprocess.Popen(["open", str(p)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            elif SYSTEM == "Windows":
                os.startfile(str(p))
            return True, f"Opened {p.name}"
        except Exception as e:
            return False, str(e)

    def find_file(self, filename: str, search_dirs: List[str] = None) -> List[str]:
        """Search for a file on the system."""
        if search_dirs is None:
            search_dirs = [
                str(Path.home()),
                str(Path.home() / "Documents"),
                str(Path.home() / "Desktop"),
                str(Path.home() / "Downloads"),
                str(Path.home() / "Projects"),
            ]

        results = []
        for search_dir in search_dirs:
            if Path(search_dir).exists():
                for match in Path(search_dir).rglob(f"*{filename}*"):
                    results.append(str(match))
                    if len(results) >= 10:
                        break

        return results

    def list_directory(self, path: str = "~") -> List[str]:
        """List contents of a directory."""
        p = Path(path).expanduser()
        if not p.exists():
            return []
        try:
            items = []
            for item in sorted(p.iterdir()):
                if item.name.startswith("."):
                    continue
                suffix = "/" if item.is_dir() else ""
                items.append(f"{item.name}{suffix}")
            return items
        except PermissionError:
            return ["[Permission denied]"]

    def create_file_at(self, path: str, content: str = "") -> Tuple[bool, str]:
        """Create a new file."""
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return True, f"Created {p.name}"
        except Exception as e:
            return False, str(e)

    # ─── Screenshot / Vision ──────────────────────────────────────────────────

    def take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return as base64 string."""
        try:
            import mss
            import mss.tools
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                screenshot = sct.grab(monitor)
                # Convert to PNG bytes
                from PIL import Image
                import io
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                # Resize for efficiency (max 1280px wide)
                max_width = 1280
                if img.width > max_width:
                    ratio = max_width / img.width
                    img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            log.warning("mss/PIL not installed for screenshots")
            # Fallback to scrot on Linux
            try:
                tmp = config.AUDIO_TEMP_DIR / "screenshot.png"
                subprocess.run(["scrot", str(tmp)], capture_output=True, timeout=5)
                if tmp.exists():
                    with open(tmp, "rb") as f:
                        return base64.b64encode(f.read()).decode()
            except Exception:
                pass
        except Exception as e:
            log.error(f"Screenshot failed: {e}")
        return None

    # ─── System Control ───────────────────────────────────────────────────────

    def set_volume(self, level: int) -> Tuple[bool, str]:
        """Set system volume (0-100)."""
        level = max(0, min(100, level))
        try:
            if SYSTEM == "Linux":
                subprocess.run(["amixer", "set", "Master", f"{level}%"],
                               capture_output=True)
            elif SYSTEM == "Darwin":
                subprocess.run(["osascript", "-e",
                                f"set volume output volume {level}"],
                               capture_output=True)
            elif SYSTEM == "Windows":
                # Requires nircmd or similar
                pass
            return True, f"Volume set to {level}%"
        except Exception as e:
            return False, str(e)

    def get_system_info(self) -> dict:
        """Get system stats."""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.5),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
                "disk_percent": psutil.disk_usage("/").percent,
                "battery": psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None,
            }
        except ImportError:
            return {"error": "psutil not installed"}

    def run_command(self, command: str, require_confirm: bool = True) -> Tuple[bool, str]:
        """Run a shell command with safety checks."""
        # Safety check
        for danger in DANGEROUS_PATTERNS:
            if danger in command:
                log.warning(f"Dangerous command blocked: {command}")
                return False, f"That command is blocked for safety: contains '{danger}'"

        if require_confirm and self._confirm_callback:
            if not self._confirm_callback(f"Run command: {command}"):
                return False, "Command cancelled"

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout or result.stderr or "Done"
            return result.returncode == 0, output[:500]
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    # ─── Clipboard ────────────────────────────────────────────────────────────

    def get_clipboard(self) -> str:
        """Get clipboard contents."""
        try:
            import pyperclip
            return pyperclip.paste()
        except Exception:
            try:
                if SYSTEM == "Linux":
                    result = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                                            capture_output=True, text=True)
                    return result.stdout
                elif SYSTEM == "Darwin":
                    result = subprocess.run(["pbpaste"], capture_output=True, text=True)
                    return result.stdout
            except Exception:
                return ""

    def set_clipboard(self, text: str):
        """Set clipboard contents."""
        try:
            import pyperclip
            pyperclip.copy(text)
        except Exception:
            try:
                if SYSTEM == "Linux":
                    proc = subprocess.Popen(["xclip", "-selection", "clipboard"],
                                            stdin=subprocess.PIPE)
                    proc.communicate(text.encode())
                elif SYSTEM == "Darwin":
                    proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                    proc.communicate(text.encode())
            except Exception as e:
                log.warning(f"Clipboard set failed: {e}")

    # ─── Notifications ────────────────────────────────────────────────────────

    def send_notification(self, title: str, message: str, urgency: str = "normal"):
        """Send a desktop notification."""
        try:
            if SYSTEM == "Linux":
                subprocess.Popen(
                    ["notify-send", "-u", urgency, title, message],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            elif SYSTEM == "Darwin":
                script = (f'display notification "{message}" '
                          f'with title "{title}"')
                subprocess.Popen(["osascript", "-e", script],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif SYSTEM == "Windows":
                # Use win10toast if available
                try:
                    from win10toast import ToastNotifier
                    ToastNotifier().show_toast(title, message, duration=5)
                except ImportError:
                    pass
        except Exception as e:
            log.warning(f"Notification failed: {e}")