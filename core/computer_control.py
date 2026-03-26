"""
Computer Control — mouse, keyboard, screenshots, app launching, OCR.
"""
import os
import subprocess
import platform
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import SCREENSHOT_DIR, MOUSE_MOVE_DURATION


class ComputerControl:
    def __init__(self):
        self.system = platform.system()  # Windows / Linux / Darwin
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def execute(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch computer control actions."""
        handlers = {
            "screenshot": self._screenshot,
            "click": self._click,
            "double_click": self._double_click,
            "right_click": self._right_click,
            "move_mouse": self._move_mouse,
            "type_text": self._type_text,
            "press_key": self._press_key,
            "hotkey": self._hotkey,
            "open_app": self._open_app,
            "run_command": self._run_command,
            "get_screen_text": self._get_screen_text,
            "scroll": self._scroll,
            "get_active_window": self._get_active_window,
            "get_clipboard": self._get_clipboard,
            "set_clipboard": self._set_clipboard,
            "get_system_info": self._get_system_info,
            "list_processes": self._list_processes,
            "kill_process": self._kill_process,
            "get_mouse_position": self._get_mouse_position,
        }

        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(handlers.keys())}

        try:
            return handler(**args)
        except Exception as e:
            return {"error": str(e), "action": action}

    def _screenshot(self, save: bool = True, region: Optional[list] = None, **_) -> Dict:
        try:
            import pyautogui
            from PIL import Image

            if region:
                img = pyautogui.screenshot(region=tuple(region))
            else:
                img = pyautogui.screenshot()

            if save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = SCREENSHOT_DIR / f"screenshot_{ts}.png"
                img.save(str(path))
                return {"success": True, "path": str(path),
                        "size": f"{img.width}x{img.height}"}
            else:
                return {"success": True, "size": f"{img.width}x{img.height}"}
        except ImportError:
            # Fallback for Linux
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOT_DIR / f"screenshot_{ts}.png"
            if self.system == "Linux":
                os.system(f"scrot '{path}' 2>/dev/null || import -window root '{path}' 2>/dev/null")
            return {"success": True, "path": str(path)}

    def _click(self, x: int = None, y: int = None, button: str = "left", **_) -> Dict:
        import pyautogui
        if x and y:
            pyautogui.click(x, y, button=button, duration=MOUSE_MOVE_DURATION)
        else:
            pyautogui.click(button=button)
        return {"success": True, "position": {"x": x, "y": y}}

    def _double_click(self, x: int = None, y: int = None, **_) -> Dict:
        import pyautogui
        if x and y:
            pyautogui.doubleClick(x, y, duration=MOUSE_MOVE_DURATION)
        else:
            pyautogui.doubleClick()
        return {"success": True}

    def _right_click(self, x: int = None, y: int = None, **_) -> Dict:
        import pyautogui
        if x and y:
            pyautogui.rightClick(x, y, duration=MOUSE_MOVE_DURATION)
        else:
            pyautogui.rightClick()
        return {"success": True}

    def _move_mouse(self, x: int, y: int, **_) -> Dict:
        import pyautogui
        pyautogui.moveTo(x, y, duration=MOUSE_MOVE_DURATION)
        return {"success": True, "moved_to": {"x": x, "y": y}}

    def _type_text(self, text: str, interval: float = 0.02, **_) -> Dict:
        import pyautogui
        pyautogui.typewrite(text, interval=interval)
        return {"success": True, "typed": text}

    def _press_key(self, key: str, **_) -> Dict:
        import pyautogui
        pyautogui.press(key)
        return {"success": True, "key": key}

    def _hotkey(self, keys: list, **_) -> Dict:
        import pyautogui
        pyautogui.hotkey(*keys)
        return {"success": True, "keys": keys}

    def _scroll(self, direction: str = "down", amount: int = 3, x: int = None, y: int = None, **_) -> Dict:
        import pyautogui
        clicks = amount if direction == "up" else -amount
        if x and y:
            pyautogui.scroll(clicks, x=x, y=y)
        else:
            pyautogui.scroll(clicks)
        return {"success": True, "direction": direction, "amount": amount}

    def _open_app(self, app: str, **_) -> Dict:
        """Open an application by name."""
        try:
            if self.system == "Windows":
                os.startfile(app)
            elif self.system == "Darwin":
                subprocess.Popen(["open", "-a", app])
            else:
                # Linux — try common methods
                subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"success": True, "opened": app}
        except Exception as e:
            # Try xdg-open
            try:
                subprocess.Popen(["xdg-open", app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"success": True, "opened": app, "method": "xdg-open"}
            except:
                return {"error": str(e)}

    def _run_command(self, cmd: str, timeout: int = 10, **_) -> Dict:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return {
                "success": True,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": str(e)}

    def _get_screen_text(self, region: Optional[list] = None, **_) -> Dict:
        """OCR the screen and return text."""
        try:
            import pyautogui
            import pytesseract

            if region:
                img = pyautogui.screenshot(region=tuple(region))
            else:
                img = pyautogui.screenshot()

            text = pytesseract.image_to_string(img)
            return {"success": True, "text": text.strip()}
        except ImportError:
            return {"error": "pytesseract not installed. Install: pip install pytesseract"}

    def _get_active_window(self, **_) -> Dict:
        try:
            if self.system == "Linux":
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True
                )
                return {"window": result.stdout.strip()}
            elif self.system == "Windows":
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return {"window": buf.value}
            else:
                return {"window": "unknown"}
        except Exception as e:
            return {"error": str(e)}

    def _get_clipboard(self, **_) -> Dict:
        try:
            import pyperclip
            return {"content": pyperclip.paste()}
        except:
            return {"error": "pyperclip not available"}

    def _set_clipboard(self, content: str, **_) -> Dict:
        try:
            import pyperclip
            pyperclip.copy(content)
            return {"success": True}
        except:
            return {"error": "pyperclip not available"}

    def _get_system_info(self, **_) -> Dict:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu_percent": cpu,
                "memory": {"total_gb": round(mem.total/1e9, 1),
                           "used_percent": mem.percent,
                           "available_gb": round(mem.available/1e9, 1)},
                "disk": {"total_gb": round(disk.total/1e9, 1),
                         "used_percent": disk.percent,
                         "free_gb": round(disk.free/1e9, 1)},
                "platform": self.system,
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1)
            }
        except ImportError:
            result = self._run_command("uname -a && free -h && df -h /")
            return result

    def _list_processes(self, **_) -> Dict:
        try:
            import psutil
            procs = []
            for p in sorted(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]),
                            key=lambda x: x.info.get("memory_percent") or 0, reverse=True)[:20]:
                procs.append(p.info)
            return {"processes": procs}
        except:
            result = self._run_command("ps aux --sort=-%mem | head -20")
            return result

    def _kill_process(self, pid: int = None, name: str = None, **_) -> Dict:
        try:
            import psutil
            if pid:
                p = psutil.Process(pid)
                p.terminate()
                return {"success": True, "killed_pid": pid}
            elif name:
                killed = []
                for p in psutil.process_iter(["pid", "name"]):
                    if name.lower() in p.info["name"].lower():
                        p.terminate()
                        killed.append(p.info["pid"])
                return {"success": True, "killed_pids": killed}
        except Exception as e:
            return {"error": str(e)}

    def _get_mouse_position(self, **_) -> Dict:
        try:
            import pyautogui
            x, y = pyautogui.position()
            return {"x": x, "y": y}
        except:
            return {"error": "pyautogui not available"}