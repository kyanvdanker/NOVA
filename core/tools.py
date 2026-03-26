"""
NOVA Tools — All built-in tool implementations.
Each tool function takes a dict of args and returns a dict result.
"""
import io
import os
import sys
import json
import time
import hashlib
import base64
import difflib
import re
import traceback
import subprocess
import platform
from typing import Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

import requests


# ─── CODE EXECUTION ───────────────────────────────────────────────────────────

def run_code_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Python code and capture output."""
    language = args.get("language", "python").lower()
    code = args.get("code", "")

    if language != "python":
        return {"error": f"Unsupported language: {language}. Only python is supported."}

    local_vars = {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        return {
            "success": True,
            "output": stdout_capture.getvalue().strip(),
            "errors": stderr_capture.getvalue().strip(),
            "locals": {k: repr(v) for k, v in local_vars.items() if k not in ("__builtins__",)},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ─── WEB SEARCH ───────────────────────────────────────────────────────────────

def search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """DuckDuckGo Instant Answer search."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing search query"}
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "t": "nova"},
            timeout=10,
        )
        data = resp.json()
        result = {
            "query": query,
            "abstract": data.get("AbstractText", ""),
            "abstract_url": data.get("AbstractURL", ""),
            "answer": data.get("Answer", ""),
            "answer_type": data.get("AnswerType", ""),
            "entity": data.get("Entity", ""),
            "related_topics": [],
        }
        for topic in data.get("RelatedTopics", [])[:8]:
            if "Text" in topic:
                result["related_topics"].append({"text": topic.get("Text"), "url": topic.get("FirstURL")})
            elif "Topics" in topic:
                for sub in topic.get("Topics", [])[:3]:
                    result["related_topics"].append({"text": sub.get("Text"), "url": sub.get("FirstURL")})
        return result
    except Exception as e:
        return {"error": str(e)}


# ─── WEATHER ──────────────────────────────────────────────────────────────────

def weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather for a city using wttr.in (no API key)."""
    city = args.get("city", args.get("location", "Amsterdam"))
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=8)
        data = resp.json()
        cur = data["current_condition"][0]
        today = data.get("weather", [{}])[0]
        return {
            "city": city,
            "temp_c": int(cur["temp_C"]),
            "temp_f": int(cur["temp_F"]),
            "feels_like_c": int(cur["FeelsLikeC"]),
            "description": cur["weatherDesc"][0]["value"],
            "humidity_pct": int(cur["humidity"]),
            "wind_kmph": int(cur["windspeedKmph"]),
            "wind_dir": cur["winddir16Point"],
            "visibility_km": int(cur["visibility"]),
            "uv_index": int(cur.get("uvIndex", 0)),
            "max_c": int(today.get("maxtempC", cur["temp_C"])),
            "min_c": int(today.get("mintempC", cur["temp_C"])),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── CALCULATOR ───────────────────────────────────────────────────────────────

def calculate_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Safe math expression evaluator with scientific functions."""
    import math
    expr = args.get("expression", args.get("expr", ""))
    if not expr:
        return {"error": "No expression provided"}
    try:
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals.update({
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "int": int, "float": float, "pow": pow,
        })
        result = eval(expr, {"__builtins__": {}}, safe_globals)
        return {"expression": expr, "result": result, "formatted": f"{result:g}" if isinstance(result, float) else str(result)}
    except Exception as e:
        return {"error": str(e), "expression": expr}


# ─── UNIT CONVERSION ──────────────────────────────────────────────────────────

def unit_convert_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between common units."""
    value = float(args.get("value", 0))
    from_unit = args.get("from", args.get("from_unit", "")).lower().strip()
    to_unit = args.get("to", args.get("to_unit", "")).lower().strip()

    conversions = {
        # Length
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("m", "inches"): lambda x: x * 39.3701,
        ("inches", "m"): lambda x: x * 0.0254,
        ("cm", "inches"): lambda x: x * 0.393701,
        ("inches", "cm"): lambda x: x * 2.54,
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        # Weight
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        # Temperature
        ("c", "f"): lambda x: x * 9/5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
        ("c", "k"): lambda x: x + 273.15,
        ("k", "c"): lambda x: x - 273.15,
        ("f", "k"): lambda x: (x - 32) * 5/9 + 273.15,
        # Volume
        ("l", "gal"): lambda x: x * 0.264172,
        ("gal", "l"): lambda x: x * 3.78541,
        ("ml", "l"): lambda x: x / 1000,
        ("l", "ml"): lambda x: x * 1000,
        ("ml", "fl_oz"): lambda x: x * 0.033814,
        ("fl_oz", "ml"): lambda x: x * 29.5735,
        # Speed
        ("kmh", "mph"): lambda x: x * 0.621371,
        ("mph", "kmh"): lambda x: x * 1.60934,
        ("ms", "kmh"): lambda x: x * 3.6,
        ("kmh", "ms"): lambda x: x / 3.6,
        # Data
        ("gb", "mb"): lambda x: x * 1024,
        ("mb", "gb"): lambda x: x / 1024,
        ("tb", "gb"): lambda x: x * 1024,
        ("gb", "tb"): lambda x: x / 1024,
        ("mb", "kb"): lambda x: x * 1024,
        ("kb", "mb"): lambda x: x / 1024,
        # Area
        ("m2", "ft2"): lambda x: x * 10.7639,
        ("ft2", "m2"): lambda x: x * 0.092903,
        ("km2", "miles2"): lambda x: x * 0.386102,
        ("miles2", "km2"): lambda x: x * 2.58999,
        # Time
        ("hours", "minutes"): lambda x: x * 60,
        ("minutes", "hours"): lambda x: x / 60,
        ("days", "hours"): lambda x: x * 24,
        ("hours", "days"): lambda x: x / 24,
        ("weeks", "days"): lambda x: x * 7,
        ("days", "weeks"): lambda x: x / 7,
    }

    key = (from_unit, to_unit)
    if key in conversions:
        result = conversions[key](value)
        return {"value": value, "from": from_unit, "to": to_unit, "result": round(result, 6)}

    return {
        "error": f"Conversion '{from_unit}' → '{to_unit}' not supported",
        "supported": [f"{a} → {b}" for a, b in conversions.keys()]
    }


# ─── HASH & CHECKSUM ──────────────────────────────────────────────────────────

def hash_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Hash text or a file."""
    text = args.get("text", "")
    file_path = args.get("file", "")
    algorithm = args.get("algorithm", "sha256").lower()

    valid_algos = ["md5", "sha1", "sha256", "sha512", "sha3_256", "blake2b"]
    if algorithm not in valid_algos:
        return {"error": f"Unknown algorithm. Use one of: {valid_algos}"}

    try:
        h = hashlib.new(algorithm)
        if file_path:
            p = Path(file_path).expanduser()
            if not p.exists():
                return {"error": f"File not found: {file_path}"}
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return {"file": str(p), "algorithm": algorithm, "hash": h.hexdigest()}
        elif text:
            h.update(text.encode("utf-8"))
            return {"text": text[:50] + "..." if len(text) > 50 else text,
                    "algorithm": algorithm, "hash": h.hexdigest()}
        else:
            return {"error": "Provide 'text' or 'file'"}
    except Exception as e:
        return {"error": str(e)}


# ─── ENCODE / DECODE ──────────────────────────────────────────────────────────

def encode_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Encode or decode data (base64, URL, hex, ROT13)."""
    text = args.get("text", "")
    mode = args.get("mode", "encode")  # encode | decode
    encoding = args.get("encoding", "base64").lower()

    try:
        if encoding == "base64":
            if mode == "encode":
                result = base64.b64encode(text.encode()).decode()
            else:
                result = base64.b64decode(text.encode()).decode()
        elif encoding == "url":
            from urllib.parse import quote, unquote
            result = quote(text) if mode == "encode" else unquote(text)
        elif encoding == "hex":
            if mode == "encode":
                result = text.encode().hex()
            else:
                result = bytes.fromhex(text).decode()
        elif encoding == "rot13":
            import codecs
            result = codecs.encode(text, "rot_13")
        elif encoding == "html":
            import html
            result = html.escape(text) if mode == "encode" else html.unescape(text)
        else:
            return {"error": f"Unknown encoding: {encoding}. Use: base64, url, hex, rot13, html"}

        return {"input": text, "encoding": encoding, "mode": mode, "result": result}
    except Exception as e:
        return {"error": str(e)}


# ─── JSON TOOLS ───────────────────────────────────────────────────────────────

def json_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Format, validate, query, or minify JSON."""
    action = args.get("action", "format")
    data = args.get("data", args.get("json", ""))
    query = args.get("query", "")  # dot-notation key path like "users.0.name"

    try:
        if action == "format":
            indent = args.get("indent", 2)
            parsed = json.loads(data)
            return {"result": json.dumps(parsed, indent=indent), "valid": True}

        elif action == "validate":
            parsed = json.loads(data)
            return {"valid": True, "type": type(parsed).__name__,
                    "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                    "length": len(parsed) if isinstance(parsed, (list, dict)) else None}

        elif action == "minify":
            parsed = json.loads(data)
            return {"result": json.dumps(parsed, separators=(',', ':'))}

        elif action == "query":
            parsed = json.loads(data)
            parts = query.split(".")
            result = parsed
            for part in parts:
                if part.isdigit():
                    result = result[int(part)]
                else:
                    result = result[part]
            return {"query": query, "result": result}

        elif action == "keys":
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return {"keys": list(parsed.keys())}
            elif isinstance(parsed, list):
                return {"length": len(parsed), "first_item_keys": list(parsed[0].keys()) if parsed and isinstance(parsed[0], dict) else None}

        else:
            return {"error": f"Unknown action: {action}. Use: format, validate, minify, query, keys"}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ─── REGEX ────────────────────────────────────────────────────────────────────

def regex_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Test regex patterns, find/replace, or validate."""
    pattern = args.get("pattern", "")
    text = args.get("text", "")
    action = args.get("action", "match")  # match | findall | replace | split | validate
    replacement = args.get("replacement", "")
    flags_str = args.get("flags", "").upper()

    flags = 0
    if "I" in flags_str: flags |= re.IGNORECASE
    if "M" in flags_str: flags |= re.MULTILINE
    if "S" in flags_str: flags |= re.DOTALL

    try:
        if action == "validate":
            re.compile(pattern)
            return {"pattern": pattern, "valid": True}

        elif action == "match":
            m = re.search(pattern, text, flags)
            if m:
                return {"matched": True, "match": m.group(), "groups": list(m.groups()),
                        "span": list(m.span())}
            return {"matched": False}

        elif action == "findall":
            matches = re.findall(pattern, text, flags)
            return {"count": len(matches), "matches": matches}

        elif action == "replace":
            result = re.sub(pattern, replacement, text, flags=flags)
            return {"result": result, "original": text}

        elif action == "split":
            parts = re.split(pattern, text, flags=flags)
            return {"parts": parts, "count": len(parts)}

        else:
            return {"error": f"Unknown action: {action}. Use: validate, match, findall, replace, split"}
    except re.error as e:
        return {"error": f"Regex error: {e}"}


# ─── DIFF ─────────────────────────────────────────────────────────────────────

def diff_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two texts or files and show differences."""
    text_a = args.get("a", args.get("text_a", ""))
    text_b = args.get("b", args.get("text_b", ""))
    file_a = args.get("file_a", "")
    file_b = args.get("file_b", "")
    mode = args.get("mode", "unified")  # unified | side_by_side | summary

    try:
        if file_a and file_b:
            text_a = Path(file_a).expanduser().read_text()
            text_b = Path(file_b).expanduser().read_text()

        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)

        if mode == "unified":
            diff = list(difflib.unified_diff(lines_a, lines_b,
                                              fromfile=file_a or "a",
                                              tofile=file_b or "b"))
            return {"diff": "".join(diff), "changed": len(diff) > 0,
                    "additions": sum(1 for l in diff if l.startswith("+")),
                    "deletions": sum(1 for l in diff if l.startswith("-"))}

        elif mode == "summary":
            matcher = difflib.SequenceMatcher(None, text_a, text_b)
            ratio = matcher.ratio()
            return {"similarity_pct": round(ratio * 100, 1),
                    "identical": ratio == 1.0,
                    "len_a": len(text_a), "len_b": len(text_b)}
    except Exception as e:
        return {"error": str(e)}


# ─── NETWORK ──────────────────────────────────────────────────────────────────

def network_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Network utilities: ping, DNS lookup, IP info, port check, HTTP status."""
    action = args.get("action", "ip_info")
    host = args.get("host", args.get("url", ""))

    try:
        if action == "ip_info":
            resp = requests.get("https://ipinfo.io/json", timeout=5)
            return resp.json()

        elif action == "ping":
            count = args.get("count", 4)
            param = "-n" if platform.system().lower() == "windows" else "-c"
            result = subprocess.run(
                ["ping", param, str(count), host],
                capture_output=True, text=True, timeout=15
            )
            return {"host": host, "output": result.stdout[-1000:],
                    "success": result.returncode == 0}

        elif action == "dns_lookup":
            import socket
            ip = socket.gethostbyname(host)
            return {"host": host, "ip": ip}

        elif action == "reverse_dns":
            import socket
            hostname = socket.gethostbyaddr(host)
            return {"ip": host, "hostname": hostname[0], "aliases": hostname[1]}

        elif action == "port_check":
            import socket
            port = int(args.get("port", 80))
            timeout = float(args.get("timeout", 3))
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            s.close()
            return {"host": host, "port": port, "open": result == 0}

        elif action == "http_status":
            resp = requests.head(host if host.startswith("http") else f"https://{host}",
                                 timeout=8, allow_redirects=True)
            return {"url": host, "status_code": resp.status_code,
                    "status": resp.reason, "headers": dict(resp.headers)}

        elif action == "my_ip":
            resp = requests.get("https://api.ipify.org?format=json", timeout=5)
            return resp.json()

        elif action == "whois":
            result = subprocess.run(["whois", host], capture_output=True, text=True, timeout=10)
            return {"host": host, "output": result.stdout[:2000]}

        else:
            return {"error": f"Unknown network action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── FILE OPERATIONS ──────────────────────────────────────────────────────────

def file_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive file operations: read, write, list, search, copy, move, delete, info."""
    action = args.get("action", "read")
    path = args.get("path", "")

    try:
        if action == "read":
            p = Path(path).expanduser()
            if not p.exists():
                return {"error": f"File not found: {path}"}
            content = p.read_text(encoding="utf-8", errors="replace")
            return {"path": str(p), "content": content[:5000],
                    "truncated": len(content) > 5000, "size": len(content),
                    "lines": content.count("\n")}

        elif action == "write":
            content = args.get("content", "")
            append = args.get("append", False)
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": str(p), "bytes_written": len(content)}

        elif action == "list":
            p = Path(path or ".").expanduser()
            if not p.exists():
                return {"error": f"Directory not found: {path}"}
            items = []
            for item in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name)):
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "type": "file" if item.is_file() else "dir",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "extension": item.suffix if item.is_file() else None,
                })
            return {"path": str(p), "items": items, "count": len(items)}

        elif action == "search":
            query = args.get("query", "")
            root = Path(path or ".").expanduser()
            pattern = args.get("pattern", "*")
            matches = []
            for p in root.rglob(pattern):
                if query.lower() in p.name.lower():
                    matches.append({"path": str(p), "name": p.name,
                                    "type": "file" if p.is_file() else "dir"})
            return {"query": query, "matches": matches[:50], "count": len(matches)}

        elif action == "grep":
            query = args.get("query", "")
            recursive = args.get("recursive", False)
            p = Path(path).expanduser()
            results = []
            paths = p.rglob("*") if recursive else [p] if p.is_file() else p.iterdir()
            for fp in paths:
                if fp.is_file():
                    try:
                        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
                        for i, line in enumerate(lines, 1):
                            if query.lower() in line.lower():
                                results.append({"file": str(fp), "line": i, "content": line.strip()})
                    except:
                        pass
            return {"query": query, "matches": results[:100], "count": len(results)}

        elif action == "info":
            p = Path(path).expanduser()
            if not p.exists():
                return {"error": f"Not found: {path}"}
            stat = p.stat()
            return {
                "path": str(p.resolve()), "name": p.name, "extension": p.suffix,
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": p.is_file(), "is_dir": p.is_dir(),
            }

        elif action == "copy":
            import shutil
            dst = args.get("destination", "")
            shutil.copy2(path, dst)
            return {"success": True, "from": path, "to": dst}

        elif action == "move":
            import shutil
            dst = args.get("destination", "")
            shutil.move(path, dst)
            return {"success": True, "from": path, "to": dst}

        elif action == "delete":
            p = Path(path).expanduser()
            if not p.exists():
                return {"error": "Not found"}
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
            return {"success": True, "deleted": str(p)}

        elif action == "mkdir":
            p = Path(path).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return {"success": True, "created": str(p)}

        elif action == "tree":
            max_depth = int(args.get("depth", 3))
            p = Path(path or ".").expanduser()
            lines = [str(p)]
            _tree_lines(p, "", max_depth, 0, lines)
            return {"tree": "\n".join(lines)}

        else:
            return {"error": f"Unknown file action: {action}"}
    except Exception as e:
        return {"error": str(e)}


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _tree_lines(path: Path, prefix: str, max_depth: int, depth: int, lines: list):
    if depth >= max_depth:
        return
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items[:50]):
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                _tree_lines(item, prefix + extension, max_depth, depth + 1, lines)
    except PermissionError:
        pass


# ─── CLIPBOARD ────────────────────────────────────────────────────────────────

def clipboard_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get or set clipboard content."""
    action = args.get("action", "get")
    try:
        import pyperclip
        if action == "get":
            content = pyperclip.paste()
            return {"content": content, "length": len(content)}
        elif action == "set":
            content = args.get("content", "")
            pyperclip.copy(content)
            return {"success": True, "copied_length": len(content)}
        elif action == "clear":
            pyperclip.copy("")
            return {"success": True}
        else:
            return {"error": f"Unknown action: {action}"}
    except ImportError:
        return {"error": "pyperclip not installed. Run: pip install pyperclip"}


# ─── SYSTEM INFORMATION ───────────────────────────────────────────────────────

def system_info_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed system information."""
    action = args.get("action", "overview")

    try:
        import psutil

        if action == "overview":
            cpu = psutil.cpu_percent(interval=0.5)
            cpu_freq = psutil.cpu_freq()
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            boot = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot

            return {
                "platform": platform.system(),
                "platform_version": platform.version()[:60],
                "processor": platform.processor()[:80],
                "cpu_cores_physical": psutil.cpu_count(logical=False),
                "cpu_cores_logical": psutil.cpu_count(logical=True),
                "cpu_percent": cpu,
                "cpu_freq_mhz": round(cpu_freq.current) if cpu_freq else None,
                "memory": {
                    "total_gb": round(mem.total / 1e9, 1),
                    "used_gb": round(mem.used / 1e9, 1),
                    "available_gb": round(mem.available / 1e9, 1),
                    "percent": mem.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / 1e9, 1),
                    "used_gb": round(disk.used / 1e9, 1),
                    "free_gb": round(disk.free / 1e9, 1),
                    "percent": disk.percent,
                },
                "uptime": str(uptime).split(".")[0],
                "boot_time": boot.strftime("%Y-%m-%d %H:%M:%S"),
            }

        elif action == "processes":
            n = int(args.get("n", 15))
            sort_by = args.get("sort_by", "memory")
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                try:
                    procs.append(p.info)
                except:
                    pass
            key = "memory_percent" if sort_by == "memory" else "cpu_percent"
            procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
            return {"processes": procs[:n]}

        elif action == "network_interfaces":
            interfaces = {}
            for name, addrs in psutil.net_if_addrs().items():
                interfaces[name] = [{"family": str(a.family), "address": a.address} for a in addrs]
            return {"interfaces": interfaces}

        elif action == "temperatures":
            try:
                temps = psutil.sensors_temperatures()
                result = {}
                for name, entries in temps.items():
                    result[name] = [{"label": e.label, "current": e.current,
                                     "high": e.high, "critical": e.critical} for e in entries]
                return {"temperatures": result}
            except:
                return {"error": "Temperature sensors not available on this platform"}

        elif action == "battery":
            bat = psutil.sensors_battery()
            if bat:
                return {"percent": bat.percent, "charging": bat.power_plugged,
                        "time_left_min": round(bat.secsleft / 60) if bat.secsleft > 0 else None}
            return {"error": "No battery detected"}

        else:
            return {"error": f"Unknown action: {action}"}
    except ImportError:
        result = subprocess.run("uname -a && free -h && df -h /",
                                shell=True, capture_output=True, text=True)
        return {"output": result.stdout}


# ─── PROCESS MANAGEMENT ───────────────────────────────────────────────────────

def process_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """List, find, kill processes."""
    action = args.get("action", "list")

    try:
        import psutil

        if action == "list":
            n = int(args.get("n", 20))
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "username"]):
                try:
                    procs.append(p.info)
                except:
                    pass
            procs.sort(key=lambda x: x.get("memory_percent") or 0, reverse=True)
            return {"processes": procs[:n], "total": len(procs)}

        elif action == "find":
            name = args.get("name", "").lower()
            results = []
            for p in psutil.process_iter(["pid", "name", "cmdline", "status"]):
                try:
                    if name in p.info["name"].lower():
                        results.append(p.info)
                except:
                    pass
            return {"name": name, "found": results, "count": len(results)}

        elif action == "kill":
            pid = args.get("pid")
            name = args.get("name", "")
            if pid:
                p = psutil.Process(int(pid))
                p.terminate()
                return {"success": True, "killed_pid": pid}
            elif name:
                killed = []
                for p in psutil.process_iter(["pid", "name"]):
                    try:
                        if name.lower() in p.info["name"].lower():
                            p.terminate()
                            killed.append(p.info["pid"])
                    except:
                        pass
                return {"success": True, "killed_pids": killed, "count": len(killed)}
            return {"error": "Provide 'pid' or 'name'"}

        elif action == "info":
            pid = int(args.get("pid"))
            p = psutil.Process(pid)
            return {
                "pid": pid, "name": p.name(), "status": p.status(),
                "cpu_percent": p.cpu_percent(interval=0.1),
                "memory_mb": round(p.memory_info().rss / 1e6, 1),
                "cmdline": " ".join(p.cmdline())[:200],
                "created": datetime.fromtimestamp(p.create_time()).isoformat(),
            }

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── TIMER / STOPWATCH ────────────────────────────────────────────────────────

_timers: Dict[str, float] = {}  # name -> start_time

def timer_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Countdown timers and stopwatches."""
    action = args.get("action", "start_countdown")
    label = args.get("label", "Timer")
    seconds = int(args.get("seconds", 60))

    if action == "start_countdown":
        import threading
        def _run():
            time.sleep(seconds)
            print(f"\n  ⏰  [{label}] {seconds}s timer done!")
        threading.Thread(target=_run, daemon=True).start()
        return {"success": True, "label": label, "seconds": seconds,
                "done_at": (datetime.now() + timedelta(seconds=seconds)).strftime("%H:%M:%S")}

    elif action == "start_stopwatch":
        _timers[label] = time.time()
        return {"success": True, "label": label, "started_at": datetime.now().strftime("%H:%M:%S")}

    elif action == "stop_stopwatch":
        if label not in _timers:
            return {"error": f"No stopwatch '{label}' running"}
        elapsed = time.time() - _timers.pop(label)
        return {"label": label, "elapsed_seconds": round(elapsed, 2),
                "elapsed_formatted": str(timedelta(seconds=int(elapsed)))}

    elif action == "lap_stopwatch":
        if label not in _timers:
            return {"error": f"No stopwatch '{label}' running"}
        elapsed = time.time() - _timers[label]
        return {"label": label, "lap_seconds": round(elapsed, 2)}

    elif action == "list":
        return {"active_timers": {k: f"{round(time.time() - v, 1)}s elapsed" for k, v in _timers.items()}}

    else:
        return {"error": f"Unknown action: {action}"}


# ─── DATE / TIME ──────────────────────────────────────────────────────────────

def datetime_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Date and time utilities."""
    action = args.get("action", "now")

    try:
        if action == "now":
            now = datetime.now()
            import time as _time
            return {
                "datetime": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "week_number": now.isocalendar()[1],
                "unix_timestamp": int(_time.time()),
                "timezone": _time.tzname[0],
            }

        elif action == "add":
            dt_str = args.get("datetime", datetime.now().isoformat())
            dt = datetime.fromisoformat(dt_str)
            delta = timedelta(
                days=int(args.get("days", 0)),
                hours=int(args.get("hours", 0)),
                minutes=int(args.get("minutes", 0)),
                seconds=int(args.get("seconds", 0)),
            )
            result = dt + delta
            return {"input": dt_str, "result": result.isoformat(), "result_human": result.strftime("%A %d %B %Y %H:%M")}

        elif action == "diff":
            a = datetime.fromisoformat(args.get("a"))
            b = datetime.fromisoformat(args.get("b"))
            diff = abs(b - a)
            return {
                "a": str(a), "b": str(b),
                "days": diff.days,
                "hours": diff.seconds // 3600,
                "minutes": (diff.seconds % 3600) // 60,
                "total_seconds": int(diff.total_seconds()),
            }

        elif action == "format":
            dt = datetime.fromisoformat(args.get("datetime", datetime.now().isoformat()))
            fmt = args.get("format", "%A, %d %B %Y at %H:%M")
            return {"result": dt.strftime(fmt)}

        elif action == "parse":
            text = args.get("text", "")
            formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y",
                       "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M"]
            for fmt in formats:
                try:
                    dt = datetime.strptime(text, fmt)
                    return {"parsed": dt.isoformat(), "format_used": fmt}
                except:
                    pass
            return {"error": f"Could not parse date: {text}"}

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── CRYPTO / STOCK PRICES ───────────────────────────────────────────────────

def price_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get cryptocurrency or stock prices."""
    symbol = args.get("symbol", "BTC").upper()
    asset_type = args.get("type", "crypto")

    try:
        if asset_type == "crypto":
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": symbol.lower(), "vs_currencies": "usd,eur",
                        "include_24hr_change": "true"},
                timeout=8
            )
            data = resp.json()
            if not data:
                # Try by ticker symbol
                search_resp = requests.get(
                    f"https://api.coingecko.com/api/v3/search?query={symbol}",
                    timeout=8
                )
                coins = search_resp.json().get("coins", [])
                if coins:
                    coin_id = coins[0]["id"]
                    resp = requests.get(
                        "https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": coin_id, "vs_currencies": "usd,eur",
                                "include_24hr_change": "true"},
                        timeout=8
                    )
                    data = resp.json()
            for key, val in data.items():
                return {
                    "symbol": symbol, "name": key, "type": "crypto",
                    "usd": val.get("usd"), "eur": val.get("eur"),
                    "change_24h_pct": round(val.get("usd_24h_change", 0), 2),
                }
            return {"error": f"Coin '{symbol}' not found on CoinGecko"}

        else:
            # Stock via Yahoo Finance (no key needed for basic data)
            resp = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8
            )
            chart = resp.json().get("chart", {})
            result = chart.get("result", [])
            if result:
                meta = result[0].get("meta", {})
                return {
                    "symbol": symbol, "type": "stock",
                    "price": meta.get("regularMarketPrice"),
                    "currency": meta.get("currency"),
                    "exchange": meta.get("exchangeName"),
                    "prev_close": meta.get("previousClose"),
                    "change_pct": round(
                        (meta.get("regularMarketPrice", 0) / meta.get("previousClose", 1) - 1) * 100, 2
                    ) if meta.get("previousClose") else None,
                }
            return {"error": f"Symbol '{symbol}' not found"}
    except Exception as e:
        return {"error": str(e)}


# ─── TEXT TOOLS ───────────────────────────────────────────────────────────────

def text_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Text manipulation utilities."""
    action = args.get("action", "stats")
    text = args.get("text", "")

    try:
        if action == "stats":
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            return {
                "characters": len(text),
                "characters_no_spaces": len(text.replace(" ", "")),
                "words": len(words),
                "sentences": len([s for s in sentences if s.strip()]),
                "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
                "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 1),
                "reading_time_min": round(len(words) / 200, 1),
            }

        elif action == "case":
            mode = args.get("mode", "upper")
            ops = {
                "upper": str.upper, "lower": str.lower, "title": str.title,
                "capitalize": str.capitalize, "swapcase": str.swapcase,
            }
            if mode not in ops:
                return {"error": f"Unknown mode. Use: {list(ops.keys())}"}
            return {"result": ops[mode](text), "mode": mode}

        elif action == "reverse":
            level = args.get("level", "chars")
            if level == "chars":
                return {"result": text[::-1]}
            elif level == "words":
                return {"result": " ".join(text.split()[::-1])}
            elif level == "lines":
                return {"result": "\n".join(text.splitlines()[::-1])}

        elif action == "extract_emails":
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            return {"emails": list(set(emails)), "count": len(set(emails))}

        elif action == "extract_urls":
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
            return {"urls": list(set(urls)), "count": len(set(urls))}

        elif action == "extract_numbers":
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return {"numbers": [float(n) for n in numbers], "count": len(numbers)}

        elif action == "truncate":
            max_len = int(args.get("max_length", 100))
            suffix = args.get("suffix", "...")
            if len(text) <= max_len:
                return {"result": text, "truncated": False}
            return {"result": text[:max_len - len(suffix)] + suffix, "truncated": True}

        elif action == "wrap":
            width = int(args.get("width", 80))
            import textwrap
            return {"result": textwrap.fill(text, width)}

        elif action == "slug":
            slug = re.sub(r'[^\w\s-]', '', text.lower())
            slug = re.sub(r'[-\s]+', '-', slug).strip('-')
            return {"result": slug}

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── QR CODE ──────────────────────────────────────────────────────────────────

def qr_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate or decode QR codes."""
    action = args.get("action", "generate")
    data = args.get("data", args.get("text", args.get("url", "")))
    output_path = args.get("output", "qrcode.png")

    try:
        import qrcode
        if action == "generate":
            qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            p = Path(output_path)
            img.save(str(p))
            return {"success": True, "path": str(p.resolve()), "data": data}
        elif action == "terminal":
            # Print QR to terminal
            qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, border=1)
            qr.add_data(data)
            qr.make(fit=True)
            f = io.StringIO()
            qr.print_ascii(out=f)
            return {"ascii": f.getvalue(), "data": data}
    except ImportError:
        return {"error": "qrcode not installed. Run: pip install qrcode[pil]"}
    except Exception as e:
        return {"error": str(e)}


# ─── GIT TOOLS ────────────────────────────────────────────────────────────────

def git_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Git repository operations."""
    action = args.get("action", "status")
    path = args.get("path", ".")
    message = args.get("message", "")
    branch = args.get("branch", "")

    try:
        def run_git(*cmd_args):
            return subprocess.run(
                ["git"] + list(cmd_args),
                capture_output=True, text=True, cwd=path, timeout=15
            )

        if action == "status":
            result = run_git("status", "--porcelain", "-b")
            return {"status": result.stdout, "path": path, "success": result.returncode == 0}

        elif action == "log":
            n = args.get("n", 10)
            result = run_git("log", f"-{n}",
                             "--pretty=format:%h|%an|%ar|%s")
            commits = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({"hash": parts[0], "author": parts[1],
                                    "when": parts[2], "message": parts[3]})
            return {"commits": commits, "count": len(commits)}

        elif action == "diff":
            result = run_git("diff")
            return {"diff": result.stdout[:3000]}

        elif action == "branches":
            result = run_git("branch", "-a")
            branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines()]
            return {"branches": branches}

        elif action == "add":
            files = args.get("files", ".")
            result = run_git("add", files)
            return {"success": result.returncode == 0, "output": result.stdout}

        elif action == "commit":
            result = run_git("commit", "-m", message)
            return {"success": result.returncode == 0, "output": result.stdout}

        elif action == "push":
            remote = args.get("remote", "origin")
            result = run_git("push", remote)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}

        elif action == "pull":
            result = run_git("pull")
            return {"success": result.returncode == 0, "output": result.stdout}

        elif action == "checkout":
            result = run_git("checkout", branch)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}

        elif action == "stash":
            stash_action = args.get("stash_action", "save")
            result = run_git("stash", stash_action)
            return {"success": result.returncode == 0, "output": result.stdout}

        elif action == "init":
            result = run_git("init")
            return {"success": result.returncode == 0, "output": result.stdout}

        else:
            return {"error": f"Unknown git action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── PACKAGE MANAGER ──────────────────────────────────────────────────────────

def package_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Python package management via pip."""
    action = args.get("action", "list")
    package = args.get("package", "")

    try:
        if action == "list":
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                                    capture_output=True, text=True, timeout=15)
            packages = json.loads(result.stdout)
            return {"packages": packages, "count": len(packages)}

        elif action == "install":
            result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                    capture_output=True, text=True, timeout=60)
            return {"success": result.returncode == 0, "output": result.stdout[-500:],
                    "error": result.stderr[-200:] if result.returncode != 0 else None}

        elif action == "uninstall":
            result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package],
                                    capture_output=True, text=True, timeout=30)
            return {"success": result.returncode == 0, "output": result.stdout}

        elif action == "info":
            result = subprocess.run([sys.executable, "-m", "pip", "show", package],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {"error": f"Package '{package}' not found"}
            info = {}
            for line in result.stdout.splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    info[k.lower().replace("-", "_")] = v
            return info

        elif action == "check":
            result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                    capture_output=True, text=True, timeout=15)
            return {"output": result.stdout, "issues": result.returncode != 0}

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── SCREENSHOT OCR ──────────────────────────────────────────────────────────

def ocr_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """OCR: extract text from images or screenshots."""
    image_path = args.get("image", args.get("path", ""))
    screenshot = args.get("screenshot", not bool(image_path))

    try:
        import pytesseract
        from PIL import Image

        if screenshot:
            import pyautogui
            img = pyautogui.screenshot()
        else:
            img = Image.open(image_path)

        text = pytesseract.image_to_string(img)
        return {"success": True, "text": text.strip(), "char_count": len(text)}
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Install: pip install pytesseract pillow"}
    except Exception as e:
        return {"error": str(e)}


# ─── RANDOM / GENERATE ───────────────────────────────────────────────────────

def generate_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate random data: passwords, UUIDs, numbers, names."""
    import random
    import string
    import uuid

    action = args.get("action", "password")

    try:
        if action == "password":
            length = int(args.get("length", 16))
            use_symbols = args.get("symbols", True)
            chars = string.ascii_letters + string.digits
            if use_symbols:
                chars += "!@#$%^&*()-_=+"
            pwd = "".join(random.choice(chars) for _ in range(length))
            return {"password": pwd, "length": length}

        elif action == "uuid":
            return {"uuid": str(uuid.uuid4()), "uuid_v1": str(uuid.uuid1())}

        elif action == "number":
            min_val = int(args.get("min", 0))
            max_val = int(args.get("max", 100))
            n = int(args.get("count", 1))
            numbers = [random.randint(min_val, max_val) for _ in range(n)]
            return {"numbers": numbers, "min": min_val, "max": max_val}

        elif action == "choice":
            items = args.get("items", [])
            if not items:
                return {"error": "Provide 'items' list"}
            return {"chosen": random.choice(items), "from": items}

        elif action == "shuffle":
            items = args.get("items", [])
            shuffled = items.copy()
            random.shuffle(shuffled)
            return {"shuffled": shuffled, "original": items}

        elif action == "token":
            length = int(args.get("length", 32))
            import secrets
            return {"token": secrets.token_hex(length // 2)}

        elif action == "string":
            length = int(args.get("length", 10))
            chars = args.get("chars", string.ascii_lowercase)
            return {"result": "".join(random.choice(chars) for _ in range(length))}

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


# ─── TRANSLATE ───────────────────────────────────────────────────────────────

def translate_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Translate text using MyMemory free API."""
    text = args.get("text", "")
    target = args.get("to", args.get("target_lang", "en"))
    source = args.get("from", args.get("source_lang", "auto"))

    if not text:
        return {"error": "Provide 'text' to translate"}

    try:
        lang_pair = f"{source}|{target}" if source != "auto" else f"autodetect|{target}"
        resp = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:500], "langpair": lang_pair},
            timeout=8
        )
        data = resp.json()
        if data.get("responseStatus") == 200:
            return {
                "original": text, "translated": data["responseData"]["translatedText"],
                "source_lang": source, "target_lang": target,
                "confidence": data["responseData"].get("match"),
            }
        return {"error": data.get("responseDetails", "Translation failed")}
    except Exception as e:
        return {"error": str(e)}


# ─── CURRENCY CONVERSION ─────────────────────────────────────────────────────

def currency_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between currencies using free exchange rate API."""
    amount = float(args.get("amount", 1))
    from_currency = args.get("from", "USD").upper()
    to_currency = args.get("to", "EUR").upper()

    try:
        resp = requests.get(
            f"https://open.er-api.com/v6/latest/{from_currency}",
            timeout=8
        )
        data = resp.json()
        if data.get("result") == "success":
            rates = data["rates"]
            if to_currency not in rates:
                return {"error": f"Unknown currency: {to_currency}"}
            rate = rates[to_currency]
            result = amount * rate
            return {
                "amount": amount, "from": from_currency, "to": to_currency,
                "rate": round(rate, 6), "result": round(result, 2),
                "updated": data.get("time_last_update_utc", "unknown"),
            }
        return {"error": "Exchange rate API unavailable"}
    except Exception as e:
        return {"error": str(e)}


# ─── SUMMARIZE / ANALYZE TEXT ────────────────────────────────────────────────

def text_analyze_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze text: sentiment keywords, readability."""
    text = args.get("text", "")
    action = args.get("action", "keywords")

    try:
        if action == "keywords":
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            stopwords = {"this", "that", "with", "have", "from", "they", "will",
                         "been", "were", "what", "when", "then", "than", "some",
                         "their", "there", "which", "would", "could", "should"}
            freq = {}
            for w in words:
                if w not in stopwords:
                    freq[w] = freq.get(w, 0) + 1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
            return {"keywords": [{"word": k, "count": v} for k, v in top]}

        elif action == "readability":
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            syllables = sum(_count_syllables(w) for w in text.split())
            if sentences > 0 and words > 0:
                flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
                grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
            else:
                flesch, grade = 0, 0
            return {
                "flesch_reading_ease": round(flesch, 1),
                "grade_level": round(grade, 1),
                "level": "Very Easy" if flesch > 90 else "Easy" if flesch > 80 else
                         "Fairly Easy" if flesch > 70 else "Standard" if flesch > 60 else
                         "Fairly Difficult" if flesch > 50 else "Difficult",
            }

        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)