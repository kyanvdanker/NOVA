"""
NOVA GUI — Flask-based web interface.
Run with: python -m gui.app  or  python main.py --gui
"""
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import ASSISTANT_NAME, USER_NAME, GUI_HOST, GUI_PORT, DATA_DIR
from core.tools import system_info_tool, file_tool

app = Flask(__name__, template_folder="public", static_folder="public")
app.config["SECRET_KEY"] = "nova-secret-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global assistant instance (set by main.py)
_assistant = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_assistant_thread: Optional[threading.Thread] = None


def set_assistant(assistant, loop):
    global _assistant, _loop
    _assistant = assistant
    _loop = loop


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           assistant_name=ASSISTANT_NAME,
                           user_name=USER_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle a chat message — streams response via SSE."""
    data = request.json
    text = data.get("message", "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400

    def generate():
        chunks = []
        done_event = threading.Event()

        def on_stream(event_type, data):
            if event_type == "chunk":
                chunks.append(data)
                socketio.emit("stream_chunk", {"chunk": data})
            elif event_type == "tool_result":
                socketio.emit("tool_result", data)
            elif event_type == "done":
                socketio.emit("stream_done", {"response": data})
                done_event.set()

        if _assistant:
            _assistant.add_stream_callback(on_stream)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _assistant.handle_input(text, voice_response=False), _loop
                )
                result = future.result(timeout=120)
            except Exception as e:
                socketio.emit("stream_done", {"response": f"Error: {e}"})
                result = f"Error: {e}"
            finally:
                if on_stream in _assistant._stream_callbacks:
                    _assistant._stream_callbacks.remove(on_stream)
        else:
            result = "Assistant not initialized"

        yield f"data: {json.dumps({'response': result})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/system_stats")
def system_stats():
    """Return current system stats for the dashboard."""
    try:
        stats = system_info_tool({"action": "overview"})
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/projects")
def get_projects():
    if not _assistant:
        return jsonify({"projects": []})
    result = _assistant.projects.execute("list", {})
    return jsonify(result)


@app.route("/api/memos")
def get_memos():
    if not _assistant:
        return jsonify({"memos": []})
    memos = _assistant.memory.list_memos()
    return jsonify({"memos": memos})


@app.route("/api/memo/<int:memo_id>")
def get_memo(memo_id):
    if not _assistant:
        return jsonify({"error": "No assistant"})
    memo = _assistant.memory.get_memo(memo_id)
    return jsonify(memo or {"error": "Not found"})


@app.route("/api/agenda")
def get_agenda():
    if not _assistant:
        return jsonify({"items": []})
    items = _assistant.memory.get_agenda_items(status="pending")
    overdue = _assistant.agenda._get_overdue()
    today = _assistant.agenda._get_today()
    upcoming = _assistant.agenda._get_upcoming(days=7)
    return jsonify({"all": items, "overdue": overdue, "today": today, "upcoming": upcoming})


@app.route("/api/files")
def list_files():
    """Browse the filesystem."""
    path = request.args.get("path", str(DATA_DIR))
    try:
        result = file_tool({"action": "list", "path": path})
        result["current_path"] = path
        result["parent_path"] = str(Path(path).parent)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/file/read")
def read_file():
    path = request.args.get("path", "")
    try:
        result = file_tool({"action": "read", "path": path})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/skills")
def get_skills():
    if not _assistant:
        return jsonify({"skills": {}})
    skills = _assistant.self_improve.get_custom_skills()
    status = _assistant.self_improve.get_status()
    return jsonify({"skills": {k: v["description"] for k, v in skills.items()},
                    "status": status})


@app.route("/api/conversation")
def get_conversation():
    if not _assistant:
        return jsonify({"messages": []})
    msgs = _assistant.memory.get_recent_messages(n=50,
                                                  session_id=_assistant._session_id)
    return jsonify({"messages": msgs})


@app.route("/api/facts")
def get_facts():
    if not _assistant:
        return jsonify({"facts": {}})
    return jsonify({"facts": _assistant.memory.get_all_facts()})


# ─── SocketIO ─────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("connected", {"status": "ok", "assistant": ASSISTANT_NAME})


@socketio.on("send_message")
def on_message(data):
    text = data.get("message", "").strip()
    if not text or not _assistant:
        emit("error", {"message": "No message or assistant not ready"})
        return

    def run():
        chunks = []

        def on_stream(event_type, payload):
            if event_type == "chunk":
                chunks.append(payload)
                socketio.emit("stream_chunk", {"chunk": payload})
            elif event_type == "tool_result":
                socketio.emit("tool_result", payload)
            elif event_type == "done":
                socketio.emit("stream_done", {"response": payload})

        _assistant.add_stream_callback(on_stream)
        try:
            future = asyncio.run_coroutine_threadsafe(
                _assistant.handle_input(text, voice_response=False), _loop
            )
            future.result(timeout=120)
        except Exception as e:
            socketio.emit("stream_done", {"response": f"Error: {e}"})
        finally:
            if on_stream in _assistant._stream_callbacks:
                _assistant._stream_callbacks.remove(on_stream)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def run_gui(assistant, loop, host=GUI_HOST, port=GUI_PORT):
    """Start the GUI server."""
    set_assistant(assistant, loop)
    print(f"\n  🌐 NOVA GUI → http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)