"""
NOVA Intent Router
Parses user intent from transcribed speech and dispatches
to the right skill handler.
Uses LLM for complex routing, regex/keywords for fast simple cases.
"""

import re
import logging
from typing import Dict, Any, Optional, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.router")


# ─── Quick Intent Patterns (no LLM needed) ────────────────────────────────────
QUICK_PATTERNS = [
    # Projects
    (r"\b(create|new|start)\s+(project|proj)\s+(.+)", "project_create"),
    (r"\b(open|show|load)\s+project\s+(.+)", "project_open"),
    (r"\b(list|show)\s+(all\s+)?(projects?)\b", "project_list"),

    # Memos/Tasks
    (r"\b(add|create|note|remember|write down)\s+(task|todo|to-do)\s*[:\-]?\s*(.+)", "task_create"),
    (r"\b(add|create|make)\s+(a\s+)?(note|memo)\s*[:\-]?\s*(.+)", "memo_create"),
    (r"\b(show|read|what are my)\s+(notes?|memos?|tasks?)\b", "memo_list"),
    (r"\b(done|complete|finished|check off)\s+task\s+(.+)", "task_complete"),

    # Screen
    (r"\b(what('s| is)?\s+on\s+(my\s+)?screen|describe\s+(my\s+)?screen|look at (my )?screen)\b", "screen_describe"),
    (r"\b(take\s+a?\s*screenshot|capture\s+screen)\b", "screenshot"),

    # Apps
    (r"\b(open|launch|start)\s+(.+)", "app_open"),
    (r"\b(close|quit|exit|kill)\s+(.+)", "app_close"),

    # Files
    (r"\b(open|show)\s+file\s+(.+)", "file_open"),
    (r"\b(find|search|where is)\s+file\s+(.+)", "file_find"),

    # System
    (r"\bvolume\s+(to\s+)?(\d+)\b", "volume_set"),
    (r"\b(system\s+info|cpu|memory|ram|battery)\b", "system_info"),

    # Memory / Learning
    (r"\b(remember|my name is|i am|call me)\s+(.+)", "remember_this"),
    (r"\b(what do you know about me|what do you remember)\b", "memory_query"),
    (r"\bforget\s+everything\b", "memory_forget"),

    # Time
    (r"\bwhat\s+(time|day|date)\b", "current_time"),

    # Help
    (r"\b(help|what can you do|commands)\b", "help"),
]


def quick_classify(text: str) -> Optional[Dict[str, Any]]:
    """Fast regex-based intent classification for common patterns."""
    text_lower = text.lower().strip()

    for pattern, intent in QUICK_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            return {"intent": intent, "groups": groups, "text": text}

    return None


class IntentRouter:
    """Routes user utterances to skill handlers."""

    def __init__(self, llm, memory, laptop_control, projects_module):
        self.llm = llm
        self.memory = memory
        self.laptop = laptop_control
        self.projects = projects_module

        self._active_project_id: Optional[int] = None
        self._active_project_name: Optional[str] = None

        # Handler map
        self._handlers = {
            "project_create": self._handle_project_create,
            "project_open": self._handle_project_open,
            "project_list": self._handle_project_list,
            "task_create": self._handle_task_create,
            "memo_create": self._handle_memo_create,
            "memo_list": self._handle_memo_list,
            "task_complete": self._handle_task_complete,
            "screen_describe": self._handle_screen_describe,
            "screenshot": self._handle_screenshot,
            "app_open": self._handle_app_open,
            "app_close": self._handle_app_close,
            "file_open": self._handle_file_open,
            "file_find": self._handle_file_find,
            "volume_set": self._handle_volume_set,
            "system_info": self._handle_system_info,
            "remember_this": self._handle_remember,
            "memory_query": self._handle_memory_query,
            "memory_forget": self._handle_memory_forget,
            "current_time": self._handle_current_time,
            "help": self._handle_help,
            "general_question": None,  # Falls through to LLM
        }

    def route(self, text: str) -> Optional[str]:
        """
        Route an utterance. Returns a response string if handled,
        or None to fall through to the LLM.
        """
        # Try quick classification first
        quick = quick_classify(text)
        if quick:
            intent = quick["intent"]
            handler = self._handlers.get(intent)
            if handler:
                try:
                    return handler(text, quick)
                except Exception as e:
                    log.error(f"Handler error ({intent}): {e}")
                    return None

        return None  # Fall through to LLM

    def set_active_project(self, project_id: int, project_name: str):
        """Set the currently active project context."""
        self._active_project_id = project_id
        self._active_project_name = project_name
        log.info(f"Active project: {project_name} (id={project_id})")

    # ─── Handlers ─────────────────────────────────────────────────────────────

    def _handle_project_create(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        name = groups[-1] if groups else text
        # Extract project name - last group
        name = name.strip(" .,!?\"'")
        project = self.projects.create_project(name)
        self.set_active_project(project["id"], project["name"])
        return (f"Project '{project['name']}' created. "
                f"It's now the active project. What's the first thing you want to add to it?")

    def _handle_project_open(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        name = groups[-1] if groups else ""
        project = self.projects.get_project_by_name(name)
        if not project:
            return f"I couldn't find a project called '{name}'. Want me to create it?"
        self.set_active_project(project["id"], project["name"])
        summary = self.projects.get_project_summary(project["id"])
        return summary

    def _handle_project_list(self, text: str, match: dict) -> str:
        projects = self.projects.list_projects("active")
        if not projects:
            return "You have no active projects. Say 'create project' followed by a name to start one."
        names = ", ".join(p["name"] for p in projects[:5])
        return f"You have {len(projects)} active project{'s' if len(projects) > 1 else ''}: {names}."

    def _handle_task_create(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        task_content = groups[-1] if groups else text
        task_content = task_content.strip(" .,!?")
        project_id = self._active_project_id
        project_hint = f" in project '{self._active_project_name}'" if self._active_project_name else ""
        self.projects.add_memo(task_content, project_id=project_id, memo_type="task")
        return f"Task added{project_hint}: {task_content}"

    def _handle_memo_create(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        content = groups[-1] if groups else text
        content = content.strip(" .,!?")
        project_id = self._active_project_id
        project_hint = f" to project '{self._active_project_name}'" if self._active_project_name else ""
        self.projects.add_memo(content, project_id=project_id, memo_type="note")
        return f"Note saved{project_hint}."

    def _handle_memo_list(self, text: str, match: dict) -> str:
        if self._active_project_id:
            memos = self.projects.get_memos(project_id=self._active_project_id)
            tasks = [m for m in memos if m["memo_type"] == "task" and m["status"] == "open"]
            notes = [m for m in memos if m["memo_type"] == "note"]
            resp = f"Project '{self._active_project_name}': "
            if tasks:
                resp += f"{len(tasks)} open tasks: " + "; ".join(t["content"][:40] for t in tasks[:3])
            if notes:
                resp += f" {len(notes)} notes."
        else:
            resp = self.projects.get_all_open_tasks_summary()
        return resp

    def _handle_task_complete(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        task_query = groups[-1] if groups else ""
        # Find task matching query
        tasks = self.projects.get_memos(memo_type="task", status="open")
        for task in tasks:
            if task_query.lower() in task["content"].lower():
                self.projects.complete_task(task["id"])
                return f"Task marked as done: {task['content'][:50]}"
        return f"Couldn't find an open task matching '{task_query}'."

    def _handle_screen_describe(self, text: str, match: dict) -> str:
        screenshot_b64 = self.laptop.take_screenshot()
        if not screenshot_b64:
            return "I couldn't take a screenshot right now."
        description = self.llm.analyze_screen(screenshot_b64,
                                               "Describe what's on the screen and what the user is working on.")
        return description

    def _handle_screenshot(self, text: str, match: dict) -> str:
        screenshot_b64 = self.laptop.take_screenshot()
        if screenshot_b64:
            return "Screenshot taken."
        return "Screenshot failed."

    def _handle_app_open(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        app = groups[-1] if groups else ""
        # Remove common filler words
        app = re.sub(r'\b(please|for me|up)\b', '', app).strip()
        success, msg = self.laptop.open_app(app)
        return msg

    def _handle_app_close(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        app = groups[-1] if groups else ""
        success, msg = self.laptop.close_app(app)
        return msg

    def _handle_file_open(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        filename = groups[-1] if groups else ""
        success, msg = self.laptop.open_file(filename)
        return msg

    def _handle_file_find(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        filename = groups[-1] if groups else ""
        results = self.laptop.find_file(filename)
        if not results:
            return f"No files found matching '{filename}'."
        return f"Found {len(results)} file{'s' if len(results) > 1 else ''}: " + \
               ", ".join(Path(r).name for r in results[:3])

    def _handle_volume_set(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        level = int(groups[-1]) if groups else 50
        success, msg = self.laptop.set_volume(level)
        return msg

    def _handle_system_info(self, text: str, match: dict) -> str:
        info = self.laptop.get_system_info()
        if "error" in info:
            return "System info unavailable. Install psutil for this feature."
        return (f"CPU at {info['cpu_percent']}%, "
                f"Memory {info['memory_percent']}% used "
                f"({info['memory_available_gb']} GB free).")

    def _handle_remember(self, text: str, match: dict) -> str:
        groups = match.get("groups", [])
        info = groups[-1] if groups else text
        self.memory.add_fact("personal", text, source="stated")
        # Check if it's a name
        name_match = re.search(r'\b(my name is|call me|i am|i\'m)\s+([a-zA-Z]+)\b', text, re.I)
        if name_match:
            name = name_match.group(2).title()
            self.memory.update_profile("name", name)
            return f"Got it, I'll call you {name}."
        return "Got it, I'll remember that."

    def _handle_memory_query(self, text: str, match: dict) -> str:
        profile = self.memory.get_user_profile_text()
        prefs = self.memory.get_all_preferences()
        stats = self.memory.get_stats()

        resp = f"I know {stats['episodes']} things from our conversations. "
        if profile:
            resp += f"About you: {profile[:200]}. "
        if prefs:
            resp += f"Your preferences: " + "; ".join(f"{k}: {v}" for k, v in list(prefs.items())[:3])
        return resp

    def _handle_memory_forget(self, text: str, match: dict) -> str:
        # This is destructive — should require confirmation
        return "That would clear all my memories of you. Say 'yes forget everything' to confirm."

    def _handle_current_time(self, text: str, match: dict) -> str:
        import time as t
        return t.strftime("It's %I:%M %p on %A, %B %d.")

    def _handle_help(self, text: str, match: dict) -> str:
        return (
            "I can manage your projects and tasks, open apps and files, "
            "describe your screen, remember things about you, answer questions, "
            "and control your computer. Just talk to me naturally. "
            f"Say '{config.WAKE_WORDS[0]}' to wake me up anytime."
        )