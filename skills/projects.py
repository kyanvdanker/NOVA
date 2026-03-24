"""
NOVA Projects & Memos
Full project management:
- Create/update/archive projects
- Add notes, tasks, and memos to projects
- Tag and search across all content
- Voice-friendly summaries
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.projects")

DB_PATH = config.STORAGE_DIR / "projects.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    conn = _conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'active',   -- active, paused, completed, archived
            priority TEXT DEFAULT 'normal', -- low, normal, high, urgent
            tags TEXT DEFAULT '[]',
            created_at REAL,
            updated_at REAL,
            due_date REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS memos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,    -- NULL = standalone memo
            title TEXT,
            content TEXT NOT NULL,
            memo_type TEXT DEFAULT 'note',  -- note, task, idea, decision, meeting
            status TEXT DEFAULT 'open',     -- open, done, dismissed
            priority TEXT DEFAULT 'normal',
            tags TEXT DEFAULT '[]',
            created_at REAL,
            updated_at REAL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS project_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            entry TEXT NOT NULL,
            created_at REAL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    """)

    conn.commit()
    conn.close()


_init()


# ─── Projects ─────────────────────────────────────────────────────────────────

def create_project(name: str, description: str = "", tags: List[str] = None,
                   priority: str = "normal", due_date: str = None) -> dict:
    """Create a new project."""
    conn = _conn()
    c = conn.cursor()
    now = time.time()

    due_ts = None
    if due_date:
        try:
            from dateutil.parser import parse
            due_ts = parse(due_date).timestamp()
        except Exception:
            pass

    c.execute(
        """INSERT INTO projects (name, description, status, priority, tags, created_at, updated_at, due_date)
           VALUES (?, ?, 'active', ?, ?, ?, ?, ?)""",
        (name, description, priority, json.dumps(tags or []), now, now, due_ts)
    )
    project_id = c.lastrowid
    conn.commit()
    conn.close()

    log.info(f"Project created: {name} (id={project_id})")
    return get_project(project_id)


def get_project(project_id: int) -> Optional[dict]:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM projects WHERE id=?", (project_id,))
    row = c.fetchone()
    conn.close()
    return _project_to_dict(row) if row else None


def get_project_by_name(name: str) -> Optional[dict]:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM projects WHERE LOWER(name) LIKE ?", (f"%{name.lower()}%",))
    row = c.fetchone()
    conn.close()
    return _project_to_dict(row) if row else None


def list_projects(status: str = "active") -> List[dict]:
    conn = _conn()
    c = conn.cursor()
    if status == "all":
        c.execute("SELECT * FROM projects ORDER BY priority DESC, updated_at DESC")
    else:
        c.execute("SELECT * FROM projects WHERE status=? ORDER BY priority DESC, updated_at DESC",
                  (status,))
    rows = c.fetchall()
    conn.close()
    return [_project_to_dict(r) for r in rows]


def update_project(project_id: int, **kwargs) -> Optional[dict]:
    """Update project fields."""
    allowed = {"name", "description", "status", "priority", "tags", "due_date"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_project(project_id)

    updates["updated_at"] = time.time()
    if "tags" in updates and isinstance(updates["tags"], list):
        updates["tags"] = json.dumps(updates["tags"])

    conn = _conn()
    c = conn.cursor()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    c.execute(f"UPDATE projects SET {set_clause} WHERE id=?",
              list(updates.values()) + [project_id])
    conn.commit()
    conn.close()
    return get_project(project_id)


def _project_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    d["tags"] = json.loads(d.get("tags", "[]"))
    if d.get("created_at"):
        d["created_at_str"] = datetime.fromtimestamp(d["created_at"]).strftime("%Y-%m-%d %H:%M")
    if d.get("due_date"):
        d["due_date_str"] = datetime.fromtimestamp(d["due_date"]).strftime("%Y-%m-%d")
    return d


def get_project_summary(project_id: int) -> str:
    """Get a voice-friendly project summary."""
    project = get_project(project_id)
    if not project:
        return "Project not found."

    memos = get_memos(project_id=project_id)
    tasks = [m for m in memos if m["memo_type"] == "task"]
    open_tasks = [t for t in tasks if t["status"] == "open"]
    notes = [m for m in memos if m["memo_type"] == "note"]

    summary = f"Project: {project['name']}. Status: {project['status']}. "
    if project.get("description"):
        summary += f"{project['description']}. "
    if open_tasks:
        summary += f"{len(open_tasks)} open tasks. "
        for t in open_tasks[:3]:
            summary += f"Task: {t['content'][:60]}. "
    if notes:
        summary += f"{len(notes)} notes. "

    return summary


# ─── Memos ────────────────────────────────────────────────────────────────────

def add_memo(content: str, project_id: int = None, title: str = None,
             memo_type: str = "note", priority: str = "normal",
             tags: List[str] = None) -> dict:
    """Add a memo, optionally attached to a project."""
    conn = _conn()
    c = conn.cursor()
    now = time.time()
    c.execute(
        """INSERT INTO memos (project_id, title, content, memo_type, status, priority, tags, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?)""",
        (project_id, title, content, memo_type, priority, json.dumps(tags or []), now, now)
    )
    memo_id = c.lastrowid

    # Log to project
    if project_id:
        c.execute(
            "INSERT INTO project_log (project_id, entry, created_at) VALUES (?, ?, ?)",
            (project_id, f"Added {memo_type}: {content[:80]}", now)
        )
        c.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))

    conn.commit()
    conn.close()
    return get_memo(memo_id)


def get_memo(memo_id: int) -> Optional[dict]:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT * FROM memos WHERE id=?", (memo_id,))
    row = c.fetchone()
    conn.close()
    return _memo_to_dict(row) if row else None


def get_memos(project_id: int = None, memo_type: str = None,
              status: str = None) -> List[dict]:
    """Get memos, optionally filtered."""
    conn = _conn()
    c = conn.cursor()
    clauses = []
    values = []

    if project_id is not None:
        clauses.append("project_id=?")
        values.append(project_id)
    if memo_type:
        clauses.append("memo_type=?")
        values.append(memo_type)
    if status:
        clauses.append("status=?")
        values.append(status)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    c.execute(f"SELECT * FROM memos {where} ORDER BY priority DESC, created_at DESC", values)
    rows = c.fetchall()
    conn.close()
    return [_memo_to_dict(r) for r in rows]


def complete_task(memo_id: int):
    """Mark a task/memo as done."""
    conn = _conn()
    c = conn.cursor()
    c.execute("UPDATE memos SET status='done', updated_at=? WHERE id=?",
              (time.time(), memo_id))
    conn.commit()
    conn.close()


def search_memos(query: str) -> List[dict]:
    """Full-text search across memos."""
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM memos WHERE content LIKE ? OR title LIKE ? ORDER BY updated_at DESC LIMIT 20",
        (f"%{query}%", f"%{query}%")
    )
    rows = c.fetchall()
    conn.close()
    return [_memo_to_dict(r) for r in rows]


def _memo_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    d["tags"] = json.loads(d.get("tags", "[]"))
    if d.get("created_at"):
        d["created_at_str"] = datetime.fromtimestamp(d["created_at"]).strftime("%Y-%m-%d %H:%M")
    return d


def get_all_open_tasks_summary() -> str:
    """Voice-friendly summary of all open tasks across projects."""
    tasks = get_memos(memo_type="task", status="open")
    if not tasks:
        return "No open tasks."

    # Group by project
    conn = _conn()
    c = conn.cursor()
    project_names = {}
    for t in tasks:
        if t["project_id"] and t["project_id"] not in project_names:
            c.execute("SELECT name FROM projects WHERE id=?", (t["project_id"],))
            row = c.fetchone()
            if row:
                project_names[t["project_id"]] = row["name"]
    conn.close()

    lines = [f"You have {len(tasks)} open tasks."]
    for t in tasks[:5]:
        proj = project_names.get(t["project_id"], "General")
        lines.append(f"In {proj}: {t['content'][:60]}")

    return " ".join(lines)


def get_project_log(project_id: int, limit: int = 10) -> List[dict]:
    """Get activity log for a project."""
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM project_log WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
        (project_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]