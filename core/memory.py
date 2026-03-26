"""
Memory System — SQLite-backed persistent memory for NOVA.
Stores conversations, memos, facts, skills log, and interaction stats.
"""
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from config.settings import DATA_DIR


DB_PATH = DATA_DIR / "NOVA.db"


class Memory:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        # Conversation history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT
            )
        """)

        # Memos
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Facts (things NOVA learned about the user/world)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                source TEXT,
                updated_at REAL NOT NULL
            )
        """)

        # Interaction stats (for self-improvement)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT,
                tool_used TEXT,
                success INTEGER DEFAULT 1,
                duration_ms INTEGER,
                timestamp REAL NOT NULL
            )
        """)

        # Skills log (self-improvement history)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS skills_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT NOT NULL,
                description TEXT,
                code TEXT,
                added_at REAL NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)

        # Agenda / todos
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agenda (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                due_date TEXT,
                due_time TEXT,
                priority INTEGER DEFAULT 2,
                status TEXT DEFAULT 'pending',
                recurrence TEXT,
                created_at REAL NOT NULL,
                completed_at REAL
            )
        """)

        self.conn.commit()

    # ── Conversations ────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, session_id: str = "default"):
        self.conn.execute(
            "INSERT INTO conversations (role, content, timestamp, session_id) VALUES (?,?,?,?)",
            (role, content, time.time(), session_id)
        )
        self.conn.commit()

    def get_recent_messages(self, n: int = 20, session_id: str = "default") -> List[Dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (session_id, n)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_total_interactions(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM conversations WHERE role='user'").fetchone()
        return row["c"]

    # ── Memos ────────────────────────────────────────────────────────────────

    def save_memo(self, title: str, content: str, tags: List[str] = None) -> int:
        tags_json = json.dumps(tags or [])
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO memos (title, content, tags, created_at, updated_at) VALUES (?,?,?,?,?)",
            (title, content, tags_json, now, now)
        )
        self.conn.commit()
        return cur.lastrowid

    def update_memo(self, memo_id: int, title: str = None, content: str = None):
        if title:
            self.conn.execute("UPDATE memos SET title=?, updated_at=? WHERE id=?",
                              (title, time.time(), memo_id))
        if content:
            self.conn.execute("UPDATE memos SET content=?, updated_at=? WHERE id=?",
                              (content, time.time(), memo_id))
        self.conn.commit()

    def get_memo(self, memo_id: int) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM memos WHERE id=?", (memo_id,)).fetchone()
        return dict(row) if row else None

    def search_memos(self, query: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM memos WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC",
            (f"%{query}%", f"%{query}%")
        ).fetchall()
        return [dict(r) for r in rows]

    def list_memos(self) -> List[Dict]:
        rows = self.conn.execute("SELECT id, title, tags, updated_at FROM memos ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_memo(self, memo_id: int):
        self.conn.execute("DELETE FROM memos WHERE id=?", (memo_id,))
        self.conn.commit()

    # ── Facts ────────────────────────────────────────────────────────────────

    def set_fact(self, key: str, value: str, source: str = "conversation"):
        self.conn.execute(
            "INSERT OR REPLACE INTO facts (key, value, source, updated_at) VALUES (?,?,?,?)",
            (key, value, source, time.time())
        )
        self.conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def get_all_facts(self) -> Dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ── Interaction Log ──────────────────────────────────────────────────────

    def log_interaction(self, intent: str = None, tool_used: str = None,
                        success: bool = True, duration_ms: int = None):
        self.conn.execute(
            "INSERT INTO interaction_log (intent, tool_used, success, duration_ms, timestamp) VALUES (?,?,?,?,?)",
            (intent, tool_used, int(success), duration_ms, time.time())
        )
        self.conn.commit()

    def get_interaction_stats(self, days: int = 7) -> Dict:
        since = time.time() - days * 86400
        rows = self.conn.execute(
            "SELECT tool_used, COUNT(*) as count FROM interaction_log WHERE timestamp > ? GROUP BY tool_used ORDER BY count DESC",
            (since,)
        ).fetchall()
        return {r["tool_used"] or "chat": r["count"] for r in rows}

    def get_recent_interactions(self, n: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM interaction_log ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Skills Log ───────────────────────────────────────────────────────────

    def add_skill(self, name: str, description: str, code: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO skills_log (skill_name, description, code, added_at) VALUES (?,?,?,?)",
            (name, description, code, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_skills(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM skills_log WHERE active=1 ORDER BY added_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Agenda ───────────────────────────────────────────────────────────────

    def add_agenda_item(self, type_: str, title: str, description: str = None,
                        due_date: str = None, due_time: str = None,
                        priority: int = 2, recurrence: str = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO agenda (type, title, description, due_date, due_time, priority, recurrence, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (type_, title, description, due_date, due_time, priority, recurrence, time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_agenda_items(self, status: str = "pending", type_: str = None) -> List[Dict]:
        if type_:
            rows = self.conn.execute(
                "SELECT * FROM agenda WHERE status=? AND type=? ORDER BY due_date, priority DESC",
                (status, type_)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM agenda WHERE status=? ORDER BY due_date, priority DESC",
                (status,)
            ).fetchall()
        return [dict(r) for r in rows]

    def complete_agenda_item(self, item_id: int):
        self.conn.execute(
            "UPDATE agenda SET status='done', completed_at=? WHERE id=?",
            (time.time(), item_id)
        )
        self.conn.commit()

    def get_due_reminders(self) -> List[Dict]:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")
        rows = self.conn.execute(
            """SELECT * FROM agenda WHERE status='pending' AND type='reminder'
               AND due_date <= ? AND (due_time IS NULL OR due_time <= ?)""",
            (today, current_time)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_agenda_item(self, item_id: int):
        self.conn.execute("DELETE FROM agenda WHERE id=?", (item_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()