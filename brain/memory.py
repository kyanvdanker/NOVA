"""
NOVA Memory System
Learns and remembers:
- User profile and preferences
- Episodic memories (past interactions)
- Facts about the user
- Project contexts
Uses SQLite for structured data + ChromaDB for semantic vector search.
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.memory")


class MemoryManager:
    """
    Persistent memory with semantic search.
    Automatically learns from conversations.
    """

    def __init__(self):
        self._db_path = config.MEMORY_DB_PATH
        self._chroma = None
        self._collection = None
        self._init_db()
        self._init_vector_store()
        log.info("Memory system initialized")

    # ─── Database ─────────────────────────────────────────────────────────────

    def _init_db(self):
        """Initialize SQLite database."""
        conn = self._conn()
        c = conn.cursor()

        # User profile
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT,
                confidence REAL DEFAULT 1.0,
                updated_at REAL
            )
        """)

        # Episodic memory (interaction history)
        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_said TEXT,
                nova_said TEXT,
                timestamp REAL,
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]'
            )
        """)

        # Explicit facts
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,   -- 'preference', 'personal', 'schedule', etc.
                fact TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,     -- 'stated', 'inferred'
                created_at REAL,
                accessed_at REAL
            )
        """)

        # Preferences
        c.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                strength REAL DEFAULT 1.0,
                updated_at REAL
            )
        """)

        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_vector_store(self):
        """Initialize ChromaDB for semantic search."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_PATH))
            self._collection = client.get_or_create_collection(
                name="nova_memories",
                metadata={"hnsw:space": "cosine"}
            )
            log.info(f"ChromaDB initialized ({self._collection.count()} memories)")
        except ImportError:
            log.warning("chromadb not installed. Semantic search disabled. "
                        "Install with: pip install chromadb")
        except Exception as e:
            log.warning(f"ChromaDB init failed: {e}. Falling back to SQLite search.")

    # ─── Core Operations ──────────────────────────────────────────────────────

    def store_interaction(self, user_said: str, nova_said: str):
        """Store a conversation turn and extract learnings."""
        timestamp = time.time()

        # Calculate importance (simple heuristic)
        importance = self._calc_importance(user_said, nova_said)

        # Store episode
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO episodes (user_said, nova_said, timestamp, importance) VALUES (?, ?, ?, ?)",
            (user_said, nova_said, timestamp, importance)
        )
        episode_id = c.lastrowid
        conn.commit()
        conn.close()

        # Add to vector store
        if self._collection:
            try:
                self._collection.add(
                    documents=[f"User: {user_said}\nNova: {nova_said}"],
                    metadatas=[{"type": "episode", "timestamp": timestamp,
                                "importance": importance}],
                    ids=[f"ep_{episode_id}"]
                )
            except Exception as e:
                log.debug(f"Vector store add failed: {e}")

        # Extract and learn from this interaction
        self._learn_from_interaction(user_said, nova_said)

    def _learn_from_interaction(self, user_said: str, nova_said: str):
        """
        Automatically extract facts and preferences from conversation.
        Uses pattern matching for quick extraction; LLM can enrich later.
        """
        text = user_said.lower()

        # Name patterns
        name_patterns = [
            "my name is ", "i'm ", "i am ", "call me ",
            "people call me ", "you can call me "
        ]
        for pattern in name_patterns:
            if pattern in text:
                rest = text.split(pattern, 1)[1].split()[0].strip(" .,!?")
                if 2 <= len(rest) <= 30 and rest.isalpha():
                    self.update_profile("name", rest.title(), source="stated")
                    log.info(f"Learned name: {rest.title()}")
                    break

        # Preference patterns
        if any(p in text for p in ["i like", "i love", "i enjoy", "i prefer", "i hate", "i dislike"]):
            self.add_fact("preference", user_said, source="stated")

        # Personal facts
        if any(p in text for p in ["i work", "i live", "i study", "my job", "my home", "i'm from"]):
            self.add_fact("personal", user_said, source="stated")

        # Schedule patterns
        if any(p in text for p in ["every morning", "usually at", "i wake up", "i go to bed",
                                    "every day", "each week"]):
            self.add_fact("schedule", user_said, source="stated")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant memories for a query.
        Uses semantic search if available, else recency + keyword.
        """
        memories = []

        if self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(k, self._collection.count()),
                    where={"type": "episode"},
                )
                if results["documents"]:
                    memories.extend(results["documents"][0][:k])
            except Exception as e:
                log.debug(f"Vector search failed: {e}")

        if not memories:
            # Fallback: keyword search in SQLite
            conn = self._conn()
            c = conn.cursor()
            # Simple keyword search
            keywords = [w for w in query.split() if len(w) > 3]
            if keywords:
                placeholders = " OR ".join([f"user_said LIKE ?" for _ in keywords])
                values = [f"%{kw}%" for kw in keywords]
                c.execute(
                    f"SELECT user_said, nova_said FROM episodes "
                    f"WHERE {placeholders} ORDER BY importance DESC, timestamp DESC LIMIT ?",
                    values + [k]
                )
                for row in c.fetchall():
                    memories.append(f"User: {row['user_said']}\nNova: {row['nova_said']}")
            conn.close()

        return memories[:k]

    def update_profile(self, key: str, value: str, confidence: float = 1.0, source: str = "stated"):
        """Update a user profile attribute."""
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            """INSERT INTO user_profile (key, value, confidence, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
               confidence=excluded.confidence, updated_at=excluded.updated_at""",
            (key, value, confidence, time.time())
        )
        conn.commit()
        conn.close()
        log.info(f"Profile updated: {key} = {value}")

    def get_profile(self) -> Dict[str, str]:
        """Get full user profile."""
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT key, value, confidence FROM user_profile ORDER BY updated_at DESC")
        profile = {row["key"]: row["value"] for row in c.fetchall()}
        conn.close()
        return profile

    def get_user_profile_text(self) -> str:
        """Get profile as readable text for LLM context."""
        profile = self.get_profile()
        if not profile:
            return ""
        lines = []
        if "name" in profile:
            lines.append(f"Owner's name: {profile['name']}")
        for k, v in profile.items():
            if k != "name":
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def add_fact(self, category: str, fact: str, confidence: float = 1.0, source: str = "stated"):
        """Store a learned fact."""
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO facts (category, fact, confidence, source, created_at, accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (category, fact, confidence, source, time.time(), time.time())
        )
        conn.commit()
        conn.close()

    def get_facts(self, category: str = None) -> List[Dict]:
        """Retrieve stored facts, optionally filtered by category."""
        conn = self._conn()
        c = conn.cursor()
        if category:
            c.execute("SELECT * FROM facts WHERE category=? ORDER BY confidence DESC",
                      (category,))
        else:
            c.execute("SELECT * FROM facts ORDER BY confidence DESC, accessed_at DESC")
        facts = [dict(row) for row in c.fetchall()]
        conn.close()
        return facts

    def set_preference(self, key: str, value: str, strength: float = 1.0):
        """Set or update a user preference."""
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            """INSERT INTO preferences (key, value, strength, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
               strength=excluded.strength, updated_at=excluded.updated_at""",
            (key, value, strength, time.time())
        )
        conn.commit()
        conn.close()

    def get_preference(self, key: str, default: str = None) -> Optional[str]:
        """Get a preference value."""
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT value FROM preferences WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row["value"] if row else default

    def get_all_preferences(self) -> Dict[str, str]:
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT key, value FROM preferences")
        prefs = {r["key"]: r["value"] for r in c.fetchall()}
        conn.close()
        return prefs

    # ─── Stats ────────────────────────────────────────────────────────────────

    def _calc_importance(self, user_said: str, nova_said: str) -> float:
        """Simple importance heuristic."""
        score = 0.3
        # Longer exchanges are more important
        combined_len = len(user_said) + len(nova_said)
        score += min(0.3, combined_len / 1000)
        # Questions are more important
        if "?" in user_said:
            score += 0.1
        # Personal info is important
        personal_keywords = ["i am", "my name", "i live", "i work", "i like", "i hate"]
        for kw in personal_keywords:
            if kw in user_said.lower():
                score += 0.1
                break
        return min(1.0, score)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as n FROM episodes")
        ep_count = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) as n FROM facts")
        fact_count = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) as n FROM user_profile")
        profile_count = c.fetchone()["n"]
        conn.close()

        return {
            "episodes": ep_count,
            "facts": fact_count,
            "profile_keys": profile_count,
            "vector_memories": self._collection.count() if self._collection else 0,
        }

    def forget(self, query: str = None, all: bool = False):
        """Forget memories. Use carefully."""
        conn = self._conn()
        c = conn.cursor()
        if all:
            c.execute("DELETE FROM episodes")
            c.execute("DELETE FROM facts")
            c.execute("DELETE FROM user_profile")
            log.warning("All memories cleared!")
        elif query:
            c.execute("DELETE FROM episodes WHERE user_said LIKE ?", (f"%{query}%",))
            c.execute("DELETE FROM facts WHERE fact LIKE ?", (f"%{query}%",))
        conn.commit()
        conn.close()