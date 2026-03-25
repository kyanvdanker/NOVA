"""
NOVA Scheduler
Manages timed events: reminders, daily briefings, background tasks.
Parses natural language times using dateutil.
"""

import time
import threading
import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.scheduler")

SCHED_DB = config.STORAGE_DIR / "scheduler.db"


@dataclass
class ScheduledJob:
    id: int
    name: str
    trigger_time: float
    repeat_sec: Optional[float]
    payload: dict
    active: bool = True


class Scheduler:
    """
    Simple but robust job scheduler.
    Persists jobs to SQLite so reminders survive restarts.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fired_callback: Optional[Callable[[ScheduledJob], None]] = None
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(SCHED_DB))
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            trigger_time REAL,
            repeat_sec REAL,
            payload TEXT,
            active INTEGER DEFAULT 1
        )""")
        conn.commit()
        conn.close()

    def set_fired_callback(self, cb: Callable):
        """Called when a job fires."""
        self._fired_callback = cb

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Scheduler started")

    def stop(self):
        self._running = False

    def _run_loop(self):
        while self._running:
            try:
                self._check_jobs()
            except Exception as e:
                log.error(f"Scheduler error: {e}")
            time.sleep(10)  # check every 10 seconds

    def _check_jobs(self):
        now = time.time()
        conn = sqlite3.connect(str(SCHED_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM jobs WHERE active=1 AND trigger_time <= ?", (now,))
        due = c.fetchall()

        for row in due:
            job = ScheduledJob(
                id=row["id"],
                name=row["name"],
                trigger_time=row["trigger_time"],
                repeat_sec=row["repeat_sec"],
                payload=json.loads(row["payload"]),
                active=bool(row["active"]),
            )
            log.info(f"Job fired: {job.name}")
            if self._fired_callback:
                threading.Thread(
                    target=self._fired_callback, args=(job,), daemon=True
                ).start()

            if job.repeat_sec:
                # Schedule next occurrence
                c.execute(
                    "UPDATE jobs SET trigger_time=? WHERE id=?",
                    (now + job.repeat_sec, job.id)
                )
            else:
                c.execute("UPDATE jobs SET active=0 WHERE id=?", (job.id,))

        conn.commit()
        conn.close()

    def add_reminder(self, message: str, when_str: str) -> Optional[dict]:
        """
        Add a reminder using natural language time.
        Returns the job info or None on failure.
        """
        trigger_time = self._parse_time(when_str)
        if not trigger_time:
            return None

        conn = sqlite3.connect(str(SCHED_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO jobs (name, trigger_time, repeat_sec, payload) VALUES (?, ?, ?, ?)",
            ("reminder", trigger_time, None, json.dumps({"message": message, "type": "reminder"}))
        )
        job_id = c.lastrowid
        conn.commit()
        conn.close()

        when_dt = datetime.fromtimestamp(trigger_time)
        log.info(f"Reminder set: '{message}' at {when_dt.strftime('%H:%M on %A')}")
        return {
            "id": job_id,
            "message": message,
            "trigger_time": trigger_time,
            "when_str": when_dt.strftime("%I:%M %p on %A, %B %d"),
        }

    def add_daily_job(self, name: str, hour: int, minute: int, payload: dict) -> int:
        """Add a recurring daily job."""
        import datetime as dt_mod
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + dt_mod.timedelta(days=1)

        conn = sqlite3.connect(str(SCHED_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO jobs (name, trigger_time, repeat_sec, payload) VALUES (?, ?, ?, ?)",
            (name, next_run.timestamp(), 86400, json.dumps(payload))
        )
        job_id = c.lastrowid
        conn.commit()
        conn.close()
        log.info(f"Daily job '{name}' scheduled at {hour:02d}:{minute:02d}")
        return job_id

    def cancel_job(self, job_id: int):
        conn = sqlite3.connect(str(SCHED_DB))
        c = conn.cursor()
        c.execute("UPDATE jobs SET active=0 WHERE id=?", (job_id,))
        conn.commit()
        conn.close()

    def list_reminders(self) -> List[dict]:
        """List upcoming active reminders."""
        now = time.time()
        conn = sqlite3.connect(str(SCHED_DB))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM jobs WHERE active=1 AND trigger_time > ? ORDER BY trigger_time",
                  (now,))
        rows = c.fetchall()
        conn.close()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            result.append({
                "id": row["id"],
                "name": row["name"],
                "message": payload.get("message", ""),
                "trigger_time": row["trigger_time"],
                "when_str": datetime.fromtimestamp(row["trigger_time"]).strftime(
                    "%I:%M %p on %A, %B %d"
                ),
            })
        return result

    def _parse_time(self, when_str: str) -> Optional[float]:
        """Parse natural language time to timestamp."""
        try:
            from dateutil.parser import parse
            from dateutil.relativedelta import relativedelta
            import re

            when_lower = when_str.lower().strip()
            now = datetime.now()

            # Relative patterns
            patterns = [
                (r"in (\d+) minute", lambda m: now.timestamp() + int(m.group(1)) * 60),
                (r"in (\d+) hour", lambda m: now.timestamp() + int(m.group(1)) * 3600),
                (r"in (\d+) second", lambda m: now.timestamp() + int(m.group(1))),
                (r"tomorrow at (.+)", lambda m: (now + relativedelta(days=1)).replace(
                    hour=parse(m.group(1)).hour,
                    minute=parse(m.group(1)).minute,
                    second=0
                ).timestamp()),
            ]

            for pattern, handler in patterns:
                match = re.search(pattern, when_lower)
                if match:
                    return handler(match)

            # Try dateutil parse
            parsed = parse(when_str, default=now)
            if parsed <= now:
                parsed = parsed + __import__('datetime').timedelta(days=1)
            return parsed.timestamp()

        except Exception as e:
            log.warning(f"Time parse failed for '{when_str}': {e}")
            return None