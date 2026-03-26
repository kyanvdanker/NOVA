"""
Agenda Manager — todos, events, reminders, with voice notifications.
"""
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable

from config.settings import REMINDER_CHECK_INTERVAL


class AgendaManager:
    def __init__(self, memory, speak_callback: Callable = None):
        self.memory = memory
        self.speak = speak_callback
        self._checker_running = False

    def execute(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "add_todo": self._add_todo,
            "add_event": self._add_event,
            "add_reminder": self._add_reminder,
            "list_todos": self._list_todos,
            "list_events": self._list_events,
            "list_all": self._list_all,
            "complete": self._complete,
            "delete": self._delete,
            "today": self._get_today,
            "upcoming": self._get_upcoming,
            "overdue": self._get_overdue,
        }
        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(handlers.keys())}
        try:
            return handler(**args)
        except Exception as e:
            return {"error": str(e)}

    def _add_todo(self, task: str, description: str = None, due: str = None,
                  priority: int = 2, **_) -> Dict:
        item_id = self.memory.add_agenda_item(
            type_="todo",
            title=task,
            description=description,
            due_date=due,
            priority=priority
        )
        return {"success": True, "id": item_id, "task": task}

    def _add_event(self, title: str, date: str, time_str: str = None,
                   description: str = None, **_) -> Dict:
        item_id = self.memory.add_agenda_item(
            type_="event",
            title=title,
            description=description,
            due_date=date,
            due_time=time_str,
            priority=3
        )
        return {"success": True, "id": item_id, "event": title, "date": date}

    def _add_reminder(self, title: str, date: str, time_str: str = None,
                      description: str = None, recurrence: str = None, **_) -> Dict:
        item_id = self.memory.add_agenda_item(
            type_="reminder",
            title=title,
            description=description,
            due_date=date,
            due_time=time_str,
            recurrence=recurrence,
            priority=3
        )
        return {"success": True, "id": item_id, "reminder": title}

    def _list_todos(self, **_) -> Dict:
        items = self.memory.get_agenda_items(status="pending", type_="todo")
        return {"todos": items, "count": len(items)}

    def _list_events(self, **_) -> Dict:
        items = self.memory.get_agenda_items(status="pending", type_="event")
        return {"events": items, "count": len(items)}

    def _list_all(self, **_) -> Dict:
        pending = self.memory.get_agenda_items(status="pending")
        return {"items": pending, "count": len(pending)}

    def _complete(self, item_id: int, **_) -> Dict:
        self.memory.complete_agenda_item(item_id)
        return {"success": True, "completed_id": item_id}

    def _delete(self, item_id: int, **_) -> Dict:
        self.memory.delete_agenda_item(item_id)
        return {"success": True}

    def _get_today(self, **_) -> Dict:
        today = datetime.now().strftime("%Y-%m-%d")
        all_items = self.memory.get_agenda_items(status="pending")
        today_items = [i for i in all_items
                       if not i.get("due_date") or i["due_date"] <= today]
        return {
            "date": today,
            "items": today_items,
            "count": len(today_items)
        }

    def _get_upcoming(self, days: int = 7, **_) -> Dict:
        now = datetime.now()
        future = (now + timedelta(days=days)).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")
        all_items = self.memory.get_agenda_items(status="pending")
        upcoming = [i for i in all_items
                    if i.get("due_date") and today <= i["due_date"] <= future]
        upcoming.sort(key=lambda x: x.get("due_date", ""))
        return {"items": upcoming, "count": len(upcoming), "days_ahead": days}

    def _get_overdue(self, **_) -> Dict:
        today = datetime.now().strftime("%Y-%m-%d")
        all_items = self.memory.get_agenda_items(status="pending")
        overdue = [i for i in all_items
                   if i.get("due_date") and i["due_date"] < today]
        return {"items": overdue, "count": len(overdue)}

    def start_reminder_checker(self):
        """Start background thread for reminder notifications."""
        self._checker_running = True
        thread = threading.Thread(target=self._reminder_loop, daemon=True)
        thread.start()

    def _reminder_loop(self):
        while self._checker_running:
            try:
                due = self.memory.get_due_reminders()
                for item in due:
                    msg = f"Reminder: {item['title']}"
                    if item.get("description"):
                        msg += f" — {item['description']}"
                    print(f"\n  🔔 {msg}")
                    if self.speak:
                        asyncio.run(self.speak(msg))
                    # Mark as done (or handle recurrence)
                    if item.get("recurrence"):
                        self._handle_recurrence(item)
                    else:
                        self.memory.complete_agenda_item(item["id"])
            except Exception as e:
                pass
            time.sleep(REMINDER_CHECK_INTERVAL)

    def _handle_recurrence(self, item: Dict):
        """Reschedule recurring reminders."""
        recurrence = item.get("recurrence", "").lower()
        try:
            due = datetime.strptime(item["due_date"], "%Y-%m-%d")
            if recurrence == "daily":
                new_due = due + timedelta(days=1)
            elif recurrence == "weekly":
                new_due = due + timedelta(weeks=1)
            elif recurrence == "monthly":
                # Rough monthly
                new_due = due + timedelta(days=30)
            else:
                self.memory.complete_agenda_item(item["id"])
                return

            self.memory.complete_agenda_item(item["id"])
            self.memory.add_agenda_item(
                type_="reminder",
                title=item["title"],
                description=item.get("description"),
                due_date=new_due.strftime("%Y-%m-%d"),
                due_time=item.get("due_time"),
                recurrence=recurrence
            )
        except Exception:
            self.memory.complete_agenda_item(item["id"])

    def stop(self):
        self._checker_running = False

    def format_agenda_summary(self) -> str:
        """Return a quick agenda summary string."""
        today_result = self._get_today()
        overdue_result = self._get_overdue()
        upcoming_result = self._get_upcoming(days=3)

        lines = []
        if overdue_result["count"] > 0:
            lines.append(f"⚠️  {overdue_result['count']} overdue item(s)")
        if today_result["count"] > 0:
            lines.append(f"📅 Today: {today_result['count']} item(s)")
            for item in today_result["items"][:3]:
                lines.append(f"   • {item['title']}")
        if upcoming_result["count"] > 0:
            lines.append(f"📆 Next 3 days: {upcoming_result['count']} item(s)")
        if not lines:
            lines.append("✅ Nothing scheduled — you're free!")
        return "\n".join(lines)