"""
Project Manager — Create, manage, and navigate engineering projects.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import DATA_DIR

PROJECTS_DIR = DATA_DIR / "projects"


class ProjectManager:
    def __init__(self):
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    def execute(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "create": self._create_project,
            "list": self._list_projects,
            "open": self._open_project,
            "get": self._get_project,
            "add_file": self._add_file,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "delete_file": self._delete_file,
            "add_note": self._add_note,
            "get_notes": self._get_notes,
            "set_status": self._set_status,
            "delete": self._delete_project,
            "search": self._search_projects,
            "add_task": self._add_task,
            "get_tasks": self._get_tasks,
            "complete_task": self._complete_task,
        }
        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}", "available": list(handlers.keys())}
        try:
            return handler(**args)
        except Exception as e:
            return {"error": str(e)}

    def _meta_path(self, name: str) -> Path:
        return PROJECTS_DIR / name / "project.json"

    def _load_meta(self, name: str) -> Optional[Dict]:
        p = self._meta_path(name)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def _save_meta(self, name: str, meta: Dict):
        p = self._meta_path(name)
        with open(p, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _create_project(self, name: str, description: str = "", language: str = "",
                        tags: List[str] = None, **_) -> Dict:
        proj_dir = PROJECTS_DIR / name
        if proj_dir.exists():
            return {"error": f"Project '{name}' already exists"}

        proj_dir.mkdir(parents=True)
        (proj_dir / "files").mkdir()
        (proj_dir / "notes").mkdir()

        meta = {
            "name": name,
            "description": description,
            "language": language,
            "tags": tags or [],
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tasks": [],
            "notes": []
        }
        self._save_meta(name, meta)

        # Create a README
        readme = f"# {name}\n\n{description}\n\nCreated: {datetime.now().strftime('%Y-%m-%d')}\n"
        with open(proj_dir / "README.md", "w") as f:
            f.write(readme)

        return {"success": True, "project": name, "path": str(proj_dir)}

    def _list_projects(self, status: str = None, **_) -> Dict:
        projects = []
        for p in PROJECTS_DIR.iterdir():
            if p.is_dir():
                meta = self._load_meta(p.name)
                if meta:
                    if status and meta.get("status") != status:
                        continue
                    projects.append({
                        "name": meta["name"],
                        "description": meta.get("description", ""),
                        "status": meta.get("status"),
                        "language": meta.get("language"),
                        "tags": meta.get("tags", []),
                        "updated_at": meta.get("updated_at")
                    })
        projects.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return {"projects": projects, "count": len(projects)}

    def _get_project(self, name: str, **_) -> Dict:
        meta = self._load_meta(name)
        if not meta:
            return {"error": f"Project '{name}' not found"}

        proj_dir = PROJECTS_DIR / name
        files = list((proj_dir / "files").rglob("*"))
        file_list = [str(f.relative_to(proj_dir)) for f in files if f.is_file()]

        return {
            **meta,
            "files": file_list,
            "path": str(proj_dir)
        }

    def _open_project(self, name: str, editor: str = "code", **_) -> Dict:
        proj_dir = PROJECTS_DIR / name
        if not proj_dir.exists():
            return {"error": f"Project '{name}' not found"}
        try:
            import subprocess
            subprocess.Popen([editor, str(proj_dir)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"success": True, "opened": str(proj_dir), "editor": editor}
        except Exception as e:
            return {"error": str(e)}

    def _add_file(self, project: str, filename: str, content: str, **_) -> Dict:
        proj_dir = PROJECTS_DIR / project
        if not proj_dir.exists():
            return {"error": f"Project '{project}' not found"}

        file_path = proj_dir / "files" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Update meta
        meta = self._load_meta(project)
        if meta:
            meta["updated_at"] = datetime.now().isoformat()
            self._save_meta(project, meta)

        return {"success": True, "file": str(file_path)}

    def _read_file(self, project: str, filename: str, **_) -> Dict:
        file_path = PROJECTS_DIR / project / "files" / filename
        if not file_path.exists():
            return {"error": f"File not found: {filename}"}
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "file": filename, "size": len(content)}

    def _list_files(self, project: str, **_) -> Dict:
        proj_dir = PROJECTS_DIR / project / "files"
        if not proj_dir.exists():
            return {"error": f"Project '{project}' not found"}
        files = []
        for f in proj_dir.rglob("*"):
            if f.is_file():
                files.append({
                    "name": str(f.relative_to(proj_dir)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        return {"files": files}

    def _delete_file(self, project: str, filename: str, **_) -> Dict:
        file_path = PROJECTS_DIR / project / "files" / filename
        if file_path.exists():
            os.remove(file_path)
            return {"success": True}
        return {"error": "File not found"}

    def _add_note(self, project: str, content: str, title: str = "", **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}

        note = {
            "id": int(time.time()),
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        meta.setdefault("notes", []).append(note)
        meta["updated_at"] = datetime.now().isoformat()
        self._save_meta(project, meta)
        return {"success": True, "note_id": note["id"]}

    def _get_notes(self, project: str, **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}
        return {"notes": meta.get("notes", [])}

    def _add_task(self, project: str, title: str, priority: int = 2,
                  assignee: str = None, **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}
        task = {
            "id": int(time.time()),
            "title": title,
            "priority": priority,
            "assignee": assignee,
            "status": "todo",
            "created_at": datetime.now().isoformat()
        }
        meta.setdefault("tasks", []).append(task)
        meta["updated_at"] = datetime.now().isoformat()
        self._save_meta(project, meta)
        return {"success": True, "task": task}

    def _get_tasks(self, project: str, status: str = None, **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}
        tasks = meta.get("tasks", [])
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        return {"tasks": tasks}

    def _complete_task(self, project: str, task_id: int, **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}
        for task in meta.get("tasks", []):
            if task["id"] == task_id:
                task["status"] = "done"
                task["completed_at"] = datetime.now().isoformat()
                break
        meta["updated_at"] = datetime.now().isoformat()
        self._save_meta(project, meta)
        return {"success": True}

    def _set_status(self, project: str, status: str, **_) -> Dict:
        meta = self._load_meta(project)
        if not meta:
            return {"error": f"Project '{project}' not found"}
        meta["status"] = status
        meta["updated_at"] = datetime.now().isoformat()
        self._save_meta(project, meta)
        return {"success": True, "status": status}

    def _delete_project(self, name: str, **_) -> Dict:
        import shutil
        proj_dir = PROJECTS_DIR / name
        if not proj_dir.exists():
            return {"error": f"Project '{name}' not found"}
        shutil.rmtree(proj_dir)
        return {"success": True}

    def _search_projects(self, query: str, **_) -> Dict:
        results = []
        q = query.lower()
        for p in PROJECTS_DIR.iterdir():
            if p.is_dir():
                meta = self._load_meta(p.name)
                if not meta:
                    continue
                if (q in meta["name"].lower() or
                        q in meta.get("description", "").lower() or
                        any(q in t.lower() for t in meta.get("tags", []))):
                    results.append(meta)
        return {"results": results}