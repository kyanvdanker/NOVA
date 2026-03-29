"""
NOVA Self-Editing Tools
-----------------------
These are the tools that let NOVA reliably edit its own source files and add
custom skills.  They are separate from tools.py so they can be imported cleanly.

Tools exposed here:
  patch_file_tool  — find-and-replace inside any file (the real "edit code" tool)
  add_skill_tool   — add a validated Python function to custom_skills.py and reload
  verify_file_tool — read a slice of a file so NOVA can confirm a change landed
"""
import ast
import importlib
import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Any

from config.settings import BASE_DIR

CUSTOM_SKILLS_PATH = BASE_DIR / "skills" / "custom_skills.py"


# ─── PATCH_FILE ───────────────────────────────────────────────────────────────

def patch_file_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find-and-replace inside a file.  Think of it as a precise scalpel for
    editing source code — much safer than rewriting whole files.

    Required args:
        path        — file to edit (absolute, or relative to project root)
        find        — exact string to find (must match literally)
        replace     — string to write in its place

    Optional args:
        allow_create — if True and file does not exist, create it (default False)
        count        — max replacements to make (default 1; 0 = all occurrences)

    Returns:
        success, replacements_made, preview (±3 lines around first change)

    Notes:
        - 'find' must appear in the file; if it does not the tool returns an error
          and makes no changes — so the model never silently corrupts a file.
        - To append to a file use FILE action "write" with append:true instead.
    """
    path_str     = args.get("path", "")
    find_str     = args.get("find", "")
    replace_str  = args.get("replace", "")
    allow_create = bool(args.get("allow_create", False))
    max_count    = int(args.get("count", 1))   # 0 = unlimited

    if not path_str:
        return {"error": "Missing required arg: path"}
    if find_str == "":
        return {"error": "Missing required arg: find  (use FILE write/append to add new content)"}

    # Resolve path safely inside project root
    p = Path(path_str)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    if not str(p).startswith(str(BASE_DIR.resolve())):
        return {"error": f"Path outside project root is not allowed: {p}"}

    if not p.exists():
        if allow_create:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(replace_str, encoding="utf-8")
            return {"success": True, "created": True, "path": str(p)}
        return {"error": f"File not found: {p}"}

    original = p.read_text(encoding="utf-8")

    if find_str not in original:
        # Give a useful diagnostic: show up to 3 lines where we'd expect the text
        snippet = original[:3000]
        return {
            "error": (
                f"'find' string not found in {p.name}. "
                "Check for extra spaces, different indentation, or wrong quotes. "
                f"File starts with:\n{snippet[:500]}"
            ),
            "file_length": len(original),
        }

    if max_count == 0:
        new_content = original.replace(find_str, replace_str)
        count_made  = original.count(find_str)
    else:
        new_content = original.replace(find_str, replace_str, max_count)
        count_made  = min(original.count(find_str), max_count)

    p.write_text(new_content, encoding="utf-8")

    # Build a ±3-line preview around the first change
    lines   = new_content.splitlines()
    idx     = new_content.find(replace_str)
    line_no = new_content[:idx].count("\n")
    start   = max(0, line_no - 3)
    end     = min(len(lines), line_no + 4)
    preview = "\n".join(f"{start+i+1}: {l}" for i, l in enumerate(lines[start:end]))

    return {
        "success":           True,
        "path":              str(p),
        "replacements_made": count_made,
        "preview":           preview,
    }


# ─── ADD_SKILL ────────────────────────────────────────────────────────────────

def add_skill_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new Python function to skills/custom_skills.py and immediately
    hot-reload it so NOVA can use it in the same session.

    Required args:
        name        — skill identifier, e.g. "format_json"  (snake_case, unique)
        description — one-line description shown in /skills list
        code        — the *body* of the function (indented with 4 spaces).
                      Do NOT include the def line — that is generated for you.
                      The function receives one argument: args (dict).
                      It must return a dict.

    Optional args:
        func_name   — Python function name (defaults to name)

    Returns:
        success, skill_name, path, or a detailed error message.

    Example:
        name: "ping_host"
        description: "Ping a hostname and return packet loss"
        code: |
            import subprocess
            host = args.get("host", "google.com")
            r = subprocess.run(["ping", "-c", "2", host],
                               capture_output=True, text=True, timeout=5)
            return {"output": r.stdout, "success": r.returncode == 0}
    """
    name        = args.get("name", "").strip()
    description = args.get("description", "").strip().replace('"', "'")
    code        = args.get("code", "")
    func_name   = args.get("func_name", name).strip() or name

    # ── Validation ────────────────────────────────────────────────────────────
    if not name:
        return {"error": "Missing required arg: name"}
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return {"error": f"name must be snake_case lowercase, got: {name!r}"}
    if not description:
        return {"error": "Missing required arg: description"}
    if not code:
        return {"error": "Missing required arg: code (the function body, indented 4 spaces)"}

    # Ensure the body is properly indented
    body_lines = code.splitlines()
    # Strip leading blank lines
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    # Re-indent to exactly 4 spaces if needed
    if body_lines:
        first_line     = body_lines[0]
        current_indent = len(first_line) - len(first_line.lstrip())
        if current_indent == 0:
            body_lines = ["    " + l for l in body_lines]
        elif current_indent != 4:
            body_lines = [l[current_indent - 4:] if l.startswith(" " * current_indent)
                          else "    " + l.lstrip()
                          for l in body_lines]
    indented_body = "\n".join(body_lines)

    # Validate Python syntax before touching the file
    full_func = f"def {func_name}(args: dict) -> dict:\n{indented_body}"
    try:
        ast.parse(full_func)
    except SyntaxError as e:
        return {
            "error": f"Syntax error in code body: {e}",
            "tip":   "Indent the body 4 spaces. Do not include the def line.",
            "parsed_function": full_func,
        }

    # ── Duplicate check ───────────────────────────────────────────────────────
    CUSTOM_SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_SKILLS_PATH.exists():
        CUSTOM_SKILLS_PATH.write_text(
            '"""\nCustom skills.\n"""\nSKILLS = {}\n\n'
            'def register_skill(name, description, func):\n'
            '    SKILLS[name] = {"description": description, "func": func}\n',
            encoding="utf-8",
        )

    existing = CUSTOM_SKILLS_PATH.read_text(encoding="utf-8")
    if f'register_skill("{name}"' in existing:
        return {"error": f"Skill '{name}' already registered. Use PATCH_FILE to modify it."}
    if f"def {func_name}(" in existing:
        return {"error": f"Function '{func_name}' already exists. Choose a different func_name."}

    # ── Write ─────────────────────────────────────────────────────────────────
    block = (
        f"\n\n"
        f"def {func_name}(args: dict) -> dict:\n"
        f'    """{description}"""\n'
        f"{indented_body}\n\n"
        f'register_skill("{name}", "{description}", {func_name})\n'
    )

    with open(CUSTOM_SKILLS_PATH, "a", encoding="utf-8") as f:
        f.write(block)

    # ── Hot-reload ────────────────────────────────────────────────────────────
    reload_error = None
    try:
        module_name = "skills.custom_skills"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            spec   = importlib.util.spec_from_file_location(module_name, CUSTOM_SKILLS_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except Exception as e:
        reload_error = str(e)

    result: Dict[str, Any] = {
        "success":    True,
        "skill_name": name,
        "func_name":  func_name,
        "path":       str(CUSTOM_SKILLS_PATH),
    }
    if reload_error:
        result["reload_warning"] = reload_error
    return result


# ─── VERIFY_FILE ─────────────────────────────────────────────────────────────

def verify_file_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a slice of a file so NOVA can confirm an edit landed correctly,
    or inspect a source file before trying to patch it.

    Args:
        path        — file to read
        contains    — (optional) string to search for; returns found:true/false
        start_line  — (optional) first line to return (1-indexed)
        end_line    — (optional) last line to return  (1-indexed, inclusive)

    If neither start_line nor contains is given, returns the first 60 lines.
    """
    path_str = args.get("path", "")
    if not path_str:
        return {"error": "Missing required arg: path"}

    p = Path(path_str)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    if not p.exists():
        return {"error": f"File not found: {p}"}

    content = p.read_text(encoding="utf-8", errors="replace")
    lines   = content.splitlines()

    result: Dict[str, Any] = {
        "path":        str(p),
        "total_lines": len(lines),
        "total_chars": len(content),
    }

    search = args.get("contains", "")
    if search:
        result["contains"] = search in content
        if search in content:
            idx     = content.find(search)
            line_no = content[:idx].count("\n")
            start   = max(0, line_no - 2)
            end     = min(len(lines), line_no + 5)
            result["found_at_line"] = line_no + 1
            result["context"] = "\n".join(
                f"{start+i+1}: {l}" for i, l in enumerate(lines[start:end])
            )

    start_line = args.get("start_line")
    end_line   = args.get("end_line")
    if start_line is not None:
        s = max(0, int(start_line) - 1)
        e = int(end_line) if end_line is not None else s + 40
        result["slice"] = "\n".join(
            f"{s+i+1}: {l}" for i, l in enumerate(lines[s:e])
        )
    elif not search:
        result["slice"] = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:60]))

    return result