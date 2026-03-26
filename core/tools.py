"""Shared tool implementations for NOVA assistant."""
import io
import sys
import traceback
from typing import Dict, Any

import requests


def run_code_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute code via the CODE tool. Currently only Python is supported."""
    language = args.get("language", "python").lower()
    code = args.get("code", "")

    if language != "python":
        return {"error": f"Unsupported language: {language}. Only python is supported."}

    local_vars = {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        output = stdout_capture.getvalue().strip()
        errors = stderr_capture.getvalue().strip()

        result = {
            "success": True,
            "output": output,
            "errors": errors,
            "locals": {k: repr(v) for k, v in local_vars.items() if k not in ("__builtins__",)},
        }
        return result
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": str(e), "traceback": tb}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a simple web search using DuckDuckGo Instant Answer API."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing search query"}

    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "t": "nova"
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {
            "query": query,
            "abstract": data.get("AbstractText", ""),
            "abstract_url": data.get("AbstractURL", ""),
            "related_topics": [],
            "entity": data.get("Entity", ""),
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
