"""
NOVA Assistant Brain — LLM interface, tool dispatch, conversation management.

Backend support:
  - "ollama"   → Ollama native /api/chat  (keep_alive + num_keep KV-cache)
  - "llamacpp" → llama.cpp /v1/chat/completions  (OpenAI-compatible SSE)
  - "vllm"     → vLLM /v1/chat/completions       (OpenAI-compatible SSE)

Switch with:  LLM_BACKEND=llamacpp  or  LLM_BACKEND=vllm  env var.
"""
import asyncio
import base64
import json
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import threading

from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT,
    LLM_TEMPERATURE, LLM_STREAM, OLLAMA_CONTEXT_LENGTH,
    ASSISTANT_NAME, USER_NAME,
    OLLAMA_KEEP_ALIVE, OLLAMA_NUM_KEEP,
    LLM_BACKEND,
)
from core.memory import Memory
from core.computer_control import ComputerControl
from core.projects import ProjectManager
from core.agenda import AgendaManager
from core.self_improvement import SelfImprovement
from core.tools import (
    run_code_tool, search_tool, weather_tool, calculate_tool, unit_convert_tool,
    hash_tool, encode_tool, json_tool, regex_tool, diff_tool, network_tool,
    file_tool, clipboard_tool, system_info_tool, process_tool, timer_tool,
    datetime_tool, price_tool, currency_tool, translate_tool, text_tool,
    qr_tool, git_tool, package_tool, generate_tool, text_analyze_tool,
)

TOOL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\|\s*(\{.*?\})", re.DOTALL)
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])$')


# ── Persistent HTTP session ───────────────────────────────────────────────────

def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=4,
        pool_maxsize=8,
        max_retries=Retry(total=0),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Warm the model into VRAM on startup (background thread, non-blocking)
    def _warmup():
        try:
            if LLM_BACKEND == "ollama":
                session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": "", "stream": False,
                          "keep_alive": OLLAMA_KEEP_ALIVE, "options": {"num_predict": 1}},
                    timeout=30,
                )
            else:
                # OpenAI-compat: a minimal completion with max_tokens=1
                session.post(
                    f"{OLLAMA_HOST}/v1/chat/completions",
                    json={"model": OLLAMA_MODEL, "stream": False, "max_tokens": 1,
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=30,
                )
        except Exception:
            pass  # Ollama/server may not be running yet

    threading.Thread(target=_warmup, daemon=True).start()
    return session


_SESSION: requests.Session = _build_session()


# ── File helpers ──────────────────────────────────────────────────────────────

def _is_image_file(f: Dict) -> bool:
    mime = f.get("type", "")
    name = f.get("name", "").lower()
    return mime in _IMAGE_MIME_TYPES or name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))


def _extract_base64_data(data_url_or_b64: str) -> str:
    if data_url_or_b64.startswith("data:"):
        _, _, b64 = data_url_or_b64.partition(",")
        return b64
    return data_url_or_b64


def _decode_text_file(f: Dict) -> str:
    raw = _extract_base64_data(f.get("data", ""))
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return ""


# ── Backend-specific request builders ────────────────────────────────────────

def _build_ollama_request(messages: List[Dict]) -> tuple[str, Dict]:
    """Returns (url, body) for the Ollama native API."""
    url = f"{OLLAMA_HOST}/api/chat"
    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": OLLAMA_CONTEXT_LENGTH,
            "num_keep": OLLAMA_NUM_KEEP,
        },
    }
    return url, body


def _build_openai_compat_request(messages: List[Dict]) -> tuple[str, Dict]:
    """
    Returns (url, body) for the OpenAI-compatible /v1/chat/completions endpoint.
    Used by both llama.cpp --server and vLLM.
    """
    url = f"{OLLAMA_HOST}/v1/chat/completions"
    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": OLLAMA_CONTEXT_LENGTH,
    }
    return url, body


def _parse_ollama_line(line: bytes) -> Optional[str]:
    """Extract content token from an Ollama streaming line."""
    try:
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        done = data.get("done", False)
        return None if (done and not content) else (content or None)
    except Exception:
        return None


def _parse_openai_sse_line(line: bytes) -> Optional[str]:
    """
    Extract content token from an OpenAI-compatible SSE line.
    Lines look like:  data: {"choices":[{"delta":{"content":"hello"}}]}
    Sentinel:         data: [DONE]
    """
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if payload == "[DONE]":
        return None  # signal end-of-stream
    try:
        data = json.loads(payload)
        delta = data.get("choices", [{}])[0].get("delta", {})
        return delta.get("content") or None
    except Exception:
        return None


# ── Assistant ─────────────────────────────────────────────────────────────────

class NOVAAssistant:
    def __init__(self):
        self.memory = Memory()
        self.computer = ComputerControl()
        self.projects = ProjectManager()
        self.voice = None
        self._session_id = f"session_{int(time.time())}"
        self.agenda = AgendaManager(self.memory)
        self.self_improve = SelfImprovement(self.memory, assistant_ref=self)
        self._initialized = False
        self._stream_callbacks: List = []

    def add_stream_callback(self, cb):
        self._stream_callbacks.append(cb)

    def remove_stream_callback(self, cb):
        if cb in self._stream_callbacks:
            self._stream_callbacks.remove(cb)

    def set_voice(self, voice_engine):
        self.voice = voice_engine
        self.agenda.speak = voice_engine.speak if voice_engine else None

    async def initialize(self):
        backend_label = {
            "ollama":   f"Ollama native  ({OLLAMA_HOST})",
            "llamacpp": f"llama.cpp      ({OLLAMA_HOST})",
            "vllm":     f"vLLM           ({OLLAMA_HOST})",
        }.get(LLM_BACKEND, OLLAMA_HOST)

        print(f"  🔗 Connecting to {backend_label}...")
        for attempt in range(5):
            try:
                if LLM_BACKEND == "ollama":
                    resp = _SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
                    models = [m["name"] for m in resp.json().get("models", [])]
                    model_base = OLLAMA_MODEL.split(":")[0]
                    if not any(model_base in m for m in models):
                        print(f"  ⚠️  Model '{OLLAMA_MODEL}' not found. Available: {models}")
                        print(f"  📥 Pull it with: ollama pull {OLLAMA_MODEL}")
                    else:
                        print(f"  ✅ Ollama ready — model '{OLLAMA_MODEL}'")
                        print(f"  ⚡ keep_alive={OLLAMA_KEEP_ALIVE}  num_keep={OLLAMA_NUM_KEEP}")
                else:
                    # OpenAI-compat health check: GET /v1/models
                    resp = _SESSION.get(f"{OLLAMA_HOST}/v1/models", timeout=5)
                    models = [m.get("id", "") for m in resp.json().get("data", [])]
                    print(f"  ✅ {LLM_BACKEND} ready — available models: {models or ['(none listed)']}")
                break
            except requests.exceptions.ConnectionError:
                if attempt == 4:
                    print(f"  ❌ Cannot connect to {LLM_BACKEND} server at {OLLAMA_HOST}")
                else:
                    print(f"  ⏳ Waiting for server... ({attempt+1}/5)")
                    await asyncio.sleep(3)
            except Exception as e:
                print(f"  ⚠️  Server check error: {e}")
                break

        self.agenda.start_reminder_checker()
        self.self_improve.start()
        self._initialized = True

    async def run(self):
        print(f"\n  {'='*50}")
        print(f"  {ASSISTANT_NAME} is online. Type your message.")
        print(f"  Commands: /agenda, /projects, /memos, /skills, /status, /tools, /exit")
        print(f"  {'='*50}\n")

        while True:
            try:
                user_input = input(f"  You: ").strip()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    await self._handle_slash_command(user_input)
                    continue
                if user_input.lower() in ("exit", "quit", "/exit"):
                    print(f"  {ASSISTANT_NAME}: Goodbye, {USER_NAME}.")
                    break
                await self.handle_input(user_input, voice_response=False)
            except (KeyboardInterrupt, EOFError):
                print(f"\n  {ASSISTANT_NAME}: Shutting down. See you, {USER_NAME}.")
                break

    async def handle_input(self, text: str, voice_response: bool = False,
                           files: List[Dict] = None) -> str:
        files = files or []
        start_time = time.time()

        image_files = [f for f in files if _is_image_file(f)]
        text_files  = [f for f in files if not _is_image_file(f)]

        user_message_text = text
        if text_files:
            parts = [text]
            for f in text_files:
                content = _decode_text_file(f)
                name = f.get("name", "file")
                ext = name.rsplit(".", 1)[-1] if "." in name else ""
                fence = f"```{ext}\n{content}\n```" if ext else f"```\n{content}\n```"
                parts.append(f"\n\n**Attached file – {name}:**\n{fence}")
            user_message_text = "".join(parts)

        self.memory.add_message("user", user_message_text, self._session_id)
        history = self.memory.get_recent_messages(n=15, session_id=self._session_id)
        messages = self._build_messages(history)

        if image_files:
            last_msg = messages[-1]
            if last_msg["role"] == "user":
                last_msg["images"] = [_extract_base64_data(f["data"]) for f in image_files]
            print(f"  🖼  Sending {len(image_files)} image(s): {', '.join(f['name'] for f in image_files)}")

        print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)

        full_response = ""
        tts_buffer = ""
        tts_tasks: List = []

        async for chunk in self._stream_response(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
            for cb in self._stream_callbacks:
                try:
                    cb("chunk", chunk)
                except Exception:
                    pass
            # Sentence-level streaming TTS
            if voice_response and self.voice:
                tts_buffer += chunk
                sentences = _SENTENCE_END.split(tts_buffer)
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        spoken = self._clean_for_speech(sentence.strip())
                        if spoken and len(spoken) > 3:
                            tts_tasks.append(asyncio.create_task(self.voice.speak(spoken)))
                    tts_buffer = sentences[-1]

        print()
        if voice_response and self.voice and tts_buffer.strip():
            spoken = self._clean_for_speech(tts_buffer.strip())
            if spoken and len(spoken) > 3:
                tts_tasks.append(asyncio.create_task(self.voice.speak(spoken)))

        # ── Tool execution ────────────────────────────────────────────────
        tool_calls = TOOL_PATTERN.findall(full_response)
        tool_results = []

        if tool_calls:
            clean_response = TOOL_PATTERN.sub("", full_response).strip()
            for tool_name, tool_args_str in tool_calls:
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}
                result = await self._execute_tool(tool_name, tool_args)
                tool_results.append({"tool": tool_name, "result": result})
                print(f"  ⚡ [{tool_name}] → {json.dumps(result)[:300]}")
                for cb in self._stream_callbacks:
                    try:
                        cb("tool_result", {"tool": tool_name, "result": result})
                    except Exception:
                        pass

            if tool_results:
                tool_summary = "\n".join(
                    f"Tool {t['tool']} returned: {json.dumps(t['result'])[:400]}"
                    for t in tool_results
                )
                follow_up_messages = messages + [
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": (
                        f"Tool results:\n{tool_summary}\n\n"
                        "Provide a natural, concise response based on these results."
                    )},
                ]
                follow_up = ""
                tts_buffer = ""
                print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)
                async for chunk in self._stream_response(follow_up_messages):
                    print(chunk, end="", flush=True)
                    follow_up += chunk
                    for cb in self._stream_callbacks:
                        try:
                            cb("chunk", chunk)
                        except Exception:
                            pass
                    if voice_response and self.voice:
                        tts_buffer += chunk
                        sentences = _SENTENCE_END.split(tts_buffer)
                        if len(sentences) > 1:
                            for sentence in sentences[:-1]:
                                spoken = self._clean_for_speech(sentence.strip())
                                if spoken and len(spoken) > 3:
                                    tts_tasks.append(asyncio.create_task(self.voice.speak(spoken)))
                            tts_buffer = sentences[-1]
                print()
                if voice_response and self.voice and tts_buffer.strip():
                    spoken = self._clean_for_speech(tts_buffer.strip())
                    if spoken:
                        tts_tasks.append(asyncio.create_task(self.voice.speak(spoken)))
                full_response = follow_up
        else:
            clean_response = full_response

        self.memory.add_message("assistant", full_response, self._session_id)
        for cb in self._stream_callbacks:
            try:
                cb("done", full_response)
            except Exception:
                pass

        duration_ms = int((time.time() - start_time) * 1000)
        tool_used = tool_calls[0][0] if tool_calls else None
        self.memory.log_interaction(intent=text[:50], tool_used=tool_used,
                                    success=True, duration_ms=duration_ms)
        return full_response

    def _build_messages(self, history: List[Dict]) -> List[Dict]:
        facts = self.memory.get_all_facts()
        facts_str = ""
        if facts:
            facts_str = "\n\nKnown facts about you and the user:\n" + \
                        "\n".join(f"- {k}: {v}" for k, v in list(facts.items())[:10])
        system = SYSTEM_PROMPT + facts_str
        messages = [{"role": "system", "content": system}]
        for msg in history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        if history:
            messages.append({"role": history[-1]["role"], "content": history[-1]["content"]})
        return messages

    async def _stream_response(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """
        Unified streaming generator.

        Ollama path   → POST /api/chat       (native JSON streaming)
        OpenAI path   → POST /v1/chat/completions  (SSE: data: {...}\\n\\n)

        The two formats differ only in how the response lines are structured,
        so we select the right parser at request time and yield plain text
        tokens identically from both.
        """
        try:
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue()

            if LLM_BACKEND == "ollama":
                url, body = _build_ollama_request(messages)
                parse_line = _parse_ollama_line
            else:
                url, body = _build_openai_compat_request(messages)
                parse_line = _parse_openai_sse_line

            def _request():
                try:
                    resp = _SESSION.post(url, json=body, stream=True, timeout=300)
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        token = parse_line(raw_line)
                        if token:
                            loop.call_soon_threadsafe(q.put_nowait, token)
                    # Signal end-of-stream
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, f"[Error: {e}]")
                    loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=_request, daemon=True).start()

            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk

        except Exception as e:
            yield f"[LLM Error: {e}]"

    # ── Tool dispatch (unchanged from original) ────────────────────────────────

    async def _execute_tool(self, tool_name: str, args: Dict) -> Dict:
        tool_name = tool_name.upper()
        action = args.pop("action", "")

        if tool_name == "COMPUTER_CONTROL":   return self.computer.execute(action, args)
        if tool_name == "PROJECT_MANAGER":    return self.projects.execute(action, args)
        if tool_name == "MEMO":               return await self._handle_memo(action, args)
        if tool_name == "AGENDA":             return self.agenda.execute(action, args)
        if tool_name == "SELF_IMPROVE":       return self._handle_self_improve(action, args)
        if tool_name == "SYSTEM":
            if action in ("overview", "processes", "battery", "temperatures", "network_interfaces"):
                return system_info_tool({"action": action, **args})
            return self.computer.execute(action, args)
        if tool_name == "MEMORY":             return self._handle_memory(action, args)
        if tool_name == "CUSTOM_SKILL":
            skill_name = args.pop("skill", action)
            return self.self_improve.execute_custom_skill(skill_name, args)
        if tool_name == "CODE":
            return run_code_tool({"language": args.get("language", "python"), "code": args.get("code", "")})
        if tool_name == "SEARCH":   return search_tool({"query": args.get("query", action)})
        if tool_name == "WEATHER":  return weather_tool({"city": args.get("city", args.get("location", action or "Amsterdam"))})
        if tool_name == "CALCULATE": return calculate_tool({"expression": args.get("expression", args.get("expr", action))})
        if tool_name == "UNIT_CONVERT": return unit_convert_tool(args)
        if tool_name == "HASH":         return hash_tool(args)
        if tool_name == "ENCODE":       return encode_tool(args)
        if tool_name == "JSON_TOOLS":   return json_tool({"action": action, **args})
        if tool_name == "REGEX":        return regex_tool({"action": action, **args})
        if tool_name == "DIFF":         return diff_tool({"action": action, **args})
        if tool_name == "NETWORK":      return network_tool({"action": action, **args})
        if tool_name == "FILE":         return file_tool({"action": action, **args})
        if tool_name == "CLIPBOARD":    return clipboard_tool({"action": action, **args})
        if tool_name == "PROCESS":      return process_tool({"action": action, **args})
        if tool_name == "TIMER":        return timer_tool({"action": action, **args})
        if tool_name == "DATETIME":     return datetime_tool({"action": action, **args})
        if tool_name == "PRICE":        return price_tool({"action": action, **args})
        if tool_name == "CURRENCY":     return currency_tool(args)
        if tool_name == "TRANSLATE":    return translate_tool(args)
        if tool_name == "TEXT":         return text_tool({"action": action, **args})
        if tool_name == "QR":           return qr_tool({"action": action, **args})
        if tool_name == "GIT":          return git_tool({"action": action, **args})
        if tool_name == "PACKAGE":      return package_tool({"action": action, **args})
        if tool_name == "GENERATE":     return generate_tool({"action": action, **args})
        if tool_name == "TEXT_ANALYZE": return text_analyze_tool({"action": action, **args})
        if tool_name == "OCR":
            from core.tools import ocr_tool
            return ocr_tool(args)
        custom = self.self_improve.get_custom_skills()
        if tool_name.lower() in custom:
            return self.self_improve.execute_custom_skill(tool_name.lower(), args)
        return {"error": f"Unknown tool: {tool_name}"}

    async def _handle_memo(self, action: str, args: Dict) -> Dict:
        if action == "save":
            memo_id = self.memory.save_memo(
                title=args.get("title", "Untitled"),
                content=args.get("content", ""),
                tags=args.get("tags", []))
            return {"success": True, "memo_id": memo_id}
        if action == "get":    return self.memory.get_memo(args.get("id", 0)) or {"error": "Not found"}
        if action == "search": return {"results": self.memory.search_memos(args.get("query", ""))}
        if action == "list":   return {"memos": self.memory.list_memos()}
        if action == "delete":
            self.memory.delete_memo(args.get("id", 0))
            return {"success": True}
        return {"error": f"Unknown memo action: {action}"}

    def _handle_self_improve(self, action: str, args: Dict) -> Dict:
        if action == "status":     return self.self_improve.get_status()
        if action == "run_cycle":  return {"result": self.self_improve.force_cycle()}
        if action == "list_skills":return {"skills": list(self.self_improve.get_custom_skills().keys())}
        return {"error": f"Unknown action: {action}"}

    def _handle_memory(self, action: str, args: Dict) -> Dict:
        if action == "set_fact":
            self.memory.set_fact(args.get("key", ""), args.get("value", ""))
            return {"success": True}
        if action == "get_fact":     return {"value": self.memory.get_fact(args.get("key", ""))}
        if action == "get_all_facts":return {"facts": self.memory.get_all_facts()}
        return {"error": f"Unknown action: {action}"}

    def _clean_for_speech(self, text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " [code block] ", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"#+\s", "", text)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        text = TOOL_PATTERN.sub("", text)
        return text.strip()[:500]

    async def _handle_slash_command(self, cmd: str):
        parts = cmd.strip("/").split()
        command = parts[0].lower()
        if command == "agenda":
            print(f"\n{self.agenda.format_agenda_summary()}\n")
        elif command == "projects":
            result = self.projects.execute("list", {})
            projs = result.get("projects", [])
            if projs:
                print(f"\n  📁 Projects ({len(projs)}):")
                for p in projs: print(f"    • {p['name']} [{p.get('status', '?')}]")
            else:
                print("  No projects yet.")
            print()
        elif command == "memos":
            memos = self.memory.list_memos()
            if memos:
                print(f"\n  📝 Memos ({len(memos)}):")
                for m in memos[:10]: print(f"    • [{m['id']}] {m['title']}")
            else:
                print("  No memos yet.")
            print()
        elif command == "skills":
            status = self.self_improve.get_status()
            print(f"\n  🧠 Self-Improvement: {status['cycles_completed']} cycles | "
                  f"{len(status['custom_skills'])} custom skills")
            for skill in status['custom_skills']: print(f"    • {skill}")
            print()
        elif command == "status":
            info = system_info_tool({"action": "overview"})
            print(f"\n  💻 System: CPU {info.get('cpu_percent','?')}% | "
                  f"RAM {info.get('memory',{}).get('percent','?')}% | "
                  f"Disk {info.get('disk',{}).get('percent','?')}%  Uptime: {info.get('uptime','?')}\n")
        elif command == "backend":
            print(f"\n  🔌 Backend: {LLM_BACKEND}  Host: {OLLAMA_HOST}  Model: {OLLAMA_MODEL}\n")
        elif command == "tools":
            print("""
  Available tools:
    COMPUTER_CONTROL, MEMO, AGENDA, PROJECT_MANAGER, CODE, SEARCH
    WEATHER, CALCULATE, UNIT_CONVERT, HASH, ENCODE, JSON_TOOLS
    REGEX, DIFF, NETWORK, FILE, CLIPBOARD, PROCESS, TIMER
    DATETIME, PRICE, CURRENCY, TRANSLATE, TEXT, QR, GIT
    PACKAGE, GENERATE, TEXT_ANALYZE, SYSTEM, MEMORY
    SELF_IMPROVE, CUSTOM_SKILL
            """)
        else:
            print(f"  Unknown command: /{command}")

    async def shutdown(self):
        print(f"  {ASSISTANT_NAME}: Saving state...")
        self.agenda.stop()
        self.self_improve.stop()
        self.memory.close()
        _SESSION.close()
        print(f"  {ASSISTANT_NAME}: Goodbye.")