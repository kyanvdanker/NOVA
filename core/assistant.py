"""
NOVA Assistant Brain — LLM interface, tool dispatch, conversation management.
Uses Ollama with streaming for low latency.
"""
import asyncio
import json
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

import requests

from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT,
    LLM_TEMPERATURE, LLM_STREAM, OLLAMA_CONTEXT_LENGTH,
    ASSISTANT_NAME, USER_NAME
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
    qr_tool, git_tool, package_tool, generate_tool, text_analyze_tool
)

# Tool regex: matches TOOL: <NAME> | <JSON>
TOOL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\|\s*(\{.*?\})", re.DOTALL)


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
        # For GUI streaming callbacks
        self._stream_callbacks: List = []

    def add_stream_callback(self, cb):
        self._stream_callbacks.append(cb)

    def remove_stream_callback(self, cb):
        self._stream_callbacks.discard(cb) if hasattr(self._stream_callbacks, 'discard') else None

    def set_voice(self, voice_engine):
        self.voice = voice_engine
        self.agenda.speak = voice_engine.speak if voice_engine else None

    async def initialize(self):
        print(f"  🔗 Connecting to Ollama ({OLLAMA_HOST})...")
        for attempt in range(5):
            try:
                resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = OLLAMA_MODEL.split(":")[0]
                if not any(model_base in m for m in models):
                    print(f"  ⚠️  Model '{OLLAMA_MODEL}' not found. Available: {models}")
                    print(f"  📥 Pull it with: ollama pull {OLLAMA_MODEL}")
                else:
                    print(f"  ✅ Ollama connected — model '{OLLAMA_MODEL}' ready")
                break
            except requests.exceptions.ConnectionError:
                if attempt == 4:
                    print("  ❌ Cannot connect to Ollama. Start it with: ollama serve")
                else:
                    print(f"  ⏳ Waiting for Ollama... ({attempt+1}/5)")
                    await asyncio.sleep(3)
            except Exception as e:
                print(f"  ⚠️  Ollama check error: {e}")
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

    async def handle_input(self, text: str, voice_response: bool = False) -> str:
        start_time = time.time()
        self.memory.add_message("user", text, self._session_id)
        history = self.memory.get_recent_messages(n=15, session_id=self._session_id)
        messages = self._build_messages(history)

        print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)

        full_response = ""
        async for chunk in self._stream_response(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
            # Notify GUI callbacks
            for cb in self._stream_callbacks:
                try:
                    cb("chunk", chunk)
                except:
                    pass

        print()

        # Execute tools
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

                result_preview = json.dumps(result)[:300]
                print(f"  ⚡ [{tool_name}] → {result_preview}")
                for cb in self._stream_callbacks:
                    try:
                        cb("tool_result", {"tool": tool_name, "result": result})
                    except:
                        pass

            if tool_results:
                tool_summary = "\n".join([
                    f"Tool {t['tool']} returned: {json.dumps(t['result'])[:400]}"
                    for t in tool_results
                ])
                follow_up_messages = messages + [
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": f"Tool results:\n{tool_summary}\n\nProvide a natural, concise response based on these results."}
                ]
                follow_up = ""
                print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)
                async for chunk in self._stream_response(follow_up_messages):
                    print(chunk, end="", flush=True)
                    follow_up += chunk
                    for cb in self._stream_callbacks:
                        try:
                            cb("chunk", chunk)
                        except:
                            pass
                print()
                full_response = follow_up
        else:
            clean_response = full_response

        self.memory.add_message("assistant", full_response, self._session_id)

        for cb in self._stream_callbacks:
            try:
                cb("done", full_response)
            except:
                pass

        if voice_response and self.voice:
            spoken = self._clean_for_speech(clean_response or full_response)
            if spoken:
                asyncio.create_task(self.voice.speak(spoken))

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
        try:
            loop = asyncio.get_event_loop()
            response_queue = asyncio.Queue()

            def _request():
                try:
                    resp = requests.post(
                        f"{OLLAMA_HOST}/api/chat",
                        json={
                            "model": OLLAMA_MODEL,
                            "messages": messages,
                            "stream": True,
                            "options": {"temperature": LLM_TEMPERATURE, "num_ctx": OLLAMA_CONTEXT_LENGTH}
                        },
                        stream=True, timeout=120
                    )
                    for line in resp.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    loop.call_soon_threadsafe(response_queue.put_nowait, content)
                                if data.get("done"):
                                    loop.call_soon_threadsafe(response_queue.put_nowait, None)
                            except:
                                pass
                except Exception as e:
                    loop.call_soon_threadsafe(response_queue.put_nowait, f"[Error: {e}]")
                    loop.call_soon_threadsafe(response_queue.put_nowait, None)

            import threading
            t = threading.Thread(target=_request, daemon=True)
            t.start()

            while True:
                chunk = await response_queue.get()
                if chunk is None:
                    break
                yield chunk

        except Exception as e:
            yield f"[LLM Error: {e}]"

    async def _execute_tool(self, tool_name: str, args: Dict) -> Dict:
        """Route tool calls to the appropriate handler."""
        tool_name = tool_name.upper()
        # Extract 'action' but keep the rest of args intact
        action = args.pop("action", "")

        # ── Core system tools ──
        if tool_name == "COMPUTER_CONTROL":
            return self.computer.execute(action, args)

        elif tool_name == "PROJECT_MANAGER":
            return self.projects.execute(action, args)

        elif tool_name == "MEMO":
            return await self._handle_memo(action, args)

        elif tool_name == "AGENDA":
            return self.agenda.execute(action, args)

        elif tool_name == "SELF_IMPROVE":
            return self._handle_self_improve(action, args)

        elif tool_name == "SYSTEM":
            # Route to system_info or computer control
            if action in ("overview", "processes", "battery", "temperatures", "network_interfaces"):
                return system_info_tool({"action": action, **args})
            return self.computer.execute(action, args)

        elif tool_name == "MEMORY":
            return self._handle_memory(action, args)

        elif tool_name == "CUSTOM_SKILL":
            skill_name = args.pop("skill", action)
            return self.self_improve.execute_custom_skill(skill_name, args)

        # ── New tools ──
        elif tool_name == "CODE":
            args["action"] = action
            return run_code_tool({"language": args.get("language", "python"),
                                   "code": args.get("code", "")})

        elif tool_name == "SEARCH":
            return search_tool({"query": args.get("query", action)})

        elif tool_name == "WEATHER":
            return weather_tool({"city": args.get("city", args.get("location", action or "Amsterdam"))})

        elif tool_name == "CALCULATE":
            return calculate_tool({"expression": args.get("expression", args.get("expr", action))})

        elif tool_name == "UNIT_CONVERT":
            return unit_convert_tool(args)

        elif tool_name == "HASH":
            return hash_tool(args)

        elif tool_name == "ENCODE":
            return encode_tool(args)

        elif tool_name == "JSON_TOOLS":
            return json_tool({"action": action, **args})

        elif tool_name == "REGEX":
            return regex_tool({"action": action, **args})

        elif tool_name == "DIFF":
            return diff_tool({"action": action, **args})

        elif tool_name == "NETWORK":
            return network_tool({"action": action, **args})

        elif tool_name == "FILE":
            return file_tool({"action": action, **args})

        elif tool_name == "CLIPBOARD":
            return clipboard_tool({"action": action, **args})

        elif tool_name == "PROCESS":
            return process_tool({"action": action, **args})

        elif tool_name == "TIMER":
            return timer_tool({"action": action, **args})

        elif tool_name == "DATETIME":
            return datetime_tool({"action": action, **args})

        elif tool_name == "PRICE":
            return price_tool({"action": action, **args})

        elif tool_name == "CURRENCY":
            return currency_tool(args)

        elif tool_name == "TRANSLATE":
            return translate_tool(args)

        elif tool_name == "TEXT":
            return text_tool({"action": action, **args})

        elif tool_name == "QR":
            return qr_tool({"action": action, **args})

        elif tool_name == "GIT":
            return git_tool({"action": action, **args})

        elif tool_name == "PACKAGE":
            return package_tool({"action": action, **args})

        elif tool_name == "GENERATE":
            return generate_tool({"action": action, **args})

        elif tool_name == "TEXT_ANALYZE":
            return text_analyze_tool({"action": action, **args})

        elif tool_name == "OCR":
            from core.tools import ocr_tool
            return ocr_tool(args)

        else:
            # Try custom skills as fallback
            custom = self.self_improve.get_custom_skills()
            if tool_name.lower() in custom:
                return self.self_improve.execute_custom_skill(tool_name.lower(), args)
            return {"error": f"Unknown tool: {tool_name}"}

    async def _handle_memo(self, action: str, args: Dict) -> Dict:
        if action == "save":
            memo_id = self.memory.save_memo(
                title=args.get("title", "Untitled"),
                content=args.get("content", ""),
                tags=args.get("tags", [])
            )
            return {"success": True, "memo_id": memo_id}
        elif action == "get":
            return self.memory.get_memo(args.get("id", 0)) or {"error": "Not found"}
        elif action == "search":
            return {"results": self.memory.search_memos(args.get("query", ""))}
        elif action == "list":
            return {"memos": self.memory.list_memos()}
        elif action == "delete":
            self.memory.delete_memo(args.get("id", 0))
            return {"success": True}
        return {"error": f"Unknown memo action: {action}"}

    def _handle_self_improve(self, action: str, args: Dict) -> Dict:
        if action == "status":
            return self.self_improve.get_status()
        elif action == "run_cycle":
            return {"result": self.self_improve.force_cycle()}
        elif action == "list_skills":
            return {"skills": list(self.self_improve.get_custom_skills().keys())}
        return {"error": f"Unknown action: {action}"}

    def _handle_memory(self, action: str, args: Dict) -> Dict:
        if action == "set_fact":
            self.memory.set_fact(args.get("key", ""), args.get("value", ""))
            return {"success": True}
        elif action == "get_fact":
            return {"value": self.memory.get_fact(args.get("key", ""))}
        elif action == "get_all_facts":
            return {"facts": self.memory.get_all_facts()}
        return {"error": f"Unknown action: {action}"}

    def _clean_for_speech(self, text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " [code block] ", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"#+\s", "", text)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        text = TOOL_PATTERN.sub("", text)
        sentences = text.split(".")
        spoken = ". ".join(sentences[:4]).strip()
        return spoken[:500] if spoken else text[:200]

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
                for p in projs:
                    print(f"    • {p['name']} [{p.get('status', '?')}]")
            else:
                print("  No projects yet.")
            print()
        elif command == "memos":
            memos = self.memory.list_memos()
            if memos:
                print(f"\n  📝 Memos ({len(memos)}):")
                for m in memos[:10]:
                    print(f"    • [{m['id']}] {m['title']}")
            else:
                print("  No memos yet.")
            print()
        elif command == "skills":
            status = self.self_improve.get_status()
            print(f"\n  🧠 Self-Improvement: {status['cycles_completed']} cycles | "
                  f"{len(status['custom_skills'])} custom skills")
            for skill in status['custom_skills']:
                print(f"    • {skill}")
            print()
        elif command == "status":
            info = system_info_tool({"action": "overview"})
            print(f"\n  💻 System:")
            print(f"    CPU: {info.get('cpu_percent', '?')}% | "
                  f"RAM: {info.get('memory', {}).get('percent', '?')}% | "
                  f"Disk: {info.get('disk', {}).get('percent', '?')}%")
            print(f"    Uptime: {info.get('uptime', '?')}")
            print()
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
        elif command == "help":
            print("""
  Commands: /agenda /projects /memos /skills /status /tools /help /exit
            """)
        else:
            print(f"  Unknown command: /{command}")

    async def shutdown(self):
        print(f"  {ASSISTANT_NAME}: Saving state...")
        self.agenda.stop()
        self.self_improve.stop()
        self.memory.close()
        print(f"  {ASSISTANT_NAME}: Goodbye.")