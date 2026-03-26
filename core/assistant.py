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
from core.tools import run_code_tool, search_tool


# Tool regex: matches TOOL: <NAME> | <JSON>
TOOL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\|\s*(\{.*?\})", re.DOTALL)


class NOVAAssistant:
    def __init__(self):
        self.memory = Memory()
        self.computer = ComputerControl()
        self.projects = ProjectManager()
        self.voice = None  # Set after voice init
        self._session_id = f"session_{int(time.time())}"
        self.agenda = AgendaManager(self.memory)
        self.self_improve = SelfImprovement(self.memory, assistant_ref=self)
        self._initialized = False

    def set_voice(self, voice_engine):
        self.voice = voice_engine
        self.agenda.speak = voice_engine.speak if voice_engine else None

    async def initialize(self):
        """Check Ollama is running and model is available."""
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
                    print(f"  ❌ Cannot connect to Ollama. Start it with: ollama serve")
                else:
                    print(f"  ⏳ Waiting for Ollama... ({attempt+1}/5)")
                    await asyncio.sleep(3)
            except Exception as e:
                print(f"  ⚠️  Ollama check error: {e}")
                break

        # Start background services
        self.agenda.start_reminder_checker()
        self.self_improve.start()
        self._initialized = True

    async def run(self):
        """Main interaction loop — text mode."""
        print(f"\n  {'='*50}")
        print(f"  {ASSISTANT_NAME} is online. Type your message.")
        print(f"  Commands: /agenda, /projects, /memos, /skills, /status, /exit")
        print(f"  {'='*50}\n")

        while True:
            try:
                user_input = input(f"  You: ").strip()
                if not user_input:
                    continue

                # Handle slash commands
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
        """Process user input and return response."""
        start_time = time.time()

        # Save to memory
        self.memory.add_message("user", text, self._session_id)

        # Get conversation history
        history = self.memory.get_recent_messages(n=15, session_id=self._session_id)

        # Build messages
        messages = self._build_messages(history)

        print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)

        full_response = ""
        tool_results = []

        # Stream response
        async for chunk in self._stream_response(messages):
            print(chunk, end="", flush=True)
            full_response += chunk

        print()  # newline after streaming

        # Extract and execute tools
        tool_calls = TOOL_PATTERN.findall(full_response)
        if tool_calls:
            clean_response = TOOL_PATTERN.sub("", full_response).strip()
            for tool_name, tool_args_str in tool_calls:
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                result = await self._execute_tool(tool_name, tool_args)
                tool_results.append({"tool": tool_name, "result": result})

                # Print tool result
                result_preview = json.dumps(result)[:200]
                print(f"  ⚡ [{tool_name}] → {result_preview}")

            # If there are tool results, get follow-up response
            if tool_results:
                tool_summary = "\n".join([
                    f"Tool {t['tool']} returned: {json.dumps(t['result'])[:300]}"
                    for t in tool_results
                ])
                follow_up_messages = messages + [
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": f"Tool results:\n{tool_summary}\nPlease provide a natural response based on these results."}
                ]
                follow_up = ""
                print(f"\n  {ASSISTANT_NAME}: ", end="", flush=True)
                async for chunk in self._stream_response(follow_up_messages):
                    print(chunk, end="", flush=True)
                    follow_up += chunk
                print()
                full_response = follow_up
        else:
            clean_response = full_response

        # Save response
        self.memory.add_message("assistant", full_response, self._session_id)

        # Voice response
        if voice_response and self.voice:
            # Clean up text for speech
            spoken = self._clean_for_speech(clean_response or full_response)
            if spoken:
                asyncio.create_task(self.voice.speak(spoken))

        # Log interaction
        duration_ms = int((time.time() - start_time) * 1000)
        tool_used = tool_calls[0][0] if tool_calls else None
        self.memory.log_interaction(
            intent=text[:50],
            tool_used=tool_used,
            success=True,
            duration_ms=duration_ms
        )

        return full_response

    def _build_messages(self, history: List[Dict]) -> List[Dict]:
        """Build message list for Ollama."""
        # Add facts as context
        facts = self.memory.get_all_facts()
        facts_str = ""
        if facts:
            facts_str = "\nKnown facts:\n" + "\n".join(f"- {k}: {v}" for k, v in list(facts.items())[:10])

        system = SYSTEM_PROMPT + facts_str

        messages = [{"role": "system", "content": system}]

        # Add conversation history (skip the very last user message — it's already in history)
        for msg in history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add latest user message
        if history:
            messages.append({"role": history[-1]["role"], "content": history[-1]["content"]})

        return messages

    async def _stream_response(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama."""
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
                            "options": {
                                "temperature": LLM_TEMPERATURE,
                                "num_ctx": OLLAMA_CONTEXT_LENGTH
                            }
                        },
                        stream=True,
                        timeout=120
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

            # Run request in thread
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
        """Route tool calls to appropriate handler."""
        tool_name = tool_name.upper()
        action = args.pop("action", "")

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
            return self.computer.execute(action, args)

        elif tool_name == "MEMORY":
            return self._handle_memory(action, args)

        elif tool_name == "CUSTOM_SKILL":
            skill_name = args.pop("skill", "")
            return self.self_improve.execute_custom_skill(skill_name, args)

        elif tool_name == "CODE":
            return run_code_tool(args)

        elif tool_name == "SEARCH":
            return search_tool(args)

        else:
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
            result = self.self_improve.force_cycle()
            return {"result": result}
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
        """Clean response text for TTS — remove markdown, code blocks, etc."""
        import re
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove markdown
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"#+\s", "", text)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        # Remove tool calls
        text = TOOL_PATTERN.sub("", text)
        # Limit length for quick response
        sentences = text.split(".")
        spoken = ". ".join(sentences[:4]).strip()
        return spoken[:500] if spoken else text[:200]

    async def _handle_slash_command(self, cmd: str):
        """Handle quick slash commands."""
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
                    print(f"    • {p['name']} [{p.get('status', '?')}] — {p.get('description', '')[:50]}")
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
            print(f"\n  🧠 Self-Improvement Status:")
            print(f"    Cycles completed: {status['cycles_completed']}")
            print(f"    Custom skills: {status['custom_skills']}")
            print(f"    Total interactions: {status['total_interactions']}")
            print()
        elif command == "status":
            info = self.computer.execute("get_system_info", {})
            print(f"\n  💻 System Status:")
            print(json.dumps(info, indent=4))
            print()
        elif command == "help":
            print("""
  Available commands:
    /agenda     — Show today's agenda
    /projects   — List all projects
    /memos      — List all memos
    /skills     — Self-improvement status
    /status     — System info
    /help       — This message
    /exit       — Quit NOVA
            """)
        else:
            print(f"  Unknown command: /{command}")

    async def shutdown(self):
        """Graceful shutdown."""
        print(f"  {ASSISTANT_NAME}: Saving state...")
        self.agenda.stop()
        self.self_improve.stop()
        self.memory.close()
        print(f"  {ASSISTANT_NAME}: Goodbye.")