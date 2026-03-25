"""
NOVA Brain — LLM Interface v2
Connects to Ollama (Llama 3.2 Vision) with:
  - Word-level streaming: yields words not waiting for sentences
    → fixes the "big delay then all at once" problem
  - Mood-adaptive system prompt injection
  - Ambient context injection (time, weather, active window)
  - Tool/function calling support
  - Vision (screenshots + camera frames)
  - Evolving personality context
"""

import json
import base64
import time
import logging
from pathlib import Path
from typing import Optional, Generator, List, Callable

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.llm")


class Message:
    def __init__(self, role: str, content: str, images: list = None):
        self.role = role
        self.content = content
        self.images = images or []

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


# Built-in tools Nova can call
NOVA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open an application on the computer",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app to open"}
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Create a new project",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a task or memo to a project or general list",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "project_name": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["task", "note", "idea", "decision"]},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a fact or preference about the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["preference", "personal", "schedule", "goal"]},
                    "fact": {"type": "string"},
                },
                "required": ["category", "fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specific time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "when": {"type": "string", "description": "Natural language time: 'in 30 minutes', 'tomorrow at 9am'"},
                },
                "required": ["message", "when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]


class LLM:
    """
    Ollama LLM interface with all intelligence systems wired in.
    """

    def __init__(self, memory_manager=None, mood_engine=None, ambient=None):
        self.memory = memory_manager
        self.mood = mood_engine
        self.ambient = ambient

        self._history: List[Message] = []
        self._system_prompt = config.NOVA_PERSONALITY

        # Tool execution callbacks (set by main orchestrator)
        self.tool_handlers = {}

        self._check_ollama()

    def _check_ollama(self):
        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = config.OLLAMA_MODEL.split(":")[0]
            if not any(model_base in m for m in models):
                log.warning(
                    f"Model '{config.OLLAMA_MODEL}' not found. "
                    f"Run: ollama pull {config.OLLAMA_MODEL}"
                )
            else:
                log.info(f"Ollama ready, model: {config.OLLAMA_MODEL}")
        except requests.exceptions.ConnectionError:
            log.error("Ollama not reachable. Start with: ollama serve")
        except Exception as e:
            log.warning(f"Ollama check: {e}")

    def _build_system_prompt(self) -> str:
        """Build dynamic system prompt with all context injections."""
        prompt = self._system_prompt

        # User profile
        if self.memory:
            profile = self.memory.get_user_profile_text()
            if profile:
                prompt += f"\n\n## Owner Profile\n{profile}"

        # Ambient context (time, weather, active window)
        if self.ambient:
            ctx = self.ambient.build_context_string()
            if ctx:
                prompt += f"\n\n## Environment\n{ctx}"
        else:
            prompt += f"\n\n## Current Time\n{time.strftime('%A, %B %d %Y, %I:%M %p')}"

        # Mood adaptation
        if self.mood and config.MOOD_ADAPTIVE_PERSONA:
            mood_ctx = self.mood.get_mood_context_str()
            if mood_ctx:
                prompt += mood_ctx

        return prompt

    def chat(
        self,
        user_input: str,
        image_path: Optional[Path] = None,
        image_base64: Optional[str] = None,
        camera_frame_b64: Optional[str] = None,
        stream: bool = True,
        use_tools: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send message to LLM. Yields word-by-word chunks for low-latency TTS.
        """
        # Retrieve relevant memories
        memories_text = ""
        if self.memory:
            memories = self.memory.retrieve(user_input, k=config.MEMORY_TOP_K)
            if memories:
                memories_text = "Relevant past context:\n" + "\n".join(
                    f"- {m[:120]}" for m in memories
                )

        # Prepare images
        images = []
        if image_base64:
            images.append(image_base64)
        elif image_path and image_path.exists():
            with open(image_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
        if camera_frame_b64:
            images.append(camera_frame_b64)

        # Build message list
        system = self._build_system_prompt()
        if memories_text:
            system += f"\n\n## Memory\n{memories_text}"

        messages = [{"role": "system", "content": system}]
        for msg in self._history[-config.CONTEXT_WINDOW:]:
            messages.append(msg.to_dict())

        user_msg = {"role": "user", "content": user_input}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.72,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 8192,
            },
        }

        if use_tools and self.tool_handlers:
            payload["tools"] = NOVA_TOOLS

        full_response = ""
        tool_calls = []

        try:
            response = requests.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json=payload,
                stream=stream,
                timeout=config.OLLAMA_TIMEOUT,
            )
            response.raise_for_status()

            if stream:
                word_buffer = ""

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")

                        if content:
                            full_response += content
                            word_buffer += content

                            # Yield word by word for low-latency TTS
                            # We split on spaces but keep punctuation attached to words
                            while " " in word_buffer or "\n" in word_buffer:
                                # Find the next word boundary
                                for sep in [" ", "\n"]:
                                    idx = word_buffer.find(sep)
                                    if idx != -1:
                                        word = word_buffer[:idx]
                                        word_buffer = word_buffer[idx + 1:]
                                        if word:
                                            yield word + " "
                                        break

                        if chunk.get("done"):
                            # Flush remaining buffer
                            if word_buffer.strip():
                                yield word_buffer
                            break

                        # Tool calls
                        tc = chunk.get("message", {}).get("tool_calls", [])
                        if tc:
                            tool_calls.extend(tc)

                    except json.JSONDecodeError:
                        continue
            else:
                data = response.json()
                full_response = data["message"]["content"]
                yield full_response
                tool_calls = data["message"].get("tool_calls", [])

        except requests.exceptions.Timeout:
            msg = "I'm thinking slowly right now. Give me a moment."
            log.error("Ollama timeout")
            yield msg
            full_response = msg
        except requests.exceptions.ConnectionError:
            msg = "I can't reach my brain. Is Ollama running?"
            log.error("Ollama connection error")
            yield msg
            full_response = msg
        except Exception as e:
            msg = f"Something went wrong: {e}"
            log.error(f"LLM error: {e}")
            yield msg
            full_response = msg

        # Update history
        self._history.append(Message("user", user_input, images))
        if full_response:
            self._history.append(Message("assistant", full_response))

        # Persist to memory
        if self.memory and full_response and user_input:
            self.memory.store_interaction(user_input, full_response)

        # Handle tool calls
        if tool_calls:
            yield from self._dispatch_tool_calls(tool_calls)

    def chat_blocking(self, user_input: str, image_path: Optional[Path] = None,
                      image_base64: Optional[str] = None,
                      camera_frame_b64: Optional[str] = None) -> str:
        """Blocking (non-streaming) chat. Returns full string."""
        return "".join(self.chat(
            user_input,
            image_path=image_path,
            image_base64=image_base64,
            camera_frame_b64=camera_frame_b64,
            stream=False,
        ))

    def _dispatch_tool_calls(self, tool_calls: list) -> Generator[str, None, None]:
        """Execute tool calls and yield results."""
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            fn_args = tc.get("function", {}).get("arguments", {})
            log.info(f"Tool call: {fn_name}({fn_args})")

            handler = self.tool_handlers.get(fn_name)
            if handler:
                try:
                    result = handler(**fn_args)
                    if result:
                        yield f" {result}"
                except Exception as e:
                    log.error(f"Tool handler error: {e}")
                    yield f" I tried to {fn_name} but encountered an error."

    def analyze_screen(self, screenshot_b64: str, question: str = None) -> str:
        """Vision analysis of a screenshot."""
        q = question or "What is on the screen? What is the user working on?"
        return self.chat_blocking(q, image_base64=screenshot_b64)

    def analyze_camera(self, frame_b64: str, question: str = None) -> str:
        """Vision analysis of a camera frame."""
        q = question or "What do you see? Is anyone present? What are they doing?"
        return self.chat_blocking(q, camera_frame_b64=frame_b64)

    def classify_intent(self, text: str) -> dict:
        """Fast intent classification."""
        prompt = f"""Classify this request into one intent. Reply ONLY with valid JSON.

Request: "{text}"

Intents: general_question, project_create, project_update, memo_create, memo_read,
task_create, task_complete, laptop_control, screen_describe, reminder_set,
preference_update, memory_query, weather_query, system_info, help

JSON format: {{"intent": "intent_name", "params": {{"key": "value"}}}}
Extract relevant params: project_name, task_content, app_name, reminder_text, reminder_time"""

        try:
            response = self.chat_blocking(prompt, use_tools=False)
            import re
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            log.debug(f"Intent classification: {e}")
        return {"intent": "general_question", "params": {}}

    def generate_autonomous(self, prompt: str) -> str:
        """
        Generate a proactive message for autonomous behavior.
        Fixed for gemma3 compatibility (no tools, minimal payload, short context).
        """
        system = self._build_system_prompt()

        # Keep system prompt short for gemma3
        if len(system) > 1500:
            system = system[:1500] + "..."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": config.OLLAMA_MODEL,   # keep your gemma3 here
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,          # don't go too high
                "repeat_penalty": 1.1,
            },
            # IMPORTANT: Do NOT include "tools" for autonomous calls with gemma3
        }

        try:
            resp = requests.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=25
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("message", {}).get("content", "").strip()

        except requests.exceptions.HTTPError as e:
            error_text = e.response.text if hasattr(e.response, 'text') else str(e)
            log.error(f"Autonomous 400 error with gemma3: {error_text}")
            # Fallback message so Nova doesn't crash
            return "Hmm, I had a small glitch there."
        except Exception as e:
            log.error(f"Autonomous generation failed: {e}")
            return ""

    def clear_history(self):
        self._history = []

    def set_ambient(self, ambient):
        self.ambient = ambient

    def set_mood(self, mood_engine):
        self.mood = mood_engine