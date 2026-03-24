"""
NOVA Brain - LLM Interface
Connects to Ollama (Llama 3.2 Vision) with:
- Streaming response support
- Tool/function calling
- Vision (screenshot analysis)
- Conversation history management
"""

import json
import base64
import time
import logging
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.llm")


class Message:
    """A single conversation message."""
    def __init__(self, role: str, content: str, images: list = None):
        self.role = role
        self.content = content
        self.images = images or []

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


class LLM:
    """
    Interface to Ollama running Llama 3.2 Vision.
    Handles conversation context, streaming, and vision.
    """

    def __init__(self, memory_manager=None):
        self.memory = memory_manager
        self._history: List[Message] = []
        self._system_prompt = config.NOVA_PERSONALITY
        self._check_ollama()

    def _check_ollama(self):
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            log.info(f"Ollama models available: {models}")

            model_base = config.OLLAMA_MODEL.split(":")[0]
            if not any(model_base in m for m in models):
                log.warning(
                    f"Model '{config.OLLAMA_MODEL}' not found. "
                    f"Run: ollama pull {config.OLLAMA_MODEL}"
                )
        except requests.exceptions.ConnectionError:
            log.error(
                "Cannot connect to Ollama! Make sure it's running: ollama serve"
            )
        except Exception as e:
            log.warning(f"Ollama check failed: {e}")

    def build_system_prompt(self, memories: str = "", user_profile: str = "") -> str:
        """Build dynamic system prompt with memories and user context."""
        prompt = self._system_prompt

        if user_profile:
            prompt += f"\n\n## Owner Profile\n{user_profile}"

        if memories:
            prompt += f"\n\n## Relevant Memories\n{memories}"

        prompt += f"\n\n## Current Time\n{time.strftime('%A, %B %d %Y, %I:%M %p')}"

        return prompt

    def chat(
        self,
        user_input: str,
        image_path: Optional[Path] = None,
        image_base64: Optional[str] = None,
        stream: bool = True,
        tools: list = None,
    ) -> Generator[str, None, None]:
        """
        Send message to LLM and stream response.
        Yields text chunks as they arrive.
        """
        # Fetch relevant memories
        memories_text = ""
        user_profile = ""
        if self.memory:
            memories = self.memory.retrieve(user_input, k=config.MEMORY_TOP_K)
            if memories:
                memories_text = "\n".join(f"- {m}" for m in memories)
            user_profile = self.memory.get_user_profile_text()

        # Prepare image
        images = []
        if image_base64:
            images = [image_base64]
        elif image_path and image_path.exists():
            with open(image_path, "rb") as f:
                images = [base64.b64encode(f.read()).decode()]

        # Build messages
        messages = [
            {"role": "system", "content": self.build_system_prompt(memories_text, user_profile)}
        ]

        # Add conversation history (context window)
        for msg in self._history[-config.CONTEXT_WINDOW:]:
            messages.append(msg.to_dict())

        # Add current user message
        user_msg = {"role": "user", "content": user_input}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        # Call Ollama
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 8192,
            },
        }

        if tools:
            payload["tools"] = tools

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
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            if content:
                                full_response += content
                                yield content
                            # Check for tool calls
                            if "tool_calls" in chunk["message"]:
                                tool_calls.extend(chunk["message"]["tool_calls"])
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            else:
                data = response.json()
                full_response = data["message"]["content"]
                yield full_response
                tool_calls = data["message"].get("tool_calls", [])

        except requests.exceptions.Timeout:
            error_msg = "My thinking is taking longer than usual. Let me try again."
            log.error("Ollama timeout")
            yield error_msg
            full_response = error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "I can't reach my brain right now. Is Ollama running?"
            log.error("Ollama connection error")
            yield error_msg
            full_response = error_msg
        except Exception as e:
            error_msg = f"Something went wrong: {e}"
            log.error(f"LLM error: {e}")
            yield error_msg
            full_response = error_msg

        # Update history
        self._history.append(Message("user", user_input, images))
        if full_response:
            self._history.append(Message("assistant", full_response))

        # Store in memory
        if self.memory and full_response:
            self.memory.store_interaction(user_input, full_response)

        # Handle tool calls
        if tool_calls:
            yield from self._handle_tool_calls(tool_calls)

    def chat_blocking(
        self,
        user_input: str,
        image_path: Optional[Path] = None,
        image_base64: Optional[str] = None,
        tools: list = None,
    ) -> str:
        """Non-streaming version. Returns full response string."""
        return "".join(self.chat(user_input, image_path, image_base64,
                                 stream=False, tools=tools))

    def _handle_tool_calls(self, tool_calls: list) -> Generator[str, None, None]:
        """Process tool/function calls from LLM."""
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            fn_args = tc.get("function", {}).get("arguments", {})
            log.info(f"Tool call: {fn_name}({fn_args})")
            yield f"\n[Tool: {fn_name}]"

    def analyze_screen(self, screenshot_base64: str, question: str = None) -> str:
        """Use vision capability to analyze a screenshot."""
        prompt = question or "Describe what's on the screen and what the user is working on."
        return self.chat_blocking(prompt, image_base64=screenshot_base64)

    def classify_intent(self, text: str) -> dict:
        """
        Classify user intent for skill routing.
        Returns dict with intent type and parameters.
        """
        prompt = f"""Classify this user request into one of these intents:
        
Intents:
- general_question: general knowledge, chitchat, advice
- project_create: create a new project
- project_update: update/add to existing project  
- memo_create: create a note or memo
- memo_read: read back notes
- laptop_control: control laptop (open/close apps, files, system)
- screen_describe: describe what's on screen
- reminder_set: set a reminder
- preference_update: user stating a preference about themselves
- memory_query: asking about past conversations

Request: "{text}"

Reply with ONLY a JSON object like: {{"intent": "general_question", "params": {{}}}}
Extract relevant params. For project intents, extract "project_name". For laptop_control, extract "action" and "target"."""

        try:
            response = self.chat_blocking(prompt)
            # Extract JSON from response
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            log.warning(f"Intent classification failed: {e}")

        return {"intent": "general_question", "params": {}}

    def clear_history(self):
        """Clear conversation history."""
        self._history = []
        log.info("Conversation history cleared")

    def get_history_summary(self) -> str:
        """Get a brief summary of conversation history."""
        if not self._history:
            return "No conversation history."
        turns = len([m for m in self._history if m.role == "user"])
        return f"{turns} exchanges in current session."

    def set_system_prompt_addition(self, addition: str):
        """Temporarily add to system prompt (e.g. for active project context)."""
        self._system_prompt = config.NOVA_PERSONALITY + "\n\n" + addition