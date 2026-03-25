#!/usr/bin/env python3
"""
NOVA Test CLI
Test individual components without full voice setup.
Usage: python3 test_nova.py [component]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_llm():
    """Test LLM connection."""
    print("Testing LLM (Ollama)...")
    from brain.memory import MemoryManager
    from brain.llm import LLM
    mem = MemoryManager()
    llm = LLM(memory_manager=mem)
    print("Sending test message...")
    response = ""
    for chunk in llm.chat("Hello Nova! Introduce yourself in one sentence."):
        print(chunk, end="", flush=True)
        response += chunk
    print("\n✓ LLM working")


def test_stt():
    """Test speech-to-text."""
    print("Testing STT (Whisper)...")
    from audio.stt import STT
    stt = STT()
    print("Say something (recording 4 seconds)...")
    import pyaudio
    import numpy as np
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(16000 / 1024 * 4)):
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()

    pcm = b"".join(frames)
    text = stt.transcribe_raw(pcm)
    print(f"Transcribed: '{text}'")
    print("✓ STT working" if text else "⚠ No speech detected")


def test_tts():
    """Test text-to-speech."""
    print("Testing TTS...")
    from audio.tts import TTS
    tts = TTS()
    print("Speaking test phrase...")
    tts.speak("Hello! I am Nova, your neural omnipresent voice assistant. I am working correctly.")
    print("✓ TTS working")


def test_memory():
    """Test memory system."""
    print("Testing Memory...")
    from brain.memory import MemoryManager
    mem = MemoryManager()
    mem.store_interaction("My name is Test User", "Nice to meet you, Test User!")
    mem.update_profile("name", "Test User")
    profile = mem.get_user_profile_text()
    stats = mem.get_stats()
    print(f"Profile: {profile}")
    print(f"Stats: {stats}")
    memories = mem.retrieve("name")
    print(f"Memories found: {len(memories)}")
    print("✓ Memory working")


def test_projects():
    """Test project management."""
    print("Testing Projects...")
    import skills.projects as proj
    p = proj.create_project("Test Project", "A test project")
    print(f"Created: {p['name']} (id={p['id']})")
    memo = proj.add_memo("Test task", project_id=p["id"], memo_type="task")
    print(f"Added task: {memo['content']}")
    summary = proj.get_project_summary(p["id"])
    print(f"Summary: {summary}")
    print("✓ Projects working")


def test_laptop():
    """Test laptop control."""
    print("Testing Laptop Control...")
    from skills.laptop_control import LaptopControl
    lc = LaptopControl()
    info = lc.get_system_info()
    print(f"System: {info}")
    screenshot = lc.take_screenshot()
    print(f"Screenshot: {'captured' if screenshot else 'failed'}")
    print("✓ Laptop control working")


def test_full():
    """Test full pipeline (text input, no voice)."""
    print("Testing full pipeline (text mode)...")
    from brain.memory import MemoryManager
    from brain.llm import LLM
    from skills.laptop_control import LaptopControl
    import skills.projects as projects
    from skills.intent_router import IntentRouter

    mem = MemoryManager()
    llm = LLM(memory_manager=mem)
    laptop = LaptopControl()
    router = IntentRouter(llm, mem, laptop, projects)

    test_inputs = [
        "What time is it?",
        "Create project Test Pipeline",
        "Add task Write unit tests",
        "List my projects",
        "What do you know about me?",
    ]

    for text in test_inputs:
        print(f"\nUser: {text}")
        response = router.route(text)
        if response:
            print(f"Nova (skill): {response}")
        else:
            resp = "".join(llm.chat(text, stream=False))
            print(f"Nova (LLM): {resp[:150]}...")

    print("\n✓ Full pipeline working")


def interactive():
    """Interactive text chat with Nova (no voice)."""
    print("Nova Interactive Mode (text)")
    print("Type your message. Ctrl+C to exit.\n")

    from brain.memory import MemoryManager
    from brain.llm import LLM
    from skills.laptop_control import LaptopControl
    import skills.projects as projects
    from skills.intent_router import IntentRouter

    mem = MemoryManager()
    llm = LLM(memory_manager=mem)
    laptop = LaptopControl()
    router = IntentRouter(llm, mem, laptop, projects)

    profile = mem.get_profile()
    name = profile.get("name", "")
    print(f"Nova: Online{f', hello {name}' if name else ''}. How can I help?\n")

    while True:
        try:
            text = input("You: ").strip()
            if not text:
                continue

            # Try skill routing
            response = router.route(text)
            if response:
                print(f"Nova: {response}\n")
            else:
                print("Nova: ", end="", flush=True)
                for chunk in llm.chat(text):
                    print(chunk, end="", flush=True)
                print("\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="NOVA Component Tester")
    parser.add_argument("component", nargs="?",
                        choices=["llm", "stt", "tts", "memory", "projects",
                                 "laptop", "full", "chat"],
                        default="full",
                        help="Component to test (default: full)")
    args = parser.parse_args()

    tests = {
        "llm": test_llm,
        "stt": test_stt,
        "tts": test_tts,
        "memory": test_memory,
        "projects": test_projects,
        "laptop": test_laptop,
        "full": test_full,
        "chat": interactive,
    }

    try:
        tests[args.component]()
    except KeyboardInterrupt:
        print("\nCancelled")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()