#!/usr/bin/env python3
"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

 Your personal AI engineering companion.
 Powered by Ollama + Gemma3 | Voice-first | Self-improving
"""
import asyncio
import signal
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import ASSISTANT_NAME, USER_NAME, OLLAMA_MODEL


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗            ║
║  ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝            ║
║  ██║███████║██████╔╝██║   ██║██║███████╗            ║
║  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║            ║
║  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║            ║
║  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝            ║
║                                                      ║
║  Personal AI Engineering Companion                   ║
║  Powered by Ollama + Gemma3                          ║
╚══════════════════════════════════════════════════════╝
""")


async def main(args):
    print_banner()

    from core.assistant import NOVAAssistant

    assistant = NOVAAssistant()

    # Initialize Ollama connection
    print("🚀 Initializing NOVA...\n")
    await assistant.initialize()

    # Voice mode
    if not args.no_voice:
        try:
            from core.voice import VoiceEngine
            print("  🎙️  Initializing voice engine...")
            voice = VoiceEngine()
            voice.initialize()
            assistant.set_voice(voice)

            if args.voice:
                # Voice-first mode with wake word
                await run_voice_mode(assistant, voice)
            else:
                # Text mode with voice responses
                await assistant.run()
        except ImportError as e:
            print(f"  ⚠️  Voice dependencies missing ({e}). Running in text mode.")
            await assistant.run()
        except Exception as e:
            print(f"  ⚠️  Voice init failed ({e}). Running in text mode.")
            await assistant.run()
    else:
        # Pure text mode
        await assistant.run()

    await assistant.shutdown()


async def run_voice_mode(assistant, voice):
    """Voice interaction loop with wake word detection."""
    from config.settings import ASSISTANT_NAME, USER_NAME
    import threading

    print(f"\n  🎙️  Voice mode active.")
    print(f"  Say '{ASSISTANT_NAME}' to activate, or type text below.")
    print(f"  Ctrl+C to quit.\n")

    wake_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_wake():
        loop.call_soon_threadsafe(wake_event.set)

    # Start wake word listener in background
    voice.start_wake_word_listener(on_wake)

    # Also allow text input in parallel
    input_queue = asyncio.Queue()

    def text_input_loop():
        while True:
            try:
                text = input("  You (text): ").strip()
                if text:
                    asyncio.get_event_loop().call_soon_threadsafe(
                        input_queue.put_nowait, text
                    )
            except (KeyboardInterrupt, EOFError):
                break

    import threading
    input_thread = threading.Thread(target=text_input_loop, daemon=True)
    input_thread.start()

    await voice.speak(f"Hello {USER_NAME}. I'm online and ready.")

    while True:
        # Wait for either wake word or text input
        wake_task = asyncio.create_task(wake_event.wait())
        text_task = asyncio.create_task(input_queue.get())

        done, pending = await asyncio.wait(
            [wake_task, text_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        if wake_task in done and wake_event.is_set():
            wake_event.clear()
            # Wake word triggered
            await voice.speak("Yes?")
            print("  🎙️  Listening...", end="", flush=True)
            audio = await asyncio.get_event_loop().run_in_executor(
                None, voice.record_until_silence
            )
            if audio:
                print(" transcribing...", end="", flush=True)
                text = await asyncio.get_event_loop().run_in_executor(
                    None, voice.transcribe, audio
                )
                if text:
                    print(f"\n  You: {text}")
                    await assistant.handle_input(text, voice_response=True)
                else:
                    print(" (couldn't hear you)")
            else:
                print(" (no audio captured)")

        elif text_task in done:
            try:
                text = text_task.result()
                if text.lower() in ("exit", "quit", "/exit"):
                    break
                await assistant.handle_input(text, voice_response=True)
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NOVA — Personal AI Engineering Companion"
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Enable voice mode with wake word detection"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable all voice features (text-only mode)"
    )
    parser.add_argument(
        "--model", "-m",
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--name",
        default=USER_NAME,
        help=f"Your name (default: {USER_NAME})"
    )

    args = parser.parse_args()

    # Override settings from args
    if args.model:
        import config.settings as s
        s.OLLAMA_MODEL = args.model
    if args.name:
        import config.settings as s
        s.USER_NAME = args.name

    # Graceful Ctrl+C
    def handle_interrupt(sig, frame):
        print(f"\n\n  Shutting down {ASSISTANT_NAME}...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    asyncio.run(main(args))