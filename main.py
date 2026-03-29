#!/usr/bin/env python3
"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
 Your personal AI engineering companion.
"""
import asyncio
import signal
import sys
import os
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import ASSISTANT_NAME, USER_NAME, OLLAMA_MODEL, GUI_HOST, GUI_PORT


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║  ██████╗  ██████╗ ██╗   ██╗ █████╗                  ║
║  ██╔══██╗██╔═══██╗██║   ██║██╔══██╗                 ║
║  ██║  ██║██║   ██║██║   ██║███████║                 ║
║  ██║  ██║██║   ██║╚██╗ ██╔╝██╔══██║                 ║
║  ██████╔╝╚██████╔╝ ╚████╔╝ ██║  ██║                 ║
║  ╚═════╝  ╚═════╝   ╚═══╝  ╚═╝  ╚═╝                 ║
║                                                      ║
║  Personal AI Engineering Companion v2.0              ║
║  Powered by Ollama + Local LLM                       ║
╚══════════════════════════════════════════════════════╝
""")


async def main(args):
    print_banner()

    from core.assistant import NOVAAssistant

    assistant = NOVAAssistant()

    print("🚀 Initializing NOVA...\n")
    await assistant.initialize()

    # Start voice if requested
    if not args.no_voice:
        try:
            from core.voice import VoiceEngine
            print("  🎙️  Initializing voice engine...")
            voice = VoiceEngine()
            voice.initialize()
            assistant.set_voice(voice)
        except ImportError as e:
            print(f"  ⚠️  Voice dependencies missing ({e}). Voice disabled.")
        except Exception as e:
            print(f"  ⚠️  Voice init failed ({e}). Voice disabled.")

    # GUI mode
    if args.gui:
        loop = asyncio.get_event_loop()
        try:
            from app import run_gui
        except ImportError:
            print("  ⚠️  GUI dependencies missing. Install: pip install flask flask-socketio")
            print("  Falling back to text mode.")
            await assistant.run()
            await assistant.shutdown()
            return

        # Run Flask in a separate thread; keep the async loop here
        gui_thread = threading.Thread(
            target=run_gui,
            args=(assistant, loop),
            kwargs={"host": GUI_HOST, "port": GUI_PORT},
            daemon=True
        )
        gui_thread.start()

        print(f"\n  🌐 GUI running at http://{GUI_HOST}:{GUI_PORT}")
        print(f"  Press Ctrl+C to stop.\n")

        if args.voice and assistant.voice:
            await run_voice_mode(assistant, assistant.voice)
        else:
            # Keep async loop alive
            try:
                while True:
                    await asyncio.sleep()
            except asyncio.CancelledError:
                pass

    elif args.voice and assistant.voice:
        await run_voice_mode(assistant, assistant.voice)
    else:
        await assistant.run()

    await assistant.shutdown()


async def run_voice_mode(assistant, voice):
    from config.settings import ASSISTANT_NAME, USER_NAME

    print(f"\n  🎙️  Voice mode active. Say the wake word to activate.")
    print(f"  You can also type below. Ctrl+C to quit.\n")

    wake_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_wake():
        loop.call_soon_threadsafe(wake_event.set)

    voice.start_wake_word_listener(on_wake)

    input_queue = asyncio.Queue()

    def text_input_loop():
        while True:
            try:
                text = input("  You (text): ").strip()
                if text:
                    loop.call_soon_threadsafe(input_queue.put_nowait, text)
            except (KeyboardInterrupt, EOFError):
                break

    input_thread = threading.Thread(target=text_input_loop, daemon=True)
    input_thread.start()

    try:
        await voice.speak(f"Hello {USER_NAME}. NOVA is online and ready.")
    except:
        pass

    while True:
        wake_task = asyncio.create_task(wake_event.wait())
        text_task = asyncio.create_task(input_queue.get())

        done, pending = await asyncio.wait(
            [wake_task, text_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if wake_task in done and wake_event.is_set():
            wake_event.clear()
            try:
                await voice.speak("Yes?")
            except:
                pass
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
                await assistant.handle_input(text, voice_response=bool(voice))
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOVA — Personal AI Engineering Companion")
    parser.add_argument("--gui", "-g", action="store_true",
                        help="Launch the web GUI (opens http://localhost:5000)")
    parser.add_argument("--voice", "-v", action="store_true",
                        help="Enable voice mode with wake word detection")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable all voice features")
    parser.add_argument("--model", "-m", default=OLLAMA_MODEL,
                        help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--name", default=USER_NAME,
                        help=f"Your name (default: {USER_NAME})")
    parser.add_argument("--port", type=int, default=GUI_PORT,
                        help=f"GUI port (default: {GUI_PORT})")

    args = parser.parse_args()

    if args.model:
        import config.settings as s
        s.OLLAMA_MODEL = args.model
    if args.name:
        import config.settings as s
        s.USER_NAME = args.name
    if args.port:
        import config.settings as s
        s.GUI_PORT = args.port

    def handle_interrupt(sig, frame):
        print(f"\n\n  Shutting down {ASSISTANT_NAME}...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    asyncio.run(main(args))