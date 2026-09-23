#!/usr/bin/env python3
"""Reachy Emotion — single Gemini-powered conversation app for Reachy Mini.

Reachy listens to you via its microphone, talks with Gemini, and reads your
emotion on-demand when Gemini decides to call the detect_emotion tool.
Emotion inference runs on-device via a local, full-PyTorch model.

Dashboard app : ReachyEmotionApp  (registered as "reachy_emotion")
CLI script    : reachy-emotion --help
"""

import argparse
import logging
import sys
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reachy Mini App Framework class  (dashboard entry point)
# ---------------------------------------------------------------------------

try:
    from reachy_mini import ReachyMini, ReachyMiniApp

    class ReachyEmotionApp(ReachyMiniApp):
        """Gemini-powered conversation app for Reachy Mini.

        Reachy listens, converses with Gemini, and uses the local
        emotion-detection-action SDK as a Gemini tool call when it wants
        to read how you're feeling.

        Configure via .env:
            GEMINI_API_KEY      — required
            EMOTION_MODEL_PATH  — local emotion checkpoint (unset = emotion disabled)
            GEMINI_LIVE_MODEL   — optional (default: gemini-3.8-live)
            GEMINI_ER_MODEL     — optional (default: gemini-robotics-er-2-preview)
        """

        custom_app_url: str | None = None

        def run(self, reachy_mini: "ReachyMini", stop_event: threading.Event) -> None:
            import asyncio

            from reachy_emotion.system_deps import check_and_warn
            from reachy_emotion.conversation_app import _resolve_emotion_client
            from reachy_emotion.live_conversation import run_live_conversation

            check_and_warn()
            emotion_client = _resolve_emotion_client(reachy_mini)
            asyncio.run(run_live_conversation(
                mini=reachy_mini,
                stop_event=stop_event,
                emotion_client=emotion_client,
            ))

except ImportError:
    class ReachyEmotionApp:  # type: ignore[no-redef]
        """Placeholder when reachy-mini SDK is not installed.

        Instantiating this class will raise ImportError with a clear message.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "reachy-mini SDK is not installed. "
                "Install it with: pip install reachy-mini"
            )


# ---------------------------------------------------------------------------
# CLI entry point  (`reachy-emotion` script)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Reachy Emotion: Gemini conversation with on-demand emotion detection"
    )
    parser.add_argument("--sim", action="store_true",
                        help="Simulation mode (start the daemon with --sim first; "
                             "macOS: mjpython -m reachy_mini.daemon.app.main --sim)")
    parser.add_argument("--text", action="store_true",
                        help="Text mode: type input (voice mode uses Gemini Live streaming)")
    parser.add_argument("--lang", default="en-US",
                        help="--text mode TTS language code (default: en-US)")
    parser.add_argument("--model", default=None,
                        help="--text mode Gemini model (default: GEMINI_MODEL env or gemini-3.5-flash; "
                             "voice mode uses GEMINI_LIVE_MODEL)")
    parser.add_argument("--media-backend", default="default",
                        choices=["default", "gstreamer", "webrtc"],
                        help="Reachy media backend")
    parser.add_argument("--prompt", default=None,
                        help="Override Gemini system prompt")
    args = parser.parse_args()

    from reachy_emotion.system_deps import check_and_warn

    check_and_warn()

    try:
        from reachy_mini import ReachyMini
    except ImportError:
        logger.error("reachy-mini is not installed. Run: pip install reachy-mini")
        sys.exit(1)

    if args.sim:
        logger.info("Simulation mode — ensure reachy-mini-daemon --sim is running")

    stop_event = threading.Event()
    with ReachyMini(media_backend=args.media_backend) as mini:
        if args.text:
            # Text mode: type input, Gemini text reply, spoken via TTS.
            from reachy_emotion.conversation_app import run_conversation_loop, _load_model
            run_conversation_loop(
                mini=mini,
                stop_event=stop_event,
                system_prompt=args.prompt,
                voice_mode=False,
                language=args.lang,
                model=args.model or _load_model(),
            )
        else:
            # Voice mode: Gemini 3.8 Live streaming session.
            import asyncio
            from reachy_emotion.conversation_app import _resolve_emotion_client
            from reachy_emotion.live_conversation import run_live_conversation
            emotion_client = _resolve_emotion_client(mini)
            try:
                asyncio.run(run_live_conversation(
                    mini=mini,
                    stop_event=stop_event,
                    system_prompt=args.prompt,
                    emotion_client=emotion_client,
                ))
            except KeyboardInterrupt:
                # Ctrl-C is how voice mode is stopped; cleanup already ran.
                logger.info("Conversation stopped")


if __name__ == "__main__":
    main()
