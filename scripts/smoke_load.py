"""W1-A3 smoke test: load the full-PyTorch EmotionDetector on MPS and run a forward pass.

Proves the local model constructs, loads the fine-tuned checkpoint, and returns a
NeuralEmotionResult — the Week-3 pipeline's core. First run downloads the emotion2vec
audio backbone from ModelScope (network required; also warms the cache = W1-B2).

Usage:
    python scripts/smoke_load.py [--device mps|cpu] [--video-only]
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

# Default checkpoint: the temperature-calibrated phase-2 weights (see PLAN.md Appendix A.2).
_DEFAULT_CKPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "emotion-detection-action"
    / "outputs"
    / "phase2_best_calibrated.pt"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ckpt", default=str(_DEFAULT_CKPT))
    ap.add_argument("--video-only", action="store_true", help="skip audio (no emotion2vec)")
    ap.add_argument("--frames", type=int, default=18, help="how many process_frame calls")
    args = ap.parse_args()

    from emotion_detection_action import Config, EmotionDetector

    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.is_file():
        print(f"FAIL: checkpoint not found: {ckpt}")
        return 1

    cfg = Config(
        two_tower_device=args.device,
        two_tower_model_path=str(ckpt),
        two_tower_pretrained=True,
        verbose=True,
    )
    detector = EmotionDetector(cfg)

    t0 = time.monotonic()
    detector.initialize()
    print(f"initialize(): {time.monotonic() - t0:.1f}s")

    rng = np.random.default_rng(0)
    sr = cfg.sample_rate
    result = None
    for i in range(args.frames):
        frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)  # BGR
        audio = None if args.video_only else rng.standard_normal(sr // 5).astype(np.float32)
        t = time.monotonic()
        result = detector.process_frame(frame, audio)
        dt = (time.monotonic() - t) * 1000
        print(f"  frame {i:2d}: {dt:6.1f} ms  dominant={result.dominant_emotion} conf={result.confidence:.2f}")

    detector.shutdown()

    assert result is not None
    from emotion_detection_action.models.fusion import NeuralFusionModel

    ok = result.dominant_emotion in NeuralFusionModel.EMOTION_ORDER
    print(f"\n{'PASS' if ok else 'FAIL'}: dominant_emotion={result.dominant_emotion!r} "
          f"metrics={result.metrics} video_missing={result.video_missing} audio_missing={result.audio_missing}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
