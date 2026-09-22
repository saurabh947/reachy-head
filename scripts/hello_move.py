"""W1-C2 SDK hello-world: move head, antennas, and body on command.

Run ON THE LAPTOP with the Reachy Mini daemon already running:
    macOS:  mjpython -m reachy_mini.daemon.app.main
    (no hardware? use `--mockup-sim` to exercise the API without a robot)

Then:  python scripts/hello_move.py

Gate (W1-C2): watch each DOF move on command; record a short clip.
Every call below is verified against reachy-mini 1.6.0 (see PLAN.md Appendix A.3):
  wake_up() reachy_mini.py:541 · goto_target(antennas=/body_yaw=) :489 ·
  media.play_sound() media_manager.py:252 · goto_sleep() :557.
"""

from __future__ import annotations

import time

import numpy as np

from reachy_mini import ReachyMini


def main() -> int:
    with ReachyMini() as mini:
        print("wake_up() — head + antennas emote + sound")
        mini.wake_up()
        time.sleep(0.5)

        print("antennas — [right, left] radians")
        for right_left in ([30, -30], [-30, 30], [0, 0]):
            mini.goto_target(antennas=np.deg2rad(right_left), duration=0.4)
            time.sleep(0.1)

        print("body_yaw — rotate left / right / centre")
        for yaw_deg in (25, -25, 0):
            mini.goto_target(body_yaw=np.deg2rad(yaw_deg), duration=0.8)
            time.sleep(0.1)

        print("play_sound")
        mini.media.play_sound("wake_up.wav")
        time.sleep(1.0)

        print("goto_sleep()")
        mini.goto_sleep()

    # Head gaze for the Week-2 face follower (verified but calibration-sensitive):
    #   mini.look_at_world(x, y, z)  — calibration-free, Reachy frame (reachy_mini.py:654)
    #   mini.look_at_image(u, v)     — needs the Reachy camera intrinsics (:589, raises without)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
