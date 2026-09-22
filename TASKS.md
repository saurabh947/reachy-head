# TASKS — Lite + Laptop migration (running tracker)

Derived from [PLAN.md](PLAN.md). One task per session; each needs its **Gate** (test/measurement).
Legend: `[ ]` todo · `[x]` done · `[~]` in progress · 🔒 = blocked on human/hardware/network (can't be done autonomously).

## Week 1 — unblock + wake-up
### Phase 1A — environment
- [x] `W1-A0` Create this TASKS.md
- [ ] `W1-A1` Consolidate to one editable venv (`pip install -e ../emotion-detection-action`) — *deferred: only needed once we edit SDK source; multimodal path needs no SDK edit yet*
- [x] `W1-A2` Fix import crash (R1): added `torch`/`torchaudio` deps + installed `torchaudio==2.11.0`. Gate PASSED: `import emotion_detection_action; import reachy_emotion.reachy_handler` OK; torch/MPS unchanged; 75 tests still green. *(guard-import sub-task dropped — unnecessary given multimodal decision keeps funasr required)*
- [x] `W1-A3` `scripts/smoke_load.py`. Gate PASSED: model loads on MPS (61.6M params, ckpt clean), returns valid `NeuralEmotionResult`. **Measured ~250–490 ms/frame (~2–4 fps) multimodal on MPS** (early R6 signal).
### Phase 1B — protect & warm assets
- [ ] `W1-B1` Back up `phase2_best_calibrated.pt` to HF Hub 🔒human (account + public/private decision)
- [x] `W1-B2` Emotion2vec cache warmed (1.05 GB pulled during W1-A3). Gate: re-init no longer downloads.
- [ ] `W1-B3` Prune `phase2_best_OLD_march4.pt` (after B1 backup) 🔒 (destructive; gated on B1)
### Phase 1C — SDK hello-world
- [x] `W1-C1` Daemon up + `/docs` reachable. CONFIRMED on hardware (2026-09-22): `reachy-mini-daemon` runs on macOS and the robot wakes — `mjpython` is NOT needed for the real robot, only for `--sim` (mujoco is imported solely by the sim backend). App connects via the LOCAL GStreamer IPC backend.
- [~] `W1-C2` `scripts/hello_move.py` written (head/antennas/body/sound, verified SDK calls). Gate: run on daemon + record 🔒hardware (SDK connection itself now proven via the running app)
- [ ] `W1-C3` Run 2–3 hub apps for feel 🔒hardware
### Phase 1D — Gemini 3.8 Live spike
- [ ] `W1-D1` `scripts/live_echo.py` (mic→Live→speaker). Gate: spoken round-trip 🔒network+creds+audio
- [ ] `W1-D2` Live + function-call spike. Gate: tool call w/o audio stall 🔒network+creds

**AUDIT after 1A+1B.**

## Week 2 — face follower
- [ ] `W2-A1` Face detect from Reachy camera (MediaPipe/FaceCropPipeline). Gate: stable bbox @ fps 🔒hardware
- [ ] `W2-B1` Pixel→gaze (`look_at_image`/`look_at_world`). Gate: head centers 🔒hardware
- [ ] `W2-B2` Track loop + smoothing. Gate: follows face (clip) 🔒hardware

**AUDIT after 1C+1D / 2A+2B.**

## Week 3 — local multimodal emotion pipeline
- [x] `W3-A1` `reachy_emotion/local_capture.py` (get_frame + get_audio_sample; BGR pass-through). Gate PASSED: 5 unit tests. *(30s no-drop check = 🔒hardware)*
- [x] `W3-B1` `reachy_emotion/local_inferencer.py` (wraps EmotionDetector, cloud-compatible dict, mono downmix, warm-up gate). Gate PASSED: 8 unit tests. *(live-labels check = 🔒hardware)*
- [ ] `W3-C1` `scripts/bench_emotion.py` latency/fps audio ON/OFF. Gate: table + verdict 🔒hardware/network
- [ ] `W3-C2` Decide live cadence (data-driven) 🔒 (needs C1 numbers)

**AUDIT after 3A+3B.**

## Week 4 — emotion → motion
- [x] `W4-A1` `reachy_emotion/emotion_moves.py` curated label→move table (suffix-tolerant). Gate PASSED: 15 unit tests; all 8 labels resolve against the real 81-move library.
- [x] `W4-B1` Fixed the LIVE path (`conversation_app._react_to_emotion`) to use the curated resolver (was 5/8 broken). Gate PASSED: 8 unit tests; `reachy_handler` tests unchanged. *(Full one-handler consolidation → W4-B2, hardware-gated; reachy_handler left as-is to avoid regressing its tests.)*
- [ ] `W4-B2` `reachy_emotion/motion_worker.py` (blocking-safe + arbitration). Gate: no body contention 🔒hardware
- [ ] `W4-B3` Transition smoothing + throttle tune. Gate: smooth clip 🔒hardware
- [x] `W4-C2a` Local inferencer wired as the emotion source in `conversation_app` (`EMOTION_MODEL_PATH`). Gate PASSED: unit tests; gemini_bridge seam (`emotion_client`) unchanged.
- [x] `W4-C1` Gemini **3.8 Live** async rewrite: `reachy_emotion/live_conversation.py` (WSS session, mic→16k PCM in, camera JPEG ≤1fps in, 24k→16k audio out, `detect_emotion` tool). Wired into `main.py` (voice→Live; `--text`→turn-based loop). **Gate PASSED end-to-end on hardware (2026-09-22):** multi-turn spoken conversation works, the tool fires reliably, the robot plays moves, audio is smooth, transcripts log. 13 unit tests + config verified vs real google-genai 1.70.0.
  - **Live hardening (from on-robot debugging):** default model → `gemini-3.5-flash` (2.5 deprecated for new users; live model `gemini-3.8-live` via `GEMINI_LIVE_MODEL`); **audio-in pump reads the mic in real time** (was pulling 1 buffer per 20 ms + sleeping → GStreamer `drop=True` discarded the rest → garbled speech); **`session.receive()` re-entered per turn** (it ends on `turn_complete`, so the session was dropping after one reply); **local model runs in a continuous warm loop** so `detect_emotion` returns a fresh read (single mic reader fanned to Gemini + a shared thread-safe `_AudioRing`; inference lock; ~0.1 s gap for smooth audio); **system prompt** tells Gemini to always call `detect_emotion` for emotion instead of eyeballing the video.
- [x] `W4-C2` Live tool answers from the local detector + plays the matching RecordedMove (`_receive_loop` → `detect_emotion` → `_react_to_emotion`); fires reliably now via the system-prompt nudge. Continuous emotion→motion loop still → W4-B2 (hardware).
- [ ] `W4-D1` Full demo video 🔒hardware

**AUDIT after 4A+4B.**

## Week 5 — harden + one extension
- [x] `W5-A1` **Deleted the cloud stack** (cloud_client.py, proto/, deploy.sh, grpc/protobuf deps, EMOTION_CLOUD_ENDPOINT, `--cloud-endpoint`, GKE docs). Gate PASSED: imports clean, no dangling refs, app dispatch intact.
- [x] `W5-A2` Test cleanup: dropped `test_cloud_client.py`, rewrote `test_conversation_app.py` (local-only), renamed bridge kwarg. Gate PASSED: `pytest` green (114).
- [ ] `W5-B1` Extension: sound-localization OR MuJoCo 🔒hardware / decision

**AUDIT after 4C+4D / 5A+5B.**

## Week 6 — ship
- [ ] `W6-1` HF Space publish 🔒human
- [ ] `W6-2` Substack + demo video 🔒human
- [ ] `W6-3` Clean-machine install check 🔒human

## Handoff — needs you (updated 2026-09-22)

**117 tests green, ruff clean, zero regressions.** The app runs the on-device model (`EMOTION_MODEL_PATH`) and,
in voice mode, a Gemini 3.8 Live streaming session — **validated end-to-end on the real robot** (Gemini access
required enabling billing on the key; the `gemini-2.5-flash` default was also swapped to `gemini-3.5-flash`).
Multi-turn conversation, the `detect_emotion` tool, and emotion→move playback all work live. All three earlier
decisions are resolved: Live built, emotion-cloud deleted (no fallback), checkpoint path reported. Remaining
work is gated on hardware / your accounts:

| Task | Why blocked | What you do |
|---|---|---|
| `W1-B1` back up the checkpoint | your storage choice | save `phase2_best_calibrated.pt` somewhere safe (only copy) |
| `W1-C2/C3` `hello_move.py` + hub apps | optional now (daemon + SDK connection proven) | `python scripts/hello_move.py` for the DOF demo |
| `W2` face follower | needs camera + intrinsics | I can build it now that the daemon camera is confirmed live |
| `W3-C1` on-hardware fps/latency | needs daemon camera+mic | run a bench on the robot (early synthetic read: ~2–4 fps multimodal on MPS) |
| `W4-B2/B3` motion worker + smoothing | validated only on hardware | I'll build the arbitration logic when we can test it on the robot |
| `W5-B1` extension: sound-localization or MuJoCo | hardware / your pick | confirm the ReSpeaker array, or `pip install reachy-mini[mujoco]` |
| `W6` HF Space + Substack + video | your accounts | — |

**Known Live limitation:** barge-in doesn't yet flush already-buffered speaker audio (interrupting keeps
playing queued speech briefly). An enhancement, not a blocker. (Model transcripts DO log — the SDK sends
`output_transcription` regardless.)

## Audit log
- **Audit #1** (after 1A + 3A/3B): reviewed `local_capture.py`, `local_inferencer.py`, pyproject change, `smoke_load.py`. 1 real bug found + fixed — check-then-return race in `LocalEmotionInferencer.detect_emotion()` could return `None` to Gemini during a concurrent `reset()` (snapshot the reference). No other real bugs; dict shape verified cloud-compatible; 88 tests green.
- **Audit #2** (after 4A + 4B1): reviewed `emotion_moves.py` + `_react_to_emotion`. No real bugs (empirically confirmed all 8 labels resolve to real moves against the full 81-move library; no cross-label prefix collisions; unclear/none handled). 1 lint fix (unused import). 111 tests green + ruff clean.
- **Audit #3** (after cloud-delete + Live rewrite): dangling-reference sweep clean (no `cloud_client`/`proto`/`EMOTION_CLOUD`/`_load_cloud_endpoint` left in src/tests/root); Live API methods + config verified against installed google-genai 1.70.0; receive-loop + format helpers unit-tested. No real bugs. Noted (not bugs): barge-in buffer flush + transcript logging deferred to the hardware gate. 114 tests green + ruff clean.
- **On-robot debugging session** (2026-09-22): brought the Live path from first-connect to a working multi-turn conversation. Fixed, in order: Gemini project access (user enabled billing) + default model `gemini-2.5-flash`→`gemini-3.5-flash`; garbled mic audio (real-time drain, no artificial sleep); session dropping after one reply (`receive()` re-entered per turn); cold/stale emotion reads (continuous warm loop + mic fan-out `_AudioRing` + inference lock); Gemini bypassing the tool (system-prompt nudge to always use `detect_emotion`). Each verified with unit tests + a live run; 117 tests green + ruff clean throughout.

---
_Autonomous scope: code + config + unit tests (mock style of existing suite). Hardware/network/human gates (🔒) are handed to Saurabh with a runnable script + exact gate._
