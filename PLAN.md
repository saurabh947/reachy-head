# Reachy Emotion — Lite + Laptop Migration Plan

> **Status:** Plan of record · **Created:** 2026-09-17 · **Owner:** Saurabh (orchestrator)
> **Supersedes:** [`initial-plan.md`](initial-plan.md) (stale — describes symbols that no longer exist).
> **Scope:** Move emotion inference off `emotion-cloud` and onto this laptop (full PyTorch), keep the
> Gemini conversation app but on **Gemini 3.8 Live**, and drive Reachy Mini's body from a local
> emotion stream. Everything below is grounded in code that was read/verified on 2026-09-17
> (see [Appendix A](#appendix-a--verified-reference-facts) for the raw facts + how to re-verify).

---

## 0. End state (target)

**One machine.** A laptop (macOS / Apple Silicon, MPS) tethered to a Reachy Mini Lite. The laptop runs:
the Reachy SDK + daemon, camera+mic capture, the full-PyTorch emotion model, the behavior/motion logic,
and a Gemini 3.8 Live conversation session. `emotion-cloud` is shelved.

**Two decisions locked (2026-09-17):**
1. **Emotion drives body + conversation.** A continuous emotion→motion reactive layer runs alongside
   Gemini; Gemini also reads emotion on demand via a `detect_emotion` tool to color its words. One shared
   `EmotionDetector` + latest-result store feeds both.
2. **Multimodal from the start.** The emotion model consumes camera frames **and** Reachy mic audio.
   The emotion2vec audio tower is CPU-bound (see [R6](#risk-register)), so audio is the Week-3 latency gate.

**Sensor path locked:** Reachy camera + mic via the `reachy_mini` daemon (`mini.media.*`), not the laptop webcam.

### Target data flow

```
                    ┌──────────────────────── Reachy Mini (tethered, USB) ────────────────────────┐
                    │  Camera ──► get_frame()          Mic ──► get_audio_sample() (16kHz PCM)      │
                    │  Speaker ◄── push_audio_sample()  Body/Head/Antennas ◄── goto_target()/play_move
                    └───────▲──────────────┬───────────────────────────┬───────────────▲──────────┘
                            │              │                           │               │
              (24kHz audio) │      frames+audio                 frames+audio      motion commands
                            │              │                           │               │
                    ┌───────┴───────┐  ┌───▼───────────────────┐  ┌────▼───────────────┴─────────┐
                    │ Gemini 3.8    │  │ Gemini Live in-streams │  │ LOCAL EmotionDetector        │
                    │ Live (WSS)    │◄─┤ mic PCM + JPEG ≤1 fps  │  │ process_frame(bgr, audio)    │
                    │ native audio  │  └────────────────────────┘  │  → NeuralEmotionResult        │
                    │ + detect_     │           reads latest ◄──────┤  (8 labels + stress/engage/  │
                    │   emotion tool│─────────────────────────────► │   arousal + embedding)       │
                    └───────────────┘                              └────────────┬─────────────────┘
                                                                                 │ latest result
                                                        ┌────────────────────────▼──────────────────┐
                                                        │ Behavior layer (motion worker thread)      │
                                                        │  emotion→move table + smoothing +          │
                                                        │  face-follower + arbitration               │
                                                        └────────────────────────────────────────────┘
```

**Honest caveat:** "no network hops" is only partly true — shelving `emotion-cloud` removes the *emotion*
hop, but **Gemini 3.8 Live is itself a cloud WebSocket** (it also replaces the current Google STT + gTTS
hops, so net hops go *down*). Fully offline is out of scope for Weeks 1–6.

---

## 1. Current state (verified)

### Repo map & dependency direction

| Repo | GitHub | Role today | Fate |
|---|---|---|---|
| `reachy-emotion` | `saurabh947/reachy-emotion` | Gemini **text** chat (`gemini-2.5-flash`); emotion is an on-demand tool that calls the cloud | Keep & rewire |
| `emotion-detection-action` | `saurabh947/emotion-detector` | The "emotion SDK": full PyTorch two-tower model + `EmotionDetector` + trained weights | **Becomes the core** |
| `emotion-cloud` | `saurabh947/emotion-cloud` | TorchServe+gRPC on GKE that **imports the SDK** and serves `process_frame()` remotely | **Shelve** |

Dependency runs **cloud → SDK**: [`emotion-cloud/models/handler.py`](../emotion-cloud/models/handler.py) does
`from emotion_detection_action import EmotionDetector, Config` then `detector.process_frame(...)`. All the
inference science (fusion, 16-frame buffer, GRU smoothing, temperature calibration, stress/engagement/arousal,
face-crop) lives in the SDK. **Shelving the cloud loses zero inference logic.**

### Current runtime path

`reachy-emotion` CLI/dashboard → [`conversation_app.py:125`](src/reachy_emotion/conversation_app.py) `run_conversation_loop`:
mic → Google STT ([`voice_input.py`](src/reachy_emotion/voice_input.py)) → [`GeminiBridge.chat`](src/reachy_emotion/gemini_bridge.py) (may call
`detect_emotion` → [`cloud_client.py`](src/reachy_emotion/cloud_client.py) gRPC to GKE, **video-only**) → reply → gTTS
([`tts_announcer.py`](src/reachy_emotion/tts_announcer.py)) → speaker; if emotion read, `_react_to_emotion` plays a RecordedMove.

### What's live / reusable / dead / broken

- **Live & keep:** the loop skeleton and both entry points ([`main.py`](src/reachy_emotion/main.py)), the Gemini `detect_emotion`
  tool *seam* ([`gemini_bridge.py:256`](src/reachy_emotion/gemini_bridge.py) `_run_emotion_detection`), the camera-capture helpers in `cloud_client.py`.
- **Dead / delete with the cloud:** `cloud_client.py` transport, `proto/` + generated stubs, `grpcio`/`grpcio-tools`/`protobuf`,
  [`deploy.sh`](deploy.sh) (SSH-to-robot + hard-requires the cloud endpoint), `initial-plan.md`, the SDK's VLA branch
  (`models/vla/`, `openvla-7b`), non-Reachy handlers (`ros_/serial_/http_handler.py`), the whole `emotion-cloud` repo.
- **Broken / half-done (verified):**
  - 🔴 **SDK is unimportable in this venv today** — `import emotion_detection_action` → `ModuleNotFoundError: No module named 'torchaudio'`
    (unguarded `from funasr import AutoModel` at [`models/backbones.py:74`](../emotion-detection-action/src/emotion_detection_action/models/backbones.py); `torchaudio` not installed). See [R1](#risk-register).
  - 🔴 **The current app never plays an emotion move** — `_react_to_emotion` lazily imports `reachy_handler`, which triggers
    the crash above; it's swallowed by a broad `except` at [`conversation_app.py:117`](src/reachy_emotion/conversation_app.py). Physical reaction is silently dead.
  - 🔴 **Emotion→move mapping is broken for 5/8 labels** — the substring match at [`conversation_app.py:113`](src/reachy_emotion/conversation_app.py)
    only lands `sad`, `surprised`, `disgusted` against the real 81-move library ([Appendix A.5](#a5--emotion-move-library-81-moves-verified)); `angry`,
    `happy`, `fearful`, `neutral`, `unclear` produce nothing.
  - 🟠 **Two divergent reaction paths** — the live `_react_to_emotion` vs the orphaned
    [`ReachyMiniActionHandler`](src/reachy_emotion/reachy_handler.py) (a proper `BaseActionHandler`, used only in tests).
  - 🟠 **Docstrings lie** — `gemini_bridge.py`/README claim continuous background streaming + `get_latest_result()`; reality is
    on-demand `detect_emotion()` and no such method.

---

## 2. Guiding principles (orchestrator rules)

1. **One task per Claude Code session.** Each task below has an ID (e.g. `W1-A2`).
2. **Prove every task with a test or measurement** — the "**Gate**" line. No gate = not done.
3. **Verify, don't assume.** Re-check signatures/versions against installed packages before relying on them
   (the facts in [Appendix A](#appendix-a--verified-reference-facts) were true on 2026-09-17; packages drift).
4. **Keep `TASKS.md` running** — the live checklist derived from this plan (created in `W1-A0`).
5. **Full PyTorch, no shrinking** — quantization/ONNX only if the Week-3 latency gate fails.
6. **Back up the weights before refactoring** — they exist as a single local copy ([R3](#risk-register)).

---

## 3. Risk register

| # | Risk | Severity | Evidence | Mitigation / owning task |
|---|---|---|---|---|
| R1 | SDK unimportable (`torchaudio`/`funasr`) | 🔴 Critical | `import emotion_detection_action` fails; `torchaudio` NOT installed; `backbones.py:74` unguarded | `W1-A2` (install `torchaudio` **and** guard the import) |
| R2 | `emotion2vec` not cached → first run needs network | 🔴 High | `~/.cache/modelscope` empty; weights not in the `.pt` | `W1-B2` (warm cache online once) |
| R3 | Trained weights are a single, gitignored, un-backed-up copy (~2.7 GB) | 🔴 High | `outputs/*.pt` gitignored, no LFS | `W1-B1` (back up to HF Hub **first**) |
| R4 | Emotion→move mapping broken 5/8 | 🟠 High | See §1 | `W4-A*` (curated table) |
| R5 | Non-editable git install + two venvs (3.12 vs 3.14) → edits don't take effect | 🟠 Med | `pip show` no "Editable"; two venvs on disk | `W1-A1` (single editable venv) |
| R6 | `emotion2vec`/FunASR is CPU-bound, MPS-immune; `quantize()` can't touch it; video tower alone ~5.8 fps on MPS | 🟠 High | Audit measurement; `backbones.py` per-sample CPU loop | `W3-C*` (measure; fallback to async-audio cadence or video-only) |
| R7 | `transformers 5.3.0` ≫ pin `>=4.35` (major drift) | 🟠 Med | `pip show transformers` = 5.3.0 | `W1-A3` (load+forward smoke test) |
| R8 | `look_at_image()` raises without Reachy camera intrinsics | 🟠 Med | [`reachy_mini.py:623`](.venv/lib/python3.12/site-packages/reachy_mini/reachy_mini.py) | `W2` (use Reachy camera via daemon, or `look_at_world`) |
| R9 | `goto_target()` **blocks** until the move completes | 🟠 Med | [`reachy_mini.py:539`](.venv/lib/python3.12/site-packages/reachy_mini/reachy_mini.py) `wait_for_task_completion` | `W4-B*` (single motion worker thread + arbitration) |
| R10 | `google-genai 1.70.0` Live-API support/shape unverified against 3.8-live | 🟠 Med | Docs read; local Live call not yet run | `W1-D1` (Live echo spike) |
| R11 | ReSpeaker DOA needs the robot's XVF3800 array (fw ≥2.1.0), not laptop mic | 🟡 Low | [`audio_doa.py:11`](.venv/lib/python3.12/site-packages/reachy_mini/media/audio_doa.py) | `W5` (check `get_input_channels()` before committing to sound-localization) |

---

## 4. Week-by-week, phase-by-phase

Legend: each task = `ID · what` → **files** → **Gate** (how you prove it). `[decision]` marks a point that may need your input.

### Week 1 — Audit ✅ + unblock + wake-up

**Goal:** one working environment where the SDK imports, the model loads, the robot moves on command, and a bare Gemini 3.8 Live audio loop works. **Deliverable:** this audit + robot moving + `EmotionDetector` loads on MPS + Live echo.

**Phase 1A — Environment**
- `W1-A0 · Create TASKS.md` from this plan (Week-1 tasks first, one-per-session, each with its gate). → **TASKS.md** → *Gate:* file exists, lists W1 tasks with checkboxes.
- `W1-A1 · Consolidate to one editable venv.` Use the py3.12 `.venv`; reinstall the SDK editable: `pip install -e ../emotion-detection-action`. Retire the SDK's py3.14 venv. → **`.venv`, `pyproject.toml`** → *Gate:* `pip show emotion-detection-action` shows an **Editable project location**; a trivial edit in the SDK source is visible from `python -c` in `.venv`.
- `W1-A2 · Fix the import crash (R1).` Add `torch`/`torchaudio` as **explicit** runtime deps of `reachy-emotion` (currently only transitive) and install `torchaudio` into `.venv`; **also** guard `from funasr import AutoModel` in [`models/backbones.py:74`](../emotion-detection-action/src/emotion_detection_action/models/backbones.py) so a video-only path can import without funasr. → **`pyproject.toml`, SDK `backbones.py`** → *Gate:* `python -c "import emotion_detection_action; import reachy_emotion.reachy_handler; print('ok')"` succeeds in `.venv`.
- `W1-A3 · Model load + forward smoke test on MPS (R7).` Load `EmotionDetector(Config(two_tower_device='mps', two_tower_model_path='../emotion-detection-action/outputs/phase2_best_calibrated.pt'))`, run one `process_frame` on a dummy frame+audio. → **new `scripts/smoke_load.py`** (throwaway ok) → *Gate:* prints a valid `NeuralEmotionResult` with a `dominant_emotion` in the 8-label set; no missing-key warnings beyond `absent_*`.

**Phase 1B — Protect & warm assets**
- `W1-B1 · Back up the checkpoint (R3).` Publish `phase2_best_calibrated.pt` (and note `phase2_last.pt`) to a private HF Hub model repo; record the URL. → **(HF Hub)** → *Gate:* re-download to a temp dir and `sha256` matches local. `[decision]` public vs private repo.
- `W1-B2 · Warm the emotion2vec cache (R2).` One online run of `W1-A3` with audio to pull `iic/emotion2vec_base` into `~/.cache/modelscope`; document size + whether an `HF_TOKEN`/modelscope auth was needed. → **(cache)** → *Gate:* `W1-A3` re-runs with the network **off** and still loads.
- `W1-B3 · Prune dead weights.` Delete `outputs/phase2_best_OLD_march4.pt` (1.2 GB) and the `._*` AppleDouble stub after backup. → *Gate:* `outputs/` no longer lists the OLD file; smoke test still passes.

**Phase 1C — SDK hello-world (robot moves on command)**
- `W1-C1 · Daemon up.` macOS: `mjpython -m reachy_mini.daemon.app.main`; verify `http://localhost:8000/docs`. For no-hardware dev use `--mockup-sim` (no MuJoCo). → *Gate:* `/docs` reachable; `ReachyMini()` connects.
- `W1-C2 · Movement primitives script.` One script exercising `wake_up()`, `goto_sleep()`, `goto_target(head=…, antennas=np.deg2rad([r,l]), body_yaw=…, method='cartoon')`, `play_sound('wake_up.wav')` — head, antennas, body independently. (Signatures: [Appendix A.3](#a3--reachy-mini-sdk-signatures-verified).) → **new `scripts/hello_move.py`** → *Gate:* video of each DOF moving on command; note that `goto_target` blocks (R9).
- `W1-C3 · Run 2–3 hub apps for feel.` (dashboard app store). → *Gate:* notes on what each does + which motions read well.

**Phase 1D — Gemini 3.8 Live spike (de-risk the rewrite)**
- `W1-D1 · Live echo (no robot) (R10).` Minimal `google-genai` server-to-server session: `client.aio.live.connect(model='gemini-3.8-live', …)`; stream 16 kHz PCM from the laptop mic in, play 24 kHz PCM out; confirm barge-in + transcription. Confirm `google-genai 1.70.0` supports it or bump. → **new `scripts/live_echo.py`** → *Gate:* a spoken sentence gets a spoken reply; measure round-trip latency.
- `W1-D2 · Live + function-call spike.` Add a dummy `detect_emotion` tool (async/NON_BLOCKING) returning a canned dict; confirm the model calls it and continues talking without stalling audio. → *Gate:* logs show a tool call + uninterrupted audio.

`[decision]` If the Live rewrite proves large, do we ship an interim **local video+audio emotion loop on the current `gemini-2.5-flash` text path** (smaller change) before the full Live migration? (Ask when W1-D1/D2 results are in.)

---

### Week 2 — Face follower (from scratch)

**Goal:** Reachy tracks your face around the room. No tracking code exists in any repo (verified). **Deliverable:** robot follows your face.

**Phase 2A — Detection**
- `W2-A1 · Face detect from the Reachy camera.` `mini.media.get_frame()` → MediaPipe face detection. Reuse the SDK's [`FaceCropPipeline`](../emotion-detection-action/src/emotion_detection_action/models/backbones.py) (`class FaceCropPipeline`, `crop(frame_rgb)` at :181) or its underlying `mp.solutions.face_detection` for a bbox/center. → **new `scripts/face_track.py`** → *Gate:* overlay shows a stable face bbox at ≥N fps (record N).

**Phase 2B — Head control**
- `W2-B1 · Pixel → gaze.` Since we own the Reachy camera via the daemon, `look_at_image(u,v)` is valid (needs intrinsics, [R8]). Fallback: a proportional controller mapping face-center error → `goto_target(head=…, body_yaw=…)` or `look_at_world(x,y,z)` (calibration-free). → *Gate:* head centers on the face; smooth, no oscillation.
- `W2-B2 · Track loop + smoothing.` Deadband + rate-limit; keep off the daemon's blocking calls (R9) via a small motion helper. → *Gate:* robot follows a moving face across the room in a recorded clip. Course lectures 6–7 = the pose math.

`[decision]` Head-only tracking vs head+body_yaw for wider range?

---

### Week 3 — Local multimodal emotion pipeline

**Goal:** live emotion labels while you talk, from the local model, camera+**mic**. Mostly wiring — the model is built. **Deliverable:** live emotion labels printing + a latency/fps number.

**Phase 3A — Capture**
- `W3-A1 · Shared capture from the daemon.` A long-lived capture that yields `(bgr_frame, audio_chunk)`: `get_frame()` + `get_audio_sample()`; confirm `get_input_audio_samplerate()` (emotion2vec wants 16 kHz mono) and `get_input_channels()` (1-ch laptop-style vs ReSpeaker array). → **new `reachy_emotion/local_capture.py`** → *Gate:* prints frame shape + audio samplerate/channels for 30 s without drops.

**Phase 3B — Wire the detector (multimodal)**
- `W3-B1 · LocalEmotionInferencer.` Wrap `EmotionDetector` (config from W1-A3) exposing `detect_emotion() -> dict` with the **same keys** the Gemini tool already consumes (`dominant_emotion, confidence, confidence_scores, stress, engagement, arousal`) — map from `NeuralEmotionResult` ([Appendix A.4](#a4--emotiondetector-api-verified)). Feed **both** frame and audio to `process_frame(bgr, audio)`. Add a warm-up gate (ignore first ~16 frames / low confidence) since the SDK repeat-pads a cold buffer. → **new `reachy_emotion/local_inferencer.py`** + unit test (replaces `test_cloud_client.py`) → *Gate:* streaming labels print live; unit test asserts the dict shape.

**Phase 3C — Measure (the real gate)**
- `W3-C1 · Latency/fps harness, audio ON vs OFF (R6).` Measure end-to-end `process_frame` latency and achievable fps on MPS with audio on and off; isolate the emotion2vec audio-tower cost. → **new `scripts/bench_emotion.py`** → *Gate:* a table of ms/inference + fps for {video-only, multimodal}; explicit "fast enough for live reaction?" verdict.
- `W3-C2 · Decide the live cadence.` Per the measurement: if multimodal misses the budget, run audio on a **slower async cadence** (emotion label updates less often than video) or fall back to video-only. Reach for `detector.quantize('dynamic')` **only** if the *video/fusion* path (not audio) is the bottleneck. → *Gate:* documented cadence + a smooth live label stream at the chosen rate.

---

### Week 4 — Emotion → motion (the real build)

**Goal:** map labels to expressive behavior; smooth transitions. This is where most effort goes (mapping is broken today). **Deliverable:** full demo video.

**Phase 4A — Mapping**
- `W4-A1 · Curated label→move table.` Build from the verified 81-move library ([Appendix A.5](#a5--emotion-move-library-81-moves-verified)) — suffix-tolerant, with a sane default. Proposed start:
  - `happy` → cheerful/enthusiastic/laughing/proud/success/grateful/loving · **behavior: antenna wiggle + lean in**
  - `sad` → downcast/sad/exhausted/tired/lonely/resigned/boredom · **slow droop**
  - `angry` → furious/rage/irritated/frustrated/displeased/reprimand · **back off (body_yaw −)**
  - `fearful` → fear/scared/anxiety · reassure/retreat
  - `disgusted` → disgusted/contempt/uncomfortable
  - `surprised` → surprised/amazed
  - `neutral` → serenity/calming/indifferent/attentive/thoughtful · idle
  - `unclear` → subtle idle / hold (confused/uncertain/lost)
  → **new `reachy_emotion/emotion_moves.py`** (move `EMOTIONS_LIBRARY` out of the orphaned handler first) → *Gate:* unit test: every one of the 8 labels resolves to an existing move name from `list_moves()`.

**Phase 4B — Motion layer + smoothing**
- `W4-B1 · Consolidate onto one handler.` Retire `_react_to_emotion`; use `ReachyMiniActionHandler` (or a slimmed version) as the emotion→motion bridge; call `handler.execute_for_emotion(result)` from the behavior loop (the detector never calls it itself — verified). → **`reachy_handler.py`, `conversation_app.py`** → *Gate:* a label stream drives correct moves; the old substring path is gone.
- `W4-B2 · Motion worker + arbitration (R9).` A single motion thread owns the robot; `goto_target`/`play_move` are blocking, so face-follower + emotion-motion + wake/sleep post prioritized targets to it. Custom gestures (antenna wiggle, lean-in, droop, back-off) via `goto_target(antennas=…, body_yaw=…, method='cartoon'/'minjerk')`. → **new `reachy_emotion/motion_worker.py`** → *Gate:* no two subsystems fight for the body; transitions look smooth (recorded).
- `W4-B3 · Transition smoothing.` Blend/queue between an in-progress move and the next; tune the current 2 s action throttle so it doesn't fight Live's audio timing. → *Gate:* side-by-side clip: jerky vs smoothed.

**Phase 4C — Gemini Live integration (the conversation half)**
- `W4-C1 · Rewrite the conversation core to Live.` Replace the turn-based `chat()` with an async `client.aio.live.connect` session: concurrent tasks for mic-in pump (16 kHz PCM), camera-in pump (JPEG ≤1 fps), and a receive loop (play 24 kHz audio out, handle tool calls). Delete STT/TTS engines; `voice_input.py`/`tts_announcer.py` become thin Reachy mic/speaker adapters. → **`conversation_app.py`, `gemini_bridge.py`, `voice_input.py`, `tts_announcer.py`, `main.py`** → *Gate:* a spoken conversation works through the robot with barge-in.
- `W4-C2 · Wire the shared detector into the Live tool.` `detect_emotion` tool returns the **latest** result from the same `EmotionDetector` feeding the motion layer (no second inference). → *Gate:* Gemini references your emotion in its words while the body reacts concurrently.

**Phase 4D — Demo**
- `W4-D1 · Full demo video.` happy=wiggle+lean, sad=droop, angry=back-off, with conversation. → *Gate:* the video.

---

### Week 5 — Harden + one extension

**Goal:** pick one extension; refactor into a clean app structure. **Deliverable:** hardened app + one extension.

**Phase 5A — Refactor**
- `W5-A1 · Delete the cloud stack.` Remove `cloud_client.py`, `proto/`, `grpcio`/`grpcio-tools`/`protobuf`, `EMOTION_CLOUD_ENDPOINT`/`--cloud-endpoint`, `deploy.sh`, GKE docs in README/`install.sh`; delete or rewrite `initial-plan.md`. → *Gate:* `pip check` clean; app runs; `grep -r grpc src/` empty.
- `W5-A2 · Clean module layout + tests.` Rewrite `conftest.py` stubs for the local model; drop `test_cloud_client.py`; add tests for capture/inferencer/mapping/motion. → *Gate:* `pytest -v` green.

**Phase 5B — One extension** `[decision: which one?]`
- `W5-B1a · Sound localization (if the ReSpeaker array is present, R11).` `mini.media.get_DoA()` → `goto_target(body_yaw=angle)` to turn toward the speaker. First confirm `get_input_channels()` shows the XVF3800 array (fw ≥2.1.0). → *Gate:* robot turns toward a clap/voice from varied angles.
- `W5-B1b · MuJoCo sim-first.` `pip install 'reachy-mini[mujoco]'` (mujoco 3.3.0 not currently installed); launch daemon `--sim` under `mjpython`. Build/validate behaviors in sim. → *Gate:* the emotion→motion demo runs in MuJoCo with no hardware.

---

### Week 6 — Ship

**Goal:** publish + write up. **Deliverable:** HF Space + Substack + video.

- `W6-1 · HF Hub / Space publish.` The `reachy_mini_apps` entry point + `index.html`/`style.css` listing already exist ([`pyproject.toml:45`](pyproject.toml)); publishing automation does **not** — build it (push the app as an HF Space). → *Gate:* app installs from the Reachy dashboard app store end-to-end.
- `W6-2 · Substack post + demo video.` → *Gate:* published links.
- `W6-3 · Confirm weights + model card live on HF (from W1-B1).` → *Gate:* a fresh clone + `install.sh` on a clean machine reaches a working app (documented).

---

## Appendix A — Verified reference facts

*All verified on 2026-09-17 against the installed packages in `reachy-emotion/.venv` (py3.12) and the local checkouts. Re-verify before relying on any of it — versions drift.*

### A.1 — Installed versions
`google-genai 1.70.0` · `reachy-mini 1.6.0` · `torch 2.11.0` (**MPS available**, CUDA no) · **`torchaudio` NOT INSTALLED** · `transformers 5.3.0` (≫ pin `>=4.35`) · `funasr 1.3.1` · `modelscope 1.35.3` · `mediapipe 0.10.33` · `opencv-python 4.13.0.92` · `numpy 2.4.4` · `huggingface-hub 1.3.0` (exact-pinned by reachy-mini) · `protobuf 6.33.6` · `grpcio 1.80.0` · `SpeechRecognition 3.16.0` · `gtts 2.5.4`. SDK `emotion-detection-action 0.2.0` installed **non-editable** from git.

### A.2 — Checkpoints (local, gitignored, no LFS)
`../emotion-detection-action/outputs/`: `phase2_best_calibrated.pt` **725 MB (use this — temperature calibration baked in, auto-applied at [`detector.py:203`](../emotion-detection-action/src/emotion_detection_action/core/detector.py))**, `phase2_last.pt` 725 MB, `phase2_best.pt` 725 MB, `phase1_best.pt`/`phase1_last.pt` 509 MB, `phase2_best_OLD_march4.pt` 1.2 GB (**delete**). `emotion2vec` weights are **not** in the `.pt` — pulled from ModelScope at load (cache currently empty).

### A.3 — Reachy Mini SDK signatures (verified)
From `.venv/.../reachy_mini/`:
- `goto_target(head: 4x4|None, antennas: [right,left] rad|None, duration=0.5, method=InterpolationTechnique(minjerk|linear|ease_in_out|cartoon), body_yaw: rad|None=0.0)` — **blocks** to completion ([`reachy_mini.py:489`, wait at :539]). Raises if head/antennas/body_yaw all None.
- `wake_up()` [:541], `goto_sleep()` [:557] — full head+antenna emotes + sound.
- `look_at_image(u, v, duration=1.0, perform_movement=True)` [:589] — **raises** if camera or `camera_specs` is None ([:623]); needs Reachy camera intrinsics.
- `look_at_world(x, y, z, duration=1.0, perform_movement=True)` [:654] — calibration-free.
- `mini.media.get_frame() -> uint8|None` [`media_manager.py:238`], `get_audio_sample() -> float32|None` [:275], `get_input_audio_samplerate() -> int` [:287], `get_input_channels() -> int` [:301], `get_DoA() -> (angle_rad, speech_bool)|None` [:375], `push_audio_sample(float32)` [:329], `start_recording()` [:268], `start_playing()` [:322], `play_sound(file)` [:252]. `release_media()`/`acquire_media()` on `mini`.
- Daemon: macOS `mjpython -m reachy_mini.daemon.app.main`; `--sim` (MuJoCo; needs `reachy-mini[mujoco]`, mujoco 3.3.0 **not installed**); `--mockup-sim` (no MuJoCo).
- DOA needs ReSpeaker XVF3800 firmware ≥2.1.0 ([`audio_doa.py:11`]).

### A.4 — EmotionDetector API (verified)
`from emotion_detection_action import EmotionDetector, Config`. `Config(two_tower_device='mps', two_tower_model_path='<abs .pt>', two_tower_pretrained=True, cache_dir=…)` — `two_tower_device` supports `cpu`/`mps`/`cuda` ([`config.py:135`](../emotion-detection-action/src/emotion_detection_action/core/config.py)). `EmotionDetector(config, action_handler)`; `.initialize()` (builds backbones + overlays checkpoint + applies temperature); `.process_frame(bgr, audio, timestamp) -> NeuralEmotionResult` (video-only if `audio=None`; 16-frame rolling buffer + GRU; `audio` = raw float32 16 kHz mono, ≤3 s); `.reset()` on subject change; `.quantize('dynamic')` optional. Result fields: `dominant_emotion, emotion_scores{8}, latent_embedding[512], metrics{stress,engagement,arousal}, confidence, video_missing, audio_missing`. **The detector does NOT call the action handler's `execute` — only `connect()`/`disconnect()`; the app must call `handler.execute_for_emotion(result)` itself.**

### A.5 — Emotion move library (81 moves, verified)
Repo `pollen-robotics/reachy-mini-emotions-library` (81 `.json` moves + 81 `.wav`), cached locally:
`amazed1 anxiety1 attentive1 attentive2 boredom1 boredom2 calming1 cheerful1 come1 confused1 contempt1 curious1 dance1 dance2 dance3 disgusted1 displeased1 displeased2 downcast1 dying1 electric1 enthusiastic1 enthusiastic2 exhausted1 fear1 frustrated1 furious1 go_away1 grateful1 helpful1 helpful2 impatient1 impatient2 incomprehensible2 indifferent1 inquiring1 inquiring2 inquiring3 irritated1 irritated2 laughing1 laughing2 lonely1 lost1 loving1 no1 no_excited1 no_sad1 oops1 oops2 proud1 proud2 proud3 rage1 relief1 relief2 reprimand1 reprimand2 reprimand3 resigned1 sad1 sad2 scared1 serenity1 shy1 sleep1 success1 success2 surprised1 surprised2 thoughtful1 thoughtful2 tired1 uncertain1 uncomfortable1 understanding1 understanding2 welcoming1 welcoming2 yes1 yes_sad1`.
Model labels (`EMOTION_ORDER`, [`fusion.py:289`](../emotion-detection-action/src/emotion_detection_action/models/fusion.py)): `angry, disgusted, fearful, happy, neutral, sad, surprised, unclear`. Metrics: `stress, engagement, arousal`.

### A.6 — Gemini 3.8 Live (from docs, 2026-09-15)
Model id `gemini-3.8-live` (stable). Stateful WebSocket (WSS). Input: audio (raw 16-bit PCM, **16 kHz**, LE), images (JPEG **≤1 FPS**), text. Output: audio (raw 16-bit PCM, **24 kHz**, LE) + optional text transcript. Function calling **supported** (async `NON_BLOCKING` default; `BLOCKING` opt-in). Barge-in, 70 languages, search grounding. **No** structured outputs, **no** caching, **no** `thinking_level`. SDK path: server-to-server via `google-genai` (`client.aio.live.connect`). Docs: <https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live>, <https://ai.google.dev/gemini-api/docs/live>.

---

## Appendix B — Open decisions log

| When | Decision | Status |
|---|---|---|
| W0 | Keep Gemini app, migrate to **Gemini 3.8 Live** | ✅ decided 2026-09-17 |
| W0 | Sensor path: **Reachy camera+mic via daemon** | ✅ decided 2026-09-17 |
| W0 | Emotion drives **body + conversation** | ✅ decided 2026-09-17 |
| W0 | Emotion model **multimodal from the start** | ✅ decided 2026-09-17 |
| W1-B1 | Weights HF repo public vs private | ⬜ open |
| W1-D | Interim local-emotion on `2.5-flash` text path before full Live rewrite? | ⬜ open (decide after W1-D spike) |
| W2 | Head-only vs head+body tracking | ⬜ open |
| W3-C2 | Live emotion cadence (multimodal vs async-audio vs video-only) | ⬜ open (data-driven) |
| W5-B | Extension: sound localization vs MuJoCo | ⬜ open |
