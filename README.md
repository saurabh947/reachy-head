# Reachy Emotion

A Gemini-powered conversation app for [Reachy Mini](https://pollen-robotics.com/reachy-mini). Talk to Reachy naturally — Gemini maintains the conversation and calls a `detect_emotion` tool when it wants to read how you're feeling.

Emotion inference runs **on-device**: a Two-Tower Multimodal Transformer (ViT-B/16 face + emotion2vec audio, full PyTorch) runs locally on the laptop the robot is tethered to — no network hop for inference. Gemini decides *when* to read the current emotion via the tool.

---

## Architecture

```
Laptop (tethered to Reachy Mini) — voice mode (default)

  Reachy mic + camera ─► Gemini 3.8 Live (speech in/out, video ≤ 1 fps) ─► Reachy speaker
                              │
                              ├─ detect_emotion ──► latest reading of the local model:
                              │     Reachy camera + mic ─► LocalEmotionInferencer (continuous)
                              │       ─► EmotionDetector (full PyTorch on MPS, Two-Tower:
                              │          ViT-B/16 face + emotion2vec audio)
                              │       ─► {dominant_emotion (8 classes), confidence,
                              │           stress / engagement / arousal}
                              │       ─► antennas / body: RecordedMove (emotion → move)
                              │
                              └─ look_at_scene / plan_task ──► Gemini Robotics ER 2
                                    (object points → head gaze; ordered arm steps)
```

**Key design:** the emotion model runs continuously in-process on the laptop (no network hop), and Gemini decides *when* to read it through the `detect_emotion` tool, which returns the latest reading. `--text` mode uses a turn-based pipeline instead: typed text → GeminiBridge (`gemini-3.5-flash`) → gTTS, with a fresh reading per tool call.

### Code structure

```
src/reachy_emotion/
├── main.py               ← ReachyEmotionApp (dashboard entry point) + CLI
├── conversation_app.py   ← core conversation loop + emotion-source resolution
├── live_conversation.py  ← voice mode: Gemini 3.8 Live streaming session + tools
├── scene_brain.py        ← Gemini Robotics ER 2: look_at_scene / plan_task + head gaze
├── gemini_bridge.py      ← Gemini chat session + detect_emotion tool (--text mode)
├── local_inferencer.py   ← LocalEmotionInferencer: in-process emotion model (default)
├── local_capture.py      ← camera+mic reader from the Reachy daemon (mini.media.*)
├── emotion_moves.py      ← emotion label → RecordedMove resolution (curated)
├── voice_input.py        ← STT from Reachy mic (energy VAD + Google STT)
├── tts_announcer.py      ← speak_text() → gTTS + pydub → Reachy speaker
├── reachy_handler.py     ← ActionCommand → RecordedMoves / Motion
└── system_deps.py        ← ffmpeg/portaudio check + install
```

---

## Prerequisites

- Python 3.10–3.12
- Reachy Mini robot (or `--text` / `--sim` for testing without hardware)
- A Gemini API key — free tier at [aistudio.google.com](https://aistudio.google.com/app/apikey)
- The local emotion model checkpoint (`phase2_best_calibrated.pt`) — set `EMOTION_MODEL_PATH` to it. Full PyTorch; runs on Apple Silicon (MPS) or CPU. The audio backbone (emotion2vec) downloads once on first run.
- System packages: `ffmpeg` (TTS) and `portaudio` (microphone)

---

## Installation

### Option A — `install.sh` (recommended)

```bash
git clone https://github.com/<your-username>/reachy-emotion
cd reachy-emotion
./install.sh
```

This single command installs system packages (`ffmpeg` + `portaudio`), all Python dependencies (including `torch` and `torchaudio`), and creates a `.env` template.

```bash
./install.sh --dry-run    # preview without making changes
./install.sh --skip-sys   # skip system packages (Python deps only)
```

### Option B — Reachy Mini Dashboard

Open Reachy Mini Control at `http://<robot-ip>:8000`, find **Reachy Emotion** in the app store, and click **Install**. Then SSH into the robot and run:

```bash
reachy-emotion-setup    # installs ffmpeg + portaudio
```

### Option C — Manual

```bash
pip install -e .        # all Python dependencies
reachy-emotion-setup    # system dependencies (ffmpeg, portaudio)
```

---

## Configuration

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and EMOTION_MODEL_PATH
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | From [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `EMOTION_MODEL_PATH` | Yes\* | Path to the local checkpoint (`.pt`), e.g. `../emotion-detection-action/outputs/phase2_best_calibrated.pt`. |
| `EMOTION_DEVICE` | No | Torch device for the local model. Default: `mps` |
| `GEMINI_MODEL` | No | `--text` mode model. Default: `gemini-3.5-flash` |
| `GEMINI_LIVE_MODEL` | No | Voice-mode Live model. Default: `gemini-3.8-live` |
| `GEMINI_ER_MODEL` | No | Gemini Robotics ER model for the scene tools (`look_at_scene`, `plan_task`). Default: `gemini-robotics-er-2-preview` |
| `GEMINI_SYSTEM_PROMPT` | No | Single-line override of Reachy's personality prompt |

\* Required for emotion detection. If unset, emotion detection is disabled and Gemini still converses.

---

## Starting the daemon

The Reachy daemon must be running before starting the app.

**Linux / on the robot:**
```bash
reachy-mini-daemon
```

**macOS (Lite / USB connection):**
```bash
# mjpython is required on macOS for MuJoCo compatibility
mjpython -m reachy_mini.daemon.app.main
```

**Simulation (no robot):**
```bash
# Linux:
reachy-mini-daemon --sim
# macOS:
mjpython -m reachy_mini.daemon.app.main --sim
```

Verify at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Usage

```bash
reachy-emotion                                        # voice mode
reachy-emotion --text                                 # text mode
reachy-emotion --sim                                  # simulation mode
reachy-emotion --lang fr-FR                           # French
```

| Flag | Description |
|---|---|
| `--text` | Type input instead of speaking |
| `--lang CODE` | `--text` mode TTS language, e.g. `en-US`, `fr-FR` (default: `en-US`) |
| `--model NAME` | `--text` mode Gemini model (voice mode uses `GEMINI_LIVE_MODEL`) |
| `--prompt TEXT` | Override system prompt for this session |
| `--sim` | Simulation mode |
| `--media-backend` | Reachy media backend: `default`, `gstreamer`, `webrtc` |

Or launch from the **Reachy Mini Dashboard** — no terminal needed.

---

## Customising Reachy's personality

**Per-run** (not saved):
```bash
reachy-emotion --prompt "You are Reachy, a robot who speaks only in haiku."
```

**Persistent** via `.env`:
```ini
GEMINI_SYSTEM_PROMPT=You are Reachy, a friendly robot. Keep answers under 10 words.
```

**Full edit**: modify `DEFAULT_SYSTEM_PROMPT` in `src/reachy_emotion/gemini_bridge.py`.

Priority: `--prompt` flag > `GEMINI_SYSTEM_PROMPT` env var > built-in default.

---

## Running tests

Requires Python 3.10–3.12 (the package requires `>=3.10`).

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The unit tests stub out all robot hardware, the Gemini API, and the emotion model — no robot, no internet connection, and no model download is needed.

---

## Troubleshooting

### Emotion detection is disabled / always "unclear"
Set `EMOTION_MODEL_PATH` to the local checkpoint (see [Configuration](#configuration)). If it is unset, emotion detection is off and Gemini simply converses without it.

### First run is slow / needs the network
On the first local run the emotion2vec audio backbone (~1 GB) downloads once to the model cache; later runs work offline. Warm it ahead of time by running the app (or `python scripts/smoke_load.py`) once while online.

### Local model fails to load
- Confirm `EMOTION_MODEL_PATH` points to an existing `.pt` checkpoint.
- Ensure `torch` + `torchaudio` are installed (they ship as dependencies).
- Try `EMOTION_DEVICE=cpu` if the `mps` device misbehaves.

### `ffmpeg not found`
Required for TTS (gTTS outputs MP3, robot speaker needs WAV).
```bash
reachy-emotion-setup   # auto-detects OS and installs ffmpeg + portaudio
```

### `ModuleNotFoundError: No module named 'reachy_mini'`
Install the Reachy Mini SDK or run in simulation mode:
```bash
reachy-emotion --sim --text
```

---

## License

MIT
