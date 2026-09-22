#!/usr/bin/env bash
# install.sh — Full installer for reachy-emotion
# Installs system dependencies (ffmpeg, portaudio) then the Python package.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Options:
#   --dry-run   Print what would be done without making changes
#   --skip-sys  Skip system dependency installation (Python package only)

set -euo pipefail

DRY_RUN=false
SKIP_SYS=false

for arg in "$@"; do
  case $arg in
    --dry-run)  DRY_RUN=true  ;;
    --skip-sys) SKIP_SYS=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run() {
  if $DRY_RUN; then
    echo "  [dry-run] $*"
  else
    "$@"
  fi
}

info()    { echo "→ $*"; }
success() { echo "✅ $*"; }
warn()    { echo "⚠️  $*"; }

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

OS="unknown"
case "$(uname -s)" in
  Linux*)  OS="linux"  ;;
  Darwin*) OS="macos"  ;;
  *)       warn "Unsupported OS: $(uname -s). Skipping system dependency install." ;;
esac

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------

install_linux_deps() {
  info "Updating package lists …"
  run sudo apt-get update -qq

  info "Installing ffmpeg …"
  run sudo apt-get install -y ffmpeg

  info "Installing portaudio (Reachy microphone support) …"
  run sudo apt-get install -y libportaudio2 portaudio19-dev
}

install_macos_deps() {
  if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Install it from https://brew.sh then re-run this script."
    warn "Or run with --skip-sys and install ffmpeg/portaudio manually."
    exit 1
  fi

  info "Installing ffmpeg …"
  run brew install ffmpeg

  info "Installing portaudio (Reachy microphone support) …"
  run brew install portaudio
}

if ! $SKIP_SYS; then
  echo ""
  echo "=== System dependencies ==="

  if [ "$OS" = "linux" ]; then
    install_linux_deps
  elif [ "$OS" = "macos" ]; then
    install_macos_deps
  fi

  success "System dependencies installed."
fi

# ---------------------------------------------------------------------------
# Python package
# ---------------------------------------------------------------------------

echo ""
echo "=== Python package ==="

# Prefer uv if available (faster), fall back to pip
if command -v uv &>/dev/null; then
  INSTALLER="uv pip"
else
  INSTALLER="pip"
fi

info "Installing reachy-emotion and all Python dependencies (using $INSTALLER) …"
run $INSTALLER install -e "$(dirname "$0")"

success "Python package installed."

# ---------------------------------------------------------------------------
# Post-install: verify system deps
# ---------------------------------------------------------------------------

echo ""
echo "=== Verifying installation ==="
if ! $DRY_RUN; then
  reachy-emotion-setup --check-only || true
fi

# ---------------------------------------------------------------------------
# .env setup reminder
# ---------------------------------------------------------------------------

ENV_FILE="$(dirname "$0")/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo ""
  warn ".env file not found. Creating from template …"
  run cp "$(dirname "$0")/.env.example" "$ENV_FILE"
  warn "Edit $ENV_FILE and set your GEMINI_API_KEY before running the app."
fi

echo ""
success "reachy-emotion is ready!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env and set:"
echo "         GEMINI_API_KEY      — from https://aistudio.google.com/app/apikey"
echo "         EMOTION_MODEL_PATH  — path to the local model checkpoint (phase2_best_calibrated.pt)"
echo "    2. Start the daemon:  reachy-mini-daemon"
echo "    3. Run the app:       reachy-emotion"
echo "       (or text mode):    reachy-emotion --text"
echo ""
