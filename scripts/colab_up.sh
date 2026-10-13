#!/usr/bin/env bash
#
# Provision a Colab GPU, install the server, and start Animagine XL 4.0 Zero on
# it in the background — then print the public URL and the local client command.
#
# Requires the Google Colab CLI:  pip install google-colab-cli
#
# Usage:
#   scripts/colab_up.sh                 # T4 GPU, session "sdxl"
#   GPU=A100 scripts/colab_up.sh        # different accelerator
#   SESSION=myrun scripts/colab_up.sh   # different session name
#
# Environment overrides:
#   SESSION    Colab session name            (default: sdxl)
#   GPU        Accelerator: T4 L4 G4 H100 A100 (default: T4; empty = CPU)
#   PORT       Server port on the VM         (default: 8000)
#   PKG        Package spec to install       (default: sdxl-runner)
#              e.g. PKG="sdxl-runner[server] @ git+https://github.com/you/sdxl-runner.git"
#   FORCE_NEW  Set to 1 to always create a fresh session
#
set -euo pipefail

SESSION="${SESSION:-sdxl}"
GPU="${GPU:-T4}"
PORT="${PORT:-8000}"
PKG="${PKG:-sdxl-runner}"
FORCE_NEW="${FORCE_NEW:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_SCRIPT="${SCRIPT_DIR}/colab_serve.py"

log() { printf '\033[1;36m[sdxl]\033[0m %s\n' "$*"; }

if ! command -v colab >/dev/null 2>&1; then
  echo "error: 'colab' CLI not found. Install it with: pip install google-colab-cli" >&2
  exit 1
fi

# --- 1. Provision (or reuse) a GPU session --------------------------------
if [ "${FORCE_NEW}" != "1" ] && colab status -s "${SESSION}" >/dev/null 2>&1; then
  log "reusing existing session '${SESSION}'"
else
  log "creating session '${SESSION}'${GPU:+ on a ${GPU} GPU}"
  if [ -n "${GPU}" ]; then
    colab new -s "${SESSION}" --gpu "${GPU}"
  else
    colab new -s "${SESSION}"
  fi
fi

# --- 2. Install dependencies ----------------------------------------------
log "installing dependencies on the VM (this can take a few minutes)"
colab install -s "${SESSION}" \
  "diffusers>=0.31" transformers accelerate safetensors peft gradio
log "installing ${PKG}"
colab install -s "${SESSION}" "${PKG}"

# --- 3. Start the server and capture the public URL -----------------------
log "starting sdxl-server + cloudflared tunnel on the VM"
OUT="$(colab exec -s "${SESSION}" --timeout 300 \
  --env "SDXL_PORT=${PORT}" -f "${SERVE_SCRIPT}")"

URL="$(printf '%s\n' "${OUT}" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1 || true)"
if [ -z "${URL}" ]; then
  echo "error: could not determine the public URL. Remote output was:" >&2
  printf '%s\n' "${OUT}" >&2
  exit 1
fi

# --- 4. Print the client command ------------------------------------------
cat <<EOF

========================================================================
 Animagine XL 4.0 Zero is running on Colab (session: ${SESSION})
========================================================================
 Public URL : ${URL}

 Generate from this machine:

   sdxl-runner generate \\
     --url ${URL} \\
     --quality --auto-size \\
     -p "1girl, arima kana, oshi no ko, looking at viewer, smile, outdoors" \\
     -o ./out

 Stop the VM when done:

   colab stop -s ${SESSION}
========================================================================
EOF
