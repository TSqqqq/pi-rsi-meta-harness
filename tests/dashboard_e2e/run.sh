#!/usr/bin/env bash
# Clicks every dashboard button in jsdom against a throwaway dashboard + fake controller.
# Needs node and jsdom (installed into a temp dir; nothing is added to the project).
# PYTHON overrides the interpreter, e.g. PYTHON="conda run --no-capture-output -n myenv python".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${DASH_E2E_DIR:-/tmp/rsi-dash-e2e}"
mkdir -p "$WORK"
[ -d "$WORK/node_modules/jsdom" ] || (cd "$WORK" && npm install --silent --no-audit --no-fund jsdom@24.1.3)
LOG="$(mktemp)"
${PYTHON:-python3} "$HERE/serve.py" > "$LOG" 2>&1 &
# conda run forks the real python; kill it by its unique script path.
trap 'pkill -f "$HERE/serve.py" 2>/dev/null || true; rm -f "$LOG"' EXIT
for _ in $(seq 40); do [ -s "$LOG" ] && break; sleep 0.25; done
PORT=$(head -1 "$LOG" | python3 -c "import sys,json;print(json.load(sys.stdin)['port'])")
DB=$(head -1 "$LOG" | python3 -c "import sys,json;print(json.load(sys.stdin)['db'])")
NODE_PATH="$WORK/node_modules" NO_PROXY=127.0.0.1 no_proxy=127.0.0.1 node "$HERE/click.js" "$PORT" "$DB" | tee /dev/stderr | grep -q ALL_PASS
