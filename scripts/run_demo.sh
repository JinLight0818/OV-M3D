#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Wayland (дэлгэцийн цонхны протокол) орчинд Qt цонхыг XWayland-аар нээнэ.
export QT_QPA_PLATFORM=xcb
if [[ ! -x .venv/bin/python ]]; then
    echo "Эхлээд bash scripts/setup_ubuntu.sh ажиллуулна уу." >&2
    exit 1
fi
exec .venv/bin/python -m src.app "$@"
