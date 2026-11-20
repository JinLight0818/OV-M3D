#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip libgl1 libglib2.0-0t64 \
    libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 xwayland fonts-dejavu-core
if ! sudo apt-get install -y intel-opencl-icd ocl-icd-libopencl1 clinfo; then
    echo "OpenCL (график процессорын тооцооллын орчин) суусангүй. Гүнийг төв процессороор бодож болно."
fi
# Системийн Python-оос тусдаа орчинд CPU (төв процессор)-ийн багцуудыг суулгана.
if [[ ! -d .venv ]]; then
    bootstrap="$(mktemp -d)"
    trap 'rm -rf "$bootstrap"' EXIT
    python3 -m venv "$bootstrap"
    "$bootstrap/bin/python" -m pip install uv
    "$bootstrap/bin/uv" python install 3.12
    "$bootstrap/bin/uv" venv --seed --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Төслийн .venv орчин Python 3.12 байх шаардлагатай."'
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
echo 'Төхөөрөмжийн эрх шаардлагатай бол sudo usermod -aG render,video,dialout "$USER" ажиллуулаад дахин нэвтэрнэ үү.'
echo 'Орчин бэлэн. source .venv/bin/activate ажиллуулна уу.'
