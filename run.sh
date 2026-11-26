#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip3 install -q fastapi uvicorn pymupdf
python3 make_manuals.py
python3 app.py
