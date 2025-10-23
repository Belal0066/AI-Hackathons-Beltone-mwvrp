#!/usr/bin/env bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

