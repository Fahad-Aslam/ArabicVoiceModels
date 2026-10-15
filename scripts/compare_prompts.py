#!/usr/bin/env python3
"""Convenience entrypoint for multi-prompt A/B runs.

Usage:
  python scripts/compare_prompts.py --backends piper_emirati --limit 3
"""

from __future__ import annotations

import sys

from arabic_voice.cli import app


if __name__ == "__main__":
    raise SystemExit(app(args=["compare-prompts", *sys.argv[1:]]))
