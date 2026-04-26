#!/usr/bin/env python3
"""
Render emotion-tagged text to audio.

Usage:
  python scripts/render.py --text "[JOY]Great day![/JOY]" --speaker martin
  python scripts/render.py --file examples/demo.txt --speaker victoria --output out.wav
  echo "[ANGER][STRESS]STOP[/STRESS] it![/ANGER]" | python scripts/render.py
  python scripts/render.py --file examples/demo.txt --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_tts import render_text
from emotion_tts.parser import parse_emotion_blocks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render emotion-tagged text to WAV")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", type=str,  help="Inline text with emotion tags")
    src.add_argument("--file", type=Path, help="Text file with emotion tags")
    p.add_argument("--speaker",   default="martin",  choices=["martin", "victoria"])
    p.add_argument("--output",    type=Path, default=Path("output.wav"))
    p.add_argument("--target-db", type=float, default=-1.0)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--lab-dir",   type=Path, default=None,
                   help="Path to repo root (overrides KOKORO_LAB_DIR)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print parsed segments without synthesizing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("No input. Use --text, --file, or pipe to stdin.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for seg in parse_emotion_blocks(text):
            print(f"  [{seg.emotion:8s}/{seg.strength:6s}]  {seg.text[:80]}")
        return

    render_text(
        text,
        speaker   = args.speaker,
        output    = args.output,
        target_db = args.target_db,
        device    = args.device,
        lab_dir   = args.lab_dir,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
