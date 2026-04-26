#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_tts.parser import parse_emotion_blocks


if __name__ == "__main__":
    text = Path(__file__).resolve().parents[1] / "examples" / "demo.txt"
    for seg in parse_emotion_blocks(text.read_text(encoding="utf-8")):
        print(f"[{seg.emotion}/{seg.strength}] {seg.text}")
