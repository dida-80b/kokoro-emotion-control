"""
Parse emotion block tags from text into EmotionSegment list.

Block syntax:
    [EMOTION]...[/EMOTION]
    [EMOTION:strength]...[/EMOTION]

Available emotions:  NEUTRAL  JOY  SADNESS  ANGER  FEAR  DISGUST  BOREDOM
Strengths:           soft  mid (default)  strong

Untagged text defaults to NEUTRAL/mid.
Prosody markers ([STRESS], [SHOUT], etc.) are preserved inside segment text
and handled downstream by prosody.synthesize_prosody().
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EMOTIONS = {"neutral", "joy", "sadness", "anger", "fear", "disgust", "boredom"}
STRENGTHS = {"soft", "mid", "strong"}
_DEFAULT_EMOTION  = "neutral"
_DEFAULT_STRENGTH = "mid"

_TAG_PATTERN = "|".join(e.upper() for e in sorted(EMOTIONS))
_BLOCK_RE = re.compile(
    rf'\[(?P<tag>{_TAG_PATTERN})(?::(?P<strength>\w+))?\]'
    rf'(?P<text>.*?)'
    rf'\[/(?P=tag)\]',
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class EmotionSegment:
    text: str      # may contain [STRESS]/[SHOUT]/etc. markers
    emotion: str   # lowercase: neutral | joy | sadness | anger | fear | disgust | boredom
    strength: str  # soft | mid | strong


def parse_emotion_blocks(text: str) -> list[EmotionSegment]:
    """
    Split *text* into EmotionSegment list.
    Text outside any emotion block is treated as neutral/mid.
    """
    segments: list[EmotionSegment] = []
    pos = 0

    for m in _BLOCK_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            segments.append(EmotionSegment(before, _DEFAULT_EMOTION, _DEFAULT_STRENGTH))

        emotion  = m.group("tag").lower()
        raw_str  = (m.group("strength") or _DEFAULT_STRENGTH).lower()
        strength = raw_str if raw_str in STRENGTHS else _DEFAULT_STRENGTH
        content  = m.group("text").strip()
        if content:
            segments.append(EmotionSegment(content, emotion, strength))
        pos = m.end()

    tail = text[pos:].strip()
    if tail:
        segments.append(EmotionSegment(tail, _DEFAULT_EMOTION, _DEFAULT_STRENGTH))

    # Plain text with no tags at all
    if not segments and text.strip():
        segments.append(EmotionSegment(text.strip(), _DEFAULT_EMOTION, _DEFAULT_STRENGTH))

    return segments
