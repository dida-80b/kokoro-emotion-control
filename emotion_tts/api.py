"""
Public API: render_text() and render_segments().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .backend import fade, normalize_master
from .parser import EmotionSegment, parse_emotion_blocks
from .registry import load_all_speakers
from .router import EmotionRouter
from .prosody import synthesize_prosody

SR = 24000
PAUSE_WITHIN  = 0.30   # between segments of the same emotion
PAUSE_BETWEEN = 0.85   # between segments of different emotions

_DEFAULT_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _find_lab_dir(hint: Optional[Path] = None) -> Path:
    if hint:
        return Path(hint)
    env = os.environ.get("KOKORO_LAB_DIR")
    if env:
        return Path(env)
    # Default: this repository root
    return Path(__file__).resolve().parents[1]


def render_text(
    text: str,
    speaker: str = "martin",
    output: Optional[Path] = None,
    target_db: float = -1.0,
    device: str = "cpu",
    configs_dir: Optional[Path] = None,
    lab_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Parse emotion blocks in *text* and synthesize audio.

    Returns mono float32 at 24 kHz. Writes WAV if *output* is given.
    """
    segments = parse_emotion_blocks(text)
    return render_segments(
        segments,
        speaker=speaker,
        output=output,
        target_db=target_db,
        device=device,
        configs_dir=configs_dir,
        lab_dir=lab_dir,
    )


def render_segments(
    segments: list[EmotionSegment],
    speaker: str = "martin",
    output: Optional[Path] = None,
    target_db: float = -1.0,
    device: str = "cpu",
    configs_dir: Optional[Path] = None,
    lab_dir: Optional[Path] = None,
) -> np.ndarray:
    """Synthesize a pre-parsed segment list."""
    import soundfile as sf
    from misaki.espeak import EspeakG2P

    configs_dir = Path(configs_dir) if configs_dir else _DEFAULT_CONFIGS
    lab_dir     = _find_lab_dir(lab_dir)

    speaker_configs = load_all_speakers(configs_dir)
    if speaker not in speaker_configs:
        raise ValueError(
            f"Unknown speaker '{speaker}'. Available: {sorted(speaker_configs)}"
        )

    router     = EmotionRouter(speaker_configs, lab_dir=lab_dir, device=device)
    phonemizer = EspeakG2P(language="de")

    parts: list[np.ndarray] = []
    prev_emotion: Optional[str] = None

    for seg in segments:
        kmodel = router.get_model(speaker, seg.emotion)
        voice  = router.get_voice(speaker, seg.emotion, seg.strength)
        speed  = router.get_speed(speaker, seg.emotion)
        volume = router.get_volume(speaker, seg.emotion)

        audio = synthesize_prosody(kmodel, phonemizer, seg.text, voice, speed)
        if audio is None:
            continue

        audio = fade(audio) * volume

        if prev_emotion is not None:
            pause = PAUSE_WITHIN if seg.emotion == prev_emotion else PAUSE_BETWEEN
            parts.append(np.zeros(int(SR * pause), dtype=np.float32))

        parts.append(audio)
        prev_emotion = seg.emotion

    if not parts:
        return np.zeros(0, dtype=np.float32)

    result = normalize_master(np.concatenate(parts), target_db=target_db)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), result, SR)

    return result
