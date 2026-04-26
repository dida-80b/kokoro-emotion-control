"""
Low-level helpers: model loading, voice blending, audio post-processing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SR = 24000
FADE_IN_MS  = 20
FADE_OUT_MS = 80


def apply_checkpoint(kmodel, ckpt_path: Path) -> None:
    """Load an F0-contour checkpoint into a KModel predictor."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key, attr in [
        ("predictor_shared", "shared"),
        ("predictor_F0",     "F0"),
        ("predictor_N",      "N"),
        ("predictor_F0_proj","F0_proj"),
        ("predictor_N_proj", "N_proj"),
    ]:
        if key in ckpt:
            getattr(kmodel.predictor, attr).load_state_dict(ckpt[key])


def build_voice(
    base_voice: torch.Tensor,
    base_style: torch.Tensor,
    neutral_style: torch.Tensor,
    style_vec: torch.Tensor,
    alpha_acoustic: float,
    alpha_prosodic: float,
) -> torch.Tensor:
    """
    Blend emotion style into a voicepack.

    voice = base_style + alpha * (style_vec - neutral_style)
    Split at dim 128: acoustic ([:128]) and prosodic ([128:]).
    """
    delta = style_vec - neutral_style
    emo = base_style.clone()
    emo[:128] = base_style[:128] + alpha_acoustic * delta[:128]
    emo[128:] = base_style[128:] + alpha_prosodic * delta[128:]
    v = base_voice.clone()
    v[:, 0, :] = emo.unsqueeze(0).expand(base_voice.shape[0], -1)
    return v


def fade(
    audio: np.ndarray,
    sr: int = SR,
    fade_in_ms: int  = FADE_IN_MS,
    fade_out_ms: int = FADE_OUT_MS,
) -> np.ndarray:
    audio = audio.copy()
    n_in  = min(int(sr * fade_in_ms  / 1000), len(audio) // 4)
    n_out = min(int(sr * fade_out_ms / 1000), len(audio) // 4)
    if n_in >= 2:
        audio[:n_in]  *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_in)))
    if n_out >= 2:
        audio[-n_out:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, n_out)))
    return audio


def normalize_master(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return np.clip(audio * (10 ** (target_db / 20) / peak), -1.0, 1.0)


def load_style_bank(style_dir: Path) -> dict[str, torch.Tensor]:
    """Load all style_*.pt files from a directory into a dict keyed by suffix."""
    bank: dict[str, torch.Tensor] = {}
    for pt in Path(style_dir).glob("style_*.pt"):
        key = pt.stem[len("style_"):]
        bank[key] = torch.load(str(pt), map_location="cpu", weights_only=True).squeeze()
    return bank
