"""
Speaker and emotion configuration loader.
Configs are YAML files in configs/ — paths are relative to lab_dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmotionConfig:
    f0_checkpoint: Optional[str]  # path relative to lab_dir, or None
    style_key: Optional[str]      # e.g. "Freude" (no strength suffix), or None
    alpha_acoustic: float = 0.20
    alpha_prosodic: float = 0.78
    speed: float = 1.0
    volume: float = 1.0


@dataclass
class SpeakerConfig:
    name: str
    model: str      # relative to lab_dir
    config: str     # relative to lab_dir (kokoro_config.json)
    voicepack: str  # relative to lab_dir
    style_dir: str  # relative to lab_dir
    emotions: dict[str, EmotionConfig] = field(default_factory=dict)


def load_speaker(yaml_path: Path) -> SpeakerConfig:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    emotions: dict[str, EmotionConfig] = {}
    for name, cfg in data.get("emotions", {}).items():
        emotions[name] = EmotionConfig(
            f0_checkpoint  = cfg.get("f0_checkpoint"),
            style_key      = cfg.get("style_key"),
            alpha_acoustic = cfg.get("alpha_acoustic", 0.20),
            alpha_prosodic = cfg.get("alpha_prosodic", 0.78),
            speed          = cfg.get("speed",  1.0),
            volume         = cfg.get("volume", 1.0),
        )
    return SpeakerConfig(
        name      = data["speaker"],
        model     = data["model"],
        config    = data["config"],
        voicepack = data["voicepack"],
        style_dir = data["style_dir"],
        emotions  = emotions,
    )


def load_all_speakers(config_dir: Path) -> dict[str, SpeakerConfig]:
    result: dict[str, SpeakerConfig] = {}
    for yaml_path in Path(config_dir).glob("*.yaml"):
        cfg = load_speaker(yaml_path)
        result[cfg.name] = cfg
    return result
