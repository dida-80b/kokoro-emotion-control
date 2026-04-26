"""
Routes (speaker, emotion, strength) to the right KModel + voice tensor.
Caches loaded models by (speaker, f0_checkpoint) to avoid redundant loads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .backend import apply_checkpoint, build_voice, load_style_bank
from .registry import SpeakerConfig


class EmotionRouter:
    def __init__(
        self,
        speaker_configs: dict[str, SpeakerConfig],
        lab_dir: Path,
        device: str = "cpu",
    ) -> None:
        self.configs  = speaker_configs
        self.lab_dir  = Path(lab_dir)
        self.device   = device
        self._models:  dict[tuple[str, Optional[str]], object] = {}
        self._voices:  dict[str, torch.Tensor] = {}
        self._styles:  dict[str, dict[str, torch.Tensor]] = {}

    def _resolve(self, rel: str) -> Path:
        return self.lab_dir / rel

    def _load_base_model(self, speaker: str):
        from kokoro import KModel  # imported here to keep module importable without kokoro
        cfg = self.configs[speaker]
        return (
            KModel(
                config=str(self._resolve(cfg.config)),
                model =str(self._resolve(cfg.model)),
            )
            .to(self.device)
            .eval()
        )

    def get_model(self, speaker: str, emotion: str):
        """Return a KModel with the correct F0 checkpoint for this emotion (cached)."""
        cfg     = self.configs[speaker]
        emo_cfg = cfg.emotions.get(emotion) or cfg.emotions.get("neutral")
        ckpt    = emo_cfg.f0_checkpoint if emo_cfg else None
        key     = (speaker, ckpt)
        if key not in self._models:
            kmodel = self._load_base_model(speaker)
            if ckpt:
                apply_checkpoint(kmodel, self._resolve(ckpt))
            self._models[key] = kmodel
        return self._models[key]

    def _style_bank(self, speaker: str) -> dict[str, torch.Tensor]:
        if speaker not in self._styles:
            cfg = self.configs[speaker]
            self._styles[speaker] = load_style_bank(self._resolve(cfg.style_dir))
        return self._styles[speaker]

    def _base_voice(self, speaker: str) -> torch.Tensor:
        if speaker not in self._voices:
            cfg = self.configs[speaker]
            self._voices[speaker] = torch.load(
                str(self._resolve(cfg.voicepack)),
                map_location="cpu",
                weights_only=True,
            )
        return self._voices[speaker]

    def get_voice(self, speaker: str, emotion: str, strength: str = "mid") -> torch.Tensor:
        """Return the voice tensor blended for the requested emotion + strength."""
        cfg     = self.configs[speaker]
        emo_cfg = cfg.emotions.get(emotion) or cfg.emotions.get("neutral")
        base_voice = self._base_voice(speaker)

        if emo_cfg is None or emo_cfg.style_key is None:
            return base_voice  # neutral: no blending

        bank        = self._style_bank(speaker)
        base_style  = base_voice[:, 0, :].mean(dim=0)
        neutral_style = bank.get("Neutral_mid", base_style)

        style_key = f"{emo_cfg.style_key}_{strength}"
        style_vec = bank.get(style_key)
        if style_vec is None:
            style_vec = bank.get(f"{emo_cfg.style_key}_mid")
        if style_vec is None:
            style_vec = neutral_style

        return build_voice(
            base_voice, base_style, neutral_style, style_vec,
            emo_cfg.alpha_acoustic, emo_cfg.alpha_prosodic,
        )

    def get_speed(self, speaker: str, emotion: str) -> float:
        cfg     = self.configs[speaker]
        emo_cfg = cfg.emotions.get(emotion) or cfg.emotions.get("neutral")
        return emo_cfg.speed if emo_cfg else 1.0

    def get_volume(self, speaker: str, emotion: str) -> float:
        cfg     = self.configs[speaker]
        emo_cfg = cfg.emotions.get(emotion) or cfg.emotions.get("neutral")
        return emo_cfg.volume if emo_cfg else 1.0
