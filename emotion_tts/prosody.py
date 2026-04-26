#!/usr/bin/env python3
"""
Prosody marker injection for Kokoro TTS.

Markers manipulate pred_dur, F0_pred, N_pred before the Kokoro decoder —
no audio-level processing, zero seam artifacts.

Syntax (case-sensitive):
    [STRESS]word or phrase[/STRESS]  — emphasized: slower, louder, +55 Hz F0
    [SHOUT]...[/SHOUT]               — high F0, much louder
    [FAST]...[/FAST]                 — faster speech
    [SLOW]...[/SLOW]                 — slower speech
    [SOFT]...[/SOFT]                 — quiet, lower F0
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

PROSODY_RULES: dict[str, dict] = {
    "stress": {"dur_scale": 1.50, "f0_add":  55.0, "n_scale": 2.00},
    "shout":  {"dur_scale": 1.10, "f0_add":  45.0, "n_scale": 1.70},
    "fast":   {"dur_scale": 0.80, "f0_add":   0.0, "n_scale": 1.05},
    "slow":   {"dur_scale": 1.25, "f0_add":   0.0, "n_scale": 1.00},
    "soft":   {"dur_scale": 1.10, "f0_add": -15.0, "n_scale": 0.70},
}

_MARKER_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\[STRESS\](.*?)\[/STRESS\]', re.DOTALL), 'stress'),
    (re.compile(r'\[SHOUT\](.*?)\[/SHOUT\]',   re.DOTALL), 'shout'),
    (re.compile(r'\[FAST\](.*?)\[/FAST\]',     re.DOTALL), 'fast'),
    (re.compile(r'\[SLOW\](.*?)\[/SLOW\]',     re.DOTALL), 'slow'),
    (re.compile(r'\[SOFT\](.*?)\[/SOFT\]',     re.DOTALL), 'soft'),
]


@dataclass
class MarkerSpan:
    marker_type: str
    text: str
    ph_start: int
    ph_end: int


def parse_markers(text: str) -> tuple[str, list[tuple[str, int, int, str]]]:
    raw: list[tuple[int, int, str, str]] = []
    for pattern, mtype in _MARKER_RE:
        for m in pattern.finditer(text):
            raw.append((m.start(), m.end(), mtype, m.group(1)))
    raw.sort(key=lambda x: x[0])

    parts: list[str] = []
    text_spans: list[tuple[str, int, int, str]] = []
    pos = 0
    for start, end, mtype, content in raw:
        parts.append(text[pos:start])
        c_start = sum(len(p) for p in parts)
        parts.append(content)
        c_end   = sum(len(p) for p in parts)
        text_spans.append((mtype, c_start, c_end, content))
        pos = end
    parts.append(text[pos:])
    return "".join(parts).strip(), text_spans


def _ph_str(phonemizer, word: str) -> str:
    raw = phonemizer(word)
    return raw[0] if isinstance(raw, tuple) else raw


def map_spans_to_phonemes(
    phonemizer,
    clean_text: str,
    text_spans: list[tuple[str, int, int, str]],
) -> list[MarkerSpan]:
    """Map character-level spans to phoneme-token index spans.

    Uses full-sentence phonemization to preserve context-sensitive G2P.
    Spaces between words count as tokens in Kokoro's vocab.
    """
    if not text_spans:
        return []

    words = clean_text.split()
    if not words:
        return []

    raw_full = phonemizer(clean_text)
    ph_full  = raw_full[0] if isinstance(raw_full, tuple) else raw_full
    ph_words = ph_full.split(' ')

    if len(ph_words) == len(words):
        word_ph_starts: list[int] = []
        pos = 0
        for i, ph_w in enumerate(ph_words):
            word_ph_starts.append(pos)
            pos += len(ph_w)
            if i < len(ph_words) - 1:
                pos += 1
        word_ph_ends = [s + len(ph_words[i]) for i, s in enumerate(word_ph_starts)]
    else:
        ph_counts = [max(len(_ph_str(phonemizer, w)), 1) for w in words]
        cum = [0]
        for c in ph_counts:
            cum.append(cum[-1] + c)
        word_ph_starts = cum[:-1]
        word_ph_ends   = cum[1:]

    word_char_starts: list[int] = []
    cursor = 0
    for w in words:
        while cursor < len(clean_text) and clean_text[cursor] == ' ':
            cursor += 1
        word_char_starts.append(cursor)
        cursor += len(w)
    word_char_ends = [s + len(w) for s, w in zip(word_char_starts, words)]

    result: list[MarkerSpan] = []
    for mtype, t_start, t_end, content in text_spans:
        ph_start: Optional[int] = None
        ph_end:   Optional[int] = None
        for i, (ws, we) in enumerate(zip(word_char_starts, word_char_ends)):
            if we > t_start and ws < t_end:
                if ph_start is None:
                    ph_start = word_ph_starts[i]
                ph_end = word_ph_ends[i]
        if ph_start is not None and ph_end is not None and ph_start < ph_end:
            result.append(MarkerSpan(mtype, content, ph_start, ph_end))

    return result


@torch.no_grad()
def forward_with_prosody(
    kmodel,
    input_ids: torch.LongTensor,
    ref_s: torch.FloatTensor,
    speed: float,
    spans: list[MarkerSpan],
    rules: dict = PROSODY_RULES,
) -> torch.FloatTensor:
    """Kokoro forward pass with prosody injection into pred_dur, F0_pred, N_pred."""
    device = kmodel.device
    input_ids = input_ids.to(device)
    ref_s     = ref_s.to(device)

    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, device=device, dtype=torch.long)
    arange    = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(arange + 1, input_lengths.unsqueeze(1))

    bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
    d_en     = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
    s        = ref_s[:, 128:]

    d        = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _     = kmodel.predictor.lstm(d)
    duration = kmodel.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    # 1) Duration modification
    for span in spans:
        rule      = rules.get(span.marker_type, {})
        dur_scale = rule.get("dur_scale", 1.0)
        if abs(dur_scale - 1.0) < 0.005:
            continue
        tok_s = max(1, min(span.ph_start + 1, pred_dur.shape[0] - 1))
        tok_e = max(1, min(span.ph_end   + 1, pred_dur.shape[0] - 1))
        if tok_s < tok_e:
            pred_dur[tok_s:tok_e] = (
                pred_dur[tok_s:tok_e].float() * dur_scale
            ).round().clamp(min=1).long()

    indices  = torch.repeat_interleave(torch.arange(T, device=device), pred_dur)
    pred_aln = torch.zeros((T, indices.shape[0]), device=device)
    pred_aln[indices, torch.arange(indices.shape[0])] = 1
    pred_aln = pred_aln.unsqueeze(0)

    en              = d.transpose(-1, -2) @ pred_aln
    F0_pred, N_pred = kmodel.predictor.F0Ntrain(en, s)

    # 2 & 3) F0 + N modification
    f0_total    = F0_pred.shape[-1]
    align_total = int(pred_dur.sum().item())
    upsample    = f0_total / align_total if align_total > 0 else 1.0

    cum_dur = torch.cat([
        torch.zeros(1, dtype=torch.long),
        pred_dur.cpu().cumsum(0),
    ])

    for span in spans:
        rule    = rules.get(span.marker_type, {})
        f0_add  = rule.get("f0_add",  0.0)
        n_scale = rule.get("n_scale", 1.0)
        if abs(f0_add) < 0.5 and abs(n_scale - 1.0) < 0.01:
            continue

        tok_s   = max(1, min(span.ph_start + 1, len(cum_dur) - 2))
        tok_e   = max(1, min(span.ph_end   + 1, len(cum_dur) - 1))
        frame_s = min(int(cum_dur[tok_s].item() * upsample), f0_total)
        frame_e = min(int(cum_dur[tok_e].item() * upsample), f0_total)
        if frame_s >= frame_e:
            continue

        seg  = frame_e - frame_s
        ramp = min(seg // 4, 12)

        if abs(f0_add) >= 0.5:
            boost = torch.full((seg,), f0_add, device=device)
            if ramp > 1:
                boost[:ramp]  *= torch.linspace(0., 1., ramp, device=device)
                boost[-ramp:] *= torch.linspace(1., 0., ramp, device=device)
            F0_pred[..., frame_s:frame_e] = F0_pred[..., frame_s:frame_e] + boost

        if abs(n_scale - 1.0) >= 0.01:
            scale = torch.full((seg,), n_scale, device=device)
            if ramp > 1:
                scale[:ramp]  = 1. + (n_scale - 1.) * torch.linspace(0., 1., ramp, device=device)
                scale[-ramp:] = 1. + (n_scale - 1.) * torch.linspace(1., 0., ramp, device=device)
            N_pred[..., frame_s:frame_e] = N_pred[..., frame_s:frame_e] * scale

    t_en  = kmodel.text_encoder(input_ids, input_lengths, text_mask)
    asr   = t_en @ pred_aln
    audio = kmodel.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
    return audio.cpu()


def synthesize_prosody(
    kmodel,
    phonemizer,
    text: str,
    voice: torch.Tensor,
    speed: float,
    rules: dict = PROSODY_RULES,
) -> np.ndarray | None:
    """Drop-in for synthesize(). Falls back to standard path when no markers present."""
    clean_text, raw_spans = parse_markers(text)

    raw_ph   = phonemizer(clean_text)
    phonemes = raw_ph[0] if isinstance(raw_ph, tuple) else raw_ph
    if not phonemes:
        return None

    input_ids_list = list(filter(
        lambda i: i is not None,
        (kmodel.vocab.get(p) for p in phonemes),
    ))
    if len(input_ids_list) + 2 > kmodel.context_length:
        return None

    ref_s = voice[min(len(phonemes) - 1, voice.shape[0] - 1)]

    if not raw_spans:
        audio = kmodel(phonemes, ref_s, speed=speed)
        if audio is None:
            return None
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy()
        if hasattr(audio, "audio"):
            return audio.audio.detach().cpu().numpy()
        return np.asarray(audio)

    spans     = map_spans_to_phonemes(phonemizer, clean_text, raw_spans)
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]])
    ref_s_2d  = ref_s.unsqueeze(0) if ref_s.ndim == 1 else ref_s
    audio     = forward_with_prosody(kmodel, input_ids, ref_s_2d, speed, spans, rules)
    return audio.numpy()
