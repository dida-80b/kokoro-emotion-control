# kokoro-emotion-control

LLM-friendly emotion TTS control for Kokoro.

## What it does
- renders tagged text to audio
- uses English emotion tags
- injects prosody directly before decoding
- loads speaker-specific checkpoints via config

## Tags
- `[NEUTRAL]`
- `[JOY]`
- `[SADNESS]`
- `[ANGER]`
- prosody: `[STRESS]`, `[SHOUT]`, `[FAST]`, `[SLOW]`, `[SOFT]`

## Install
Use this repo directly.

## Run
```bash
python scripts/render.py --text "[ANGER][STRESS]STOP[/STRESS][/ANGER]" --speaker martin --output out.wav
```

## Train
See `docs/training.md`.
The repo ships the training code and base checkpoints; you still need the prepared dataset under `data/processed/emodb_augmented/`.

## Notes
- `neutral` is the plain default path.
- checkpoints are stored with Git LFS.

## License
Code and scripts: MIT.
