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
Use this repo directly. The old lab repo is optional as a legacy testbed.

## Run
```bash
python scripts/render.py --text "[ANGER][STRESS]STOP[/STRESS][/ANGER]" --speaker martin --output out.wav
```

## Notes
- `neutral` is the plain default path.
- checkpoints are stored with Git LFS.
- the old lab repo stays as the testbed.
