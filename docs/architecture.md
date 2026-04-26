# Architecture

Text -> emotion parser -> speaker/emotion router -> Kokoro + checkpoint -> prosody injection -> audio

## Core idea
- emotion blocks select speaker-specific checkpoints and styles
- prosody markers modify duration, F0, and energy inside the model
- neutral text stays plain

## Files
- `emotion_tts/parser.py`
- `emotion_tts/router.py`
- `emotion_tts/prosody.py`
- `emotion_tts/api.py`
