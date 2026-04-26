# Usage

## Text example
```text
[JOY]Das ist gut.[/JOY]
[ANGER][STRESS]STOP[/STRESS] jetzt.[/ANGER]
```

## CLI
```bash
python scripts/render.py --text "[JOY]Das ist gut.[/JOY]" --speaker victoria --output out.wav
```

## Default behavior
- untagged text -> neutral
- `neutral` uses the plain path, no emotion blend
