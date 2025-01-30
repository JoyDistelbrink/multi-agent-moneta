## Dependency Overview

The project dependencies are managed with [uv](https://github.com/astral-sh/uv)

To install dependencies run:
```sh
uv sync
```

To genereate requirements.txt:
```
uv pip compile pyproject.toml --no-deps
```
