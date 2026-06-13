# AGENTS.md

## Overview

Open Interpreter — Python CLI (3.9+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI. See `README.md`.

## Development

```bash
pip install -e ".[server]"                         # editable install
interpreter                                        # run CLI
interpreter --server                               # WebSocket server on port 8000
```

Build backend is Poetry but CI uses pip. May migrate to uv — check `pyproject.toml` and `.github/workflows/python-package.yml` for current approach.

## Code change guidelines

- **Always write or update unit tests.** New functions need tests; bug fixes need regression tests.
- **Push and monitor CI** — CI is the source of truth. Keep checking until it passes and fix any failures.
- **Update all relevant documentation**: code comments, docstrings, `docs/` pages, and all translated READMEs (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, `docs/README_UK.md`, `docs/README_IN.md`, `docs/README_DE.md`, `docs/README_VN.md`).
- **One logical change per commit** for easy review. Clear commit messages.

## Do not touch

- User profile YAML files in platform-specific data dirs — these are user config, not project code.
- `poetry.lock` — do not regenerate unless explicitly asked.
