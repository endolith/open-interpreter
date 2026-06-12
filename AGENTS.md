# Agent instructions

## Branch naming

Create branches as `cursor/<topic>-6eeb` (for example, `cursor/fix-foo-6eeb`).

## Before opening a PR

1. Run `ruff check interpreter tests` and fix any reported issues.
2. Run `pytest -m "not integration"` and fix failures until green.
3. If `OPENAI_API_KEY` is available locally, also run `pytest -m integration`.
4. Commit and push your branch.
5. Open a pull request targeting `main`.
6. Do **not** merge the PR yourself.

## Install

Use editable pip install, not Poetry:

```shell
pip install -e ".[server]"
pip install pytest ruff websockets requests
```
