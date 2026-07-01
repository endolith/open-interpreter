# AGENTS.md

Open Interpreter — Python CLI (3.9+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

## Quick commands

- `pytest -k "test_name"` – run a single test
- `ruff check .` – lint
- `ruff format .` – auto-format

## Development setup

See `docs/CONTRIBUTING.md` for setup instructions. CI workflow in `.github/workflows/python-package.yml` is the source of truth for how to install, lint, and test.

The build backend is Poetry, but CI installs via pip. The package manager may change (e.g. to uv) — always check `pyproject.toml` and CI for the current approach.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- **Autonomous agents must monitor CI** after pushing: keep checking the workflow status until it passes, fixing any failures before considering the work complete.

### Documentation

Don't delete or omit comments while changing things.  Comments are just as important as code.

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Commits

- **One logical change per commit.** Break up multi-part work into separate commits that are each easy to review.
- Write clear, descriptive commit messages.

### PRs and Issues

- All changes must be submitted as PRs so they can be revised independently.
- Check if there are any Issues related to the change you are making, and if so, mention it in the PR and write `Fixes #…` in the relevant commit message, so that the Issue will be auto-closed on merge.

### Done checklist

- [ ] Local tests pass
- [ ] CI is green (monitor until it passes)
- [ ] New logic has a test with a docstring
- [ ] All affected docs updated (READMEs, `docs/`, docstrings, AGENTS.md)
- [ ] PR references related Issue with `Fixes #`
