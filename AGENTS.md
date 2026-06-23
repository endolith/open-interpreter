# AGENTS.md

## Development setup

See `docs/CONTRIBUTING.md` for setup instructions. CI workflow in `.github/workflows/python-package.yml` is the source of truth for how to install, lint, and test.

The build backend is Poetry, but CI installs via pip. The package manager may change (e.g. to uv) — always check `pyproject.toml` and CI for the current approach.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Autonomous agents must monitor CI** after pushing: keep checking the workflow status until it passes, and fix any failures before considering the work complete.

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change.

### Commits

- **One logical change per commit.** Break up multi-part work into separate commits that are each easy to review.
- Write clear, descriptive commit messages.
