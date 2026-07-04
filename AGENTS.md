# AGENTS.md

Open Interpreter — Python CLI (3.9+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

This repo (`endolith/open-interpreter`) is the community-maintained home of OI Classic (Python). Contributions belong here (branches `main` and `classic/develop`). The original home [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) is now an unrelated Rust coding agent (Codex fork). Do not treat it as upstream or copy its architecture. OpenInterpreter.com is now dedicated to an unrelated desktop tool.

## Quick commands

- `pytest -m "not integration"` – fast local unit tests (skip LLM calls)
- `pytest -k "test_name"` – run a single test
- `ruff check .` – lint
- `ruff format .` – auto-format
- `OI_RUN_INTEGRATION=1 pytest -m integration` – integration tests (call LLMs; use sparingly locally)

## Development setup

See `docs/CONTRIBUTING.md` for setup instructions. CI workflow in `.github/workflows/python-package.yml` is the source of truth for how to install, lint, and test.

The build backend is Poetry, but CI installs via pip. The package manager may change (e.g. to uv) — always check `pyproject.toml` and CI for the current approach.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- **Autonomous agents must monitor CI** after pushing: keep checking the workflow status until it passes, fixing any failures before considering the work complete.
- **Shell tests on Unix** that feed bash-syntax snippets to `subprocess_language` should call `require_bash_compatible_shell()` from `tests.helpers`:  Non-bash `$SHELL` (e.g. fish) hangs instead of failing.
- **Shared test helpers** live in `tests/helpers.py` (import `from tests.helpers import …`). Do not import from `conftest.py` — it is not a stable import path on all platforms.
- **Computer subsystem tests** use `COMPUTER_TOOL_SUBSYSTEMS` in `tests/helpers.py` — update when `_get_all_computer_tools_list` changes (until [#101](https://github.com/endolith/open-interpreter/issues/101) lands).
- **Integration tests** (`@pytest.mark.integration`) call an LLM and auto-execute generated code. Locally, set `OI_RUN_INTEGRATION=1` to run them. CI sets the same variable in the integration workflow job (see `docs/CONTRIBUTING.md`).
- **Platform-specific tests** use `linux_ci`, `windows_ci`, or `darwin_ci` markers. Add OS-only coverage to `tests/test_platform_ci.py` (or the appropriate language file) rather than running the full suite on every CI runner.
- **Manual harness tests** in `tests/test_interpreter.py` (e.g. `@pytest.mark.skip(reason="Mac only")`) are legacy developer smokes — they `assert False`, read private data (SMS), or need a display. **Do not enable them in macOS CI**; write deterministic `darwin_ci` tests in `test_platform_ci.py` instead (AppleScript and shell quoting are already covered there).

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea—and nothing else** (i.e., both atomic and logical). This includes documentation and tests related to the change—keep them in the same commit so the code, its tests, and its documentation remain in sync and can be reverted together. Break up multi-part work into separate commits that are each easy to review.
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the *context* that makes the diff understandable—why each change exists and what it achieves. Avoid meta-commentary about the commit itself (e.g., "fixing my commit according to instructions"). Keep process discussion in chat.
- **Use Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.

### Comments

- **Code comments** explain *why*: the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable. Remove meta-commentary about the development process (e.g., "fixing my commit according to instructions" or "now following the directions"). Keep process discussions in chat, not in comments or commit messages.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### PRs and Issues

- All changes must be submitted as PRs so they can be revised independently.
- Check if there are any Issues related to the change you are making, and if so, mention it in the PR and write `Fixes #…` in the relevant commit message, so that the Issue will be auto-closed on merge.

### Done checklist

- [ ] Local tests pass (`pytest -m "not integration"`)
- [ ] CI is green (monitor until it passes)
- [ ] New logic has a test with a docstring
- [ ] All affected docs updated (READMEs, `docs/`, docstrings, AGENTS.md)
- [ ] PR references related Issue with `Fixes #`
