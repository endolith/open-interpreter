# AGENTS.md

Open Interpreter — Python CLI (3.10+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

This repo (`endolith/open-interpreter`) is the community-maintained home of OI Classic (Python). Contributions belong here (branches `main` and `classic/develop`). The original home [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) is now an unrelated Rust coding agent (Codex fork). Do not treat it as upstream or copy its architecture. OpenInterpreter.com is now dedicated to an unrelated desktop tool.

## Quick commands

- `pytest -m "not integration"` – fast local unit tests (skip LLM calls)
- `pytest -k "test_name"` – run a single test
- `ruff check .` – lint
- `ruff format .` – auto-format
- `OI_RUN_INTEGRATION=1 pytest -m integration` – integration tests (also needs `OPENAI_API_KEY`; use sparingly locally)

## Development setup

See `docs/CONTRIBUTING.md` for setup instructions. CI workflow in `.github/workflows/python-package.yml` is the source of truth for how to install, lint, and test.

The build backend is Poetry, but CI installs via pip. The package manager may change (e.g. to uv) — always check `pyproject.toml` and CI for the current approach.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.
- **Autonomous agents must monitor CI** after pushing: keep checking the workflow status until it passes, fixing any failures before considering the work complete.
- **Integration tests** (`@pytest.mark.integration`) call an LLM and auto-execute generated code. Locally they need both `OI_RUN_INTEGRATION=1` and `OPENAI_API_KEY`; without either, pytest skips them. CI sets both in the integration workflow job (see `docs/CONTRIBUTING.md`).
- **Optional test gates skip; they do not fail.** Integration tests, OS markers (`linux_ci` / `windows_ci` / `darwin_ci`), and missing external binaries should produce a pytest skip with a clear reason—not a test failure—when prerequisites are absent.
- **Shell tests on Unix** that feed bash-syntax snippets to `subprocess_language` should call `require_bash_compatible_shell()` from `tests.helpers`: Non-bash `$SHELL` (e.g. fish) hangs instead of failing.
- **Shared test helpers** live in `tests/helpers.py` (import `from tests.helpers import …`). Do not import from `conftest.py` — it is not a stable import path on all platforms.
- **Computer subsystem tests** use `COMPUTER_TOOL_SUBSYSTEMS` in `tests/helpers.py` — update when `_get_all_computer_tools_list` changes (until [#101](https://github.com/endolith/open-interpreter/issues/101) lands).
- Most of the unit tests were written after the fact by AI, with the assumption that the current state of the code was correct (which is likely not true in all cases).  Keep that in mind when a test fails.  Is your code actually wrong, or was the test written to validate incorrect code?
- **Platform-specific tests** use `linux_ci`, `windows_ci`, or `darwin_ci` markers. Add OS-only coverage to `tests/test_platform_ci.py` (or the appropriate language file) rather than running the full suite on every CI runner.
- **Manual harness tests** in `tests/test_interpreter.py` (e.g. `@pytest.mark.skip(reason="Mac only")`) are legacy developer smokes — they `assert False`, read private data (SMS), or need a display. **Do not enable them in macOS CI**; write deterministic `darwin_ci` tests in `test_platform_ci.py` instead (AppleScript and shell quoting are already covered there).

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea—and nothing else** (i.e., both atomic and logical). Unrelated edits belong in separate commits even when each is small (e.g. a workflow trigger change and a pytest marker are two commits). A commit's idea includes everything that supports it—its tests, documentation, and any CI/workflow changes for it—so keep those in the same commit as the code they describe, not in a later commit for a different feature, and reviewers can read commit-by-commit while `git revert <commit>` undoes one idea cleanly.
- **Fold follow-up fixes into the commit that caused the problem.** If a commit in an unmerged branch needs a small fix (e.g. it broke CI), squash that fix into the original commit with `git rebase -i` (mark it `fixup`) rather than stacking a "fix CI" commit on top. History should read as if each commit was correct the first time, so every commit is one coherent, CI-green unit.
- **Never commit agent-generated scratch files.** AI assistants sometimes write summary/status notes into the repo (e.g. `*_summary.md`, session notes, chat dumps); these are not part of the codebase and must not be committed. If one is accidentally committed in an unmerged branch, remove it from history with `git rebase -i` rather than adding a "delete file" commit that leaves an add-then-delete pair.
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the *context* that makes the diff understandable—why each change exists and what it achieves. Avoid meta-commentary about the commit itself (e.g., "fixing my commit according to instructions"). Keep process discussion in chat.
- **Use Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.

### Comments

- **Code comments** explain *why*: the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable. Remove meta-commentary about the development process (e.g., "fixing my commit according to instructions" or "now following the directions"). Keep process discussions in chat, not in comments or commit messages.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### PRs and Issues

- **Always open a PR for your work.** All changes must be submitted as PRs so they can be revised independently. This applies to every agent, including cloud agents working in ephemeral sandboxed sessions: never finish a task without opening a PR (even docs-only changes), since the sandbox filesystem is lost when the session ends.
- **Rework an existing PR in place.** When asked to rebase or fix a PR, modify the PR's actual head branch and force-push with `--force-with-lease`—don't create a new parallel branch and point people at it. First push a backup of the head branch to the remote (e.g. `git push origin <head-branch>:backup/<head-branch>`) and confirm the backup ref exists, so the original is recoverable even if the local environment is lost; only then rewrite the branch. If anything goes wrong, restore from the backup.
- **Prefer small, reviewable PRs.** Split large efforts into stacked PRs with a clear merge order. Each PR should have one scope that is easy to review and merge.
- **Don't expand a PR's scope because a reviewer pointed something out.** Reviewer comments may flag pre-existing or out-of-scope issues; if a comment isn't about code this PR changed, fix it in its own PR and rebase on top of that, rather than scope-creeping the PR under review.
- Check if there are any Issues related to the change you are making, and if so, mention it in the PR and write `Fixes #…` in the relevant commit message, so that the Issue will be auto-closed on merge.
- **PR descriptions** should stand alone for a reviewer who has not read the issue or agent chat. Use short sections: **Background** (what should work), **Problem** (what is wrong), **Visible symptoms** (what users or CI observe), **What this PR changes** (scope and non-goals), **Tests** (what was added or updated). Add **Related work** only when stacked PRs or merge order matter. Split unrelated fixes into separate PRs; cross-link siblings when you do.

#### CodeRabbit review workflow

CodeRabbit auto-reviews on every push and posts a new review each round. It re-reviews each commit, so expect a fresh "Actionable comments posted" summary after every force-push; older review threads stay open even after they're superseded.

- **Verify each finding against the current code before acting.** Some comments are stale (superseded by a later push) or out of scope (e.g. a comment pointing at code that belongs to a different open PR — fix it there, not in the PR under review).
- **When a thread is addressed, resolve it WITH an explanatory reply** stating why it's resolved — don't just mark it resolved silently.
- Each round only surfaces new or still-unresolved findings, and may re-flag a finding as a "duplicate" when it iterates on its own earlier suggestion.
- CodeRabbit's "🪄 Autofix" checkboxes and "🤖 Prompt for AI agents" sections are optional conveniences; the `🤖 Prompt for AI agents` block is a machine-readable list of the findings to verify/fix.

### Done checklist

- [ ] Local tests pass (`pytest -m "not integration"`)
- [ ] CI is green (monitor until it passes)
- [ ] New logic has a test with a docstring
- [ ] All affected docs updated (READMEs, `docs/`, docstrings, AGENTS.md)
- [ ] PR references related Issue with `Fixes #`
