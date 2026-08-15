# AGENTS.md

Open Interpreter — Python CLI (3.10+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

This repo (`endolith/open-interpreter`) is the community-maintained home of OI Classic (Python). Contributions belong here (branches `main` and `classic/develop`). The original home [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) is now an unrelated Rust coding agent (Codex fork). Do not treat it as upstream or copy its architecture. OpenInterpreter.com is now dedicated to an unrelated desktop tool.

## Development setup

This is the classic/develop branch which is a mess. Don't bother running tests and stuff, those only work in the main branch.

## Code change guidelines

### Testing

- **Always write or update unit tests** when changing code. New functions/methods need tests; bug fixes need regression tests.
- **Every test function must have a docstring** explaining what behavior it verifies and why. Someone who breaks the test must be able to understand what they broke and what the intended behavior is.

### Documentation

When changing code, update **all** relevant documentation:

- **Code comments and docstrings** — keep them accurate and up-to-date.
- **`docs/` folder** — update any affected documentation pages.
- **README files** — the project has READMEs in multiple languages (`README.md`, `docs/README_ZH.md`, `docs/README_JA.md`, `docs/README_ES.md`, …). If you change something documented in the English README, update the translated versions too.
- **`AGENTS.md`** — update this file if the development guidelines change or there is something non-obvious that an agent needs to know in future jobs.

### Commits

- **Make every commit a small, self-contained, working unit that completes one coherent idea—and nothing else** (i.e., both atomic and logical). Unrelated edits belong in separate commits even when each is small (e.g. a workflow trigger change and a pytest marker are two commits). A commit's idea includes everything that supports it—its tests, documentation, and any CI/workflow changes for it—so keep those in the same commit as the code they describe, not in a later commit for a different feature, and reviewers can read commit-by-commit while `git revert <commit>` undoes one idea cleanly.
- **Fold follow-up fixes into the commit that caused the problem.** If a commit in an unmerged branch needs a small fix (e.g. it broke CI), squash that fix into the original commit with `git rebase -i` (mark it `fixup`) rather than stacking a "fix CI" commit on top. History should read as if each commit was correct the first time, so every commit is one coherent, CI-green unit.  (Actually in `develop` if the fix is several commits later, just leave it clearly marked so it can be folded into the feature PR later.)
- **Write comprehensive commit messages.** The subject line is a concise summary; the body must explain the problem being solved, the chosen approach, and any trade-offs. Provide the *context* that makes the diff understandable—why each change exists and what it achieves. Avoid meta-commentary about the commit itself (e.g., "fixing my commit according to instructions"). Keep process discussion in chat.
- **Use Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`) to categorize changes and enable automated changelog generation.

### Comments

- **Code comments** explain *why*: the intent, non-obvious reasoning, edge cases, and business logic. If a comment is needed to restate what the code does, rewrite the code to be clearer instead. Historical context that explains current behavior is acceptable. Remove meta-commentary about the development process (e.g., "fixing my commit according to instructions" or "now following the directions"). Keep process discussions in chat, not in comments or commit messages.
- Don't delete or omit comments while changing things. Comments are just as important as code.

### PRs and Issues

- In the develop branch, don't create PRs.  Features will be merged into `main` as PRs eventually.

#### CodeRabbit review workflow

No PRs, so no CodeRabbit. :D
