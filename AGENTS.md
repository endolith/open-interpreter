# AGENTS.md

Open Interpreter — Python CLI (3.10+) that lets LLMs execute code locally. Uses LiteLLM, ipykernel/jupyter-client, Rich, FastAPI.

This repo (`endolith/open-interpreter`) is the community-maintained home of OI Classic (Python). Contributions belong here (branches `main` and `classic/develop`). The original home [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) is now an unrelated Rust coding agent (Codex fork). Do not treat it as upstream or copy its architecture. OpenInterpreter.com is now dedicated to an unrelated desktop tool.

## Development setup

Features from this branch will eventually be merged into `main`.

Any feature branches that target `classic/develop` should have a `develop/` prefix.  Most changes should just be committed directly to `classic/develop`, though.

Tests **do** run on this branch — run them. The suite is currently green
(`python -m pytest -m "not integration"` — 583 passed, 0 failed), so a failure is
yours until shown otherwise. Skips are expected and legitimate: they gate on
things a given machine does not have (`pwsh`, `R`, `yq`, `gawk`, `patch`, `poke`,
`comby`), and each says which.

Most unit tests here were written after the fact by an AI, assuming the code was
correct — so a surprising failure is more often a bad test than a regression (see
Testing below). Check which before assuming either way.

**CI does not run for this branch, and that is expected.**
`.github/workflows/python-package.yml` triggers only on `push` and `pull_request`
against `main`, so pushing to `classic/develop` runs nothing. The "CI is green"
line in the done checklist is a standing non-item here — do not wait for a run
that will never start, and do not claim one. Verify locally instead:

    python -m ruff check interpreter tests     # the same lint gate CI runs
    python -m pytest -m "not integration" -q

The branch **is** lint-clean: `ruff check interpreter tests` passes, so any error it
reports is one you introduced. That was not always true. Six standing
`F821`/`F601` errors used to be documented here as an accepted baseline, and all
six have since been fixed — and all six were real defects, not style:

- `display.py` called `get_monitors()`, a name never imported (the module lazy-imports
  `screeninfo`), so `toolbox.display.view()` raised `NameError` on every call.
- `point.py` read `model_path` three times and never assigned it. It never fired
  because `fast_model = True` made the block dead, so it sat latent behind a flag.
- `terminal_interface.py`'s edit `extension_map` listed `cmd`/`bash` twice. The
  duplicates agreed, so nothing broke — but disagreeing duplicates would have
  silently changed which syntax highlighting the editor used.

If you hit lint errors and believe they predate your work, check `git log` before
dismissing them. Do not reintroduce the "standing baseline" framing to avoid a
drive-by fix; give the fix its own commit instead.

## Codebase map

- `interpreter/terminal_interface/start_terminal_interface.py:main` — CLI entry
  (`interpreter` command). Parses args, builds the `Interpreter`, starts it.
- `interpreter/terminal_interface/terminal_interface.py` — the interactive REPL
  (chat loop, confirmations, display). `%`-commands in `magic_commands.py`;
  first-run contribution prompts in `contributing_conversations.py`; profiles
  (`--profile os`, `--local`, …) in `profiles/defaults` +
  `validate_llm_settings.py`.
- `interpreter/core/core.py` — the `Interpreter` object (config, `chat()`).
- `interpreter/core/respond.py` — the respond loop (LLM chunks → code blocks →
  execution → approval). `_respond_and_store` is the seam most tests patch.
- `interpreter/core/llm/` — LiteLLM wrapper (`llm.py`; see Model providers above),
  streaming runners (`run_tool_calling_llm.py`, `run_function_calling_llm.py`,
  `run_text_llm.py`), message conversion in `utils/`
  (`convert_to_openai_messages`, `merge_deltas`, `parse_partial_json`).
- `interpreter/core/async_core.py` — the async HTTP/WebSocket server
  (`AsyncInterpreter`, `Server`, `create_router`; FastAPI + janus queues + worker
  threads). Owns the approval handshake (`pending_confirmation`, `_approval_event`),
  the output queue, and the OpenAI-compatible endpoints. Largest under-covered
  file; `docs/ROADMAP.md` Phase 1b has the workflow (pin behaviour first, flip
  pins in the fix PR).
- `interpreter/core/computer/` — execution backends. Shell/Python/etc. go through
  `terminal/languages/subprocess_language.py`); desktop control lives in
  `core/toolbox/` (`mouse/`, `keyboard/`, `display/`, `vision/`, `mail/`, `sms/`,
  `calendar/`, `contacts/`, `clipboard/`, `files/`, `browser/`, `web/`, `os/`) and
  is the mostly-uncovered OS-integration cluster. `interpreter/computer_use/` at
  top level is the deprecated Anthropic demo — excluded from coverage, do not
  touch. **There is no `core/computer/` on this branch**; that layout is `main`'s.
- `interpreter/core/utils/` — small helpers (`truncate_output`, `prompt_choice`
  for y/n prompts, telemetry). `prompt_choice` takes only exact single letters
  and re-prompts on anything else.
- `interpreter/core/tools/file_edit.py` — file-editing tool.
- `scripts/wtf.py` — standalone "fix my last terminal error" CLI.
- **Dead or fragile live code**: `Skills.run()` (in `core/toolbox/skills`),
  `api.openinterpreter.com` callers, `aifs`-based search, torch-based icon
  grounding. macOS-only (AppleScript/DB-gated): mail, sms, calendar, contacts.
  Moondream/EasyOCR vision is **not** present here — that is `main`.

## Model providers

Prefixes that route to a specific provider are wired in `Llm.load()`
(`interpreter/core/llm/llm.py`), next to the dashscope and deepseek blocks:

- `openai/…`, `openrouter/…`, `anthropic/…` etc. — resolved by LiteLLM itself, no OI code.
- `dashscope-us/`, `dashscope-intl/`, `deepseek/` — OI sets `api_base`/`api_key` from env vars.
- `opencode_go/…` — as above, plus a required `x-opencode-session` header, a hard
  requirement on `OPENCODE_GO_API_KEY` (never `OPENAI_API_KEY`), a refusal for the
  Go models that are not served over chat completions, and a hardcoded answer for
  the one vision-capable model. See `docs/settings/all-settings.mdx`.
- `i` — Open Interpreter's own hosted model, special-cased in `Llm.run()`.

Three things to know before adding another prefix:

- **Do not rewrite the model name to make LiteLLM route it.** `self.model` is
  read by every provider gate in the codebase — tool-call format, reasoning
  padding, vision, secret sanitising — so overwriting it with a different
  provider's name makes all of them decide wrongly and quietly. `opencode_go/`
  used to be rewritten to `openai/<model>`, and that single overloading caused a
  run of unrelated-looking misroutings. The correct shape is: keep `self.model`
  truthful, and pass the wire name plus `custom_llm_provider` in the params
  inside `run()`. LiteLLM has no `opencode_go` provider, so both parts are
  needed. The outgoing request is identical to the rewritten form — verified
  against the gateway.
- **A provider prefix that LiteLLM does not know needs its metadata supplied
  explicitly.** LiteLLM has no entry for these ids, so anything it would
  otherwise answer (context window, vision support, tool-call format) comes back
  as a miss. A miss is worse than an answer when the consequence is silent: an
  unknown context window means trimming to 8000 tokens, and unknown vision means
  images replaced by text descriptions. `opencode_go/` hardcodes the endpoint
  split and the one vision-capable model, and reads context/output limits from
  the upstream catalog (`models.dev`, provider `opencode-go`) with a static copy
  as fallback. Note the gateway's own `/zen/go/v1/models` returns ids only — it
  carries no limits, which is why the upstream catalog is used instead.
- Pointing a model at an `api_base` makes LiteLLM resolve an unset `api_key` from
  the environment, so an ambient `OPENAI_API_KEY` gets sent as `Authorization:
  Bearer` to that host. A prefix with its own credential must set `api_key`
  explicitly or fail loudly.
- `start_terminal_interface.py` has a hardcoded allowlist of prefixes that must not be
  re-prefixed when `--api_base` is set; a new prefix needs adding there too.

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
