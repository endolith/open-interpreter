# Porting `classic/develop` → `main`

`classic/develop` is a daily-driver branch (~589 commits ahead of `main` as of 2026-06). It is **not** the merge target. Land work as **one feature per PR** against `main`.

The **authoritative feature inventory** is the fork changelog at the top of `README.md` on `classic/develop`. This doc maps those bullets to port clusters and tracks status.

## Rules

- One isolated PR per cluster below (or smaller).
- Rebase onto `main`; do not merge `classic/develop` wholesale.
- CI must pass on `main` without `OPENAI_API_KEY` for unit tests.
- Agents: open PRs only — **do not merge**.

## Priority order (suggested)

| P | Cluster | Notes |
|---|---------|-------|
| 0 | CI on `main` | `cursor/ci-main-ruff-pytest-6eeb` |
| 1 | Conversation (`%rename`, auto-title, atomic save, truncation) | PR #9 in flight |
| 2 | Headless / pynput | Issue #3; port to `main` |
| 3 | LLM providers (DeepSeek, OpenRouter, DashScope, reasoning) | **Issue #6 is done on `classic/develop`** — port + close |
| 4 | `computer` → `toolbox` rename | Large breaking change; own PR(s) |
| 5 | Web search toolbox | LinkUp/Tavily/SerpApi/Brave/Serper, result classes |
| 6 | `view_image` | Vision tool + resize/shrink prompts |
| 7 | Incremental markdown rendering | Streaming UI; includes code-fence fixes |
| 8 | Shell split (`bash` / `cmd`) | Replaces ambiguous `shell` |
| 9 | Edit tool + pre-run command editing | sed/ed/gawk/jq/yq/comby/patch, `$EDITOR` |
| 10 | Windows support | UTF-8, Notepad fallback, `bat` highlighting, etc. |
| 11 | Terminal / REPL UX | Size detection, REPL state output, timestamps |
| 12 | System message + secret redaction + geolocation | |
| 13 | `ai2` delegation module | `boolean_query`, `choice_query`, etc. |
| 14 | Auto-run allowlist | Tri-state + exact-match allowlist |
| 15 | PowerShell improvements | Profile load, prompt filtering |
| 16 | Profile validation + telemetry off | |
| 17 | `TextFileReader` | Low priority; LLMs don't use it |
| 18 | Python 3.13 + test fixes | May overlap with CI work |

## Feature inventory (from README)

Status key: **done** = on `classic/develop`; **port** = needs isolated PR to `main`; **PR** = in flight.

### Core renames / breaking

| Feature | Status | Port notes |
|---------|--------|------------|
| `computer` → `toolbox` rename | done | Entire API rename; docstring first line + Returns in system message; encourage `help()` |
| `shell` → `bash` / `cmd` split | done | Distinct languages per platform |

### Web search toolbox

| Feature | Status | Port notes |
|---------|--------|------------|
| Multi-backend search (LinkUp, Tavily, SerpApi, Brave, Serper) | done | Fallback chain |
| `search`, `answer`, `structured_output`, `fetch` | done | |
| Compact result repr (`FetchResult`, etc.) | done | Avoid context flooding |
| Locale detection for regional results | done | |

### LLM API / providers

| Feature | Status | Port notes |
|---------|--------|------------|
| Reasoning models (`reasoning_content`, Thinking panels, params) | done | OpenRouter `extra_body`, DeepSeek V4 thinking |
| OpenRouter (`openrouter/...`, `OPENROUTER_API_KEY`) | done | |
| **DeepSeek API** (`deepseek/...`, `DEEPSEEK_API_KEY`) | done | **Issue #6 — implemented; close after port to `main`** |
| DashScope / Qwen (`dashscope-us`, `dashscope-intl`) | done | Vision for Qwen 3.5 |
| Mistral compatibility | done | Tool ID length, image role mapping |
| API error handling | done | Panels, markdown/HTML errors, retry, auto-retry |
| Usage tracking (`%usage`) | done | Token stats |

### Tools / execution

| Feature | Status | Port notes |
|---------|--------|------------|
| `view_image` | done | Resize/shrink prompts for large files |
| Edit commands before running | done | Temp file + `$EDITOR` |
| Edit tool (sed, ed, gawk, jq, yq, comby, patch, poke) | done | Split into sub-PRs if too large |
| Python REPL state output | done | Variables, modules, CWD; restart alerts |
| `ai2` module | done | `boolean_query`, `choice_query`, `single_response` |
| `TextFileReader` | done | Encoding autodetect; unused by LLMs |

### Terminal / UI

| Feature | Status | Port notes |
|---------|--------|------------|
| Incremental markdown rendering | done | Avoid flicker; stream large code outputs |
| Terminal size detection | done | Resize without breaking layout |
| User message timestamps | done | Date / elapsed time context |
| HTML output suppression | done | Don't open browser unnecessarily |

### Conversation

| Feature | Status | Port notes |
|---------|--------|------------|
| "New Conversation" in `--conversations` navigator | done | |
| Atomic file saving | done | Corruption-resistant |
| Cache-aware truncation (`truncation_step`) | done | KV/prefix-cache; large cost win |
| Auto-title conversation files | done | LLM-generated filename after N messages |
| `%rename` command | done | PR #9 → `main` |

### Safety / config

| Feature | Status | Port notes |
|---------|--------|------------|
| Secret redaction | done | Don't send env passwords to LLM servers |
| System message enhancements | done | Geolocation, REPL encouragement, etc. |
| Profile validation | done | Warn on invalid config attrs |
| Telemetry disable in profile | done | |

### Windows

| Feature | Status | Port notes |
|---------|--------|------------|
| Downloads folder via `SHGetKnownFolderPath` | done | |
| UTF-8 code page for shell | done | |
| `bat` syntax highlighting | done | |
| Notepad fallback / "edit" verb for `.html`/`.htm`/`.bat` | done | |

### Headless / bugs

| Feature | Status | Port notes |
|---------|--------|------------|
| pynput lazy import masks errors (issue #3) | done | Port to `main`; supersedes PR #5 |

### Platform / maintenance

| Feature | Status | Port notes |
|---------|--------|------------|
| Python 3.13 support + test fixes | done | Overlaps CI matrix |

## Open issues vs `classic/develop`

| Issue | Title | On `classic/develop`? | Action |
|-------|-------|----------------------|--------|
| #3 | pynput / headless traceback | Yes | Port to `main`, then close |
| #6 | DeepSeek API | **Yes** | Port to `main`, then close (do not re-implement) |
| #2 | REPL-like code | Partially (system prompts, REPL state) | Port relevant hunks; close when satisfied |

## Superseded agent PRs (wrong base)

| Old PR | Base | Replacement |
|--------|------|-------------|
| #4 AGENTS.md | `classic/develop` | AGENTS.md on `main` after CI PR |
| #5 pynput | `classic/develop` | Port → `main` |
| #7 Rich streaming | `classic/develop` | Port incremental markdown cluster → `main` |
| #8 CI | `classic/develop` | CI on `main` only |

## Dead branches

- `development` — do not target in workflows or PRs.
