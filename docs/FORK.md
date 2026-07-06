# Fork maintenance (endolith/open-interpreter)

This repository is the **community-maintained Python** edition of Open Interpreter.

The upstream [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) project now focuses on a **Rust rewrite** (default branch `oix`). Killian Lucas linked here as the Python fork in [this commit](https://github.com/openinterpreter/openinterpreter/commit/632f2a36ef06ff30431368c637ef488bbe5d508c).

## Branches

| Branch | Role |
| ------ | ---- |
| **`main`** | **Merge target.** Clean, reviewable feature PRs land here. CI runs here. |
| **`classic/develop`** | Maintainer daily driver. Large accumulated diff (~hundreds of commits ahead of `main`). **Not** the merge target — port isolated features to `main` one PR at a time. Default branch today for convenience only. |
| **`development`** | Legacy upstream branch. **Frozen / not becoming `main`.** Scheduled to be renamed `archive/development` (or deleted) once nothing references it. |
| `classic/*` | Topic experiments; may be stale. |
| `cursor/*` | Cloud agent branches (`cursor/<topic>-6eeb`). |

See also [PORT_CLUSTERS.md](PORT_CLUSTERS.md) for how to port features from `classic/develop` → `main`.

## PyPI

The PyPI package name is still [`open-interpreter`](https://pypi.org/project/open-interpreter/). Transfer/rename is **TBD** (pending coordination with upstream). Until then, installs may still point at this repo or upstream depending on release tags.

## OpenRouter attribution

When provider code sets OpenRouter HTTP headers, use this fork's URL:

- `OR_SITE_URL` → `https://github.com/endolith/open-interpreter`
- `OR_APP_NAME` → `Open Interpreter` (or `Open Interpreter (Python fork)`)

## CI and contributions

- PRs target **`main`**.
- CI: ruff + pytest (unit tests without API keys; integration optional with `OPENAI_API_KEY` secret). See [AGENTS.md](../AGENTS.md) once merged.
- Lint/format: **ruff** (not black) on `main` going forward.

## Upstream docs

Hosted-model docs under `docs/language-models/` may still reference upstream URLs. Provider behavior on `main` may lag `classic/develop` until feature ports land.
