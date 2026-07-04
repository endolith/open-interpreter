# ●

**Open Interpreter is large, open-source initiative to build a standard interface between language models and computers.**

There are many ways to contribute, from helping others on [Github](https://github.com/endolith/open-interpreter/issues) or [Discord](https://discord.gg/6p3fD6rBVm), writing documentation, or improving code.

We depend on contributors like you. Let's build this.

## Which version is this?

The name "Open Interpreter" has been used for several related but distinct projects. If you are reading this file, you are in the **community-maintained fork of the classic Python version** at [endolith/open-interpreter](https://github.com/endolith/open-interpreter).

| Project | Where | Language | Status | What it does |
| --- | --- | --- | --- | --- |
| **Open Interpreter (Classic)** | `main` | Python | Active | The OG Python project: a terminal chatbot that runs code and shell commands locally in an interactive, REPL-like session.  **Contributions should target this branch.** |
| **OI Classic `develop`** | `classic/develop` | Python | Active | The day-to-day development branch of OI Classic. Same codebase as `main`, but with in-progress changes.  This is what I use regularly, and its features will be merged into `main` eventually. |
| **Open Interpreter 1.0 (abandoned)** | `development` | Python | Inactive | An incomplete rewrite that was meant to become OI 1.0, using Anthropic-native tools like Claude **computer use**, and a new TUI. Some features or architecture may be cherry-picked into `main` later. |
| **oiv2** | [Notnaton/oiv2](https://github.com/Notnaton/oiv2) | Python | Inactive | A proof of concept aimed at **small local LLMs**. Uses simple XML tool tags, a persistent Python environment, and a grid-overlay screenshot system (`A1`, `B2`, …) so weaker models can click the screen without fine-grained vision. |
| **Open Interpreter (Rust)** | [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) | Rust | Active | Originally the home of this project, now repurposed as a **fork of OpenAI Codex** acting as a terminal **coding agent**. (Edits files in a project, runs sandboxed commands, MCP, etc. Somewhat different focus from OI Classic?) |
| **Interpreter (desktop app)** | [openinterpreter.com](https://www.openinterpreter.com/) | Unknown | Active | A desktop product focused on document/knowledge work (Word, Excel, PDFs, browser). [Open Interpreter's blog](https://www.openinterpreter.com/blog/open-interpreter-1-0) says it is "built on the new Open Interpreter runtime." The public Rust repo's releases ship the terminal `interpreter` CLI, not the desktop installer. |

**Who maintains what?** Killian Lucas created the original Python Open Interpreter and now focuses on the Rust agent and desktop app. He is no longer maintaining the Python version. This fork (`endolith/open-interpreter`) is the community home for **OI Classic** going forward.

**Contributing here:** open issues and PRs against **`main`** (OI Classic) in [endolith/open-interpreter](https://github.com/endolith/open-interpreter). Do not assume the Rust repo, the `development` branch, or `oiv2` share the same architecture or contribution process.

## What should I work on?

First, please familiarize yourself with our [project scope](ROADMAP.md#whats-in-our-scope). Then, pick up a task from our [roadmap](ROADMAP.md) or work on solving an [issue](https://github.com/endolith/open-interpreter/issues).

If you encounter a bug or have a feature in mind, don't hesitate to [open a new issue](https://github.com/endolith/open-interpreter/issues/new/choose).

## Philosophy

This is a minimalist, **tightly scoped** project that places a premium on simplicity. We're skeptical of new extensions, integrations, and extra features. We would rather not extend the system if it adds nonessential complexity.

# Contribution Guidelines

1. Before taking on significant code changes, please discuss your ideas on [Discord](https://discord.gg/6p3fD6rBVm) to ensure they align with our vision. We want to keep the codebase simple and unintimidating for new users.
2. Fork the repository and create a new branch for your work.
3. Follow the [Running Your Local Fork](CONTRIBUTING.md#running-your-local-fork) guide below.
4. Make changes with clear code comments explaining your approach. Try to follow existing conventions in the code.
5. Follow the [Code Formatting and Linting](CONTRIBUTING.md#code-formatting-and-linting) guide below.
6. Open a PR into `main` linking any related issues. Provide detailed context on your changes.

We will review PRs when possible and work with you to integrate your contribution. Please be patient as reviews take time. Once approved, your code will be merged.

## Running Your Local Fork

**Note: for anyone testing the new `--local`, `--os`, and `--local --os` modes: When you run `poetry install` you aren't installing the optional dependencies and it'll throw errors. To test `--local` mode, run `poetry install -E local`. To test `--os` mode, run `poetry install -E os`. To test `--local --os` mode, run `poetry install -E local -E os`. You can edit the system messages for these modes in `interpreter/terminal_interface/profiles/defaults`.**

Once you've forked the code and created a new branch for your work, you can run the fork in CLI mode by following these steps:

1. CD into the project folder by running `cd open-interpreter`.
2. Install `poetry` [according to their documentation](https://python-poetry.org/docs/#installing-with-pipx), which will create a virtual environment for development + handle dependencies.
3. Install dependencies by running `poetry install`.
4. Run the program with `poetry run interpreter`. Run tests with `poetry run pytest -s -x`.


### CI test layout

GitHub Actions splits work by OS so Linux runs the full suite without re-running everything on Windows and macOS:

| Job | Runner | What runs |
| --- | --- | --- |
| **Unit tests (Linux)** | `ubuntu-latest` | Full unit suite (`pytest -m "not integration and not windows_ci and not darwin_ci"`), Python 3.10–3.14 |
| **Integration** | `ubuntu-latest` | LLM tests (`pytest -m integration`), same-repo PRs and `main` only; sets `OI_RUN_INTEGRATION=1` |
| **Windows CI smoke** | `windows-latest` | Only `@pytest.mark.windows_ci` — `cmd.exe`, PowerShell, Windows paths, `tests` import path |
| **macOS CI smoke** | `macos-latest` | Only `@pytest.mark.darwin_ci` — real `osascript`, Unix `$SHELL` |

**Integration tests** (`pytest -m integration`) call an LLM and auto-run generated code. They are skipped by default locally even when `OPENAI_API_KEY` is set — pass `OI_RUN_INTEGRATION=1` to run them. CI sets `OI_RUN_INTEGRATION=1` in the integration workflow job.

```bash
# Linux/macOS
OI_RUN_INTEGRATION=1 pytest -m integration

# Windows
set OI_RUN_INTEGRATION=1
pytest -m integration
```

Locally: `pytest -m "not integration"` runs the full unit suite. `linux_ci` / `windows_ci` / `darwin_ci` tests are skipped automatically on the wrong OS.


**Note**: This project uses [`black`](https://black.readthedocs.io/en/stable/index.html) and [`isort`](https://pypi.org/project/isort/) via a [`pre-commit`](https://pre-commit.com/) hook to ensure consistent code style. If you need to bypass it for some reason, you can `git commit` with the `--no-verify` flag.

### Installing New Dependencies

If you wish to install new dependencies into the project, please use `poetry add package-name`.

### Installing Developer Dependencies

If you need to install dependencies specific to development, like testing tools, formatting tools, etc. please use `poetry add package-name --group dev`. Also add the package to the `dev` extra in `[tool.poetry.extras]` and as an optional dependency so `pip install -e ".[dev]"` stays in sync.

### Known Issues

For some, `poetry install` might hang on some dependencies. As a first step, try to run the following command in your terminal:

`export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring`

Then run `poetry install` again. If this doesn't work, please join our [Discord community](https://discord.gg/6p3fD6rBVm) for help.

## Code Formatting and Linting

Our project uses `black` for code formatting and `isort` for import sorting. To ensure consistency across contributions, please adhere to the following guidelines:

1. **Install Pre-commit Hooks**:

   If you want to automatically format your code every time you make a commit, install the pre-commit hooks.

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   After installing, the hooks will automatically check and format your code every time you commit.

2. **Manual Formatting**:

   If you choose not to use the pre-commit hooks, you can manually format your code using:

   ```bash
   black .
   isort .
   ```

# Licensing

Contributions to Open Interpreter would be under the MIT license before version 0.2.0, or under AGPL for subsequent contributions.

# Questions?

Join our [Discord community](https://discord.gg/6p3fD6rBVm) and post in the #General channel to connect with contributors. We're happy to guide you through your first open source contribution to this project!

**Thank you for your dedication and understanding as we continue refining our processes. As we explore this extraordinary new technology, we sincerely appreciate your involvement.**
