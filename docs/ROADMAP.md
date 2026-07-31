# Roadmap

## Plan

This repository (`endolith/open-interpreter`) is the community-maintained home of **OI Classic**, the Python edition of Open Interpreter. The upstream [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) repo was rewritten as an unrelated Rust project, so OI Classic now evolves independently.

Work is organized in phases so that each step makes the next one more trustworthy.

### Phase 1: CI and test foundation (in progress)

Automation so that pull requests can be reviewed and merged with confidence. Much of this is already in place on `main`:

- [x] Unit tests run in CI without an API key (`pytest -m "not integration"`)
- [x] Ruff lint job with `pyproject.toml` configuration
- [x] Python 3.10–3.14 test matrix with `fail-fast: false`
- [x] Coverage reports with optional Codecov upload
- [x] Mock OpenAI server so LLM-dependent smoke tests need no real key
- [x] Integration tests gated behind `OI_RUN_INTEGRATION` + `OPENAI_API_KEY`, run in their own job
- [x] Shared test helpers (`tests/helpers.py`) and `linux_ci`/`windows_ci`/`darwin_ci` platform markers
- [x] [CodeRabbit](https://github.com/apps/coderabbitai) AI code review on pull requests
- [x] Hundreds of unit tests across `interpreter/` — coverage rose significantly over recent commits
- [ ] Close the remaining coverage gaps (see [#141](https://github.com/endolith/open-interpreter/issues/141)) and keep every new feature test-backed

### Phase 2: Port features from `classic/develop` into `main`

`classic/develop` is the maintainer's daily driver: it contains most recent features and is what most users should install for now. Its features are scattered across one long linear history, so each one is ported to `main` as its own isolated PR. As features land, `main` gradually becomes the better (and eventually the recommended) install for everyone:

- [ ] File-edit tools (sed, gawk, jq, yq, comby, patch) with dry-run previews
- [ ] Tool-calling instruction refactor and generated tool schemas
- [ ] PowerShell prompt detection, profile loading, and shell improvements
- [ ] `reasoning_content` streaming
- [ ] Web toolbox (`web.search` / `web.fetch` / `web.answer`) with pluggable backends
- [ ] Conversation title generation, `%rename`, and atomic conversation saving
- [ ] `view_image` tool with user approval flow
- [ ] Tri-state `auto_run` with allowlist
- [ ] Profile validation and telemetry opt-out
- [ ] Provider error retry logic and improved error rendering
- [ ] Conversation undo improvements
- [ ] … and the other feature clusters in `classic/develop`

### Phase 3: Pythonic refactor

Once the develop features are in, refactor the codebase to be more Pythonic and better written in general. This must wait until after Phase 2, because `classic/develop` is based on the older, less Pythonic `main` code.

### Phase 4: Mine the abandoned OI 1.0 rewrite (maybe)

The `development` branch (and its continuation `develop_1.0`) was meant to become Open Interpreter 1.0, but depends on Anthropic computer-use APIs that are now obsolete and was never feature-complete. After the earlier phases, consider cherry-picking any worthwhile ideas from it.

## Documentation

- [ ] Work with Mintlify to translate docs. How does Mintlify let us translate our documentation automatically? I know there's a way.
- [ ] Better comments throughout the package (they're like docs for contributors)
- [ ] Show how to replace interpreter.llm so you can use a custom llm

## New features

- [ ] Figure out how to get OI to answer to user input requests like python's `input()`. Do we somehow detect a delay in the output..? Is there some universal flag that TUIs emit when they expect user input? Should we do this semantically with embeddings, then ask OI to review it and respond..?
- [ ] Placeholder text that gives a compelling example OI request. Probably use `textual`
- [ ] Everything else `textual` offers, like could we make it easier to select text? Copy paste in and out? Code editing interface?
- [x] Let people turn off the active line highlighting
- [ ] Add a --plain flag which doesn't use rich, just prints stuff in plain text
- [ ] Use iPython stuff to track the active line, instead of inserting print statements, which makes debugging weird (From ChatGPT: For deeper insights into what's happening behind the scenes, including which line of code is being executed, you can increase the logging level of the IPython kernel. You can configure the kernel's logger to a more verbose setting, which logs each execution request. However, this requires modifying the kernel's startup settings, which might involve changing logging configurations in the IPython kernel source or when launching the kernel.)
- [ ] Let people edit the code OI writes. Could just open it in the user's preferred editor. Simple. [Full description of how to implement this here.](https://github.com/OpenInterpreter/open-interpreter/pull/830#issuecomment-1854989795)
- [ ] Display images in the terminal interface
- [ ] There should be a function that just renders messages to the terminal, so we can revive conversation navigator, and let people look at their conversations
- [ ] ^ This function should also render the last like 5 messages once input() is about to be run, so we don't get those weird stuttering `rich` artifacts
- [ ] Let OI use OI, add `interpreter.chat(async=True)` bool. OI can use this to open OI on a new thread
  - [ ] Also add `interpreter.await()` which waits for `interpreter.running` (?) to = False, and `interpreter.result()` which returns the last assistant messages content.
- [ ] Allow for limited functions (`interpreter.functions`) using regex
  - [ ] If `interpreter.functions != []`:
    - [ ] set `interpreter.computer.languages` to only use Python
    - [ ] Use regex to ensure the output of code blocks conforms to just using those functions + other python basics
- [ ] (Maybe) Allow for a custom embedding function (`interpreter.embed` or `computer.ai.embed`) which will let us do semantic search
- [ ] (Maybe) if a git is detected, switch to a mode that's good for developers, like showing nested file structure in dynamic system message, searching for relevant functions (use computer.files.search)
- [x] Allow for integrations somehow (you can replace interpreter.llm.completions with a wrapped completions endpoint for any kind of logging. need to document this tho)
  - [ ] Document this^
- [ ] Expand "safe mode" to have proper, simple Docker support, or maybe Cosmopolitan LibC
- [ ] Make it so core can be run elsewhere from terminal package — perhaps split over HTTP (this would make docker easier too)
- [ ] For OS mode, experiment with screenshot just returning active window, experiment with it just showing the changes, or showing changes in addition to the whole thing, etc. GAIA should be your guide

## Future-proofing

- [ ] Really good tests / optimization framework, to be run less frequently than Github actions tests
  - [x] Figure out how to run us on [GAIA](https://huggingface.co/gaia-benchmark)
    - [x] How do we just get the questions out of this thing?
    - [x] How do we assess whether or not OI has solved the task?
  - [ ] Loop over GAIA, use a different language model every time (use Replicate, then ask LiteLLM how they made their "mega key" to many different LLM providers)
  - [ ] Loop over that ↑ using a different prompt each time. Which prompt is best across all LLMs?
  - [ ] (For the NCU) might be good to use a Google VM with a display
  - [ ] (Future future) Use GPT-4 to assess each result, explaining each failure. Summarize. Send it all to GPT-4 + our prompt. Let it redesign the prompt, given the failures, rinse and repeat
- [ ] Stateless (as in, doesn't use the application directory) core python package. All `appdir` or `platformdirs` stuff should be only for the TUI
  - [ ] `interpreter.__dict__` = a dict derived from config is how the python package should be set, and this should be from the TUI. `interpreter` should not know about the config
  - [ ] Move conversation storage out of the core and into the TUI. When we exit or error, save messages same as core currently does
- [ ] Further split TUI from core (some utils still reach across)
- [ ] Better storage of different model keys in TUI / config file. All keys, to multiple providers, should be stored in there. Easy switching
  - [ ] Automatically migrate users from old config to new config, display a message of this
- [ ] On update, check for new system message and ask user to overwrite theirs, or only let users pass in "custom instructions" which adds to our system message
  - [ ] I think we could have a config that's like... system_message_version. If system_message_version is below the current version, ask the user if we can overwrite it with the default config system message of that version. (This somewhat exists now but needs to be robust)

## What's in our scope?

This repository (`endolith/open-interpreter`) is the **Python** edition of Open Interpreter. The upstream [openinterpreter/openinterpreter](https://github.com/openinterpreter/openinterpreter) repo is a separate **Rust** rewrite (default branch `oix`).

**Branches**

- **`main`** — merge target; PRs and CI land here. Default branch and CI badge. As features are ported over from `classic/develop` (see the [Plan](#plan)), this becomes the recommended install for everyone.
- **`classic/develop`** — the maintainer's daily driver and currently the best install for most users. Features are ported to `main` as isolated PRs, not merged wholesale (see the [Plan](#plan) Phase 2).
- **`development`** — abandoned attempt at Open Interpreter 1.0 (see the [Plan](#plan) Phase 4); not maintained and not becoming `main`.
- **`develop_1.0`** — experimental continuation of `development` (the maintainer's work on it before it was abandoned); also abandoned.

Open Interpreter contains two projects which support each other, whose scopes are as follows:

1. `core`, which is dedicated to figuring out how to get LLMs to safely control a computer. Right now, this means creating a real-time code execution environment that language models can operate.
2. `terminal_interface`, a text-only way for users to direct the code-running LLM running inside `core`. This includes functions for connecting the `core` to various local and hosted LLMs (which the `core` itself should not know about).

## What's not in our scope?

Our guiding philosophy is minimalism, so we have also decided to explicitly consider the following as **out of scope**:

1. Additional functions in `core` beyond running code.
2. More complex interactions with the LLM in `terminal_interface` beyond text (but file paths to more complex inputs, like images or video, can be included in that text).

---

This roadmap gets pretty rough from here. More like working notes.

## Working Notes

## \* Roughly, how to build `computer.browser`:

First I think we should have a part, like `computer.browser.ask(query)` which just hits up [perplexity](https://www.perplexity.ai/) for fast answers to questions.

Then we want these sorts of things:

- `browser.open(url)`
- `browser.screenshot()`
- `browser.click()`

It should actually be based closely on Selenium. Copy their API so the LLM knows it.

Other than that, basically should be = to the computer module itself, at least the IO / keyboard and mouse parts.

However, for non vision models, `browser.screenshot()` can return the accessibility tree, not an image. And for `browser.click(some text)` we can use the HTML to find that text.

**Here's how GPT suggests we implement the first steps of this:**

Creating a Python script that automates the opening of Chrome with the necessary flags and then interacts with it to navigate to a URL and retrieve the accessibility tree involves a few steps. Here's a comprehensive approach:

1. **Script to Launch Chrome with Remote Debugging**:

   - This script will start Chrome with the `--remote-debugging-port=9222` flag.
   - It will handle different platforms (Windows, macOS, Linux).

2. **Python Script for Automation**:
   - This script uses `pychrome` to connect to the Chrome instance, navigate to a URL, and retrieve the accessibility tree.

### Step 1: Launching Chrome with Remote Debugging

You'll need a script to launch Chrome. This script varies based on the operating system. Below is an example for Windows. You can adapt it for macOS or Linux by changing the path and command to start Chrome.

```python
import subprocess
import sys
import os

def launch_chrome():
    chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"  # Update this path for your system
    url = "http://localhost:9222/json/version"
    subprocess.Popen([chrome_path, '--remote-debugging-port=9222'], shell=True)
    print("Chrome launched with remote debugging on port 9222.")

if __name__ == "__main__":
    launch_chrome()
```

### Step 2: Python Script to Navigate and Retrieve Accessibility Tree

Next, you'll use `pychrome` to connect to this Chrome instance. Ensure you've installed `pychrome`:

```bash
pip install pychrome
```

Here's the Python script:

```python
import pychrome
import time

def get_accessibility_tree(tab):
    # Enable the Accessibility domain
    tab.call_method("Accessibility.enable")

    # Get the accessibility tree
    tree = tab.call_method("Accessibility.getFullAXTree")
    return tree

def main():
    # Create a browser instance
    browser = pychrome.Browser(url="http://127.0.0.1:9222")

    # Create a new tab
    tab = browser.new_tab()

    # Start the tab
    tab.start()

    # Navigate to a URL
    tab.set_url("https://www.example.com")
    time.sleep(3)  # Wait for page to load

    # Retrieve the accessibility tree
    accessibility_tree = get_accessibility_tree(tab)
    print(accessibility_tree)

    # Stop the tab (closes it)
    tab.stop()

    # Close the browser
    browser.close()

if __name__ == "__main__":
    main()
```

This script will launch Chrome, connect to it, navigate to "https://www.example.com", and then print the accessibility tree to the console.

**Note**: The script to launch Chrome assumes a typical installation path on Windows. You will need to modify this path according to your Chrome installation location and operating system. Additionally, handling different operating systems requires conditional checks and respective commands for each OS.
