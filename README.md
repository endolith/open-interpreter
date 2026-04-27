# This is Endolith's fork of Open Interpreter

Since upstream maintenance has been very slow, I've just been piling up vibe-coded changes (of dubious quality) in my `classic/develop` branch, which I've now made the default. I use this branch very frequently, and it works pretty well. While Open Interpreter Classic is old and has its flaws, I haven't found another tool that really does the same things.

Major changes vs the upstream branch (`classic/main`):

- **computer → toolbox**: Renamed in the hope that the LLMs understand how to use it better. Breaking change, renamed entire API.
  - First line of docstring and Returns are always shown to LLM in system message to hint at how to use tools
  - As well as encouragement to use `help()` to RTFM (which they rarely do)
- **Web search tools** for the AI to find info online:
  - Multiple backends with fallbacks: LinkUp, Tavily, SerpApi, Brave, Serper
  - `search`: list of search results
  - `answer`: AI-synthesized answers with citations
  - `structured_output`: JSON schema extraction via LinkUp
  - `fetch`: web page as markdown
  - Result classes with compact repr to avoid flooding the context window
  - Automatic locale detection for country/language-specific results
- **ai2 module**: for OI to delegate tasks in loops: `boolean_query`, `choice_query`, `single_response`
  - They don't use this on their own, you have to tell them.
- Better LLM API stuff:
  - **Reasoning models support**: `reasoning_content` streaming, cyan "Thinking" panels, `include_reasoning`/`reasoning_effort` params on `interpreter.llm`, OpenRouter extra_body reasoning, DeepSeek V4 thinking-mode compatibility
  - **OpenRouter support**: `--model openrouter/openai/gpt-4.1-mini ` with `OPENROUTER_API_KEY` env var set
  - **DashScope integration**: Qwen models, vision enabled for newer models. `DASHSCOPE_API_KEY` env var. Use `dashscope-us/...` or `dashscope-intl/...` to pick the US vs Singapore OpenAI-compatible endpoint.
  - **Mistral compatibility**: tool ID length, image role mapping
  - **API error handling**: Error presented in a panel with rendering of markdown/HTML, retry prompts, auto-retry on temporary provider errors, clean exits
  - **Usage tracking**: `%usage` command with token stats
- **Profile validation**: warns about invalid configuration attributes
- **Edit commands before running**: temp file editing with `$EDITOR` env var support
- **`view_image` tool**: Allow vision-capable LLMs to request to view image files; includes image resize/shrink prompts for large files
- **Incremental markdown rendering**: raw streaming / rendering of markdown blocks one at a time to avoid screen flickering
- **Better terminal size detection:** You can resize window without breaking all subsequent text formatting
- **Python REPL state output**: shows variables, modules, CWD for the LLM's context, alerts when REPL was restarted
- **Conversation improvements**:
  - User message timestamps so it knows the date and how much time has passed
  - "New Conversation" menu option in `--conversations` navigator (in case you change your mind)
  - Atomic file saving (corruption-resistant)
- **Secret redaction**: Try to avoid sending passwords and secrets environment variables to the LLM's server
- **System message enhancements**: rough geolocation, encourage REPL-like coding, etc.
- **Windows support**: Downloads folder detection via `SHGetKnownFolderPath`, UTF-8 code page for shell, `bat` syntax highlighting
- **HTML output suppression**: prevents browser from opening unnecessarily
- **TextFileReader**: convenience class with encoding auto-detect. (The LLMs never actually use this; I don't know if it works.)
- **Python 3.13 support?** and a bunch of fixed tests in the process. Dubious.

## Models I've been using:

- `dashscope-us/qwen3.5-plus` == `dashscope-us/qwen3.5-397b-a17b`: Good and cheap and has vision, but no input token caching, so more expensive than it should be.
- `deepseek/deepseek-v4-flash`: New favorite for text?  Cached and cheap, calls tools well, actually uses python as a REPL instead of writing a brand new script each time.  No vision.
- `openrouter/xiaomi/mimo-v2-pro`: Pretty good. No vision.
- `openrouter/xiaomi/mimo-v2-flash`: Surprisingly good for how cheap it is. Fast. No vision.
- `openrouter/openai/gpt-4.1-mini`: Better than `gpt-5-mini`.
- `openrouter/minimax/minimax-m2.5`: Seems ok. Table formatted wrong. Pretty autonomous. `dir` recursively by default
- `openrouter/xiaomi/mimo-v2-omni`: OK?  Vision.  Prints thoughts and asks for the same image more than once and other weird things.

(Note that lack of caching of input tokens affects the cost a lot since conversations get very long and it doesn't have `responses` API (yet).)

## Not as good:

- `dashscope-us/qwen3.6-plus`: Misusing commands, trying to import a module it knows isn't installed, hallucinating arguments
- `dashscope-us/qwen3.5-122b-a10b`: Incorrect tool calls
- `deepseek/deepseek-v3.2-exp` ≈ `deepseek/deepseek-v3.2`: Incorrect tool calls, kinda slow, dumb
- `google/gemini-3-flash` ≈ `gemini-3.1-flash-lite-preview`: Surprisingly bad at using tools
- `openrouter/minimax/minimax-m2.7` : Hallucinating URLs and not using convenience functions, asking dumb questions
- `google/gemma-4-31b-it`: Slow, yappy, markdown formatting issues
- `openrouter/openrouter/auto`: "temporarily rate-limited upstream" over and over

----

# ● Open Interpreter

<p align="center">
    <a href="https://discord.gg/Hvz9Axh84z">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <a href="docs/README_JA.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
    <a href="docs/README_ZH.md"><img src="https://img.shields.io/badge/文档-中文版-white.svg" alt="ZH doc"/></a>
    <a href="docs/README_ES.md"> <img src="https://img.shields.io/badge/Español-white.svg" alt="ES doc"/></a>
    <a href="docs/README_UK.md"><img src="https://img.shields.io/badge/Українська-white.svg" alt="UK doc"/></a>
    <a href="docs/README_IN.md"><img src="https://img.shields.io/badge/Hindi-white.svg" alt="IN doc"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/static/v1?label=license&message=AGPL&color=white&style=flat" alt="License"/></a>
    <br>
    <br><a href="https://0ggfznkwh4j.typeform.com/to/G21i9lJ2">Get early access to the desktop app</a>‎ ‎ |‎ ‎ <a href="https://docs.openinterpreter.com/">Documentation</a><br>
</p>
<img alt="local_explorer" src="https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a">

**Open Interpreter** lets LLMs run code (Python, Javascript, Shell, and more) locally. You can chat with Open Interpreter through a ChatGPT-like interface in your terminal by running `$ interpreter` after installing.

This provides a natural-language interface to your computer's general-purpose capabilities:

- Create and edit photos, videos, PDFs, etc.
- Control a Chrome browser to perform research
- Plot, clean, and analyze large datasets
- ...etc.

**⚠️ Note: You'll be asked to approve code before it's run.**

## Demo

[Demo video](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### An interactive demo is also available on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### Along with an example voice interface, inspired by _Her_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## Quick Start

### Install

This repository is Endolith's fork of the classic Python Open Interpreter.

This command will install **`main`** which is the default branch (stable base, CI, and merge target for ported changes):

```shell
pip install git+https://github.com/OpenInterpreter/open-interpreter.git
```

> See our [setup guide](https://docs.openinterpreter.com/getting-started/setup) for optional dependencies.

For day-to-day use, however, you probably want to install **`classic/develop`** instead — that's the unstable branch maintained and used daily, with many changes and features vs the main branch, such as support for reasoning models, OpenRouter/DeepSeek/Qwen, web search tools, etc.:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

For fork-specific features, model notes, and setup details, see the [`classic/develop` README](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md).

### Terminal

After installation, simply run `interpreter`:

```shell
interpreter
```

Open Interpreter will default to OpenAI's **GPT-4o** and will ask you to enter a key, which you can obtain from [OpenAI's API keys page](https://platform.openai.com/api-keys). For other providers or local models, see below.

### Python

```python
from interpreter import interpreter

interpreter.chat("Plot AAPL and META's normalized stock prices") # Executes a single command
interpreter.chat() # Starts an interactive chat
```

### GitHub Codespaces

Press the <kbd>,</kbd> key on this repository's GitHub page to create a codespace. After a moment, you'll receive a cloud virtual machine environment pre-installed with open-interpreter. You can then start interacting with it directly and freely confirm its execution of system commands without worrying about damaging the system.

## Comparison to ChatGPT's Code Interpreter

OpenAI's release of [Code Interpreter](https://openai.com/blog/chatgpt-plugins#code-interpreter) with GPT-4 presents a fantastic opportunity to accomplish real-world tasks with ChatGPT.

However, OpenAI's service is hosted, closed-source, and heavily restricted:

- No internet access.
- [Limited set of pre-installed packages](https://wfhbrian.com/artificial-intelligence/mastering-chatgpts-code-interpreter-list-of-python-packages/).
- 100 MB maximum upload, 120.0 second runtime limit.
- State is cleared (along with any generated files or links) when the environment dies.

---

Open Interpreter overcomes these limitations by running in your local environment. It has full access to the internet, isn't restricted by time or file size, and can utilize any package or library.

This combines the power of GPT-4's Code Interpreter with the flexibility of your local development environment.

## Commands

### Interactive Chat

To start an interactive chat in your terminal, either run `interpreter` from the command line:

```shell
interpreter
```

Or `interpreter.chat()` from a .py file:

```python
interpreter.chat()
```

**You can also stream each chunk:**

```python
message = "What operating system are we on?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Programmatic Chat

For more precise control, you can pass messages directly to `.chat(message)`:

```python
interpreter.chat("Add subtitles to all videos in /videos.")

# ... Streams output to your terminal, completes task ...

interpreter.chat("These look great but can you make the subtitles bigger?")

# ...
```

### Start a New Chat

In Python, Open Interpreter remembers conversation history. If you want to start fresh, you can reset it:

```python
interpreter.messages = []
```

### Save and Restore Chats

`interpreter.chat()` returns a List of messages, which can be used to resume a conversation with `interpreter.messages = messages`:

```python
messages = interpreter.chat("My name is Killian.") # Save messages to 'messages'
interpreter.messages = [] # Reset interpreter ("Killian" will be forgotten)

interpreter.messages = messages # Resume chat from 'messages' ("Killian" will be remembered)
```

### Customize System Message

You can inspect and configure Open Interpreter's system message to extend its functionality, modify permissions, or give it more context.

```python
interpreter.system_message += """
Run shell commands with -y so the user doesn't have to confirm them.
"""
print(interpreter.system_message)
```

### Change your Language Model

Open Interpreter uses [LiteLLM](https://docs.litellm.ai/docs/providers/) to connect to hosted language models.

You can change the model by setting the model parameter:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

In Python, set the model on the object:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[Find the appropriate "model" string for your language model here.](https://docs.litellm.ai/docs/providers/)

### Running Open Interpreter locally

#### Running with Local Models

Open Interpreter can use OpenAI-compatible server to run models locally (in LM Studio, Jan.ai, Ollama, etc.)

Simply run `interpreter` with the `api_base` URL of your inference server (for LM Studio it is `http://localhost:1234/v1` by default):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

Alternatively you can use Llamafile without installing any third party software just by running

```shell
interpreter --local
```

for a more detailed guide check out [this video by Mike Bird](https://www.youtube.com/watch?v=CEs51hGWuGU)

**How to run LM Studio in the background.**

1. Download [LM Studio](https://lmstudio.ai/) then start it.
2. Select a model then click **↓ Download**.
3. Click the **↔️** button on the left (below 💬).
4. Select your model at the top, then click **Start Server**.

Once the server is running, you can begin your conversation with Open Interpreter.

> **Note:** Local mode sets your `context_window` to 3000, and your `max_tokens` to 1000. If your model has different requirements, set these parameters manually (see below).

#### Python

Our Python package gives you more control over each setting. To replicate and connect to LM Studio, use these settings:

```python
from interpreter import interpreter

interpreter.offline = True # Disables online features like Open Procedures
interpreter.llm.model = "openai/x" # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = "fake_key" # LiteLLM, which we use to talk to LM Studio, requires this
interpreter.llm.api_base = "http://localhost:1234/v1" # Point this at any OpenAI compatible server

interpreter.chat()
```

#### Context Window, Max Tokens

You can modify the `max_tokens` and `context_window` (in tokens) of locally running models.

For local mode, smaller context windows will use less RAM, so we recommend trying a much shorter window (~1000) if it's failing / if it's slow. Make sure `max_tokens` is less than `context_window`.

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Verbose mode

To help you inspect Open Interpreter we have a `--verbose` mode for debugging.

You can activate verbose mode by using its flag (`interpreter --verbose`), or mid-chat:

```shell
$ interpreter
...
> %verbose true <- Turns on verbose mode

> %verbose false <- Turns off verbose mode
```

### Interactive Mode Commands

In the interactive mode, you can use the below commands to enhance your experience. Here's a list of available commands:

**Available Commands:**

- `%verbose [true/false]`: Toggle verbose mode. Without arguments or with `true` it
  enters verbose mode. With `false` it exits verbose mode.
- `%reset`: Resets the current session's conversation.
- `%undo`: Removes the previous user message and the AI's response from the message history.
- `%tokens [prompt]`: (_Experimental_) Calculate the tokens that will be sent with the next prompt as context and estimate their cost. Optionally calculate the tokens and estimated cost of a `prompt` if one is provided. Relies on [LiteLLM's `cost_per_token()` method](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) for estimated costs.
- `%help`: Show the help message.

### Configuration / Profiles

Open Interpreter allows you to set default behaviors using `yaml` files.

This provides a flexible way to configure the interpreter without changing command-line arguments every time.

Run the following command to open the profiles directory:

```shell
interpreter --profiles
```

You can add `yaml` files there. The default profile is named `default.yaml`.

#### Multiple Profiles

Open Interpreter supports multiple `yaml` files, allowing you to easily switch between configurations:

```shell
interpreter --profile my_profile.yaml
```

## Sample FastAPI Server

The generator update enables Open Interpreter to be controlled via HTTP REST endpoints:

```python
# server.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from interpreter import interpreter

app = FastAPI()

@app.get("/chat")
def chat_endpoint(message: str):
    def event_stream():
        for result in interpreter.chat(message, stream=True):
            yield f"data: {result}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/history")
def history_endpoint():
    return interpreter.messages
```

```shell
pip install fastapi uvicorn
uvicorn server:app --reload
```

You can also start a server identical to the one above by simply running `interpreter.server()`.

## Android

The step-by-step guide for installing Open Interpreter on your Android device can be found in the [open-interpreter-termux repo](https://github.com/MikeBirdTech/open-interpreter-termux).

## Safety Notice

Since generated code is executed in your local environment, it can interact with your files and system settings, potentially leading to unexpected outcomes like data loss or security risks.

**⚠️ Open Interpreter will ask for user confirmation before executing code.**

You can run `interpreter -y` or set `interpreter.auto_run = True` to bypass this confirmation, in which case:

- Be cautious when requesting commands that modify files or system settings.
- Watch Open Interpreter like a self-driving car, and be prepared to end the process by closing your terminal.
- Consider running Open Interpreter in a restricted environment like Google Colab or Replit. These environments are more isolated, reducing the risks of executing arbitrary code.

When using a **remote API model**, message text (including code output) is sanitized with [bc-detect-secrets](https://github.com/bridgecrewio/detect-secrets) so API keys and passwords are redacted before being sent to the API. Use `--sanitize_secrets off` to disable, or `--sanitize_secrets on` to enable for local models too. See [All Settings — Sanitize secrets](https://docs.openinterpreter.com/settings/all-settings#sanitize-secrets-api-keys-passwords).

There is **experimental** support for a [safe mode](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/SAFE_MODE.md) to help mitigate some risks.

## How Does it Work?

Open Interpreter equips a [function-calling language model](https://platform.openai.com/docs/guides/function-calling) with an `exec()` function, which accepts a `language` (like "Python" or "JavaScript") and `code` to run.

We then stream the model's messages, code, and your system's outputs to the terminal as Markdown.

## Access Documentation Offline

The full [documentation](https://docs.openinterpreter.com/) is accessible on-the-go without the need for an internet connection.

[Node](https://nodejs.org/en) is a pre-requisite:

- Version 18.17.0 or any later 18.x.x version.
- Version 20.3.0 or any later 20.x.x version.
- Any version starting from 21.0.0 onwards, with no upper limit specified.

Install [Mintlify](https://mintlify.com/):

```bash
npm i -g mintlify@latest
```

Change into the docs directory and run the appropriate command:

```bash
# Assuming you're at the project's root directory
cd ./docs

# Run the documentation server
mintlify dev
```

A new browser window should open. The documentation will be available at [http://localhost:3000](http://localhost:3000) as long as the documentation server is running.

## Contributing

Thank you for your interest in contributing! We welcome involvement from the community.

Please see our [contributing guidelines](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/CONTRIBUTING.md) for more details on how to get involved.

## Roadmap

Visit [our roadmap](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/ROADMAP.md) to preview the future of Open Interpreter.

**Note**: This software is not affiliated with OpenAI.

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> Having access to a junior programmer working at the speed of your fingertips ... can make new workflows effortless and efficient, as well as open the benefits of programming to new audiences.
>
> — _OpenAI's Code Interpreter Release_
