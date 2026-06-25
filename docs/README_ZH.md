<h1 align="center">● Open Interpreter（开放解释器）</h1>

<p align="center">
    <a href="https://discord.gg/Hvz9Axh84z">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <a href="../README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
    <a href="README_JA.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
    <a href="README_ZH.md"><img src="https://img.shields.io/badge/文档-中文版-white.svg" alt="ZH doc"/></a>
    <a href="README_ES.md"> <img src="https://img.shields.io/badge/Español-white.svg" alt="ES doc"/></a>
    <a href="README_UK.md"><img src="https://img.shields.io/badge/Українська-white.svg" alt="UK doc"/></a>
    <a href="README_IN.md"><img src="https://img.shields.io/badge/Hindi-white.svg" alt="IN doc"/></a>
    <a href="../LICENSE"><img src="https://img.shields.io/static/v1?label=license&message=AGPL&color=white&style=flat" alt="License"/></a>
    <a href="https://github.com/endolith/open-interpreter/actions/workflows/python-package.yml">
        <img alt="Build and Test" src="https://github.com/endolith/open-interpreter/actions/workflows/python-package.yml/badge.svg"/></a>
    <a href="https://codecov.io/gh/endolith/open-interpreter">
        <img alt="codecov" src="https://codecov.io/gh/endolith/open-interpreter/branch/main/graph/badge.svg"/></a>
    <br>
    <br><a href="https://www.openinterpreter.com/">桌面应用</a> | <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a> | <a href=".">文档</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Open Interpreter（开放解释器）** 可以让大语言模型在本地运行代码和 Shell 命令（Python、JavaScript、Bash、cmd、PowerShell、Ruby、R、Java 等）。安装后，在终端中运行 `interpreter`，即可通过聊天机器人界面与 Open Interpreter 交互。

本软件为计算机的通用功能提供了自然语言界面：

- 创建和编辑照片、视频、PDF 等
- 控制 Chrome 浏览器进行搜索
- 绘制、清理和分析大型数据集
- ... 等

**⚠️ 注意：默认情况下，代码运行前会要求您批准。**

## 与其他工具的比较

Open Interpreter 早于许多其他 AI 编程工具，既有相似之处，也有不同之处：

- 虽然它可以编写代码并执行 Shell 命令，类似于 [Claude Code](https://claude.ai/code)、[Cursor](https://cursor.sh)、[Devin](https://www.devin.ai) 等编程代理，但 Open Interpreter 的重点不在于通过修补源代码文件来维护项目代码库，而更多是在持久、交互式的 REPL 式会话中完成一次性任务（更接近 Jupyter 笔记本，而非 IDE）。
- 与 [OpenClaw](https://openclaw.ai/)、[Hermes Agent](https://hermes-agent.org/) 等不同，它通常以交互方式使用，而非作为自主代理。
- 它不是像 [Claude Desktop](https://claude.ai/download) 那样通过 MCP 工具与世界交互，而是直接运行代码片段或 [Shell 命令](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html)。
- 它类似于 [ShellGPT](https://github.com/ther1d/shell_gpt) 或 [cmd-ai](https://github.com/BrodaNoel/cmd-ai) 等自然语言 Shell 翻译器，但不局限于 Shell，并使用交互式聊天机器人界面，因此您可以在命令运行前审查、拒绝（`n`）或编辑（`e`），并要求模型修改。
- 网页聊天机器人中的代码解释器功能（[OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter)、[Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter)、[Grok](https://docs.x.ai/developers/tools/code-execution)、[Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution) 等）在远程沙盒环境中运行代码，该环境为闭源且受限。文件必须单独上传，结果再下载。执行的代码通常无法访问互联网，仅限于预装软件包，容器在不活动后会过期，导致进度和数据丢失。Open Interpreter 通过在本地环境中运行来克服这些限制。它可以完全访问互联网，不受时间或文件大小限制，可以使用任何软件包或库，甚至自行安装对特定任务有用的库。

## 演示

[演示视频](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### Google Colab 上也提供了交互式演示

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### 此外还有一个受 _Her_ 启发的语音界面示例

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## 快速开始

### 安装

这是由社区维护的 Open Interpreter Python 版本。

以下命令将安装 **`main`**，即默认分支（稳定基础、CI 以及移植更改的合并目标）：

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> 有关可选依赖项，请参阅我们的[设置指南](getting-started/setup.mdx)。

不过，对于日常使用，您可能更希望安装 **`classic/develop`** —— 这是每日维护和使用的不稳定分支，相比 main 分支有许多变化和功能，例如对推理模型、OpenRouter/DeepSeek/Qwen、网络搜索工具等的支持：

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

有关分支特有功能、模型说明和设置详情，请参阅 [`classic/develop` README](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md)。

### 终端

安装后，运行 `interpreter`：

```shell
interpreter
```

Open Interpreter 默认使用 OpenAI 的 **GPT-4o**，并会要求您输入 API 密钥，您可以在 [OpenAI API 密钥页面](https://platform.openai.com/api-keys) 获取。有关其他提供商或本地模型，请参阅下文。

### Python

```python
from interpreter import interpreter

interpreter.chat("绘制 AAPL 和 META 的标准化股价") # 执行单一命令
interpreter.chat() # 开始交互式聊天
```

### GitHub Codespaces

在此仓库的 GitHub 页面上按下 <kbd>,</kbd> 键即可创建 codespace。片刻之后，您将获得一个预装 Open Interpreter 的云虚拟机环境。然后您可以直接与其交互，并自由确认其系统命令的执行，而无需担心损坏系统。

## 命令

### 交互式聊天

要在终端中开始交互式聊天，从命令行运行 `interpreter`：

```shell
interpreter
```

或者从 .py 文件中运行 `interpreter.chat()`：

```python
interpreter.chat()
```

**您还可以流式传输每个数据块：**

```python
message = "我们使用的是什么操作系统？"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### 程序化聊天

为了更精确的控制，您可以通过 `.chat(message)` 直接传递消息：

```python
interpreter.chat("为 /videos 中的所有视频添加字幕。")

# ... 将输出流式传输到终端，完成任务 ...

interpreter.chat("这些看起来不错，但你能把字幕调大一些吗？")

# ...
```

### 开始新的聊天

在 Python 中，Open Interpreter 会记录对话历史。如果您想从头开始，可以重置它：

```python
interpreter.messages = []
```

### 保存和恢复聊天

`interpreter.chat()` 返回消息列表，可用于通过 `interpreter.messages = messages` 恢复对话：

```python
messages = interpreter.chat("My name is Killian.") # 将消息保存到 'messages'
interpreter.messages = [] # 重置解释器（"Killian" 将被遗忘）

interpreter.messages = messages # 从 'messages' 恢复聊天（"Killian" 将被记住）
```

### 自定义系统消息

您可以检查并配置 Open Interpreter 的系统消息，以扩展其功能、修改权限或赋予其更多上下文。

```python
interpreter.system_message += """
使用 -y 运行 shell 命令，这样用户就不必确认它们。
"""
print(interpreter.system_message)
```

### 更改语言模型

Open Interpreter 使用 [LiteLLM](https://docs.litellm.ai/docs/providers/) 连接到托管的语言模型。

您可以通过设置 model 参数来更改模型：

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

在 Python 中，在对象上设置模型：

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[在此查找适合您语言模型的 model 字符串。](https://docs.litellm.ai/docs/providers/)

### 在本地运行 Open Interpreter

#### 终端

Open Interpreter 可以使用 OpenAI 兼容服务器在本地运行模型（LM Studio、Jan.ai、Ollama 等）。

只需使用推理服务器的 `api_base` URL 运行 `interpreter`（LM Studio 默认为 `http://localhost:1234/v1`）：

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

或者，您可以通过运行以下命令使用 Llamafile，而无需安装任何第三方软件：

```shell
interpreter --local
```

有关更详细的指南，请参阅 [Mike Bird 的这段视频](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H)

**如何在后台运行 LM Studio。**

1. 下载 [LM Studio](https://lmstudio.ai/) 然后启动它。
2. 选择一个模型，然后点击 **↓ Download**。
3. 点击左侧的 **↔️** 按钮（在 💬 下方）。
4. 在顶部选择您的模型，然后点击 **Start Server**。

服务器运行后，您就可以开始与 Open Interpreter 对话了。

> **注意：** 本地模式将 `context_window` 设置为 3000，`max_tokens` 设置为 1000。如果您的模型有不同的要求，请手动设置这些参数（见下文）。

#### Python

我们的 Python 包让您可以更好地控制每项设置。要复制并连接到 LM Studio，请使用以下设置：

```python
from interpreter import interpreter

interpreter.offline = True # 禁用在线功能（例如更新检查、遥测）
interpreter.llm.model = "openai/x" # 告诉 OI 以 OpenAI 格式发送消息
interpreter.llm.api_key = "fake_key" # LiteLLM（我们用于与 LM Studio 通信）需要此参数
interpreter.llm.api_base = "http://localhost:1234/v1" # 指向任何 OpenAI 兼容服务器

interpreter.chat()
```

#### 上下文窗口、最大 Token 数

您可以修改本地运行模型的 `max_tokens` 和 `context_window`（以 token 为单位）。

对于本地模式，较小的上下文窗口会使用更少的 RAM，因此如果失败或运行缓慢，我们建议尝试更短的窗口（约 1000）。请确保 `max_tokens` 小于 `context_window`。

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### 详细模式

为帮助您检查 Open Interpreter，我们提供了用于调试的 `--verbose` 模式。

您可以使用标志（`interpreter --verbose`）激活详细模式，或在聊天过程中：

```shell
$ interpreter
...
> %verbose true <- 开启详细模式

> %verbose false <- 关闭详细模式
```

### 交互模式命令

在交互模式下，您可以使用以下命令来增强体验。以下是可用命令列表：

**可用命令：**

- `%% [command]`：在系统 Shell 中运行命令（绕过 LLM）。
- `%verbose [true/false]`：切换详细模式。无参数或使用 `true` 时进入详细模式。使用 `false` 时退出详细模式。
- `%auto_run [true/false]`：切换代码是否无需确认即可运行。无参数或使用 `true` 时进入 auto_run 模式。使用 `false` 时退出 auto_run 模式。
- `%reset`：重置当前会话的对话。
- `%undo`：从消息历史中移除上一条用户消息和 AI 的回复。
- `%save_message [path]`：将消息保存到指定的 JSON 路径。如果未提供路径，默认为 'messages.json'。
- `%load_message [path]`：从指定的 JSON 路径加载消息。如果未提供路径，默认为 'messages.json'。
- `%tokens [prompt]`：（_实验性_）计算将作为上下文随下一条提示发送的 token 数并估算其成本。如果提供了 `prompt`，还可选计算该提示的 token 数和估算成本。估算成本依赖 [LiteLLM 的 `cost_per_token()` 方法](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token)。
- `%jupyter`：将对话导出为 Jupyter 笔记本文件。
- `%markdown [path]`：将对话导出到指定的 Markdown 路径。如果未提供路径，将保存到 Downloads 文件夹并使用生成的对话名称。
- `%info`：显示系统和解释器信息。
- `%help`：显示帮助消息。

### 配置 / 配置文件

Open Interpreter 允许您使用 `yaml` 文件设置默认行为。

这提供了一种灵活的配置方式，无需每次都更改命令行参数。

运行以下命令打开配置文件目录：

```
interpreter --profiles
```

您可以在那里添加 `yaml` 文件。默认配置文件名为 `default.yaml`。

#### 多个配置文件

Open Interpreter 支持多个 `yaml` 文件，让您可以轻松切换配置：

```
interpreter --profile my_profile.yaml
```

## 示例 FastAPI 服务器

Open Interpreter 可以通过 HTTP REST 端点进行控制：

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

您还可以通过运行 `interpreter --server` 启动带有 WebSocket 支持和 Web UI 的内置服务器（需要 `[server]` 额外依赖）。

## Android

在 Android 设备上安装 Open Interpreter 的分步指南可在 [open-interpreter-termux 仓库](https://github.com/MikeBirdTech/open-interpreter-termux) 中找到。

## 安全提示

由于生成的代码在本地环境中执行，它可以与您的文件和系统设置交互，可能导致数据丢失或安全风险等意外结果。

**⚠️ Open Interpreter 会在执行代码前要求用户确认。**

您可以运行 `interpreter -y` 或设置 `interpreter.auto_run = True` 来绕过此确认，在这种情况下：

- 在请求修改文件或系统设置的命令时要谨慎。
- 像关注自动驾驶汽车一样关注 Open Interpreter，并随时准备通过关闭终端来结束进程。
- 考虑在 Google Colab 或 Replit 等受限环境中运行 Open Interpreter。这些环境更加隔离，降低了执行任意代码的风险。

对 [安全模式](SAFE_MODE.md) 有**实验性**支持，有助于降低某些风险。

## 它是如何工作的？

Open Interpreter 为[函数调用语言模型](https://platform.openai.com/docs/guides/function-calling)配备了 `execute` 工具，该工具接受 `language`（如 "Python" 或 "JavaScript"）和要运行的 `code`。（不支持函数调用的模型也可通过 Markdown 代码块使用。）

然后，我们将模型的消息、代码和您系统的输出以 Markdown 形式流式传输到终端。

## 离线访问文档

完整的[文档](.)可在没有互联网连接的情况下随时访问。

[Node](https://nodejs.org/en) 是前提条件：

- 版本 18.17.0 或任何更新的 18.x.x 版本。
- 版本 20.3.0 或任何更新的 20.x.x 版本。
- 任何从 21.0.0 开始的版本，无上限。

安装 [Mintlify](https://mintlify.com/)：

```bash
npm i -g mintlify@latest
```

进入 docs 目录并运行相应命令：

```bash
# 假设您位于项目根目录
cd ./docs

# 运行文档服务器
mintlify dev
```

应该会打开一个新的浏览器窗口。只要文档服务器在运行，文档将在 [http://localhost:3000](http://localhost:3000) 上可用。

## 贡献

感谢您对贡献的兴趣！我们欢迎社区的参与。

请参阅我们的[贡献准则](CONTRIBUTING.md)，了解如何参与的更多详情。

## 路线图

访问[我们的路线图](ROADMAP.md)以预览 Open Interpreter 的未来。

**请注意**：此软件与 OpenAI 无关。

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> 拥有一位以指尖速度工作的初级程序员 ... 可以让新的工作流程变得轻松高效，并将编程的好处带给新的受众。
>
> — _OpenAI 的 Code Interpreter 发布_

<br>
