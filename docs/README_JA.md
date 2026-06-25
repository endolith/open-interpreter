<h1 align="center">● Open Interpreter</h1>

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
    <br><a href="https://www.openinterpreter.com/">デスクトップアプリ</a> | <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a> | <a href=".">ドキュメント</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Open Interpreter** は、LLM がコードやシェルコマンドをローカルで実行できるようにします（Python、JavaScript、Bash、cmd、PowerShell、Ruby、R、Java など）。インストール後、`interpreter` を実行すると、ターミナル上のチャットボットインターフェースを通じて Open Interpreter と対話できます。

これにより、コンピュータの汎用機能を自然言語インターフェースで操作できます:

- 写真、動画、PDF などの作成や編集
- Chrome ブラウザを制御してリサーチを行う
- 大規模なデータセットのプロット、クリーニング、分析
- ...など

**⚠️ 注意: デフォルトでは、コードを実行する前に承認を求められます。**

## 他のツールとの比較

Open Interpreter は他の多くの AI コーディングツールよりも先に登場しており、類似点と相違点があります:

- コードを書いたりシェルコマンドを実行したりできる点では、[Claude Code](https://claude.ai/code)、[Cursor](https://cursor.sh)、[Devin](https://www.devin.ai) などのコーディングエージェントと似ていますが、Open Interpreter はソースコードファイルをパッチしてプロジェクトのコードベースを維持することよりも、永続的で対話型の REPL 的なセッションで単発タスクを完了することに重点を置いています（IDE より Jupyter ノートブックに近い）。
- [OpenClaw](https://openclaw.ai/)、[Hermes Agent](https://hermes-agent.org/) などとは異なり、通常は自律型エージェントではなく、対話的に使用されます。
- [Claude Desktop](https://claude.ai/download) のように MCP ツールを通じて世界とやり取りするのではなく、コードスニペットや[シェルコマンドを直接実行](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html)します。
- [ShellGPT](https://github.com/ther1d/shell_gpt) や [cmd-ai](https://github.com/BrodaNoel/cmd-ai) などの自然言語シェル翻訳ツールに似ていますが、シェルに限定されず、対話型チャットボットインターフェースを使用するため、実行前にコマンドを確認・拒否（`n`）・編集（`e`）でき、モデルに修正を依頼することもできます。
- Web チャットボットの Code Interpreter 機能（[OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter)、[Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter)、[Grok](https://docs.x.ai/developers/tools/code-execution)、[Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution) など）は、クローズドソースで制限のあるリモートのサンドボックス環境でコードを実行します。ファイルは個別にアップロードし、結果をダウンロードする必要があります。実行されたコードは一般にインターネットにアクセスできず、プリインストールされたパッケージのセットに限定され、コンテナは非アクティブ後に期限切れとなり、進捗やデータが失われます。Open Interpreter はローカル環境で実行することでこれらの制限を克服します。インターネットにフルアクセスでき、時間やファイルサイズの制限を受けず、任意のパッケージやライブラリを使用でき、タスクに有用なライブラリを自らインストールすることもできます。

## デモ

[デモ動画](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### Google Colab でも対話形式のデモを利用できます

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### _Her_ にインスパイアされた音声インターフェースの例もあります

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## クイックスタート

### インストール

これはコミュニティがメンテナンスする Open Interpreter の Python 版です。

次のコマンドは、デフォルトブランチである **`main`**（安定版のベース、CI、移植変更のマージ先）をインストールします:

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> オプションの依存関係については、[セットアップガイド](getting-started/setup.mdx)を参照してください。

ただし、日常的な使用では、代わりに **`classic/develop`** をインストールすることをお勧めします。これは毎日メンテナンスされ使用されている不安定なブランチで、推論モデル、OpenRouter/DeepSeek/Qwen、ウェブ検索ツールなど、main ブランチと比べて多くの変更と機能があります:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

フォーク固有の機能、モデルに関する注記、セットアップの詳細については、[`classic/develop` README](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md)を参照してください。

### ターミナル

インストール後、`interpreter` を実行するだけです:

```shell
interpreter
```

Open Interpreter はデフォルトで OpenAI の **GPT-4o** を使用し、API キーの入力を求めます。キーは [OpenAI の API キーページ](https://platform.openai.com/api-keys)から取得できます。他のプロバイダーやローカルモデルについては、下記を参照してください。

### Python

```python
from interpreter import interpreter

interpreter.chat("Plot AAPL and META's normalized stock prices") # 単一コマンドを実行
interpreter.chat() # 対話型チャットを開始
```

### GitHub Codespaces

このリポジトリの GitHub ページで <kbd>,</kbd> キーを押すと、Codespace を作成できます。しばらくすると、Open Interpreter がプリインストールされたクラウド仮想マシン環境が用意されます。システムを損傷する心配なく、直接対話し、システムコマンドの実行を自由に確認できます。

## コマンド

### 対話型チャット

ターミナルで対話型チャットを開始するには、コマンドラインから `interpreter` を実行します:

```shell
interpreter
```

または、.py ファイルから `interpreter.chat()` を実行します:

```python
interpreter.chat()
```

**各チャンクをストリーミングすることもできます:**

```python
message = "What operating system are we on?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### プログラム的なチャット

より精確な制御のために、メッセージを直接 `.chat(message)` に渡すことができます:

```python
interpreter.chat("Add subtitles to all videos in /videos.")

# ... ターミナルに出力をストリームし、タスクを完了 ...

interpreter.chat("These look great but can you make the subtitles bigger?")

# ...
```

### 新しいチャットを開始

Python では、Open Interpreter は会話履歴を記憶します。最初からやり直したい場合は、リセットできます:

```python
interpreter.messages = []
```

### チャットの保存と復元

`interpreter.chat()` はメッセージのリストを返し、`interpreter.messages = messages` で会話を再開できます:

```python
messages = interpreter.chat("My name is Killian.") # 'messages' にメッセージを保存
interpreter.messages = [] # インタープリタをリセット（"Killian" は忘れられる）

interpreter.messages = messages # 'messages' からチャットを再開（"Killian" は記憶される）
```

### システムメッセージのカスタマイズ

Open Interpreter のシステムメッセージを確認・設定することで、機能を拡張したり、権限を変更したり、より多くのコンテキストを与えたりできます。

```python
interpreter.system_message += """
Run shell commands with -y so the user doesn't have to confirm them.
"""
print(interpreter.system_message)
```

### 言語モデルの変更

Open Interpreter は、ホストされた言語モデルへの接続に [LiteLLM](https://docs.litellm.ai/docs/providers/) を使用しています。

model パラメータを設定することで、モデルを変更できます:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

Python では、オブジェクト上でモデルを設定します:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[言語モデルに適した "model" 文字列はこちらで検索してください。](https://docs.litellm.ai/docs/providers/)

### Open Interpreter をローカルで実行する

#### ターミナル

Open Interpreter は OpenAI 互換サーバーを使用してモデルをローカルで実行できます（LM Studio、Jan.ai、Ollama など）。

推論サーバーの `api_base` URL を指定して `interpreter` を実行するだけです（LM Studio の場合、デフォルトは `http://localhost:1234/v1`）:

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

または、サードパーティのソフトウェアをインストールせずに、次を実行するだけで Llamafile を使用できます:

```shell
interpreter --local
```

より詳細なガイドについては、[Mike Bird によるこの動画](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H)をご覧ください。

**LM Studio をバックグラウンドで実行する方法**

1. [LM Studio](https://lmstudio.ai/) をダウンロードして起動します。
2. モデルを選択し、**↓ Download** をクリックします。
3. 左側の **↔️** ボタン（💬 の下）をクリックします。
4. 上部でモデルを選択し、**Start Server** をクリックします。

サーバーが稼働したら、Open Interpreter との会話を開始できます。

> **注意:** ローカルモードでは、`context_window` を 3000、`max_tokens` を 1000 に設定します。モデルによって異なる要件がある場合、これらのパラメータを手動で設定してください（下記参照）。

#### Python

Python パッケージでは、各設定をより細かく制御できます。LM Studio に接続するには、次の設定を使用します:

```python
from interpreter import interpreter

interpreter.offline = True # オンライン機能を無効化（例: 更新チェック、テレメトリ）
interpreter.llm.model = "openai/x" # OI に OpenAI 形式でメッセージを送信させる
interpreter.llm.api_key = "fake_key" # LM Studio との通信に使用する LiteLLM で必要
interpreter.llm.api_base = "http://localhost:1234/v1" # OpenAI 互換サーバーの URL を指定

interpreter.chat()
```

#### コンテキストウィンドウ、最大トークン数

ローカルで実行しているモデルの `max_tokens` と `context_window`（トークン単位）を変更できます。

ローカルモードでは、小さいコンテキストウィンドウは RAM を少なく使用するため、失敗する場合や遅い場合は、より短いウィンドウ（〜1000）を試すことをお勧めします。`max_tokens` が `context_window` より小さいことを確認してください。

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### 詳細モード

Open Interpreter を調査するために、`--verbose` モードを用意しています。

フラグ（`interpreter --verbose`）で有効にするか、チャット中に切り替えられます:

```shell
$ interpreter
...
> %verbose true <- 詳細モードをオン

> %verbose false <- 詳細モードをオフ
```

### 対話モードのコマンド

対話モードでは、以下のコマンドで操作を便利にできます。利用可能なコマンドの一覧:

**利用可能なコマンド:**

- `%% [command]`: システムシェルでコマンドを実行（LLM をバイパス）。
- `%verbose [true/false]`: 詳細モードの切り替え。引数なしまたは `true` で詳細モードに入り、`false` で終了します。
- `%auto_run [true/false]`: 確認なしでコードを実行するかどうかの切り替え。引数なしまたは `true` で auto_run モードに入り、`false` で終了します。
- `%reset`: 現在のセッションの会話をリセットします。
- `%undo`: メッセージ履歴から前のユーザーメッセージと AI の応答を削除します。
- `%save_message [path]`: メッセージを指定した JSON パスに保存します。パスが指定されていない場合、デフォルトは `messages.json` です。
- `%load_message [path]`: 指定した JSON パスからメッセージを読み込みます。パスが指定されていない場合、デフォルトは `messages.json` です。
- `%tokens [prompt]`: （_実験的_）次のプロンプトのコンテキストとして送信されるトークンを計算し、コストを見積もります。`prompt` が指定された場合は、そのトークン数と見積もりコストも計算します。見積もりコストは [LiteLLM の `cost_per_token()` メソッド](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token)に依存します。
- `%jupyter`: 会話を Jupyter ノートブックファイルにエクスポートします。
- `%markdown [path]`: 会話を指定した Markdown パスにエクスポートします。パスが指定されていない場合、生成された会話名で Downloads フォルダに保存されます。
- `%info`: システムとインタープリタの情報を表示します。
- `%help`: ヘルプメッセージを表示します。

### 設定 / プロファイル

Open Interpreter では、`yaml` ファイルを使用してデフォルトの動作を設定できます。

これにより、毎回コマンドライン引数を変更することなく、柔軟にインタープリタを設定できます。

次のコマンドを実行してプロファイルディレクトリを開きます:

```
interpreter --profiles
```

そこに `yaml` ファイルを追加できます。デフォルトのプロファイル名は `default.yaml` です。

#### 複数プロファイル

Open Interpreter は複数の `yaml` ファイルをサポートしており、設定を簡単に切り替えられます:

```
interpreter --profile my_profile.yaml
```

## FastAPI サーバーのサンプル

Open Interpreter は HTTP REST エンドポイント経由で制御できます:

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

WebSocket サポートと Web UI を備えた組み込みサーバーは、`interpreter --server` で起動できます（`[server]` エクストラが必要です）。

## Android

Android デバイスへの Open Interpreter のインストール手順は、[open-interpreter-termux リポジトリ](https://github.com/MikeBirdTech/open-interpreter-termux)に記載されています。

## 安全に関する注意

生成されたコードはローカル環境で実行されるため、ファイルやシステム設定と相互作用する可能性があり、データ損失やセキュリティリスクなど予期せぬ結果につながる可能性があります。

**⚠️ Open Interpreter はコードを実行する前にユーザーの確認を求めます。**

この確認を回避するには、`interpreter -y` を実行するか、`interpreter.auto_run = True` を設定します。その場合:

- ファイルやシステム設定を変更するコマンドを要求するときは注意してください。
- Open Interpreter を自動運転車のように監視し、ターミナルを閉じてプロセスを終了できるように準備しておいてください。
- Google Colab や Replit のような制限された環境で Open Interpreter を実行することを検討してください。これらの環境はより隔離されており、任意のコードの実行に関連するリスクを軽減します。

一部のリスクを軽減するための[セーフモード](SAFE_MODE.md)の **実験的** サポートがあります。

## Open Interpreter はどのように機能するのか？

Open Interpreter は、[関数呼び出し対応の言語モデル](https://platform.openai.com/docs/guides/function-calling)に `execute` ツールを装備し、実行する `language`（"Python" や "JavaScript" など）と `code` を受け取ります。（関数呼び出し非対応モデルは Markdown コードブロック経由でもサポートされます。）

その後、モデルのメッセージ、コード、システムの出力を Markdown としてターミナルにストリーミングします。

## オフラインでドキュメントにアクセスする

[ドキュメント](.)全体は、インターネット接続なしでいつでも閲覧できます。

[Node](https://nodejs.org/en) が前提条件です:

- バージョン 18.17.0、またはそれ以降の 18.x.x バージョン。
- バージョン 20.3.0、またはそれ以降の 20.x.x バージョン。
- バージョン 21.0.0 以降（上限なし）。

[Mintlify](https://mintlify.com/) をインストールします:

```bash
npm i -g mintlify@latest
```

docs ディレクトリに移動し、次のコマンドを実行します:

```bash
# プロジェクトのルートディレクトリにいると仮定
cd ./docs

# ドキュメントサーバーを起動
mintlify dev
```

新しいブラウザウィンドウが開きます。ドキュメントサーバーが稼働している間、[http://localhost:3000](http://localhost:3000) でドキュメントにアクセスできます。

## 貢献

貢献にご関心をお寄せいただき、ありがとうございます！コミュニティからの参加を歓迎しています。

参加方法の詳細については、[貢献ガイドライン](CONTRIBUTING.md)をご覧ください。

## ロードマップ

Open Interpreter の未来を一足先に見るには、[ロードマップ](ROADMAP.md)をご覧ください。

**注意**: このソフトウェアは OpenAI とは関連していません。

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> 指先の速度で動くジュニアプログラマーにアクセスできることで、... 新しいワークフローを楽で効率的なものにし、プログラミングの恩恵を新しい層にも広げることができます。
>
> — _OpenAI の Code Interpreter リリース_

<br>
