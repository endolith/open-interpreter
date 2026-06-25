<h1 align="center">● Open Interpreter</h1>

<p align="center">
    <a href="https://discord.gg/Hvz9Axh84z">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <a href="../README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
    <a href="README_VN.md"><img src="https://img.shields.io/badge/Tài%20liệu-Tiếng%20Việt-white.svg" alt="VN doc"/></a>
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
    <br><a href="https://www.openinterpreter.com/">Ứng dụng máy tính</a> ‎ ‎ |‎ ‎ <a href="https://github.com/openinterpreter/openinterpreter">Open Interpreter (Rust)</a>‎ ‎ |‎ ‎ <a href=".">Tài liệu</a><br>
</p>

<br>

![local_explorer](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/d941c3b4-b5ad-4642-992c-40edf31e2e7a)

<br>

**Open Interpreter** cho phép các LLM chạy mã và lệnh shell cục bộ (Python, JavaScript, Bash, cmd, PowerShell, Ruby, R, Java và hơn thế nữa). Bạn tương tác với Open Interpreter qua giao diện chatbot trong terminal bằng cách chạy `interpreter` sau khi cài đặt.

Điều này cung cấp giao diện ngôn ngữ tự nhiên cho các khả năng đa dụng của máy tính:

- Tạo và chỉnh sửa ảnh, video, PDF, v.v.
- Điều khiển trình duyệt Chrome để nghiên cứu
- Vẽ biểu đồ, làm sạch và phân tích các bộ dữ liệu lớn
- ...v.v.

**⚠️ Lưu ý: Mặc định, bạn sẽ được yêu cầu phê duyệt mã trước khi chạy.**

## So sánh với các công cụ khác

Open Interpreter ra đời trước nhiều công cụ lập trình AI khác, và có những điểm tương đồng cũng như khác biệt:

- Mặc dù có thể viết mã và thực thi lệnh shell, tương tự các agent lập trình như [Claude Code](https://claude.ai/code), [Cursor](https://cursor.sh), [Devin](https://www.devin.ai) và các công cụ tương tự, Open Interpreter ít tập trung vào việc duy trì codebase dự án bằng cách vá các tệp mã nguồn, mà tập trung hơn vào hoàn thành các tác vụ một lần trong phiên tương tác, bền vững giống REPL (gần với Jupyter notebook hơn là IDE).
- Khác với [OpenClaw](https://openclaw.ai/), [Hermes Agent](https://hermes-agent.org/), v.v., Open Interpreter thường được dùng theo cách tương tác chứ không phải như agent tự động.
- Thay vì tương tác với thế giới qua các công cụ MCP, như [Claude Desktop](https://claude.ai/download), Open Interpreter chạy các đoạn mã hoặc [lệnh shell trực tiếp](https://ejholmes.github.io/2026/02/28/mcp-is-dead-long-live-the-cli.html).
- Tương tự các công cụ dịch shell bằng ngôn ngữ tự nhiên như [ShellGPT](https://github.com/ther1d/shell_gpt) hoặc [cmd-ai](https://github.com/BrodaNoel/cmd-ai), nhưng không giới hạn ở shell, và dùng giao diện chatbot tương tác — bạn có thể xem xét, từ chối (`n`), hoặc chỉnh sửa (`e`) lệnh trước khi chạy, và yêu cầu mô hình sửa lại.
- Tính năng Code Interpreter trong chatbot web ([OpenAI](https://developers.openai.com/api/docs/guides/tools-code-interpreter), [Mistral](https://docs.mistral.ai/studio-api/agents/agent-tools/code_interpreter), [Grok](https://docs.x.ai/developers/tools/code-execution), [Gemini](https://ai.google.dev/gemini-api/docs/interactions/code-execution), v.v.) chạy mã trong môi trường từ xa, sandbox, mã nguồn đóng và bị hạn chế. Tệp phải được tải lên từng cái và kết quả phải tải xuống sau đó. Mã được thực thi thường không truy cập được internet, bị giới hạn bởi bộ gói cài sẵn, và container hết hạn sau thời gian không hoạt động, làm mất tiến trình và dữ liệu. Open Interpreter vượt qua các hạn chế này bằng cách chạy trong môi trường cục bộ của bạn. Nó có quyền truy cập internet đầy đủ, không bị giới hạn thời gian hay kích thước tệp, và có thể dùng bất kỳ gói hay thư viện nào — thậm chí tự cài các thư viện hữu ích cho từng tác vụ.

## Demo

[Video demo](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/37152071-680d-4423-9af3-64836a6f7b60)

### Bản demo tương tác cũng có sẵn trên Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing)

### Cùng với ví dụ giao diện giọng nói, lấy cảm hứng từ _Her_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NojYGHDgxH6Y1G1oxThEBBb2AtyODBIK)

## Bắt đầu nhanh

### Cài đặt

Đây là phiên bản Python do cộng đồng duy trì của Open Interpreter.

Lệnh này sẽ cài đặt **`main`**, nhánh mặc định (cơ sở ổn định, CI và đích hợp nhất cho các thay đổi được chuyển):

```shell
pip install git+https://github.com/endolith/open-interpreter.git
```

> Xem [hướng dẫn thiết lập](getting-started/setup.mdx) của chúng tôi để biết các phụ thuộc tùy chọn.

Tuy nhiên, để sử dụng hàng ngày, bạn có thể muốn cài đặt **`classic/develop`** thay thế — đó là nhánh không ổn định được duy trì và sử dụng hàng ngày, với nhiều thay đổi và tính năng so với nhánh main, chẳng hạn như hỗ trợ mô hình suy luận, OpenRouter/DeepSeek/Qwen, công cụ tìm kiếm web, v.v.:

```shell
pip install git+https://github.com/endolith/open-interpreter.git@classic/develop
```

Để biết các tính năng riêng của fork, ghi chú về mô hình và chi tiết thiết lập, hãy xem [README `classic/develop`](https://github.com/endolith/open-interpreter/blob/classic/develop/README.md).

### Terminal

Sau khi cài đặt, chỉ cần chạy `interpreter`:

```shell
interpreter
```

Open Interpreter mặc định dùng **GPT-4o** của OpenAI và sẽ yêu cầu bạn nhập khóa API, có thể lấy tại [trang khóa API của OpenAI](https://platform.openai.com/api-keys). Đối với nhà cung cấp khác hoặc mô hình cục bộ, xem bên dưới.

### Python

```python
from interpreter import interpreter

interpreter.chat("Plot AAPL and META's normalized stock prices") # Thực thi một lệnh
interpreter.chat() # Bắt đầu chat tương tác
```

### GitHub Codespaces

Nhấn phím <kbd>,</kbd> trên trang GitHub của kho lưu trữ này để tạo codespace. Sau một lúc, bạn sẽ nhận được môi trường máy ảo trên cloud đã cài sẵn open-interpreter. Bạn có thể bắt đầu tương tác trực tiếp và thoải mái xác nhận việc thực thi lệnh hệ thống mà không lo hư hại máy.

## Lệnh

### Chat tương tác

Để bắt đầu chat tương tác trong terminal, chạy `interpreter` từ dòng lệnh:

```shell
interpreter
```

Hoặc `interpreter.chat()` từ tệp `.py`:

```python
interpreter.chat()
```

**Bạn cũng có thể stream từng phần:**

```python
message = "Chúng ta đang dùng hệ điều hành nào?"

for chunk in interpreter.chat(message, display=False, stream=True):
  print(chunk)
```

### Chat lập trình

Để kiểm soát chính xác hơn, bạn có thể truyền tin nhắn trực tiếp vào `.chat(message)`:

```python
interpreter.chat("Thêm phụ đề cho tất cả video trong /videos.")

# ... Stream đầu ra ra terminal, hoàn thành tác vụ ...

interpreter.chat("Trông ổn rồi, nhưng bạn có thể làm phụ đề to hơn không?")

# ...
```

### Bắt đầu chat mới

Trong Python, Open Interpreter ghi nhớ lịch sử hội thoại. Nếu muốn bắt đầu lại, bạn có thể đặt lại:

```python
interpreter.messages = []
```

### Lưu và khôi phục chat

`interpreter.chat()` trả về danh sách tin nhắn, có thể dùng để tiếp tục cuộc trò chuyện với `interpreter.messages = messages`:

```python
messages = interpreter.chat("Tên tôi là Killian.") # Lưu tin nhắn vào 'messages'
interpreter.messages = [] # Đặt lại interpreter ("Killian" sẽ bị quên)

interpreter.messages = messages # Tiếp tục chat từ 'messages' ("Killian" sẽ được nhớ)
```

### Tùy chỉnh tin nhắn hệ thống

Bạn có thể kiểm tra và cấu hình tin nhắn hệ thống của Open Interpreter để mở rộng chức năng, thay đổi quyền, hoặc cung cấp thêm ngữ cảnh.

```python
interpreter.system_message += """
Chạy lệnh shell với -y để người dùng không phải xác nhận.
"""
print(interpreter.system_message)
```

### Thay đổi mô hình ngôn ngữ

Open Interpreter dùng [LiteLLM](https://docs.litellm.ai/docs/providers/) để kết nối với các mô hình ngôn ngữ được lưu trữ.

Bạn có thể đổi mô hình bằng cách đặt tham số model:

```shell
interpreter --model gpt-3.5-turbo
interpreter --model claude-2
interpreter --model command-nightly
```

Trong Python, đặt mô hình trên đối tượng:

```python
interpreter.llm.model = "gpt-3.5-turbo"
```

[Tìm chuỗi "model" phù hợp cho mô hình ngôn ngữ của bạn tại đây.](https://docs.litellm.ai/docs/providers/)

### Chạy Open Interpreter cục bộ

#### Terminal

Open Interpreter có thể dùng máy chủ tương thích OpenAI để chạy mô hình cục bộ (LM Studio, Jan.ai, Ollama, v.v.)

Chỉ cần chạy `interpreter` với URL `api_base` của máy chủ suy luận (với LM Studio, mặc định là `http://localhost:1234/v1`):

```shell
interpreter --api_base "http://localhost:1234/v1" --api_key "fake_key"
```

Hoặc bạn có thể dùng Llamafile mà không cần cài phần mềm bên thứ ba, chỉ cần chạy:

```shell
interpreter --local
```

Để xem hướng dẫn chi tiết hơn, hãy xem [video này của Mike Bird](https://www.youtube.com/watch?v=CEs51hGWuGU&si=cN7f6QhfT4edfG5H)

**Cách chạy LM Studio ở chế độ nền.**

1. Tải [LM Studio](https://lmstudio.ai/) rồi khởi động.
2. Chọn một mô hình rồi nhấn **↓ Download**.
3. Nhấn nút **↔️** bên trái (dưới 💬).
4. Chọn mô hình ở phía trên, rồi nhấn **Start Server**.

Khi máy chủ đã chạy, bạn có thể bắt đầu trò chuyện với Open Interpreter.

> **Lưu ý:** Chế độ cục bộ đặt `context_window` thành 3000 và `max_tokens` thành 1000. Nếu mô hình của bạn có yêu cầu khác, hãy đặt các tham số này thủ công (xem bên dưới).

#### Python

Gói Python của chúng tôi cho bạn kiểm soát chi tiết hơn từng cài đặt. Để kết nối với LM Studio, dùng các cài đặt sau:

```python
from interpreter import interpreter

interpreter.offline = True # Tắt tính năng trực tuyến (ví dụ: kiểm tra cập nhật, telemetry)
interpreter.llm.model = "openai/x" # Báo cho OI gửi tin nhắn theo định dạng OpenAI
interpreter.llm.api_key = "fake_key" # LiteLLM, dùng để nói chuyện với LM Studio, yêu cầu giá trị này
interpreter.llm.api_base = "http://localhost:1234/v1" # Trỏ tới bất kỳ máy chủ tương thích OpenAI nào

interpreter.chat()
```

#### Cửa sổ ngữ cảnh, Max Tokens

Bạn có thể sửa `max_tokens` và `context_window` (tính bằng token) của các mô hình chạy cục bộ.

Ở chế độ cục bộ, cửa sổ ngữ cảnh nhỏ hơn sẽ dùng ít RAM hơn, vì vậy chúng tôi khuyên thử cửa sổ ngắn hơn nhiều (~1000) nếu bị lỗi hoặc chạy chậm. Đảm bảo `max_tokens` nhỏ hơn `context_window`.

```shell
interpreter --local --max_tokens 1000 --context_window 3000
```

### Chế độ verbose

Để giúp bạn kiểm tra Open Interpreter, chúng tôi có chế độ `--verbose` để gỡ lỗi.

Bạn có thể bật chế độ verbose bằng cờ (`interpreter --verbose`), hoặc giữa phiên chat:

```shell
$ interpreter
...
> %verbose true <- Bật chế độ verbose

> %verbose false <- Tắt chế độ verbose
```

### Lệnh chế độ tương tác

Trong chế độ tương tác, bạn có thể dùng các lệnh sau để cải thiện trải nghiệm. Đây là danh sách các lệnh có sẵn:

**Các lệnh có sẵn:**

- `%% [lệnh]`: Chạy lệnh trong shell hệ thống (bỏ qua LLM).
- `%verbose [true/false]`: Bật/tắt chế độ verbose. Không có tham số hoặc với `true` thì vào chế độ verbose. Với `false` thì thoát chế độ verbose.
- `%auto_run [true/false]`: Bật/tắt việc chạy mã không cần xác nhận. Không có tham số hoặc với `true` thì vào chế độ auto_run. Với `false` thì thoát chế độ auto_run.
- `%reset`: Đặt lại cuộc trò chuyện của phiên hiện tại.
- `%undo`: Xóa tin nhắn người dùng trước đó và phản hồi của AI khỏi lịch sử tin nhắn.
- `%save_message [path]`: Lưu tin nhắn vào đường dẫn JSON chỉ định. Nếu không có đường dẫn, mặc định là `messages.json`.
- `%load_message [path]`: Tải tin nhắn từ đường dẫn JSON chỉ định. Nếu không có đường dẫn, mặc định là `messages.json`.
- `%tokens [prompt]`: (_Thử nghiệm_) Tính số token sẽ được gửi cùng prompt tiếp theo làm ngữ cảnh và ước tính chi phí. Tùy chọn tính token và chi phí ước tính của `prompt` nếu được cung cấp. Dựa vào [phương thức `cost_per_token()` của LiteLLM](https://docs.litellm.ai/docs/completion/token_usage#2-cost_per_token) để ước tính chi phí.
- `%jupyter`: Xuất cuộc trò chuyện ra tệp Jupyter notebook.
- `%markdown [path]`: Xuất cuộc trò chuyện ra đường dẫn Markdown chỉ định. Nếu không có đường dẫn, sẽ lưu vào thư mục Downloads với tên cuộc trò chuyện được tạo tự động.
- `%info`: Hiển thị thông tin hệ thống và interpreter.
- `%help`: Hiển thị thông báo trợ giúp.

### Cấu hình / Hồ sơ

Open Interpreter cho phép bạn đặt hành vi mặc định bằng tệp `yaml`.

Điều này cung cấp cách linh hoạt để cấu hình interpreter mà không cần thay đổi đối số dòng lệnh mỗi lần.

Chạy lệnh sau để mở thư mục hồ sơ:

```
interpreter --profiles
```

Bạn có thể thêm tệp `yaml` ở đó. Hồ sơ mặc định có tên `default.yaml`.

#### Nhiều hồ sơ

Open Interpreter hỗ trợ nhiều tệp `yaml`, cho phép bạn dễ dàng chuyển đổi giữa các cấu hình:

```
interpreter --profile my_profile.yaml
```

## Máy chủ FastAPI mẫu

Open Interpreter có thể được điều khiển qua các endpoint HTTP REST:

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

Bạn cũng có thể khởi động máy chủ tích hợp với hỗ trợ WebSocket và giao diện web bằng cách chạy `interpreter --server` (yêu cầu extra `[server]`).

## Android

Hướng dẫn từng bước cài đặt Open Interpreter trên thiết bị Android có trong [kho open-interpreter-termux](https://github.com/MikeBirdTech/open-interpreter-termux).

## Lưu ý an toàn

Vì mã được tạo chạy trong môi trường cục bộ của bạn, nó có thể tương tác với tệp và cài đặt hệ thống, dẫn đến kết quả không mong muốn như mất dữ liệu hoặc rủi ro bảo mật.

**⚠️ Open Interpreter sẽ yêu cầu xác nhận của người dùng trước khi thực thi mã.**

Bạn có thể chạy `interpreter -y` hoặc đặt `interpreter.auto_run = True` để bỏ qua xác nhận này, trong trường hợp đó:

- Hãy thận trọng khi yêu cầu các lệnh sửa đổi tệp hoặc cài đặt hệ thống.
- Theo dõi Open Interpreter như xe tự lái, và sẵn sàng kết thúc tiến trình bằng cách đóng terminal.
- Cân nhắc chạy Open Interpreter trong môi trường bị hạn chế như Google Colab hoặc Replit. Các môi trường này cô lập hơn, giảm rủi ro khi chạy mã tùy ý.

Có hỗ trợ **thử nghiệm** cho [chế độ an toàn](SAFE_MODE.md) để giúp giảm thiểu một số rủi ro.

## Cách hoạt động

Open Interpreter trang bị [mô hình ngôn ngữ gọi hàm](https://platform.openai.com/docs/guides/function-calling) với công cụ `execute`, nhận `language` (như "Python" hoặc "JavaScript") và `code` để chạy. (Các mô hình không gọi hàm cũng được hỗ trợ qua khối mã markdown.)

Sau đó, chúng tôi stream tin nhắn, mã của mô hình và đầu ra hệ thống của bạn ra terminal dưới dạng Markdown.

## Truy cập tài liệu ngoại tuyến

Toàn bộ [tài liệu](.) có thể truy cập mọi lúc mà không cần kết nối internet.

[Node](https://nodejs.org/en) là điều kiện tiên quyết:

- Phiên bản 18.17.0 hoặc bất kỳ phiên bản 18.x.x sau đó.
- Phiên bản 20.3.0 hoặc bất kỳ phiên bản 20.x.x sau đó.
- Bất kỳ phiên bản nào từ 21.0.0 trở lên, không giới hạn trên.

Cài đặt [Mintlify](https://mintlify.com/):

```bash
npm i -g mintlify@latest
```

Chuyển vào thư mục docs và chạy lệnh phù hợp:

```bash
# Giả sử bạn đang ở thư mục gốc của dự án
cd ./docs

# Chạy máy chủ tài liệu
mintlify dev
```

Một cửa sổ trình duyệt mới sẽ mở. Tài liệu sẽ có tại [http://localhost:3000](http://localhost:3000) miễn là máy chủ tài liệu đang chạy.

## Đóng góp

Cảm ơn bạn đã quan tâm đến việc đóng góp! Chúng tôi hoan nghênh sự tham gia từ cộng đồng.

Vui lòng xem [hướng dẫn đóng góp](CONTRIBUTING.md) để biết thêm chi tiết cách tham gia.

## Lộ trình

Truy cập [lộ trình](ROADMAP.md) của chúng tôi để xem trước tương lai của Open Interpreter.

**Lưu ý**: Phần mềm này không liên kết với OpenAI.

![thumbnail-ncu](https://github.com/OpenInterpreter/open-interpreter/assets/63927363/1b19a5db-b486-41fd-a7a1-fe2028031686)

> Có quyền truy cập vào một lập trình viên junior làm việc nhanh như tốc độ đầu ngón tay của bạn... có thể khiến quy trình làm việc mới trở nên dễ dàng và hiệu quả, đồng thời mở rộng lợi ích lập trình cho nhiều đối tượng hơn.
>
> — _Phát hành Code Interpreter của OpenAI_

<br>
