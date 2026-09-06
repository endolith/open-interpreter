"""Minimal OpenAI-compatible chat API for CI (no real LLM)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _message_text(messages: list) -> str:
    """Flatten chat message contents into one string for scenario matching."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _last_user_text(messages: list) -> str:
    """Return the content of the most recent user message."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def pick_reply(body: dict) -> str:
    """Return a canned assistant reply based on prompt keywords (level-2 scenarios)."""
    messages = body.get("messages") or []
    text = _last_user_text(messages).lower()
    ran_code = any(
        message.get("role") == "computer" and message.get("type") == "console"
        for message in messages
    )

    if ran_code and "read file.txt" not in text:
        # End the respond() loop after auto_run executes mocked code once.
        return "The task is done."

    if "code output:" in text:
        # OI injects console output as a follow-up user turn; stop looping.
        return "The task is done."

    if "hello, world" in text or "just the words hello, world" in text:
        return "Hello, World!"

    if "washington" in text and "file.txt" in text and (
        "write" in text or "save" in text
    ):
        return (
            "```python\n"
            "with open('file.txt', 'w') as f:\n"
            "    f.write('Washington')\n"
            "```"
        )

    if "read file.txt" in text or (
        "read" in text and "file.txt" in text and "washington" not in text
    ):
        return "Washington"

    if "use python" in text and "print" not in text:
        # Simple math smoke: integration tests ask the model to compute via Python.
        return "```python\nprint(42)\n```"

    return "Hello, World!"


def _user_history_text(messages: list) -> str:
    """All user message text, for multi-turn scenario detection.

    Follow-up turns (e.g. injected console output) do not repeat the original
    prompt, so scenarios spanning several turns must look at the whole history,
    not just the last user message.
    """
    return "\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    )


def _messages_since_errand(messages: list) -> list:
    """Messages from the errand prompt onward, or [] when there is no errand.

    An interpreter may have chatted about other things before starting the
    errand; only turns after the errand prompt belong to this scenario.
    """
    for i, message in enumerate(messages):
        if (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and "errand" in message["content"].lower()
        ):
            return messages[i:]
    return []


def _assistant_count(messages: list) -> int:
    """Count assistant messages since the errand prompt (completed errand turns).

    The request body holds OpenAI-format messages, where executed code shows
    up as assistant tool calls / code content — never as computer console
    entries — so completed turns are what we count.
    """
    return sum(
        1 for message in _messages_since_errand(messages) if message.get("role") == "assistant"
    )


_ERRAND_PYTHON_CODE = 'with open("step1.txt", "w") as f:\n    f.write("one")'
_ERRAND_SHELL_CODE = "echo two > step2.txt"


def _tool_call_delta(call_id, name, arguments, index=0):
    """One streaming tool-call delta entry in OpenAI wire shape."""
    return {
        "tool_calls": [
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    }


def merge_tool_calls(deltas: list) -> list:
    """Merge streaming tool_calls deltas into one OpenAI message list.

    Entries are grouped by index with arguments concatenated, mirroring how
    clients reassemble a streamed call. Entries without a usable function are
    skipped.
    """
    merged: dict[int, dict] = {}
    order: list[int] = []
    for delta in deltas:
        for entry in delta.get("tool_calls") or []:
            index = entry.get("index", 0)
            if index not in merged:
                merged[index] = {
                    "index": index,
                    "id": entry.get("id"),
                    "type": entry.get("type", "function"),
                    "function": {"name": None, "arguments": ""},
                }
                order.append(index)
            function = entry.get("function") or {}
            if function.get("name") is not None:
                merged[index]["function"]["name"] = function["name"]
            merged[index]["function"]["arguments"] += function.get("arguments") or ""
    return [
        merged[index]
        for index in order
        if merged[index]["function"]["name"] is not None
    ]


def _split_tool_call_deltas(call_id, name, arguments):
    """Split a tool call across two deltas (name + partial args, then rest).

    Mirrors how providers stream large arguments, exercising client-side
    reassembly of the merged function_call.
    """
    cut = len(arguments) // 2
    return [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments[:cut]},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[cut:]}}]},
    ]


def errand_tool_deltas(messages: list) -> list[dict] | None:
    """Streaming deltas for the multi-turn errand scenario, or None.

    Turn state comes from executed-code count: first a python tool call, then
    a shell tool call, then plain talking. Returns delta dicts (no envelope).
    """
    history = _user_history_text(messages).lower()
    if "errand" not in history:
        return None
    turns = _assistant_count(messages)
    if turns == 0:
        arguments = json.dumps({"language": "python", "code": _ERRAND_PYTHON_CODE})
        return _split_tool_call_deltas("call_step1", "execute", arguments)
    if turns == 1:
        arguments = json.dumps({"language": "shell", "code": _ERRAND_SHELL_CODE})
        return [_tool_call_delta("call_step2", "execute", arguments)]
    return [{"content": "Errand complete."}]


def errand_text_reply(messages: list) -> str | None:
    """Plain-text reply for the errand scenario in code-block mode, or None."""
    history = _user_history_text(messages).lower()
    if "errand" not in history:
        return None
    turns = _assistant_count(messages)
    if turns == 0:
        return "```python\n" + _ERRAND_PYTHON_CODE + "\n```"
    if turns == 1:
        return "```shell\n" + _ERRAND_SHELL_CODE + "\n```"
    return "Errand complete."


def stream_reply_chunks(content: str) -> list[str]:
    """Split assistant text into streaming deltas that run_text_llm can parse.

    run_text_llm defers processing while accumulated text ends with a backtick
    (waiting for more of a fence). Split the opening fence from the language line
    so the yielded code body does not include leading ``` markers.
    """
    if "```" not in content:
        return [content]

    before, rest = content.split("```", 1)
    chunks: list[str] = []
    if before:
        chunks.append(before)

    if "\n" in rest:
        language, code = rest.split("\n", 1)
        chunks.append("```")
        if code.endswith("```"):
            body, _ = code.rsplit("```", 1)
            chunks.append(f"{language}\n{body}")
            chunks.append("```")
        else:
            chunks.append(f"{language}\n{code}")
    else:
        chunks.append(f"```{rest}")

    return [chunk for chunk in chunks if chunk]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        stream = body.get("stream", False)
        messages = body.get("messages") or []

        # Tool-call mode (function calling): the request carries a tools
        # parameter. Serve streaming tool_calls deltas for known scenarios.
        if body.get("tools"):
            deltas = errand_tool_deltas(messages)
            if deltas is None:
                deltas = [{"content": "Hello, World!"}]
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta in deltas:
                    chunk = {"choices": [{"delta": delta, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            tool_calls = merge_tool_calls(deltas)
            if tool_calls:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
                finish_reason = "tool_calls"
            else:
                text = "".join(
                    delta.get("content", "")
                    for delta in deltas
                    if isinstance(delta.get("content"), str)
                )
                message = {"role": "assistant", "content": text}
                finish_reason = "stop"
            payload = {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ]
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        content = errand_text_reply(messages)
        if content is None:
            content = pick_reply(body)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in stream_reply_chunks(content):
                chunk = {
                    "choices": [{"delta": {"content": piece}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return

        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MockOpenAIServer:
    """In-process HTTP server that mimics OpenAI chat completions for tests."""

    def __init__(self):
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def api_base(self) -> str:
        if self._httpd is None:
            raise RuntimeError("server not started")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/v1"

    def start(self):
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
