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
