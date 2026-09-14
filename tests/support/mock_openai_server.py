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


# Multi-turn scenarios, keyed by a phrase in the user prompt that starts them.
# Each step is one assistant turn: a dict describes an execute tool call
# (rendered as streaming tool_calls deltas in tool mode and as a fenced code
# block in text mode, from the same source so the two modes cannot drift);
# a plain string is a talking turn. Optional dict fields shape the tool-mode
# wire format only (text mode has no equivalent):
#   name_first: open with a name-only entry whose arguments are "", as
#       OpenAI's first delta does, before any argument text arrives.
#   cut_after: markers; the argument JSON continues in a new delta right
#       after the first occurrence of each, so cuts can land mid-token.
#   second_call: a parallel {language, code, call_id} at tool_calls index 1,
#       sent in the opening delta, or in a later delta of its own when
#       second_call_later is set. Text mode renders the first call only.
SCENARIOS: dict[str, list] = {
    "errand": [
        {
            "language": "python",
            "code": 'with open("step1.txt", "w") as f:\n    f.write("one")',
            "call_id": "call_step1",
            "cut_after": ['"code": "with open('],
        },
        # Deliberately reads step1.txt: the shell step must observe the
        # filesystem state left by the python step, proving execution state
        # persists across tool calls (and across languages) rather than each
        # step running isolated.
        {
            "language": "shell",
            "code": "echo $(cat step1.txt)-two > step2.txt",
            "call_id": "call_step2",
        },
        {"language": "python", "code": "print(undefined_name)", "call_id": "call_step3"},
        {"language": "python", "code": 'print("recovered")', "call_id": "call_step4"},
        "Errand complete.",
    ],
    # Split persistence: part one defines a python value and a shell value in
    # one chat, part two uses them in the next chat.
    "persistence check part one": [
        {"language": "python", "code": "persist_num = 40 + 2", "call_id": "persist_step1"},
        {"language": "shell", "code": "export PERSIST_WORD=hello", "call_id": "persist_step2"},
        "values defined.",
    ],
    "persistence check part two": [
        {"language": "python", "code": "print(persist_num)", "call_id": "persist_step3"},
        {"language": "shell", "code": "echo $PERSIST_WORD", "call_id": "persist_step4"},
        "values verified.",
    ],
    # OpenAI announces a call (id, name, arguments "") before streaming any
    # argument text; the client must not treat the empty opener as a call.
    "name-first opener": [
        {"language": "python", "code": 'print("opener ok")', "call_id": "opener_1", "name_first": True},
        "opener done.",
    ],
    # Cuts inside the language token and inside a \uXXXX escape: partial
    # JSON must not be trusted for the language until it is complete, and a
    # half-escape must not corrupt the non-ASCII characters it encodes.
    "unicode cut": [
        {
            "language": "python",
            "code": 'print("café ✓")',
            "call_id": "unicode_1",
            "cut_after": ['"pyth', "\\u00e"],
        },
        "unicode done.",
    ],
    # Parallel tool calls: OpenAI may return two calls in one turn, either
    # both announced in the opening delta or the second in a later delta.
    "two calls at once": [
        {
            "language": "python",
            "code": 'print("first of two")',
            "call_id": "pair_a",
            "second_call": {"language": "shell", "code": "echo second-of-two", "call_id": "pair_b"},
        },
        "pair done.",
    ],
    "two calls staggered": [
        {
            "language": "python",
            "code": 'print("first of two")',
            "call_id": "stagger_a",
            "second_call": {"language": "shell", "code": "echo second-of-two", "call_id": "stagger_b"},
            "second_call_later": True,
        },
        "staggered done.",
    ],
}


def _scenario_step(messages: list):
    """The registry step this request is owed, or None when no scenario applies.

    User prompts are scanned newest-first and the first registry keyword found
    owns the turn: a repeated scenario restarts at step zero, and a scenario
    started after another one takes over regardless of registry order.

    The turn index is the number of assistant messages since the owning
    prompt. Request bodies hold OpenAI-format messages, where executed code
    shows up as assistant tool calls / code content — never as computer
    console entries — so completed assistant turns are what count. Past the
    last step the scenario is over: later prompts fall through to the normal
    fallback instead of repeating the completion.
    """
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        text = message["content"].lower()
        for keyword, steps in SCENARIOS.items():
            if keyword in text:
                turn = sum(1 for m in messages[i:] if m.get("role") == "assistant")
                return steps[turn] if turn < len(steps) else None
    return None


def _call_entry(call_id: str, arguments: str, index: int = 0) -> dict:
    """The opening tool_calls entry of one call: id, type, name and first arguments."""
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": arguments},
    }


def _split_after(text: str, markers) -> list[str]:
    """Pieces of text, broken right after the first occurrence of each marker.

    A missing marker raises so a scenario typo fails loudly instead of
    quietly streaming the arguments in one piece.
    """
    cuts = sorted(text.index(marker) + len(marker) for marker in markers)
    bounds = [0, *cuts, len(text)]
    return [text[start:end] for start, end in zip(bounds, bounds[1:])]


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


def _tool_deltas(step: dict) -> list[dict]:
    """Streaming tool_calls deltas (no envelope) for one code step.

    The first delta carries the call's id, type and name; continuation
    deltas carry only index and further argument text, as providers stream
    large arguments. The optional step fields documented on SCENARIOS are
    all applied here so a new wire shape costs one field, not a renderer.
    """
    pieces = _split_after(_arguments(step), step.get("cut_after", ()))
    if step.get("name_first"):
        pieces.insert(0, "")
    deltas = [{"tool_calls": [_call_entry(step["call_id"], pieces[0])]}]
    deltas += [{"tool_calls": [{"index": 0, "function": {"arguments": piece}}]} for piece in pieces[1:]]
    if "second_call" in step:
        second = step["second_call"]
        entry = _call_entry(second["call_id"], _arguments(second), index=1)
        if step.get("second_call_later"):
            deltas.append({"tool_calls": [entry]})
        else:
            deltas[0]["tool_calls"].append(entry)
    return deltas


def _arguments(step: dict) -> str:
    """The execute tool's JSON arguments for one code step."""
    return json.dumps({"language": step["language"], "code": step["code"]})


def scenario_tool_deltas(messages: list) -> list[dict] | None:
    """Streaming deltas for the scenario turn this request is owed, or None."""
    step = _scenario_step(messages)
    if step is None:
        return None
    return [{"content": step}] if isinstance(step, str) else _tool_deltas(step)


def scenario_text_reply(messages: list) -> str | None:
    """Plain-text reply (code-block mode) for the scenario turn, or None."""
    step = _scenario_step(messages)
    if step is None:
        return None
    return step if isinstance(step, str) else "```%s\n%s\n```" % (step["language"], step["code"])


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
            deltas = scenario_tool_deltas(messages)
            if deltas is None:
                deltas = [{"content": "Hello, World!"}]
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta in deltas:
                    chunk = {"choices": [{"delta": delta, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                # A stream that requested tool calls terminates with the
                # tool_calls reason, mirroring the OpenAI wire contract;
                # text-only turns terminate with stop.
                terminal_reason = (
                    "tool_calls"
                    if any(
                        delta.get("tool_calls") for delta in deltas
                    )
                    else "stop"
                )
                done = {"choices": [{"delta": {}, "finish_reason": terminal_reason}]}
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

        content = scenario_text_reply(messages)
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
