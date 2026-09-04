import json
import urllib.request

import pytest

from tests.support.mock_openai_server import (
    MockOpenAIServer,
    errand_tool_deltas,
    merge_tool_calls,
    pick_reply,
    stream_reply_chunks,
)


@pytest.fixture
def running_server():
    """A live mock server for direct HTTP assertions."""
    server = MockOpenAIServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _post(server, body):
    """POST a chat-completions body and return the decoded JSON payload."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        server.api_base + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_pick_reply_hello_world():
    """pick_reply returns a plain greeting for the integration hello-world prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Please reply with just the words Hello, World! and nothing else.",
            }
        ]
    }
    assert pick_reply(body) == "Hello, World!"


def test_pick_reply_write_file_returns_python():
    """pick_reply returns a Python code block for the write-to-file integration prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Write the word 'Washington' to a .txt file called file.txt.",
            }
        ]
    }
    reply = pick_reply(body)
    assert reply.startswith("```python")
    assert "file.txt" in reply
    assert "Washington" in reply


def test_pick_reply_read_file_returns_content():
    """pick_reply returns file contents for a read-file follow-up prompt."""
    body = {
        "messages": [
            {"role": "user", "content": "Read file.txt in the current directory."}
        ]
    }
    assert pick_reply(body) == "Washington"


def test_stream_reply_chunks_splits_fenced_code():
    """stream_reply_chunks emits multiple deltas for fenced code so run_text_llm parses it."""
    chunks = stream_reply_chunks("```python\nprint(1)\n```")
    assert chunks == ["```", "python\nprint(1)\n", "```"]


def _errand_messages(n_assistant_turns=0):
    """History with an errand prompt plus completed assistant turns."""
    messages = [{"role": "user", "content": "Please run this errand."}]
    for _ in range(n_assistant_turns):
        messages.append({"role": "assistant", "content": ""})
    return messages


def test_merge_tool_calls_reassembles_split_arguments():
    """merge_tool_calls groups split deltas by index into one message entry."""
    arguments = '{"language": "python", "code": "print(1)"}'
    cut = len(arguments) // 2
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "execute", "arguments": arguments[:cut]},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[cut:]}}]},
    ]
    assert merge_tool_calls(deltas) == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute", "arguments": arguments},
        }
    ]


def test_merge_tool_calls_skips_functionless_entries():
    """merge_tool_calls drops entries with no usable function."""
    assert merge_tool_calls([{"tool_calls": [{"index": 0, "function": None}]}]) == []


def test_errand_tool_deltas_talk_after_two_turns():
    """The errand ends by talking once two assistant turns exist."""
    messages = _errand_messages(n_assistant_turns=2)
    assert errand_tool_deltas(messages) == [{"content": "Errand complete."}]


def test_nonstream_tool_turn_returns_populated_tool_calls(running_server):
    """Non-stream tool requests return message.tool_calls with finish_reason tool_calls."""
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Please run this errand."}],
        "tools": [{"type": "function", "function": {"name": "execute"}}],
        "stream": False,
    }
    payload = _post(running_server, body)
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = payload["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "execute"
    assert '"language": "python"' in tool_calls[0]["function"]["arguments"]


def test_nonstream_unknown_prompt_returns_text_stop(running_server):
    """Non-stream tool requests without a scenario return text, not an empty tool_calls claim."""
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hello."}],
        "tools": [{"type": "function", "function": {"name": "execute"}}],
        "stream": False,
    }
    payload = _post(running_server, body)
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == "Hello, World!"
    assert "tool_calls" not in payload["choices"][0]["message"]


def test_nonstream_talk_turn_returns_text_stop(running_server):
    """Non-stream completion turns after the errand return the final text."""
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": _errand_messages(n_assistant_turns=2),
        "tools": [{"type": "function", "function": {"name": "execute"}}],
        "stream": False,
    }
    payload = _post(running_server, body)
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == "Errand complete."
