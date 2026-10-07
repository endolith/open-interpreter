"""Regression tests for #324: chat_completion dispatch defects.

Three defects lived in POST /openai/chat/completions: every request
gained a "." user turn (model saw it and history kept it), the
context_mode/{START} branch was dead code after the str/list chain, and
parts-list content crashed .lower() with a 500 while code was pending.
Only respond() is faked; it records the history handed to the LLM.
"""

import copy
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from interpreter.core.async_core import AsyncInterpreter, Server


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Hide INTERPRETER_API_KEY so the auth middleware stays out of the way."""
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)


def _make(auto_run=True):
    interpreter = AsyncInterpreter()
    interpreter.auto_run = auto_run
    interpreter.disable_telemetry = True
    interpreter.conversation_history = False
    client = TestClient(Server(interpreter).app, raise_server_exceptions=False)
    return interpreter, client


def _post(client, content, **extra):
    return client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": content}], **extra},
    )


def test_no_junk_turn_injected():
    """The model sees the real turn only; history keeps no '.' turns."""
    seen = []

    def fake_respond(interp):
        seen.append(copy.deepcopy(interp.messages))
        yield {"role": "assistant", "type": "message", "content": "4"}

    with mock.patch("interpreter.core.core.respond", fake_respond):
        interpreter, client = _make()
        assert _post(client, "what is 2+2").status_code == 200
        assert _post(client, "and 3+3?").status_code == 200

    assert [m["content"] for m in seen[-1] if m["role"] == "user"] == [
        "what is 2+2",
        "and 3+3?",
    ]
    assert [(m["role"], m["content"]) for m in interpreter.messages] == [
        ("user", "what is 2+2"),
        ("assistant", "4"),
        ("user", "and 3+3?"),
        ("assistant", "4"),
    ]


def test_context_mode_accumulates_until_start():
    """Background accumulates silently; {START} answers once without leaking."""
    seen = []

    def fake_respond(interp):
        seen.append(copy.deepcopy(interp.messages))
        yield {"role": "assistant", "type": "message", "content": "4"}

    with mock.patch("interpreter.core.core.respond", fake_respond):
        _, client = _make()
        assert _post(client, "{CONTEXT_MODE_ON}").status_code == 200
        assert _post(client, "some background").status_code == 200
        assert len(seen) == 0
        assert _post(client, "{START}").status_code == 200

    assert len(seen) == 1
    assert [m["content"] for m in seen[-1] if m["role"] == "user"] == [
        "some background"
    ]


def test_parts_list_yes_while_code_pending():
    """Parts-list 'yes' with pending code answers instead of 500ing."""
    seen = []

    def code_respond(interp):
        seen.append(copy.deepcopy(interp.messages))
        yield {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
        }
        yield {
            "role": "computer",
            "type": "confirmation",
            "content": {"format": "python", "content": "print(1)"},
        }

    with mock.patch("interpreter.core.core.respond", code_respond):
        interpreter, client = _make(auto_run=False)
        assert _post(client, "print 1", stream=True).status_code == 200
        assert interpreter.messages[-1]["type"] == "code"
        response = _post(client, [{"type": "text", "text": "yes"}])

    assert response.status_code == 200
