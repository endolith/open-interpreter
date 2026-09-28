"""Tests for custom HTTP headers on outgoing LLM requests.

OI could not send provider-specific HTTP headers at all before `extra_headers`
existed. Some gateways reject requests that omit one (e.g. OpenCode Go's
`x-opencode-session`), so the passthrough is a capability in its own right.
"""

from interpreter.core.core import OpenInterpreter


def _capture_params(interpreter, **llm_attrs):
    """Drive Llm.run() far enough to build the request, and return the params dict.

    Replaces the completions callable with a stub that records its kwargs and
    yields a single done-ish chunk, so no network call is made.
    """
    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter.llm.completions = fake_completions
    for key, value in llm_attrs.items():
        setattr(interpreter.llm, key, value)

    messages = [
        {"role": "system", "type": "message", "content": "sys"},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    list(interpreter.llm.run(messages))
    return captured


def test_extra_headers_reach_the_request():
    """Headers set on llm.extra_headers are passed through to litellm verbatim.

    This is the whole point of the setting: a gateway that requires a session or
    tenant header cannot get one otherwise, because params is a hand-enumerated
    dict with no other way to inject request-level HTTP headers.
    """
    interpreter = OpenInterpreter()
    captured = _capture_params(interpreter, extra_headers={"x-opencode-session": "abc123"})
    assert captured["extra_headers"] == {"x-opencode-session": "abc123"}


def test_no_extra_headers_key_when_unset():
    """With no headers configured the key is omitted rather than sent as None.

    litellm treats an explicit None differently from an absent key, and OI should
    not add a parameter for a setting the user never touched.
    """
    interpreter = OpenInterpreter()
    captured = _capture_params(interpreter)
    assert "extra_headers" not in captured


def test_extra_headers_default_is_none():
    """A fresh Llm has no headers configured, so existing profiles are unaffected."""
    assert OpenInterpreter().llm.extra_headers is None
