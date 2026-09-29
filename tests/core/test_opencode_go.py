"""Tests for the opencode_go/ model prefix.

OpenCode Go is an OpenAI-compatible gateway that (a) requires an API key of its
own, (b) rejects requests with no `x-opencode-session` header, and (c) serves its
catalog over three different wire formats, only some of which OI speaks. These
tests pin each of those behaviours, especially the two ways it can go wrong
quietly: sending the wrong credential, and silently getting a different model.
"""

import pytest

from interpreter.core.core import OpenInterpreter


def _configure(model, monkeypatch, **env):
    """Build an interpreter aimed at opencode_go/<model> and return it, unloaded."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    interpreter = OpenInterpreter()
    interpreter.llm.model = model
    return interpreter


def _request_params(interpreter):
    """Drive one Llm.run() and return the request params it produced."""
    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter.llm.completions = fake_completions
    list(
        interpreter.llm.run(
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "h"},
            ]
        )
    )
    return captured


def test_prefix_rewrites_to_openai_compatible_model(monkeypatch):
    """opencode_go/<model> becomes openai/<model> with Go's base URL, like DashScope."""
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch, OPENCODE_GO_API_KEY="k")
    interpreter.llm.load()
    assert interpreter.llm.model == "openai/deepseek-v4-flash"
    assert interpreter.llm.api_base == "https://opencode.ai/zen/go/v1"


def test_api_base_is_overridable(monkeypatch):
    """An explicit api_base (here from OPENCODE_GO_API_BASE) wins over the default."""
    interpreter = _configure(
        "opencode_go/glm-5.1",
        monkeypatch,
        OPENCODE_GO_API_KEY="k",
        OPENCODE_GO_API_BASE="https://proxy.local/v1",
    )
    interpreter.llm.load()
    assert interpreter.llm.api_base == "https://proxy.local/v1"


def test_stale_api_base_from_another_provider_is_replaced(monkeypatch):
    """A leftover api_base must not capture the Go key.

    Regression test for a real failure: with `api_base: https://api.openai.com/v1`
    left in default.yaml (or a previous `--api_base`), the model was still
    rewritten to openai/<model> but the request went to OpenAI, which answered
    `AuthenticationError: OpenAIException - Invalid API key` -- an error that
    names neither Go nor the actual cause. The prefix identifies the provider,
    so a base belonging to a different one is replaced.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "oc_sk_go")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_base = "https://api.openai.com/v1"
    interpreter.llm.load()
    assert interpreter.llm.api_base == "https://opencode.ai/zen/go/v1"


def test_stale_local_api_base_is_kept_and_documented(monkeypatch):
    """A localhost base is deliberately NOT replaced, unlike OpenAI's.

    A localhost base is ambiguous: it is either a local model server left over
    from an earlier setup (LM Studio, Ollama) or a deliberate proxy in front of
    Go. There is no way to tell the two apart, and overriding it would break the
    second case silently. So localhost is preserved, and if it is wrong the
    gateway's own error names the host that was actually contacted -- unlike the
    OpenAI case, where a 401 blamed the key. Set OPENCODE_GO_API_BASE to
    override explicitly.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "oc_sk_go")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_base = "http://localhost:1234/v1"
    interpreter.llm.load()
    assert interpreter.llm.api_base == "http://localhost:1234/v1"


def test_explicit_go_base_beats_stale_foreign_base(monkeypatch):
    """OPENCODE_GO_API_BASE is honoured even when another base is already set."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "oc_sk_go")
    monkeypatch.setenv("OPENCODE_GO_API_BASE", "https://opencode.ai/zen/go/v2")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_base = "https://api.openai.com/v1"
    interpreter.llm.load()
    assert interpreter.llm.api_base == "https://opencode.ai/zen/go/v2"


def test_session_header_carries_the_conversation_id(monkeypatch):
    """Go rejects requests without x-opencode-session, so it is set from the id.

    The value must be the conversation id, not a per-request value: the header is
    how Go pins provider routing and prompt caching to a conversation.
    """
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch, OPENCODE_GO_API_KEY="k")
    interpreter.llm.load()
    params = _request_params(interpreter)
    assert params["extra_headers"]["x-opencode-session"] == interpreter.conversation_id


def test_user_supplied_session_header_wins(monkeypatch):
    """An explicit header from the profile is respected rather than overwritten."""
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch, OPENCODE_GO_API_KEY="k")
    interpreter.llm.extra_headers = {"x-opencode-session": "pinned-by-user"}
    interpreter.llm.load()
    params = _request_params(interpreter)
    assert params["extra_headers"]["x-opencode-session"] == "pinned-by-user"


def test_other_user_headers_are_preserved(monkeypatch):
    """A caller's unrelated headers ride along with the session header."""
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch, OPENCODE_GO_API_KEY="k")
    interpreter.llm.extra_headers = {"X-Tenant": "acme"}
    interpreter.llm.load()
    params = _request_params(interpreter)
    assert params["extra_headers"]["X-Tenant"] == "acme"
    assert "x-opencode-session" in params["extra_headers"]


def test_session_header_follows_a_reset(monkeypatch):
    """After %reset the header must carry the new conversation id.

    reset() re-mints conversation_id to match the cleared history. The header is
    filled in per request, so a value captured at load() time would have gone
    stale and kept advertising the old conversation.
    """
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch, OPENCODE_GO_API_KEY="k")
    interpreter.llm.load()
    before = _request_params(interpreter)["extra_headers"]["x-opencode-session"]

    interpreter.reset()

    after = _request_params(interpreter)["extra_headers"]["x-opencode-session"]
    assert after == interpreter.conversation_id
    assert after != before


def test_missing_api_key_fails_loudly(monkeypatch):
    """A missing Go key raises instead of falling back to OPENAI_API_KEY.

    The rewrite to openai/<model> makes litellm resolve an unset key from the
    environment, which would send the user's OpenAI credential to opencode.ai --
    an auth failure at best and a credential leak at worst.
    """
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated-openai-key")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)

    with pytest.raises(ValueError, match="OPENCODE_GO_API_KEY"):
        interpreter.llm.load()


def test_openai_key_is_not_used_for_go(monkeypatch):
    """An ambient OPENAI_API_KEY alone is never enough to authenticate to Go."""
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated-openai-key")
    interpreter = _configure("opencode_go/glm-5.1", monkeypatch)

    with pytest.raises(ValueError):
        interpreter.llm.load()
    assert interpreter.llm.api_key != "sk-unrelated-openai-key"


@pytest.mark.parametrize(
    "model",
    ["minimax-m3", "qwen3.6-plus", "qwen3.8-max", "grok-4.6", "gpt-5.6-luna"],
)
def test_non_chat_models_are_refused_rather_than_mis_routed(model, monkeypatch):
    """Models served only on /messages or /responses are rejected with a clear error.

    The gateway does not reliably reject these on /chat/completions; a report
    against another client found it silently answered with a *different* model.
    Failing here is strictly better than returning a confident wrong answer.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "k")
    interpreter = _configure(f"opencode_go/{model}", monkeypatch)

    with pytest.raises(ValueError, match="does not speak"):
        interpreter.llm.load()


def test_chat_completions_models_are_accepted(monkeypatch):
    """The chat-completions families load without complaint."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "k")
    for model in ("deepseek-v4-pro", "glm-5.1", "kimi-k2.6", "mimo-v2.5"):
        interpreter = _configure(f"opencode_go/{model}", monkeypatch)
        interpreter.llm.load()
        assert interpreter.llm.model == f"openai/{model}"
