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


def test_stale_profile_key_is_replaced_by_env_key(monkeypatch):
    """An explicit OPENCODE_GO_API_KEY wins over whatever the profile holds.

    Regression test for the second half of the reported failure: with a key left
    in default.yaml from another provider, the original `if self.api_key is None`
    guard kept it and the Go gateway rejected it with `Invalid API key.` -- which
    blames the Go key that was never sent.

    This was originally written to assert that the environment wins only over a
    *non-Go-shaped* profile key, on the assumption that Go keys carry an `oc_`
    prefix and could be told apart from an OpenAI `sk-` key. That premise is
    wrong: Go keys are `sk-` plus 64 characters, and every other major provider
    uses `sk-` too, so the two cannot be distinguished by shape. The rule is
    therefore stated unconditionally -- an explicit environment key wins -- which
    is both predictable and what the documentation promises.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-from-env")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_key = "sk-" + "9" * 64

    interpreter.llm.load()

    assert interpreter.llm.api_key == "sk-from-env"


def test_unrecognised_profile_key_without_env_key_fails_loudly(monkeypatch):
    """No OPENCODE_GO_API_KEY and an unrecognisable profile key: refuse, never send.

    The key is not shaped like any provider credential, so sending it would earn
    the same `Invalid API key.` failure while burying the real problem. Clearing
    it and raising names the actual fix instead.

    Renamed and re-keyed from the earlier version of this test: it treated
    `sk-…` as the foreign shape, which was based on the incorrect belief that Go
    keys do not start with `sk-`. Since they do, `sk-` is now a *recognised*
    shape and the case is covered by
    test_go_key_from_profile_is_kept; what is left to test is a key that is not
    a credential of any recognisable form.
    """
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_key = "not-a-recognised-credential"

    with pytest.raises(ValueError, match="requires an OpenCode Go API key"):
        interpreter.llm.load()
    assert interpreter.llm.api_key is None


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


# --- Key shape, and the Go catalog -----------------------------------------
#
# Both of these came out of reading the published docs rather than the code, and
# both contradicted what the code assumed. See the two docstrings below.


def test_real_go_key_shape_is_recognised():
    """Go keys are `sk-` + 64 chars, so a real key must not be treated as foreign.

    This predicate decides whether a key already in the profile is kept or
    discarded as belonging to another provider. It previously only recognised an
    `oc_` prefix, which OpenCode does not issue: the console's `Key.create`
    produces `sk-` plus 64 characters, and anomalyco/opencode#40343 describes a
    Go key as "sk-... API key from the Zen console". The effect was that a valid
    Go key saved in a profile was discarded and the user was told to set
    OPENCODE_GO_API_KEY instead, so the profile route could never work.
    """
    from interpreter.core.llm.llm import _is_opencode_go_key

    assert _is_opencode_go_key("sk-" + "a" * 64) is True
    # Still accepted, for profiles written while the oc_ assumption held.
    assert _is_opencode_go_key("oc_sk_abc123") is True
    # Only a key that is neither shape is rejected.
    assert _is_opencode_go_key("") is False
    assert _is_opencode_go_key(None) is False


def test_go_key_from_profile_is_kept(monkeypatch):
    """End to end: a real-shaped Go key in a profile survives load.

    This is the path docs/settings/all-settings.mdx tells users to take ("set
    llm.api_key in your profile"), so it has to work without the environment
    variable also being set.
    """
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    key = "sk-" + "b" * 64
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_key = key

    interpreter.llm.load()

    assert interpreter.llm.api_key == key


def test_no_key_at_all_still_fails_loudly(monkeypatch):
    """Widening the accepted key shapes must not turn into a silent default.

    Refusing when there is genuinely no key is the whole point of the check; the
    prefix test only rejects foreign-looking keys, so the empty case is what
    carries the guarantee.
    """
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)
    interpreter.llm.api_key = None

    with pytest.raises(ValueError, match="requires an OpenCode Go API key"):
        interpreter.llm.load()


def test_vision_capable_go_model_is_detected(monkeypatch):
    """A Go model that takes images must not have them turned into text.

    LiteLLM has no entry for the `openai/<go-model>` spelling, so its vision
    probe returns False for every Go model. Without an explicit answer, OI
    renders images to text descriptions up front, so the capability is lost
    silently, with no error to notice.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "k")
    interpreter = _configure("opencode_go/deepseek-v4-flash-vision-exp", monkeypatch)

    interpreter.llm.load()

    assert interpreter.llm.supports_vision is True


def test_non_vision_go_model_is_left_to_auto_detection(monkeypatch):
    """Only the published vision model is hardcoded; others are not guessed.

    Claiming vision for a model that does not have it produces the mirror-image
    failure -- raw image parts sent to a text-only model.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "k")
    interpreter = _configure("opencode_go/deepseek-v4-flash", monkeypatch)

    interpreter.llm.load()

    assert interpreter.llm.supports_vision is None


# The endpoint split below is transcribed from the table in opencode.ai/docs/go
# (retrieved 2026-09-29). The live catalog is GET /zen/go/v1/models; this copy
# exists because OI must refuse an unsupported model before making a request.
MESSAGES_MODELS = {
    "minimax-m2.5",
    "minimax-m2.7",
    "minimax-m3",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-flash",
    "qwen3.8-max",
}

RESPONSES_MODELS = {
    "grok-4.6",
    "grok-4.7",
    "gpt-5.6-luna",
    "gpt-6-luna",
    "muse-spark-1.2-contributor",
    "muse-spark-1.3-contributor",
}

CHAT_COMPLETIONS_MODELS = {
    "glm-5.1",
    "glm-5.2",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "kimi-k3",
    "longcat-2.0",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "mimo-v2.6-flash",
    "mimo-v2.6-pro",
    "hy3",
    "hy4-preview",
    "space-bunny-free",
}


def test_guard_lists_match_the_published_endpoint_table():
    """The three lists must partition the documented catalog exactly.

    An earlier version generated the Qwen ids with a range comprehension, which
    invented models that do not exist (qwen3.5-plus, qwen3.6-max, qwen3.8-plus)
    while the range drifted out of step with the catalog. Explicit sets cannot
    drift that way without a deliberate edit, and this test fails if one of them
    is edited inconsistently with the docs.
    """
    from interpreter.core.llm.llm import (
        _OPENCODE_GO_MESSAGES_MODELS,
        _OPENCODE_GO_RESPONSES_MODELS,
    )

    assert set(_OPENCODE_GO_MESSAGES_MODELS) == MESSAGES_MODELS
    assert set(_OPENCODE_GO_RESPONSES_MODELS) == RESPONSES_MODELS
    # No model may be in two lists, or the refusal would be ambiguous.
    assert not MESSAGES_MODELS & RESPONSES_MODELS
    assert not MESSAGES_MODELS & CHAT_COMPLETIONS_MODELS
    assert not RESPONSES_MODELS & CHAT_COMPLETIONS_MODELS


@pytest.mark.parametrize("model", sorted(CHAT_COMPLETIONS_MODELS))
def test_every_documented_chat_model_loads(model, monkeypatch):
    """Each chat-completions model in the catalog resolves to its openai/ form.

    Parametrised over the whole catalog so a newly documented model cannot be
    added to the guard lists without also being checked here, and so a model that
    is wrongly listed as unsupported fails loudly instead of being refused at
    request time.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "k")
    interpreter = _configure(f"opencode_go/{model}", monkeypatch)

    interpreter.llm.load()

    assert interpreter.llm.model == f"openai/{model}"
