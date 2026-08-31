import pytest

import interpreter.core.llm.llm as llm_mod
from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles import profiles


@pytest.fixture
def capture_text_params(monkeypatch):
    """Capture the params run() hands to the streaming backend."""

    captured = {}

    def fake_run_text_llm(self, params):
        captured["params"] = params
        return iter(
            [("message", {"role": "assistant", "type": "message", "content": "stubbed"})]
        )

    monkeypatch.setattr(llm_mod, "run_text_llm", fake_run_text_llm)
    return captured


@pytest.fixture
def stub_openrouter_entry(monkeypatch):
    """Stub the OpenRouter model-metadata lookup and reset per-process warning state.

    _openrouter_model_entry normally hits https://openrouter.ai/api/v1/models,
    which would make these tests slow and network-dependent. Returns a controller
    so each test can set the entry it needs (e.g. reasoning.mandatory for the
    GLM-style endpoints that reject reasoning disabling).
    """

    state = {"entry": None}

    def fake_entry(self, model):
        if not model.lower().startswith("openrouter/"):
            return None
        return state["entry"]

    monkeypatch.setattr(llm_mod.Llm, "_openrouter_model_entry", fake_entry)
    monkeypatch.setattr(llm_mod, "_openrouter_model_entries", {})
    monkeypatch.setattr(llm_mod, "_warned_mandatory_reasoning", set())
    monkeypatch.setattr(llm_mod, "_warned_unsupported_effort", set())
    return state


def _run_one_turn(interpreter):
    messages = [
        {"role": "system", "type": "message", "content": "You are helpful."},
        {"role": "user", "type": "message", "content": "hi"},
    ]
    next(interpreter.llm.run(messages))


def test_profile_reasoning_effort_flows_to_request(
    capture_text_params, stub_openrouter_entry
):
    """A profile's llm.reasoning_effort reaches the outgoing request.

    The profile YAML is the user-facing way to tune how hard reasoning models
    think. apply_profile must copy llm.reasoning_effort onto interpreter.llm and
    run() must forward it in the request: top-level for LiteLLM's mapping, and
    inside extra_body.reasoning.effort for OpenRouter. Without this a
    "reasoning_effort: low" profile silently does nothing and the model keeps
    thinking at its default (often high) effort.
    """
    interpreter = OpenInterpreter()
    interpreter.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/deepseek/deepseek-v4-flash-latest",
            "reasoning_effort": "low",
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert params["reasoning_effort"] == "low"
    assert params["extra_body"]["reasoning"]["effort"] == "low"


def test_profile_include_reasoning_false_disables_effort(
    capture_text_params, stub_openrouter_entry
):
    """include_reasoning: false disables reasoning and drops any reasoning_effort.

    Turning reasoning off is the profile-level escape hatch for models that think
    endlessly. For models where reasoning CAN be disabled, the request must carry
    include_reasoning=false AND suppress a reasoning_effort that was also set in
    the profile, since sending an effort alongside "enabled": false is
    contradictory and some backends reject it.
    """
    interpreter = OpenInterpreter()
    interpreter.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/deepseek/deepseek-v4-flash-latest",
            "include_reasoning": False,
            "reasoning_effort": "low",
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert params["include_reasoning"] is False
    assert params["extra_body"]["include_reasoning"] is False
    assert params["extra_body"]["reasoning"]["enabled"] is False
    assert "reasoning_effort" not in params
    assert "effort" not in params["extra_body"]["reasoning"]


def test_mandatory_reasoning_ignores_include_reasoning_false(
    capture_text_params, stub_openrouter_entry
):
    """Endpoints with mandatory reasoning never receive reasoning.enabled:false.

    GLM-5.3-family endpoints (z-ai/glm-5.3-flash and friends) report
    reasoning.mandatory=true on OpenRouter's model list and reject any request
    that tries to disable reasoning with a 400 ("Reasoning is mandatory for this
    endpoint and cannot be disabled"). The request builder must detect the
    mandatory flag and omit the disable (plus the effort, which is also dropped
    when disabling), falling back to the model's default reasoning instead of
    failing the turn.
    """
    stub_openrouter_entry["entry"] = {
        "id": "z-ai/glm-5.3-flash",
        "reasoning": {
            "mandatory": True,
            "default_enabled": True,
            "supported_efforts": ["max", "high", "low"],
            "default_effort": "max",
        },
    }

    interpreter = OpenInterpreter()
    interpreter.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/z-ai/glm-5.3-flash",
            "include_reasoning": False,
            "reasoning_effort": "low",
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert "include_reasoning" not in params
    assert "reasoning_effort" not in params
    extra_body = params.get("extra_body") or {}
    assert extra_body.get("include_reasoning") is None
    reasoning = extra_body.get("reasoning")
    assert reasoning is None or reasoning.get("enabled") is not False


def test_mandatory_reasoning_still_sends_supported_effort(
    capture_text_params, stub_openrouter_entry
):
    """On a mandatory-reasoning endpoint, a supported effort is still forwarded.

    Reasoning cannot be disabled on GLM-5.3-family endpoints, but effort CAN be
    lowered (low is the cheapest supported level and the documented default is
    max). With include_reasoning left unset (None, the default), the request must
    carry reasoning: {effort: "low"} so the user's "think less" intent is honored.
    """
    stub_openrouter_entry["entry"] = {
        "id": "z-ai/glm-5.3-flash",
        "reasoning": {
            "mandatory": True,
            "default_enabled": True,
            "supported_efforts": ["max", "high", "low"],
            "default_effort": "max",
        },
    }

    interpreter = OpenInterpreter()
    interpreter.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/z-ai/glm-5.3-flash",
            "reasoning_effort": "low",
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert params["reasoning_effort"] == "low"
    assert params["extra_body"]["reasoning"]["effort"] == "low"
    assert "include_reasoning" not in params


def test_unsupported_effort_dropped_with_warning(
    capture_text_params, stub_openrouter_entry
):
    """An effort level the model doesn't support is dropped, not sent.

    GLM-5.3-family endpoints only accept low/high/max and 400 on anything else
    (OpenRouter's error for "medium" is the misleading "reasoning cannot be
    disabled"). The request builder must consult supported_efforts and skip the
    unsupported value rather than fail the turn.
    """
    stub_openrouter_entry["entry"] = {
        "id": "z-ai/glm-5.3-flash",
        "reasoning": {
            "mandatory": True,
            "default_enabled": True,
            "supported_efforts": ["max", "high", "low"],
            "default_effort": "max",
        },
    }

    interpreter = OpenInterpreter()
    interpreter.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/z-ai/glm-5.3-flash",
            "reasoning_effort": "medium",
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert "reasoning_effort" not in params
    extra_body = params.get("extra_body") or {}
    assert extra_body.get("reasoning") is None or (
        "effort" not in extra_body["reasoning"]
    )


def test_profile_reasoning_validation_accepts_reasoning_effort(capfd):
    """_validate_profile does not warn about llm.reasoning_effort.

    Both reasoning_effort and include_reasoning are real attributes on the Llm
    class, so a profile setting them must not trigger the "attribute doesn't exist
    ... setting ignored" warning that fires for misspelled/unknown keys.
    """
    interpreter = OpenInterpreter()
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "gpt-4.1",
            "reasoning_effort": "low",
            "include_reasoning": True,
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")

    _, err = capfd.readouterr()
    assert "doesn't exist" not in err
    assert interpreter.llm.reasoning_effort == "low"
    assert interpreter.llm.include_reasoning is True
