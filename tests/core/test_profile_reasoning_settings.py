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
    monkeypatch.setattr(llm_mod, "_warned_effort_without_reasoning", set())
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
    interpreter.llm.supports_functions = False
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


def test_include_reasoning_false_still_sends_effort_and_notes_it(
    capture_text_params, stub_openrouter_entry, capfd
):
    """include_reasoning: false still sends a configured effort, with a note.

    A model told not to think cannot honor an effort level, and providers ignore
    the effort rather than erroring (verified against OpenRouter/DeepSeek:
    reasoning {enabled: false, effort: "low"} returns 200 with no reasoning). We
    pass the configured values through unchanged rather than silently stripping the
    effort, and print a one-time note so the user is not surprised it did nothing.
    """
    interpreter = OpenInterpreter()
    interpreter.llm.supports_functions = False
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
    # The configured effort is passed through, not stripped ...
    assert params["reasoning_effort"] == "low"
    assert params["extra_body"]["reasoning"]["effort"] == "low"
    # ... and the user is told it will not take effect.
    out, _ = capfd.readouterr()
    assert "reasoning_effort" in out


def test_include_reasoning_false_is_passed_through(
    capture_text_params, stub_openrouter_entry
):
    """reasoning.enabled:false is sent as-is, even for a mandatory-reasoning model.

    Some OpenRouter endpoints report reasoning.mandatory=true and reject a
    disable with a 400. We deliberately do not special-case that from model
    metadata: the value is passed through and the provider's error reaches the
    user like any other, rather than being silently swallowed here. Model
    metadata also changes over time, so pre-filtering on it is brittle.
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
    interpreter.llm.supports_functions = False
    interpreter.llm.supports_vision = False
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "openrouter/z-ai/glm-5.3-flash",
            "include_reasoning": False,
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")
    _run_one_turn(interpreter)

    params = capture_text_params["params"]
    assert params["include_reasoning"] is False
    assert params["extra_body"]["include_reasoning"] is False
    assert params["extra_body"]["reasoning"]["enabled"] is False


def test_unsupported_effort_is_passed_through_not_stripped(
    capture_text_params, stub_openrouter_entry
):
    """An effort outside the model's advertised set is still sent, not dropped.

    OpenRouter advertises a restricted set of effort levels per model (GLM-5.3:
    low/high/max), but we deliberately do not pre-filter against it: the set can
    change over time, and OpenRouter remaps or rejects the value server-side.
    Passing it through means the user's setting reaches the provider and any
    rejection surfaces as a normal error rather than being silently ignored.
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
    interpreter.llm.supports_functions = False
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
    assert params["reasoning_effort"] == "medium"
    assert params["extra_body"]["reasoning"]["effort"] == "medium"


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
