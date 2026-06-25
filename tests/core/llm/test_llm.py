from unittest import mock

from interpreter import OpenInterpreter

_MESSAGES = [
    {"role": "system", "type": "message", "content": "system"},
    {"role": "user", "type": "message", "content": "hello"},
]


def _capture_llm_params(interpreter, temperature):
    interpreter.llm.temperature = temperature
    interpreter.llm.supports_functions = True
    interpreter.llm.supports_vision = False
    interpreter.llm._is_loaded = True
    interpreter.llm.model = "gpt-4o-mini"

    captured = {}

    def capture_params(llm, params):
        captured["params"] = params
        return iter(())

    with mock.patch(
        "interpreter.core.llm.llm.run_tool_calling_llm", side_effect=capture_params
    ):
        list(interpreter.llm.run(_MESSAGES))

    return captured["params"]


def test_temperature_zero_is_sent_to_llm_api():
    """temperature=0.0 must reach the API; `if self.temperature:` skipped it (falsy)."""

    params = _capture_llm_params(OpenInterpreter(), 0.0)
    assert params["temperature"] == 0.0


def test_temperature_none_omitted_from_llm_api():
    params = _capture_llm_params(OpenInterpreter(), None)
    assert "temperature" not in params
