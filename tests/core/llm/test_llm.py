import pytest
from types import SimpleNamespace
from unittest import mock

from interpreter import OpenInterpreter
from interpreter.core.llm.llm import Llm, SuppressDebugFilter

from tests.conftest import TEST_LLM_MODEL

_MESSAGES = [
    {"role": "system", "type": "message", "content": "system"},
    {"role": "user", "type": "message", "content": "hello"},
]


def _capture_llm_params(interpreter, temperature):
    interpreter.llm.temperature = temperature
    interpreter.llm.supports_functions = True
    interpreter.llm.supports_vision = False
    interpreter.llm._is_loaded = True
    interpreter.llm.model = TEST_LLM_MODEL

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
    """Unset temperature must not be sent; API should use its own default."""
    params = _capture_llm_params(OpenInterpreter(), None)
    assert "temperature" not in params


def test_suppress_debug_filter_blocks_cost_map_messages():
    filt = SuppressDebugFilter()
    record = mock.Mock()
    record.getMessage.return_value = "loading cost map data"
    assert filt.filter(record) is False
    record.getMessage.return_value = "normal log line"
    assert filt.filter(record) is True


def test_llm_clamps_max_tokens_to_context_window():
    interpreter = SimpleNamespace(
        shrink_images=True,
        display_message=mock.Mock(),
        computer=SimpleNamespace(vision=SimpleNamespace(query=mock.Mock())),
        os=False,
        verbose=False,
        debug=False,
    )
    llm = Llm(interpreter)
    llm._is_loaded = True
    llm.context_window = 1000
    llm.max_tokens = 5000
    llm.supports_functions = False
    llm.supports_vision = False

    messages = [{"role": "system", "type": "message", "content": "sys"}]

    with mock.patch.object(llm, "load"):
        with mock.patch(
            "interpreter.core.llm.llm.convert_to_openai_messages",
            return_value=[{"role": "system", "content": "sys"}],
        ):
            with mock.patch("interpreter.core.llm.llm.tt.trim", return_value=([], {})):
                with mock.patch(
                    "interpreter.core.llm.llm.run_text_llm", return_value=iter([])
                ):
                    list(llm.run(messages))
    assert llm.max_tokens == 200


def test_llm_run_requires_system_first():
    interpreter = SimpleNamespace(
        shrink_images=True,
        display_message=mock.Mock(),
        computer=SimpleNamespace(vision=SimpleNamespace(query=mock.Mock())),
    )
    llm = Llm(interpreter)
    with pytest.raises(AssertionError, match="system"):
        list(llm.run([{"role": "user", "type": "message", "content": "hi"}]))
