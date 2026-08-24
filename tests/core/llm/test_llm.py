from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter import OpenInterpreter
from interpreter.core.llm.llm import Llm, SuppressDebugFilter
from tests.helpers import TEST_LLM_MODEL


@pytest.fixture(autouse=True)
def _isolate_litellm_drop_params():
    """fixed_litellm_completions mutates the process-global litellm.drop_params;
    restore it after each test so it can't leak between tests."""
    import litellm

    original = getattr(litellm, "drop_params", None)
    yield
    litellm.drop_params = original

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
    """SuppressDebugFilter hides LiteLLM cost-map debug noise from logs."""
    filt = SuppressDebugFilter()
    record = mock.Mock()
    record.getMessage.return_value = "loading cost map data"
    assert filt.filter(record) is False
    record.getMessage.return_value = "normal log line"
    assert filt.filter(record) is True


def test_llm_clamps_max_tokens_to_context_window():
    """max_tokens is capped to context_window // 2 before calling the API."""
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
    """llm.run rejects message lists that do not start with a system message."""
    interpreter = SimpleNamespace(
        shrink_images=True,
        display_message=mock.Mock(),
        computer=SimpleNamespace(vision=SimpleNamespace(query=mock.Mock())),
    )
    llm = Llm(interpreter)
    with pytest.raises(AssertionError, match="system"):
        list(llm.run([{"role": "user", "type": "message", "content": "hi"}]))


def _interp(**attrs):
    interp = SimpleNamespace(
        shrink_images=True,
        os=False,
        verbose=False,
        debug=False,
        display_message=mock.Mock(),
        computer=SimpleNamespace(
            vision=SimpleNamespace(
                query=mock.Mock(return_value="description"),
                ocr=mock.Mock(return_value="ocr text"),
            ),
            import_computer_api=True,
        ),
    )
    for name, value in attrs.items():
        setattr(interp, name, value)
    return interp


def _run_capture(llm, messages, function_calling=True):
    """Run llm.run() with the LLM plumbing stubbed out and return a dict whose
    "params" key holds the params that would be sent to the underlying runner."""
    target = "run_tool_calling_llm" if function_calling else "run_text_llm"
    captured = {}

    def fake_run(llm_obj, params):
        captured["params"] = params
        return iter(())

    with mock.patch(f"interpreter.core.llm.llm.{target}", side_effect=fake_run):
        with mock.patch(
            "interpreter.core.llm.llm.convert_to_openai_messages",
            side_effect=lambda msgs, **kwargs: msgs,
        ):
            with mock.patch(
                "interpreter.core.llm.llm.tt.trim", side_effect=lambda msgs, **kwargs: msgs
            ):
                list(llm.run(messages))
    return captured


def test_run_remaps_claude_35_model():
    """llm.run rewrites legacy claude-3.5 model names before calling the API."""
    llm = Llm(_interp())
    llm.model = "claude-3.5"
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False

    captured = _run_capture(
        llm,
        [
            {"role": "system", "type": "message", "content": "s"},
            {"role": "user", "type": "message", "content": "hi"},
        ],
    )

    assert captured["params"]["model"] == "claude-sonnet-4-6"
    assert llm.model == "claude-sonnet-4-6"


def test_run_dispatches_to_text_runner_without_functions():
    """llm.run uses the text runner when the model doesn't support functions."""
    llm = Llm(_interp())
    llm._is_loaded = True
    llm.supports_functions = False
    llm.supports_vision = False

    captured = _run_capture(
        llm,
        [
            {"role": "system", "type": "message", "content": "s"},
            {"role": "user", "type": "message", "content": "hi"},
        ],
        function_calling=False,
    )

    assert captured["params"]["stream"] is True
    assert captured["params"]["messages"][0]["content"] == "hi"


def test_run_includes_optional_llm_params():
    """llm.run forwards api_key/api_base/api_version/max_tokens/temperature and
    the conversation_id when they're set."""
    interp = _interp()
    interp.conversation_id = "conv-1"
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.api_key = "key"
    llm.api_base = "http://x"
    llm.api_version = "2024"
    llm.max_tokens = 100
    llm.temperature = 0.5

    captured = _run_capture(
        llm,
        [
            {"role": "system", "type": "message", "content": "s"},
            {"role": "user", "type": "message", "content": "hi"},
        ],
    )

    params = captured["params"]
    assert params["api_key"] == "key"
    assert params["api_base"] == "http://x"
    assert params["api_version"] == "2024"
    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.5
    assert params["conversation_id"] == "conv-1"


def test_run_auto_detects_function_support():
    """llm.run queries litellm to decide function support when unset."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = None
    llm.supports_vision = False

    with mock.patch(
        "interpreter.core.llm.llm.litellm.supports_function_calling",
        return_value=True,
    ):
        captured = _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
        )

    assert llm.supports_functions is True
    assert "messages" in captured["params"]


def test_run_trims_images_in_os_mode_to_last_two():
    """In OS mode llm.run keeps only the last two image messages."""
    interp = _interp(os=True, verbose=True)
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = True
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        {"role": "user", "type": "image", "content": "0"},
        {"role": "user", "type": "image", "content": "1"},
        {"role": "user", "type": "image", "content": "2"},
    ]

    _run_capture(llm, messages)

    remaining = [m["content"] for m in messages if m["type"] == "image"]
    assert remaining == ["1", "2"]


def test_run_trims_middle_images_outside_os_mode():
    """llm.run keeps the first and last two image messages, dropping the middle."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = True
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        {"role": "user", "type": "image", "content": "0"},
        {"role": "user", "type": "image", "content": "1"},
        {"role": "user", "type": "image", "content": "2"},
        {"role": "user", "type": "image", "content": "3"},
        {"role": "user", "type": "image", "content": "4"},
    ]

    _run_capture(llm, messages)

    remaining = [m["content"] for m in messages if m["type"] == "image"]
    assert remaining == ["0", "3", "4"]


def test_run_renders_path_image_for_non_vision_model():
    """Without vision support, llm.run replaces a path image with a text
    description from the vision renderer plus OCR."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    img_msg = {
        "role": "user",
        "type": "image",
        "format": "path",
        "content": "/tmp/x.png",
    }
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        img_msg,
    ]

    _run_capture(llm, messages)

    assert img_msg["format"] == "description"
    assert "description" in img_msg["content"]
    assert "ocr text" in img_msg["content"]
    assert "The image I'm referring to" in img_msg["content"]
    interp.display_message.assert_any_call("\n  *Viewing image...*\n")


def test_run_import_error_on_vision_blanks_image():
    """llm.run blanks the image content when the vision renderer exists but
    raises ImportError."""
    interp = _interp()
    interp.computer.vision.query.side_effect = ImportError
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    img_msg = {"role": "user", "type": "image", "format": "path", "content": "/tmp/x.png"}
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        img_msg,
    ]

    _run_capture(llm, messages)

    assert img_msg["format"] == "description"
    assert img_msg["content"] == ""


def test_model_setter_resets_load_state():
    """Assigning a new model marks the LLM as needing a reload."""
    llm = Llm(_interp())
    llm._is_loaded = True
    llm.model = "gpt-4o"
    assert llm._is_loaded is False


def test_load_fetches_context_window_from_litellm():
    """llm.load fills in context_window/max_tokens from litellm's model info."""
    interp = _interp()
    llm = Llm(interp)
    llm.model = "gpt-4o"
    llm.context_window = None
    llm.max_tokens = None
    with mock.patch(
        "interpreter.core.llm.llm.litellm.get_model_info",
        return_value={"max_input_tokens": 8000, "max_output_tokens": 4000},
    ):
        llm.load()

    assert llm.context_window == 8000
    assert llm.max_tokens == 1600


def test_load_ollama_downloads_and_pings_model():
    """llm.load for an ollama model pulls it if missing, reads its context
    window, and pings it to force a load."""
    interp = _interp()
    interp.computer.ai = SimpleNamespace(chat=mock.Mock())
    llm = Llm(interp)
    llm.model = "ollama/llama3"
    llm.api_base = "http://ollama:11434"

    tags_response = mock.Mock()
    tags_response.ok = True
    tags_response.json.return_value = {"models": []}  # llama3 not downloaded
    show_response = mock.Mock()
    show_response.json.return_value = {"model_info": {"llama3.context_length": 8192}}

    with mock.patch("interpreter.core.llm.llm.requests.get", return_value=tags_response):
        with mock.patch(
            "interpreter.core.llm.llm.requests.post",
            side_effect=[mock.Mock(), show_response],
        ) as post:
            llm.load()

    post.assert_any_call(
        "http://ollama:11434/api/pull", json={"name": "llama3:latest"}
    )
    assert llm.context_window == 8192
    assert llm.max_tokens == 1638
    interp.computer.ai.chat.assert_called_once_with("ping")
    interp.display_message.assert_any_call("*Model loaded.*\n")


def test_load_ollama_exits_when_ollama_unreachable():
    """llm.load exits (SystemExit) with a download hint when ollama isn't
    reachable."""
    interp = _interp()
    llm = Llm(interp)
    llm.model = "ollama/llama3"
    with mock.patch(
        "interpreter.core.llm.llm.requests.get", side_effect=Exception("no ollama")
    ):
        with mock.patch("builtins.exit", side_effect=SystemExit):
            with pytest.raises(SystemExit):
                llm.load()

    assert "Ollama not found" in interp.display_message.call_args[0][0]
    assert "ollama.com" in interp.display_message.call_args[0][0]


def test_fixed_litellm_completions_retries_then_raises_first_error():
    """fixed_litellm_completions retries all attempts and re-raises the first error."""
    from interpreter.core.llm.llm import fixed_litellm_completions

    with mock.patch(
        "interpreter.core.llm.llm.litellm.completion",
        side_effect=[RuntimeError(f"failure {i}") for i in range(4)],
    ) as completion:
        with pytest.raises(RuntimeError, match="failure 0"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[]))

    assert completion.call_count == 4


def test_fixed_litellm_completions_uses_dummy_key_on_auth_error():
    """A missing API key triggers one retry with a dummy key."""
    from interpreter.core.llm.llm import fixed_litellm_completions

    auth_error = type("AuthenticationError", (Exception,), {})
    calls = []

    def fake_completion(**params):
        calls.append(params)
        if len(calls) == 1:
            raise auth_error("no key")
        return iter(())

    with mock.patch("interpreter.core.llm.llm.litellm.completion", side_effect=fake_completion):
        with mock.patch(
            "interpreter.core.llm.llm.litellm.exceptions.AuthenticationError", auth_error
        ):
            list(fixed_litellm_completions(model="gpt-4o", messages=[]))

    assert "api_key" not in calls[0]
    assert calls[1]["api_key"] == "x"
    assert len(calls) == 2


def test_fixed_litellm_completions_strips_latest_and_sets_stop_for_local():
    """Local models get stop tokens and :latest suffixes are stripped."""
    from interpreter.core.llm.llm import fixed_litellm_completions

    captured = {}
    with mock.patch(
        "interpreter.core.llm.llm.litellm.completion",
        side_effect=lambda **params: (captured.update(params), iter(()))[1],
    ):
        list(fixed_litellm_completions(model="local-llama3:latest", messages=[], stop=None))

    assert captured["model"] == "local-llama3"
    assert captured["stop"] == ["<|assistant|>", "<|end|>", "<|eot_id|>"]


def test_run_model_i_sets_open_interpreter_endpoint():
    """llm.run remaps model `i` to openai/i and configures the Open Interpreter endpoint."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.model = "i"

    captured = _run_capture(
        llm,
        [
            {"role": "system", "type": "message", "content": "s"},
            {"role": "user", "type": "message", "content": "hi"},
        ],
    )

    assert captured["params"]["model"] == "openai/i"
    assert llm.context_window == 7000
    assert llm.api_key == "x"
    assert llm.max_tokens == 1000
    assert llm.api_base == "https://api.openinterpreter.com/v0"


def test_run_auto_detects_vision_support_true():
    """llm.run queries litellm to decide vision support when unset (true)."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = None

    with mock.patch("interpreter.core.llm.llm.litellm.supports_vision", return_value=True):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
        )

    assert llm.supports_vision is True


def test_run_auto_detects_vision_support_false():
    """llm.run queries litellm to decide vision support when unset (false)."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = None

    with mock.patch("interpreter.core.llm.llm.litellm.supports_vision", return_value=False):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
        )

    assert llm.supports_vision is False


def test_run_auto_detect_vision_falls_back_on_error():
    """llm.run defaults supports_vision to False when litellm's check raises."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = None

    with mock.patch(
        "interpreter.core.llm.llm.litellm.supports_vision", side_effect=Exception
    ):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
        )

    assert llm.supports_vision is False


def test_run_auto_detect_function_support_falls_back_on_error():
    """llm.run defaults supports_functions to False when litellm's check raises."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = None
    llm.supports_vision = False

    with mock.patch(
        "interpreter.core.llm.llm.litellm.supports_function_calling", side_effect=Exception
    ):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
            function_calling=False,
        )

    assert llm.supports_functions is False


def test_run_auto_detect_function_support_false():
    """llm.run sets supports_functions False when litellm says the model can't."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = None
    llm.supports_vision = False

    with mock.patch(
        "interpreter.core.llm.llm.litellm.supports_function_calling", return_value=False
    ):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
            function_calling=False,
        )

    assert llm.supports_functions is False


def test_run_trims_to_context_window_when_max_tokens_unset():
    """With max_tokens unset, llm.run trims to the full context window."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.context_window = 2000
    llm.max_tokens = None

    trim_kwargs = {}
    with mock.patch(
        "interpreter.core.llm.llm.tt.trim",
        side_effect=lambda msgs, **kwargs: (trim_kwargs.update(kwargs), msgs)[1],
    ):
        with mock.patch(
            "interpreter.core.llm.llm.run_tool_calling_llm", return_value=iter(())
        ):
            with mock.patch(
                "interpreter.core.llm.llm.convert_to_openai_messages",
                side_effect=lambda msgs, **kwargs: msgs,
            ):
                list(
                    llm.run(
                        [
                            {"role": "system", "type": "message", "content": "s"},
                            {"role": "user", "type": "message", "content": "hi"},
                        ]
                    )
                )

    assert trim_kwargs["max_tokens"] == 2000


def test_run_shows_terminal_hint_when_trim_unknown_model():
    """llm.run falls back to 8000 tokens and shows a hint when the model is unknown."""
    interp = _interp(in_terminal_interface=True)
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.context_window = None
    llm.max_tokens = None

    trim_calls = {"count": 0}

    def failing_trim(*args, **kwargs):
        trim_calls["count"] += 1
        if trim_calls["count"] == 1:
            raise Exception("unknown model")
        return [{"role": "user", "content": "hi"}]

    captured = {}
    with mock.patch(
        "interpreter.core.llm.llm.run_tool_calling_llm",
        side_effect=lambda llm_obj, params: (captured.update({"params": params}), iter(()))[1],
    ):
        with mock.patch(
            "interpreter.core.llm.llm.convert_to_openai_messages",
            side_effect=lambda msgs, **kwargs: msgs,
        ):
            with mock.patch("interpreter.core.llm.llm.tt.trim", side_effect=failing_trim):
                list(
                    llm.run(
                        [
                            {"role": "system", "type": "message", "content": "s"},
                            {"role": "user", "type": "message", "content": "hi"},
                        ]
                    )
                )

    assert "We were unable to determine the context window" in interp.display_message.call_args[0][0]
    assert "interpreter --context_window" in interp.display_message.call_args[0][0]


def test_run_shows_python_hint_when_trim_unknown_model_not_terminal():
    """The non-terminal fallback hint mentions self.context_window instead."""
    interp = _interp(in_terminal_interface=False)
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.context_window = None
    llm.max_tokens = None

    trim_calls = {"count": 0}

    def failing_trim(*args, **kwargs):
        trim_calls["count"] += 1
        if trim_calls["count"] == 1:
            raise Exception("unknown model")
        return [{"role": "user", "content": "hi"}]

    with mock.patch(
        "interpreter.core.llm.llm.run_tool_calling_llm",
        return_value=iter(()),
    ):
        with mock.patch(
            "interpreter.core.llm.llm.convert_to_openai_messages",
            side_effect=lambda msgs, **kwargs: msgs,
        ):
            with mock.patch("interpreter.core.llm.llm.tt.trim", side_effect=failing_trim):
                list(
                    llm.run(
                        [
                            {"role": "system", "type": "message", "content": "s"},
                            {"role": "user", "type": "message", "content": "hi"},
                        ]
                    )
                )

    assert "We were unable to determine the context window" in interp.display_message.call_args[0][0]
    assert "self.context_window" in interp.display_message.call_args[0][0]


def test_run_reunites_system_message_when_trim_always_fails():
    """When all trimming fails, llm.run reunites the system message with the messages."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.context_window = 2000
    llm.max_tokens = 1000

    def always_fail(*args, **kwargs):
        raise Exception("trim always fails")

    captured = {}
    with mock.patch(
        "interpreter.core.llm.llm.run_tool_calling_llm",
        side_effect=lambda llm_obj, params: (captured.update({"params": params}), iter(()))[1],
    ):
        with mock.patch(
            "interpreter.core.llm.llm.convert_to_openai_messages",
            side_effect=lambda msgs, **kwargs: msgs,
        ):
            with mock.patch("interpreter.core.llm.llm.tt.trim", side_effect=always_fail):
                list(
                    llm.run(
                        [
                            {"role": "system", "type": "message", "content": "s"},
                            {"role": "user", "type": "message", "content": "hi"},
                        ]
                    )
                )

    assert captured["params"]["messages"][0] == {"role": "system", "content": "s"}


def test_run_sets_litellm_max_budget():
    """llm.run forwards max_budget to the litellm global budget manager."""
    import interpreter.core.llm.llm as llm_mod

    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    llm.max_budget = 10.0

    with mock.patch.object(llm_mod.litellm, "max_budget", None):
        _run_capture(
            llm,
            [
                {"role": "system", "type": "message", "content": "s"},
                {"role": "user", "type": "message", "content": "hi"},
            ],
        )
        assert llm_mod.litellm.max_budget == 10.0


def test_load_returns_early_when_already_loaded():
    """llm.load is a no-op when the model is already loaded."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.load()
    assert llm._is_loaded is True


def test_load_swallows_get_model_info_error():
    """llm.load leaves context_window None when litellm can't identify the model."""
    interp = _interp()
    llm = Llm(interp)
    llm.model = "unknown-model"
    llm.context_window = None
    with mock.patch(
        "interpreter.core.llm.llm.litellm.get_model_info", side_effect=Exception
    ):
        llm.load()
    assert llm.context_window is None


def test_fixed_litellm_completions_keeps_drop_params_for_i():
    """For the `i` model with a conversation_id, litellm.drop_params must stay False."""
    import interpreter.core.llm.llm as llm_mod
    from interpreter.core.llm.llm import fixed_litellm_completions

    with mock.patch.object(
        llm_mod.litellm, "completion", side_effect=lambda **params: iter(())
    ):
        list(fixed_litellm_completions(model="i", conversation_id="c1", messages=[]))
    assert llm_mod.litellm.drop_params is False


def test_fixed_litellm_completions_exits_on_keyboard_interrupt():
    """A KeyboardInterrupt during completion calls sys.exit(0)."""
    import interpreter.core.llm.llm as llm_mod
    from interpreter.core.llm.llm import fixed_litellm_completions

    with mock.patch.object(llm_mod.litellm, "completion", side_effect=KeyboardInterrupt):
        with mock.patch.object(
            llm_mod.sys, "exit", side_effect=SystemExit
        ) as exit_mock:
            with pytest.raises(SystemExit):
                list(fixed_litellm_completions(model="gpt-4o", messages=[]))
    # The first KeyboardInterrupt terminates the generator via sys.exit(0);
    # assert it was called exactly once, so retries don't continue silently.
    exit_mock.assert_called_once_with(0)


def test_run_verbose_logs_image_removal():
    """In verbose mode llm.run logs each removed middle image."""
    interp = _interp(verbose=True)
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = True
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        {"role": "user", "type": "image", "content": "0"},
        {"role": "user", "type": "image", "content": "1"},
        {"role": "user", "type": "image", "content": "2"},
        {"role": "user", "type": "image", "content": "3"},
    ]

    with mock.patch("interpreter.core.llm.llm.print") as print_mock:
        _run_capture(llm, messages)

    print_mock.assert_any_call("Removing image message!")


def test_run_path_image_without_import_api_has_no_postcursor():
    """Without the computer API, a path image gets no vision-query postcursor."""
    interp = _interp()
    interp.computer.import_computer_api = False
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    img_msg = {"role": "user", "type": "image", "format": "path", "content": "/tmp/x.png"}
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        img_msg,
    ]

    _run_capture(llm, messages)

    assert img_msg["format"] == "description"
    assert "The image I'm referring to" in img_msg["content"]
    assert "computer.vision.query" not in img_msg["content"]


def test_run_base64_image_uses_imagine_precursor():
    """Non-path images get the 'Imagine...' precursor when vision isn't supported."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    img_msg = {
        "role": "user",
        "type": "image",
        "format": "base64",
        "content": "data:image/png;base64,xxx",
    }
    messages = [
        {"role": "system", "type": "message", "content": "s"},
        img_msg,
    ]

    _run_capture(llm, messages)

    assert "Imagine I have just shown you an image" in img_msg["content"]


def test_run_prepends_empty_system_message():
    """llm.run re-adds an empty system message to keep the system role first."""
    interp = _interp()
    llm = Llm(interp)
    llm._is_loaded = True
    llm.supports_functions = True
    llm.supports_vision = False
    messages = [
        {"role": "system", "type": "message", "content": ""},
        {"role": "user", "type": "message", "content": "hi"},
    ]

    captured = _run_capture(llm, messages)

    assert captured["params"]["messages"][0]["role"] == "system"
    assert captured["params"]["messages"][0]["content"] == ""