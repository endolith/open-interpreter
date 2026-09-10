import json
import os

import litellm

import interpreter.core.llm.llm as llm_mod
from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles.profiles import apply_profile_to_object


class _FakeChunk:
    """Stand-in for a litellm streaming chunk; only model_dump() is needed."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_request_and_response_dumps_correlate_by_request_id(monkeypatch, tmp_path):
    """Opt-in logging writes the outgoing request and the streamed response, paired by id.

    Debugging the DeepSeek reasoning loop requires seeing both sides of each
    call: the exact wire messages (to confirm no consecutive assistant turns)
    and the raw model output (to see whether it emitted a tool call or just
    narrated). The two JSONL files must therefore share a request_id.
    """
    monkeypatch.setattr(llm_mod, "get_storage_path", lambda sub=None: str(tmp_path / (sub or "")))
    monkeypatch.setenv("OI_LOG_LITELLM_REQUESTS", "1")

    def fake_completion(**params):
        yield _FakeChunk({"choices": [{"delta": {"reasoning_content": "plan"}, "finish_reason": None}]})
        yield _FakeChunk({"choices": [{"delta": {"content": "I'll check."}, "finish_reason": None}]})
        yield _FakeChunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})

    monkeypatch.setattr(litellm, "completion", fake_completion)

    list(
        llm_mod.fixed_litellm_completions(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "do it"}],
        )
    )

    logs = tmp_path / "logs"
    requests = _read_jsonl(logs / "litellm_requests.jsonl")
    responses = _read_jsonl(logs / "litellm_responses.jsonl")

    assert len(requests) == 1
    assert len(responses) == 1
    assert requests[0]["request_id"] == responses[0]["request_id"]
    assert requests[0]["messages"] == [{"role": "user", "content": "do it"}]

    # The raw stream is captured verbatim for inspection.
    assert responses[0]["chunk_count"] == 3
    assert responses[0]["chunks"][0]["choices"][0]["delta"]["reasoning_content"] == "plan"
    assert responses[0]["chunks"][2]["choices"][0]["finish_reason"] == "stop"


def test_logging_is_silent_when_disabled(monkeypatch, tmp_path):
    """Without OI_LOG_LITELLM_REQUESTS=1 no debug files are written and nothing changes.

    Logging must be strictly opt-in so normal runs neither pay the cost nor leak
    prompt contents to disk.
    """
    monkeypatch.setattr(llm_mod, "get_storage_path", lambda sub=None: str(tmp_path / (sub or "")))
    monkeypatch.delenv("OI_LOG_LITELLM_REQUESTS", raising=False)

    def fake_completion(**params):
        yield _FakeChunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(litellm, "completion", fake_completion)

    list(
        llm_mod.fixed_litellm_completions(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    logs = tmp_path / "logs"
    assert not (logs / "litellm_requests.jsonl").exists()
    assert not (logs / "litellm_responses.jsonl").exists()


def test_param_flag_enables_dumping_without_env_var(monkeypatch, tmp_path):
    """The params flag (from llm.log_litellm_requests) enables dumping with no env var.

    The profile setting must work on its own for users who never touch the
    environment; fixed_litellm_completions reads the forwarded flag and strips it
    so it never reaches the provider.
    """
    monkeypatch.setattr(llm_mod, "get_storage_path", lambda sub=None: str(tmp_path / (sub or "")))
    monkeypatch.delenv("OI_LOG_LITELLM_REQUESTS", raising=False)

    def fake_completion(**params):
        yield _FakeChunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(litellm, "completion", fake_completion)

    list(
        llm_mod.fixed_litellm_completions(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
            _oi_log_requests=True,
        )
    )

    logs = tmp_path / "logs"
    assert (logs / "litellm_requests.jsonl").exists()
    assert (logs / "litellm_responses.jsonl").exists()


def test_profile_sets_log_litellm_requests():
    """`llm.log_litellm_requests` is a real, profile-settable attribute defaulting off.

    Profiles are applied by attribute assignment, so the setting only works if the
    Llm object actually defines it. This locks the name and default so a profile
    key like `llm: {log_litellm_requests: true}` takes effect.
    """
    interpreter = OpenInterpreter()
    assert interpreter.llm.log_litellm_requests is False

    apply_profile_to_object(interpreter, {"llm": {"log_litellm_requests": True}})
    assert interpreter.llm.log_litellm_requests is True


def test_logs_use_platform_config_dir_not_dot_config(monkeypatch, tmp_path):
    """Logs land in the platform config dir, not `~/.config/open-interpreter`.

    On Windows the OI config dir is under `%LOCALAPPDATA%`, which is not
    `~/.config/open-interpreter`. Hardcoding the latter sent debug logs to a
    different tree than profiles/ and conversations/. Logs must resolve through
    the same storage path so all OSes agree.
    """
    home = tmp_path / "home"
    storage = tmp_path / "storage"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(llm_mod, "get_storage_path", lambda sub=None: str(storage / (sub or "")))
    monkeypatch.setenv("OI_LOG_LITELLM_REQUESTS", "1")

    def fake_completion(**params):
        yield _FakeChunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(litellm, "completion", fake_completion)

    list(
        llm_mod.fixed_litellm_completions(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert (storage / "logs" / "litellm_requests.jsonl").exists()
    assert (storage / "logs" / "litellm_responses.jsonl").exists()
    assert not (home / ".config" / "open-interpreter" / "logs").exists()


