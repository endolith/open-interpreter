import json
import os

import litellm

import interpreter.core.llm.llm as llm_mod


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
    monkeypatch.setenv("HOME", str(tmp_path))
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

    logs = tmp_path / ".config" / "open-interpreter" / "logs"
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
    monkeypatch.setenv("HOME", str(tmp_path))
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

    logs = tmp_path / ".config" / "open-interpreter" / "logs"
    assert not (logs / "litellm_requests.jsonl").exists()
    assert not (logs / "litellm_responses.jsonl").exists()
