"""A scripted stand-in for Llm.completions.

Llm.run() only ever touches the provider through `llm.completions(**params)`,
which normally is fixed_litellm_completions. Replacing that one attribute lets
the whole loop -- system message assembly, trimming, streaming reassembly, code
execution, output feedback, termination -- run for real while the model's
replies come from a list.

This is the in-process counterpart to tests/support/mock_openai_server.py. The
HTTP mock proves the provider wire format; this proves the loop. Because the
replies are an ordered script rather than keyword-matched, a test can drive an
exact multi-turn workflow and assert on what each request carried.
"""


from tests.support.mock_openai_server import stream_reply_chunks


class FakeCompletions:
    """Yields scripted replies as streaming deltas, recording each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # the params dict of each request, in order

    def __call__(self, **params):
        self.calls.append(params)
        if not self.replies:
            raise AssertionError(
                "FakeCompletions ran out of scripted replies: the loop asked the "
                f"model {len(self.calls)} times but only {len(self.calls) - 1} "
                "replies were scripted. Either the loop failed to terminate or "
                "the test needs another reply."
            )
        text = self.replies.pop(0)
        # Emit several deltas rather than one, so streaming reassembly is
        # exercised the way a real stream does it. Split on fence boundaries
        # using the same helper the HTTP mock uses: run_text_llm defers while
        # the accumulated text ends with a backtick, so a chunk that cuts a
        # ``` fence in half is parsed as prose and the code never runs.
        for piece in stream_reply_chunks(text):
            yield {"choices": [{"delta": {"content": piece}}]}
        yield {"choices": [{"delta": {}}]}


def install_fake_llm(interpreter, replies):
    """Point an interpreter at a scripted reply list instead of a provider.

    Returns the FakeCompletions instance so a test can inspect `.calls` -- the
    requests the loop actually built, which is where cross-function bugs show
    up even when the visible answer looks right.
    """
    fake = FakeCompletions(replies)
    llm = interpreter.llm
    llm.completions = fake
    llm.model = "openai/fake"
    llm.api_key = "fake"
    # Unroutable on purpose: if the seam is ever bypassed the test fails loudly
    # instead of quietly reaching the network.
    llm.api_base = "http://127.0.0.1:9"
    llm.supports_functions = False
    llm.supports_vision = False
    llm.context_window = 32000
    llm.max_tokens = 4096
    llm._is_loaded = True
    interpreter.offline = True
    interpreter.disable_telemetry = True
    interpreter.auto_run = True
    return fake
