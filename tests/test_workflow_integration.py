"""Whole-workflow tests: the loop from a typed message to a finished run.

Unit tests cover functions in isolation, which is why a change can leave every
one of them green and still break the product -- the defects live in the joins:
what the loop puts in the next request, whether execution output comes back,
whether the run ever stops. These tests drive the real loop end to end with the
model's replies scripted, so each one pins a join rather than a function.

No network and no API key: the script is installed at `llm.completions`, the
single seam Llm.run() uses to reach a provider, so system message assembly,
trimming, streaming reassembly, code execution and output feedback all run for
real. That also means these run in ordinary CI, which is the point -- a
regression net only helps if it is always on.

The companion tests/test_mock_llm.py drives the same loop over HTTP to prove
the provider wire format; this file proves the loop's own behaviour.
"""

import pytest

from interpreter import OpenInterpreter
from tests.support.fake_llm import install_fake_llm


@pytest.fixture
def workflow_interpreter():
    """An interpreter that runs code for real but never reaches a provider."""
    interpreter = OpenInterpreter(disable_telemetry=True)
    interpreter.offline = True
    interpreter.auto_run = True
    # chat() persists the message list every turn, and the default path is the
    # developer's real ~/.config/open-interpreter/conversations. No test here
    # reads the saved copy back, and none should leave files there.
    interpreter.conversation_history = False
    try:
        yield interpreter
    finally:
        # Each interpreter owns a Jupyter kernel; without this they accumulate
        # for the length of the run.
        try:
            interpreter.computer.terminate()
        except Exception:
            pass


def _request_text(call):
    """Flatten one recorded request's messages into searchable text."""
    parts = []
    for message in call.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def _console_text(messages):
    """All console output the run produced, as one string."""
    return "".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("type") == "console"
    )


@pytest.mark.timeout(120)
def test_code_output_is_fed_back_into_the_next_request(workflow_interpreter):
    """What the code printed must reach the model's next request.

    This is the core of the agent loop: run code, show the model what happened,
    let it answer. If the output never makes it into the following request the
    model is answering blind, and no unit test of the executor or of message
    conversion alone would notice -- both halves work, the join does not.
    """
    fake = install_fake_llm(
        workflow_interpreter,
        ["Let me compute that.\n```python\nprint(21 * 2)\n```", "The answer is 42."],
    )

    messages = workflow_interpreter.chat("what is 21*2", display=False, stream=False)

    assert "42" in _console_text(messages)
    assert messages[-1]["content"] == "The answer is 42."
    # Two requests: one to get the code, one after running it.
    assert len(fake.calls) == 2
    assert "42" in _request_text(fake.calls[1])


@pytest.mark.timeout(120)
def test_a_traceback_reaches_the_model_and_the_run_recovers(workflow_interpreter):
    """A failing block feeds its error back, and the run continues.

    Recovery is the whole reason output is returned to the model. If a raised
    exception ends the run, or is swallowed before the next request, the agent
    cannot fix its own mistake -- the single most common thing it has to do.
    """
    fake = install_fake_llm(
        workflow_interpreter,
        [
            "```python\nraise ValueError('boom')\n```",
            "Let me fix that.\n```python\nprint('recovered')\n```",
            "Recovered.",
        ],
    )

    messages = workflow_interpreter.chat("do the thing", display=False, stream=False)

    assert "recovered" in _console_text(messages)
    assert messages[-1]["content"] == "Recovered."
    # The request that followed the failure must carry the error text.
    assert "ValueError" in _request_text(fake.calls[1]) or "boom" in _request_text(
        fake.calls[1]
    )


@pytest.mark.timeout(120)
def test_variables_persist_across_turns_in_one_session(workflow_interpreter):
    """A name bound in one turn is still bound in the next.

    The session is one kernel, not one kernel per block. If it is restarted
    between turns the model's code silently loses its state and every
    multi-step plan breaks halfway through.
    """
    install_fake_llm(
        workflow_interpreter,
        [
            "```python\nkeep = 7\n```",
            "Stored.",
            "```python\nprint(keep * 6)\n```",
            "That is 42.",
        ],
    )

    workflow_interpreter.chat("remember 7", display=False, stream=False)
    messages = workflow_interpreter.chat("multiply it by 6", display=False, stream=False)

    assert "42" in _console_text(messages)


@pytest.mark.timeout(60)
def test_a_plain_reply_ends_the_run_after_one_request(workflow_interpreter):
    """A reply with no code stops the loop immediately.

    The loop continues while there is code to run. A termination bug shows up
    here as an extra request -- the scripted list runs dry and FakeCompletions
    raises -- which is exactly the runaway that costs real money in production.
    """
    fake = install_fake_llm(workflow_interpreter, ["Hello, World!"])

    messages = workflow_interpreter.chat("say hello", display=False, stream=False)

    assert len(fake.calls) == 1
    assert messages[-1]["content"] == "Hello, World!"


@pytest.mark.timeout(60)
def test_every_request_starts_with_the_system_message(workflow_interpreter):
    """The system message leads every request, on the first turn and later ones.

    Trimming runs on each turn and has dropped the system message before. When
    it does, the model loses its instructions mid-conversation and the failure
    looks like the model "getting worse" rather than a bug.
    """
    fake = install_fake_llm(
        workflow_interpreter,
        ["```python\nprint('x')\n```", "Done.", "Still here."],
    )

    workflow_interpreter.chat("run something", display=False, stream=False)
    workflow_interpreter.chat("anything else?", display=False, stream=False)

    assert len(fake.calls) >= 2
    for call in fake.calls:
        first = call["messages"][0]
        assert first["role"] == "system"
        assert first["content"].strip()


@pytest.mark.timeout(60)
def test_the_second_turn_carries_the_first_turns_history(workflow_interpreter):
    """Turn two sends turn one's exchange, in order, ahead of the new message.

    History assembly is rebuilt from scratch on every turn. Dropping or
    reordering it is invisible to the user until the model contradicts itself,
    and invisible to unit tests that only check one conversion in isolation.
    """
    fake = install_fake_llm(workflow_interpreter, ["First answer.", "Second answer."])

    workflow_interpreter.chat("first question", display=False, stream=False)
    workflow_interpreter.chat("second question", display=False, stream=False)

    second = _request_text(fake.calls[1])
    assert "first question" in second
    assert "First answer." in second
    assert "second question" in second
    assert second.index("first question") < second.index("second question")


@pytest.mark.timeout(60)
def test_a_streamed_reply_reassembles_into_one_message(workflow_interpreter):
    """Deltas arriving in pieces become one clean message, not fragments.

    The script yields each reply in several chunks, like a real stream. If the
    reassembly duplicates or drops a piece the stored message is subtly wrong
    and every later request repeats the corruption.
    """
    sentence = "The quick brown fox jumps over the lazy dog."
    install_fake_llm(workflow_interpreter, [sentence])

    messages = workflow_interpreter.chat("say the sentence", display=False, stream=False)

    assistant = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"] == sentence


@pytest.mark.timeout(120)
def test_a_file_written_in_one_turn_is_read_in_the_next(
    workflow_interpreter, monkeypatch, tmp_path
):
    """Work lands on disk and is still there for the following turn.

    The end-to-end shape of most real tasks: produce something, then use it.
    It exercises execution, output feedback and session continuity together,
    which is where a change to any one of them shows up as a broken product.
    """
    monkeypatch.chdir(tmp_path)
    install_fake_llm(
        workflow_interpreter,
        [
            "```python\nopen('notes.txt', 'w').write('Washington')\n```",
            "Written.",
            "```python\nprint(open('notes.txt').read())\n```",
            "It says Washington.",
        ],
    )

    workflow_interpreter.chat("write Washington to notes.txt", display=False, stream=False)
    assert (tmp_path / "notes.txt").read_text() == "Washington"

    messages = workflow_interpreter.chat("read it back", display=False, stream=False)
    assert "Washington" in _console_text(messages)
