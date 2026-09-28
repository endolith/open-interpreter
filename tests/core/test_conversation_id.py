"""Tests for the per-conversation identifier.

OI mints one id per conversation and re-mints it on reset. It is read by two
different audiences: the hosted `i` model (as a request body field) and
providers that key prompt caching/routing on a per-conversation id (as an HTTP
header). Both must see a value that is stable within a conversation and distinct
between conversations.
"""

from interpreter.core.core import OpenInterpreter


def test_conversation_id_exists_from_construction():
    """A conversation id is available before the first request is ever made.

    Previously the id was only created lazily, and only for the hosted `i` model
    (inside Llm.run). Anything that needs it on turn one -- such as attaching a
    routing header to the very first request -- had nothing to read.
    """
    interpreter = OpenInterpreter()
    assert isinstance(interpreter.conversation_id, str)
    assert interpreter.conversation_id


def test_conversation_id_is_stable_across_turns():
    """Every turn of one conversation reports the same id.

    This is the property prompt caching depends on: a per-turn id would make the
    provider treat each request as an unrelated conversation and forfeit the
    cached prefix.
    """
    interpreter = OpenInterpreter()
    first = interpreter.conversation_id
    assert interpreter.conversation_id == first
    interpreter.messages.append({"role": "user", "type": "message", "content": "hello"})
    assert interpreter.conversation_id == first


def test_conversation_ids_differ_between_conversations():
    """Two independent conversations must not share an id, or they share routing."""
    assert OpenInterpreter().conversation_id != OpenInterpreter().conversation_id


def test_reset_mints_a_new_conversation_id():
    """%reset starts a new conversation, so it also starts a new id.

    reset() clears the message history. Keeping the old id would leave the
    provider's routing and prompt cache pointed at a conversation whose history
    no longer exists, which is exactly the coupling the id is meant to express.
    """
    interpreter = OpenInterpreter()
    before = interpreter.conversation_id
    interpreter.reset()
    assert interpreter.conversation_id != before


def test_i_model_still_configures_its_endpoint():
    """The hosted `i` model keeps its api_base/api_key/context_window defaults.

    Regression guard: that block used to be nested inside
    `if not hasattr(interpreter, "conversation_id")`. Now that every conversation
    has an id from construction, that guard is always false and would silently
    skip the whole block, leaving `i` pointed at the default OpenAI endpoint.
    """
    interpreter = OpenInterpreter()
    interpreter.llm.model = "i"

    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter.llm.completions = fake_completions
    list(
        interpreter.llm.run(
            [
                {"role": "system", "type": "message", "content": "sys"},
                {"role": "user", "type": "message", "content": "hi"},
            ]
        )
    )

    assert captured["api_base"] == "https://api.openinterpreter.com/v0"
    assert captured["api_key"] == "x"
    assert interpreter.llm.context_window == 7000
    assert interpreter.llm.max_tokens == 1000


def test_i_model_sends_conversation_id_in_the_body():
    """The `i` API receives the id as a request field, as it always has."""
    interpreter = OpenInterpreter()
    interpreter.llm.model = "i"

    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter.llm.completions = fake_completions
    list(
        interpreter.llm.run(
            [
                {"role": "system", "type": "message", "content": "sys"},
                {"role": "user", "type": "message", "content": "hi"},
            ]
        )
    )

    assert captured["conversation_id"] == interpreter.conversation_id


def test_non_i_models_do_not_send_conversation_id_in_the_body():
    """conversation_id stays out of the request body for every other provider.

    It is a field only the hosted `i` API understands. Now that the id exists for
    all models, an unscoped `hasattr` check would start sending an unknown field
    to OpenAI, Anthropic, and everyone else.
    """
    interpreter = OpenInterpreter()
    interpreter.llm.model = "gpt-4o"

    captured = {}

    def fake_completions(**params):
        captured.update(params)
        return iter(())

    interpreter.llm.completions = fake_completions
    list(
        interpreter.llm.run(
            [
                {"role": "system", "type": "message", "content": "sys"},
                {"role": "user", "type": "message", "content": "hi"},
            ]
        )
    )

    assert "conversation_id" not in captured
