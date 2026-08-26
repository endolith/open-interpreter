import asyncio
import os
from unittest import mock

import pytest

from interpreter.core.async_core import AsyncInterpreter, authenticate_function


def test_authenticate_function_no_env_key_allows_all():
    """When INTERPRETER_API_KEY is unset, the async server accepts any client key."""
    with mock.patch.dict(os.environ, {}, clear=True):
        os.environ.pop("INTERPRETER_API_KEY", None)
        assert authenticate_function("anything") is True


def test_authenticate_function_requires_matching_key():
    """When INTERPRETER_API_KEY is set, only that exact key is accepted."""
    with mock.patch.dict(os.environ, {"INTERPRETER_API_KEY": "secret"}):
        assert authenticate_function("secret") is True
        assert authenticate_function("wrong") is False


def test_accumulate_creates_message_on_start():
    """Streaming chunks must begin with start=True before content is stored."""
    async_i = AsyncInterpreter()
    async_i.messages = []
    async_i.accumulate({"role": "user", "type": "message", "start": True})
    async_i.accumulate({"role": "user", "type": "message", "content": "hello"})
    assert async_i.messages[-1]["content"] == "hello"


def test_accumulate_appends_same_type_content():
    """Consecutive chunks of the same type/format append to one message."""
    async_i = AsyncInterpreter()
    async_i.messages = [{"role": "user", "type": "message", "content": "hi"}]
    async_i.accumulate({"role": "user", "type": "message", "content": " there"})
    assert async_i.messages[-1]["content"] == "hi there"


def test_accumulate_requires_start_before_content():
    """accumulate() rejects content chunks that arrive before a start=True chunk for that message."""
    async_i = AsyncInterpreter()
    async_i.messages = []
    with pytest.raises(Exception, match="start"):
        async_i.accumulate({"role": "user", "type": "message", "content": "oops"})


def test_input_start_chunk_stops_active_response():
    """A new start chunk should stop the active response before buffering the new turn."""
    async_i = AsyncInterpreter()
    mock_thread = mock.Mock()
    mock_thread.is_alive.return_value = True
    async_i.respond_thread = mock_thread
    async_i.messages = []

    asyncio.run(async_i.input({"role": "user", "type": "message", "start": True}))

    mock_thread.join.assert_called_once()
    assert async_i.stop_event.is_set()
    assert async_i.messages == [
        {"role": "user", "type": "message", "content": ""},
    ]


def test_input_end_chunk_with_queued_stop_command_joins_response_thread():
    """An end chunk should honor a queued stop command without starting respond()."""
    async_i = AsyncInterpreter()
    mock_thread = mock.Mock()
    async_i.respond_thread = mock_thread
    async_i.messages = [{"role": "user", "type": "command", "content": "stop"}]

    with mock.patch("interpreter.core.async_core.threading.Thread") as thread_cls:
        asyncio.run(async_i.input({"role": "user", "type": "message", "end": True}))

    mock_thread.join.assert_called_once()
    assert async_i.stop_event.is_set()
    thread_cls.assert_not_called()
    assert async_i.messages == []


def test_accumulate_accepts_bytes_content():
    """accumulate() stores bytes content on the current message as bytes."""
    async_i = AsyncInterpreter()
    async_i.messages = [{"role": "user", "type": "message", "content": ""}]
    async_i.accumulate(b"\x00\x01")
    assert async_i.messages[-1]["content"] == b"\x00\x01"


def test_accumulate_appends_bytes_to_bytes_content():
    """accumulate() concatenates bytes onto an existing bytes message."""
    async_i = AsyncInterpreter()
    async_i.messages = [{"role": "user", "type": "message", "content": b"\x00"}]
    async_i.accumulate(b"\x01")
    assert async_i.messages[-1]["content"] == b"\x00\x01"


def test_respond_emits_error_message_when_generator_raises():
    """respond() must surface generator exceptions as an error chunk on the output queue."""
    async_i = AsyncInterpreter()
    mock_q = mock.MagicMock()
    async_i.output_queue = mock.MagicMock(sync_q=mock_q)

    def failing_generator():
        raise RuntimeError("kaboom")
        yield  # pragma: no cover

    with mock.patch.object(async_i, "_respond_and_store", failing_generator):
        async_i.respond()

    error_chunks = [
        call.args[0] for call in mock_q.put.call_args_list if call.args[0]["type"] == "error"
    ]
    assert len(error_chunks) == 1
    assert "kaboom" in error_chunks[0]["content"]


def test_respond_retries_when_generator_is_empty():
    """respond() must retry when the generator yields nothing and then raise a final error."""
    async_i = AsyncInterpreter()
    mock_q = mock.MagicMock()
    async_i.output_queue = mock.MagicMock(sync_q=mock_q)
    async_i.auto_run = True

    def empty_generator():
        if False:
            yield  # pragma: no cover

    with mock.patch.object(async_i, "_respond_and_store", empty_generator):
        with mock.patch("interpreter.core.async_core.time.sleep"):
            with pytest.raises(Exception, match="No chunks sent"):
                async_i.respond()

    error_chunks = [
        call.args[0]
        for call in mock_q.put.call_args_list
        if call.args[0]["type"] == "error"
    ]
    assert len(error_chunks) >= 1


def test_respond_emits_error_when_no_generator_defined():
    """respond() must put an error message when the interpreter has no messages to respond to."""
    async_i = AsyncInterpreter()
    mock_q = mock.MagicMock()
    async_i.output_queue = mock.MagicMock(sync_q=mock_q)
    async_i.auto_run = True
    async_i.messages = []

    with mock.patch.object(async_i, "_respond_and_store", return_value=[]):
        with mock.patch("interpreter.core.async_core.time.sleep"):
            with pytest.raises(Exception, match="No chunks sent"):
                async_i.respond()

    error_chunks = [
        call.args[0]
        for call in mock_q.put.call_args_list
        if call.args[0]["type"] == "error"
    ]
    assert len(error_chunks) >= 1
