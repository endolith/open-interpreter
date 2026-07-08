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

    asyncio.run(async_i.input({"role": "user", "type": "message", "start": True}))

    mock_thread.join.assert_called_once()
    assert async_i.stop_event.is_set()


def test_input_stop_command_joins_response_thread():
    """A stop command should wait for the active response thread before returning."""
    async_i = AsyncInterpreter()
    mock_thread = mock.Mock()
    async_i.respond_thread = mock_thread
    async_i.messages = [{"role": "user", "type": "command", "content": "stop"}]

    asyncio.run(async_i.input({"role": "user", "type": "message", "end": True}))

    mock_thread.join.assert_called_once()
