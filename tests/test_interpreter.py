import os
import platform
import re
import signal
import socket
import time
from random import randint

import pytest

from tests.helpers import require_bash_compatible_shell

#####
from interpreter import AsyncInterpreter, OpenInterpreter
from interpreter.terminal_interface.utils.count_tokens import (
    count_messages_tokens,
    count_tokens,
)

interpreter = OpenInterpreter()
interpreter.conversation_history = False
#####

import multiprocessing
import threading
import time

import pytest
from websocket import create_connection

# Use spawn, not fork. After earlier integration tests run, the pytest process may
# have background threads (asyncio, litellm, etc.). fork() in a threaded parent is
# unsafe on Linux and can leave the child uvicorn process unable to accept connections.
_MP_SPAWN = multiprocessing.get_context("spawn")
_SERVER_HOST = "127.0.0.1"
_SERVER_PORT = 8000


def _allocate_server_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_SERVER_HOST, 0))
        return sock.getsockname()[1]


def _server_ws_url():
    return f"ws://{_SERVER_HOST}:{_SERVER_PORT}/"


def _server_http_url(path=""):
    return f"http://{_SERVER_HOST}:{_SERVER_PORT}{path}"


def _start_server_subprocess(target):
    global _SERVER_PORT
    _SERVER_PORT = _allocate_server_port()
    os.environ["INTERPRETER_PORT"] = str(_SERVER_PORT)
    process = _MP_SPAWN.Process(target=target)
    process.start()
    return process


def _wait_for_server(process, timeout=120):
    import urllib.error
    import urllib.request

    url = f"http://{_SERVER_HOST}:{_SERVER_PORT}/heartbeat"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process.is_alive():
            raise RuntimeError(
                f"Server subprocess exited before becoming ready (exit code {process.exitcode})"
            )
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.URLError:
            time.sleep(0.5)
    raise TimeoutError(
        f"Server at {_SERVER_HOST}:{_SERVER_PORT} did not respond within {timeout}s"
    )


def _stop_server_subprocess(process):
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()


async def _wait_for_websocket_complete(
    websocket,
    max_messages=500,
    recv_timeout=300.0,
    acknowledge=False,
    phase="unknown",
    server_process=None,
):
    """Read WebSocket chunks until the server sends a 'complete' status.

    The old while True loops hung forever when 'complete' never arrived (for example
    on auth failure). Bound both message count and per-recv wait time.
    """

    import asyncio
    import json

    accumulated_content = ""
    messages_received = 0
    for _ in range(max_messages):
        if server_process is not None and not server_process.is_alive():
            raise RuntimeError(
                f"Server subprocess exited during WebSocket wait (phase: {phase}, "
                f"exit code {server_process.exitcode}, port {_SERVER_PORT})"
            )
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=recv_timeout)
        except asyncio.TimeoutError as exc:
            raise Exception(
                f"No WebSocket message within {recv_timeout}s waiting for 'complete' "
                f"(phase: {phase}, messages received: {messages_received}, "
                f"port {_SERVER_PORT})"
            ) from exc
        messages_received += 1

        message_data = json.loads(message)
        if acknowledge and "id" in message_data:
            await websocket.send(json.dumps({"ack": message_data["id"]}))
        if "error" in message_data:
            raise Exception(message_data["content"])
        print("Received from WebSocket:", message_data)
        content = message_data.get("content")
        if type(content) == str:
            accumulated_content += content
        elif content:
            accumulated_content += str(content)
        if (
            message_data.get("role") == "server"
            and message_data.get("type") == "status"
            and message_data.get("content") == "complete"
        ):
            print("Received expected message from server")
            return accumulated_content

    raise Exception(
        f"Never received 'complete' status after {max_messages} messages "
        f"(phase: {phase}, port {_SERVER_PORT})"
    )


def _last_assistant_message(messages):
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("type") == "message":
            return str(message.get("content", ""))
    return ""


def _last_assistant_text(messages):
    """Last assistant message or code block (models often reply with code only)."""

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("type") in ("message", "code"):
            content = message.get("content")
            if content:
                return str(content)
    return ""


@pytest.mark.timeout(120)
def test_hallucinations():
    """Common LLM code hallucinations are normalized before execution.

    Covers executeexecute suffixes, JSON code blocks, functions.execute()
    wrappers, and loose object literals — each should run and produce the
    expected numeric or print output."""

    code = """10+12executeexecute\n"""

    interpreter.messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": code}
    ]
    for chunk in interpreter._respond_and_store():
        if chunk.get("format") == "output":
            assert chunk.get("content") == "22"
            break

    code = """{
    "language": "python",
    "code": "10+12"
  }"""

    interpreter.messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": code}
    ]
    for chunk in interpreter._respond_and_store():
        if chunk.get("format") == "output":
            assert chunk.get("content") == "22"
            break

    code = """functions.execute({
    "language": "python",
    "code": "10+12"
  })"""

    interpreter.messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": code}
    ]
    for chunk in interpreter._respond_and_store():
        if chunk.get("format") == "output":
            assert chunk.get("content") == "22"
            break

    code = """{language: "python", code: "print('hello')" }"""

    interpreter.messages = [
        {"role": "assistant", "type": "code", "format": "python", "content": code}
    ]
    for chunk in interpreter._respond_and_store():
        if chunk.get("format") == "output":
            assert chunk.get("content").strip() == "hello"
            break


def test_streaming_output_chunks_are_incremental():
    """
    Regression test for issue #73: multi-line shell output must stream
    incremental deltas. Previously, _respond_and_store reused the same dict
    objects for yielded chunks and accumulated messages, so consumers that
    held references to earlier chunks saw growing content on every new line.
    """
    from unittest.mock import patch

    output_lines = ["file1\n", "file2\n", "file3\n"]
    respond_chunks = [
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": line,
        }
        for line in output_lines
    ]

    original_messages = interpreter.messages
    try:
        interpreter.messages = [
            {
                "role": "assistant",
                "type": "code",
                "format": "shell",
                "content": "ls -l",
            }
        ]

        with patch("interpreter.core.core.respond", return_value=iter(respond_chunks)):
            yielded_output_chunks = []
            for chunk in interpreter._respond_and_store():
                if chunk.get("format") == "output":
                    yielded_output_chunks.append(chunk)

            assert [c["content"] for c in yielded_output_chunks] == output_lines

            # Earlier yielded chunks must not be mutated as later lines arrive.
            assert yielded_output_chunks[0]["content"] == "file1\n"
            assert yielded_output_chunks[1]["content"] == "file2\n"

            # Conversation history should still accumulate the full output.
            assert interpreter.messages[-1]["content"] == "".join(output_lines)
    finally:
        interpreter.messages = original_messages


def run_auth_server():
    os.environ["INTERPRETER_REQUIRE_ACKNOWLEDGE"] = "True"
    os.environ["INTERPRETER_API_KEY"] = "testing"
    async_interpreter = AsyncInterpreter()
    async_interpreter.print = False
    async_interpreter.server.run()


# @pytest.mark.skip(reason="Requires uvicorn, which we don't require by default")
@pytest.mark.integration
def test_authenticated_acknowledging_breaking_server():
    """Test the server when we have authentication and acknowledging one.

    I know this is bad, just trying to test quickly!"""


    # Start the server in a new process

    process = _start_server_subprocess(run_auth_server)

    _wait_for_server(process)

    import asyncio
    import json

    import requests
    import websockets

    async def test_fastapi_server():
        import asyncio

        async with websockets.connect(_server_ws_url()) as websocket:
            print("Connected to WebSocket")

            await websocket.send(json.dumps({"auth": "testing"}))

            post_url = _server_http_url("/settings")
            settings = {
                "llm": {
                    "model": "gpt-4o-mini",
                    "execution_instructions": "",
                    "supports_functions": False,
                },
                "system_message": "You are a poem writing bot. Do not do anything but respond with a poem.",
                "auto_run": True,
            }
            response = requests.post(
                post_url, json=settings, headers={"X-API-KEY": "testing"}
            )
            print("POST request sent, response:", response.json())

            await websocket.send(
                json.dumps({"role": "user", "type": "message", "start": True})
            )
            await websocket.send(
                json.dumps(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "Write a short poem about Seattle.",
                    }
                )
            )
            await websocket.send(
                json.dumps({"role": "user", "type": "message", "end": True})
            )
            print("WebSocket chunks sent")

            max_chunks = 5

            poem = ""
            while True:
                max_chunks -= 1
                if max_chunks == 0:
                    break
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=300.0)
                except asyncio.TimeoutError as exc:
                    raise Exception("Timed out waiting for early poem chunks") from exc
                message_data = json.loads(message)
                if "id" in message_data:
                    await websocket.send(json.dumps({"ack": message_data["id"]}))
                if "error" in message_data:
                    raise Exception(str(message_data))
                print("Received from WebSocket:", message_data)
                if type(message_data.get("content")) == str:
                    poem += message_data.get("content")
                    print(message_data.get("content"), end="", flush=True)
                if (
                    message_data.get("role") == "server"
                    and message_data.get("type") == "status"
                    and message_data.get("content") == "complete"
                ):
                    raise (
                        Exception(
                            "It shouldn't have finished this soon, accumulated_content is: "
                            + poem
                        )
                    )

            await websocket.close()
            print("Disconnected from WebSocket")

        time.sleep(3)

        # Now let's hilariously keep going
        print("RESUMING")

        async with websockets.connect(_server_ws_url()) as websocket:
            print("Connected to WebSocket")

            await websocket.send(json.dumps({"auth": "testing"}))

            poem += await _wait_for_websocket_complete(
                websocket,
                acknowledge=True,
                phase="auth_server_resume_poem",
                server_process=process,
            )

            time.sleep(1)
            print("Is this a normal poem?")
            print(poem)
            time.sleep(1)

    # asyncio.get_event_loop() raises RuntimeError on Python 3.12+ when there
    # is no current event loop (e.g. pytest has not set one up). asyncio.run()
    # creates a fresh loop, runs the coroutine, and closes it cleanly.
    try:
        asyncio.run(test_fastapi_server())
    finally:
        _stop_server_subprocess(process)


def run_server():
    os.environ["INTERPRETER_REQUIRE_ACKNOWLEDGE"] = "False"
    if "INTERPRETER_API_KEY" in os.environ:
        del os.environ["INTERPRETER_API_KEY"]
    async_interpreter = AsyncInterpreter()
    async_interpreter.print = False
    async_interpreter.server.run()


# @pytest.mark.skip(reason="Requires uvicorn, which we don't require by default")
@pytest.mark.integration
@pytest.mark.timeout(900)
def test_server():
    """FastAPI/WebSocket server accepts settings, streams chat, and completes cleanly.

    Spins up AsyncInterpreter in a subprocess (spawn context), posts settings,
    sends a user message over WebSocket, and verifies poem-style responses
    arrive without authentication when INTERPRETER_REQUIRE_ACKNOWLEDGE is off."""

    process = _start_server_subprocess(run_server)

    _wait_for_server(process)

    import asyncio
    import json

    import requests
    import websockets

    async def test_fastapi_server():
        import asyncio

        async with websockets.connect(_server_ws_url()) as websocket:
            print("Connected to WebSocket")

            await websocket.send(json.dumps({"auth": "dummy-api-key"}))

            # POST /settings rejects messages, system_message, and auto_run as
            # sensitive settings (SENSITIVE_SERVER_SETTINGS in async_core.py), so
            # only non-sensitive keys are sent here. The subprocess's default
            # auto_run=False already satisfies the math turn's no-execute intent.
            post_url = _server_http_url("/settings")
            settings = {
                "llm": {"model": "gpt-4o-mini"},
            }
            response = requests.post(post_url, json=settings, timeout=30)
            response.raise_for_status()
            print("POST request sent, response:", response.json())

            await websocket.send(
                json.dumps({"role": "user", "type": "message", "start": True})
            )
            await websocket.send(
                json.dumps(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "What's the secret word?",
                    }
                )
            )
            await websocket.send(
                json.dumps({"role": "user", "type": "message", "end": True})
            )
            print("WebSocket chunks sent")

            # Wait for a specific response
            accumulated_content = await _wait_for_websocket_complete(
                websocket, phase="secret_word_crunk", server_process=process
            )

            post_url = _server_http_url("/settings")
            settings = {
                "llm": {"model": "gpt-4o-mini"},
            }
            response = requests.post(post_url, json=settings, timeout=30)
            response.raise_for_status()
            print("POST request sent, response:", response.json())

            await websocket.send(
                json.dumps({"role": "user", "type": "message", "start": True})
            )
            await websocket.send(
                json.dumps(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "What's the secret word?",
                    }
                )
            )
            await websocket.send(
                json.dumps({"role": "user", "type": "message", "end": True})
            )
            print("WebSocket chunks sent")

            # Wait for a specific response
            accumulated_content = await _wait_for_websocket_complete(
                websocket, phase="secret_word_barloney", server_process=process
            )

            post_url = _server_http_url("/settings")
            settings = {
                "custom_instructions": "",
                "verbose": False,
            }
            response = requests.post(post_url, json=settings, timeout=30)
            response.raise_for_status()
            print("POST request sent, response:", response.json())

            await websocket.send(
                json.dumps({"role": "user", "type": "message", "start": True})
            )
            await websocket.send(
                json.dumps(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "What's 239023*79043? Use Python.",
                    }
                )
            )
            await websocket.send(
                json.dumps({"role": "user", "type": "message", "end": True})
            )
            print("WebSocket chunks sent")

            accumulated_content = await _wait_for_websocket_complete(
                websocket, phase="math_code_auto_run_false", server_process=process
            )
            get_url = _server_http_url("/settings/messages")
            response = requests.get(get_url)
            print("GET request sent, response:", response.json())

            # Assert that the last message has a type of 'code'
            response_json = response.json()
            if isinstance(response_json, str):
                response_json = json.loads(response_json)
            messages = response_json["messages"] if "messages" in response_json else []
            assert messages[-1]["type"] == "code"
            assert "18893094989" not in accumulated_content.replace(",", "")

            # The math turn used auto_run=False, so the model wrote Python code but did
            # not execute it. "go" tells the server to run that pending code block now.
            await websocket.send(
                json.dumps({"role": "user", "type": "command", "start": True})
            )
            await websocket.send(
                json.dumps(
                    {
                        "role": "user",
                        "type": "command",
                        "content": "go",
                    }
                )
            )
            await websocket.send(
                json.dumps({"role": "user", "type": "command", "end": True})
            )

            # Wait for a specific response
            accumulated_content = await _wait_for_websocket_complete(
                websocket, phase="go_execute_code", server_process=process
            )

            # File turn: check the model's plain-text answer about a file path
            # (judged via computer.ai.chat). auto_run stays off so no shell code
            # auto-executes and hangs the server before 'complete'; custom_instructions
            # steers plain-text replies and _last_assistant_text catches code blocks.
            post_url = _server_http_url("/settings")
            settings = {
                "custom_instructions": (
                    "Answer in plain text only. Do not write or run code."
                ),
            }
            response = requests.post(post_url, json=settings, timeout=30)
            response.raise_for_status()
            print("POST request sent, response:", response.json())

            user_start = {"role": "user", "start": True}
            file_question = {
                "role": "user",
                "type": "message",
                "content": "Does this file exist?",
            }
            file_path = {
                "role": "user",
                "type": "file",
                "format": "path",
                "content": "/something.txt",
            }

            await websocket.send(json.dumps(user_start))
            print("sent", user_start)
            await websocket.send(json.dumps(file_question))
            print("sent", file_question)
            await websocket.send(json.dumps(file_path))
            print("sent", file_path)
            await websocket.send(json.dumps({"role": "user", "end": True}))
            print("WebSocket chunks sent")

            # WebSocket stream may arrive before GET /messages is consistent; keep
            # accumulated_content as a fallback when picking the assistant reply.
            accumulated_content = await _wait_for_websocket_complete(
                websocket, phase="file_exists", server_process=process
            )

            # Get messages
            get_url = _server_http_url("/settings/messages")
            response_json = requests.get(get_url).json()
            print("GET request sent, response:", response_json)
            if isinstance(response_json, str):
                response_json = json.loads(response_json)
            messages = response_json["messages"]

            # Prefer structured messages; fall back to the WebSocket stream when the
            # model replies with a code block or GET /messages lags behind the stream.
            last_assistant = _last_assistant_text(messages) or accumulated_content
            assert last_assistant, "expected assistant response after file turn"
            response = interpreter.computer.ai.chat(
                last_assistant
                + "\n\nBased on the assistant response above, does the assistant think the file exists? Yes or no? Only reply with one word— 'yes' or 'no'."
            )
            assert response.strip(" \n.").lower() == "no"

            #### TEST IMAGES ####

            # POST /settings rejects {"messages": []} (messages is guarded as a
            # sensitive setting since PR #96), so the conversation cannot be reset
            # that way. Start a fresh, separate server for an isolated vision turn
            # and reconnect on a new WebSocket; `vision_process` tracks that server
            # so it can be stopped in a finally block even if a step below raises.
            await websocket.close()
            _stop_server_subprocess(process)
            vision_process = _start_server_subprocess(run_server)
            try:
                _wait_for_server(vision_process)

                base64png = "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"

                async with websockets.connect(_server_ws_url()) as image_websocket:
                    await image_websocket.send(json.dumps({"auth": "dummy-api-key"}))

                    await image_websocket.send(json.dumps({"role": "user", "start": True}))
                    await image_websocket.send(
                        json.dumps(
                            {
                                "role": "user",
                                "type": "message",
                                "content": (
                                    "What do you see in this image? Reply with only one letter.\n"
                                    "A) a cat\n"
                                    "B) a color gradient\n"
                                    "C) a table of numbers\n"
                                    "D) a black rectangle"
                                ),
                            }
                        )
                    )
                    await image_websocket.send(
                        json.dumps(
                            {
                                "role": "user",
                                "type": "image",
                                "format": "base64.png",
                                "content": base64png,
                            }
                        )
                    )
                    await image_websocket.send(json.dumps({"role": "user", "end": True}))
                    print("WebSocket chunks sent")

                    accumulated_content = await _wait_for_websocket_complete(
                        image_websocket, phase="vision_mcq", server_process=vision_process
                    )

                # _wait_for_websocket_complete appends the status content "complete".
                vision_reply = accumulated_content.removesuffix("complete")
                assert vision_reply, "expected assistant response after image turn"
                assert re.search(
                    r"\bB\b", vision_reply, re.IGNORECASE
                ), f"expected vision model to answer B (gradient), got: {vision_reply!r}"

                # Exercise the /run endpoint: the payload signals the vision server's
                # Python process with SIGINT so it shuts down, and the finally block
                # below stops the subprocess regardless of how this turn ends.
                post_url = _server_http_url("/run")
                code_data = {
                    "code": "import os, signal; os.kill(os.getpid(), signal.SIGINT)",
                    "language": "python",
                }
                response = requests.post(post_url, json=code_data, timeout=30)
                print("POST request sent, response:", response.json())

            finally:
                # The /run payload sends SIGINT to the vision server, but this
                # finally still stops the subprocess on any exit path (assertions,
                # WebSocket timeouts, and the /run request itself).
                _stop_server_subprocess(vision_process)

    # asyncio.get_event_loop() raises RuntimeError on Python 3.12+ when there
    # is no current event loop (e.g. pytest has not set one up). asyncio.run()
    # creates a fresh loop, runs the coroutine, and closes it cleanly.
    try:
        asyncio.run(test_fastapi_server())
    finally:
        _stop_server_subprocess(process)


@pytest.mark.skip(reason="Mac only — manual harness; use darwin_ci in test_platform_ci.py for CI")
def test_sms():
    """Manual Mac-only smoke for reading and searching SMS via AppleScript.

    Not suitable for CI: reads real Messages, needs TCC permission, ends assert False."""

    sms = interpreter.computer.sms

    # Get the last 5 messages
    messages = sms.get(limit=5)
    print(messages)

    # Search messages for a substring
    search_results = sms.get(substring="i love you", limit=100)
    print(search_results)

    assert False


@pytest.mark.skip(reason="Mac only — manual harness; use darwin_ci in test_platform_ci.py for CI")
def test_pytes():
    """Manual Mac-only smoke for vision OCR on a Desktop PNG (developer harness).

    Not suitable for CI: needs a Desktop PNG on a logged-in GUI session, assert False."""

    import os

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    files_on_desktop = [f for f in os.listdir(desktop_path) if f.endswith(".png")]
    if files_on_desktop:
        first_file = files_on_desktop[0]
        first_file_path = os.path.join(desktop_path, first_file)
        print(first_file_path)
        ocr = interpreter.computer.vision.ocr(path=first_file_path)
        print(ocr)
        print("what")
    else:
        print("No files found on Desktop.")

    assert False


@pytest.mark.integration
def test_ai_chat():
    """Integration smoke: computer.ai.chat returns a response for a simple greeting."""

    print(interpreter.computer.ai.chat("hi"))


@pytest.mark.integration
def test_generator():
    """Sends two messages, makes sure everything is correct with display both on and off."""


    interpreter.llm.model = "gpt-4o-mini"

    for tests in [
        {"query": "What's 38023*40334? Use Python", "display": True},
        {"query": "What's 2334*34335555? Use Python", "display": True},
        {"query": "What's 3545*22? Use Python", "display": False},
        {"query": "What's 0.0021*3433335555? Use Python", "display": False},
    ]:
        assistant_message_found = False
        console_output_found = False
        active_line_found = False
        flag_checker = []

        for chunk in interpreter.chat(
            tests["query"]
            + "\nNo talk or plan, just immediately code, then tell me the answer.",
            stream=True,
            display=True,
        ):
            print(chunk)
            # Check if chunk has the right schema
            assert "role" in chunk, "Chunk missing 'role'"
            assert "type" in chunk, "Chunk missing 'type'"
            if "start" not in chunk and "end" not in chunk:
                assert "content" in chunk, "Chunk missing 'content'"
            if "format" in chunk:
                assert isinstance(chunk["format"], str), "'format' should be a string"

            flag_checker.append(chunk)

            # Check if assistant message, console output, and active line are found
            if chunk["role"] == "assistant" and chunk["type"] == "message":
                assistant_message_found = True
            if chunk["role"] == "computer" and chunk["type"] == "console":
                console_output_found = True
            if "format" in chunk:
                if (
                    chunk["role"] == "computer"
                    and chunk["type"] == "console"
                    and chunk["format"] == "active_line"
                ):
                    active_line_found = True

        # Ensure all flags are proper
        assert (
            flag_checker.count(
                {"role": "assistant", "type": "code", "format": "python", "start": True}
            )
            == 1
        ), "Incorrect number of 'assistant code start' flags"
        assert (
            flag_checker.count(
                {"role": "assistant", "type": "code", "format": "python", "end": True}
            )
            == 1
        ), "Incorrect number of 'assistant code end' flags"
        assert (
            flag_checker.count({"role": "assistant", "type": "message", "start": True})
            == 1
        ), "Incorrect number of 'assistant message start' flags"
        assert (
            flag_checker.count({"role": "assistant", "type": "message", "end": True})
            == 1
        ), "Incorrect number of 'assistant message end' flags"
        assert (
            flag_checker.count({"role": "computer", "type": "console", "start": True})
            == 1
        ), "Incorrect number of 'computer console output start' flags"
        assert (
            flag_checker.count({"role": "computer", "type": "console", "end": True})
            == 1
        ), "Incorrect number of 'computer console output end' flags"

        # Assert that assistant message, console output, and active line were found
        assert assistant_message_found, "No assistant message was found"
        assert console_output_found, "No console output was found"
        assert active_line_found, "No active line was found"


@pytest.mark.skip(reason="Requires open-interpreter[local]")
def test_localos():
    """Manual smoke for local OS view with images disabled then re-enabled."""

    interpreter.computer.emit_images = False
    interpreter.computer.view()
    interpreter.computer.emit_images = True
    assert False


@pytest.mark.skip(reason="Requires open-interpreter[local]")
def test_m_vision():
    """Manual local-model smoke: chat with a base64 image when supports_vision is off."""

    base64png = "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"
    messages = [
        {"role": "user", "type": "message", "content": "describe this image"},
        {
            "role": "user",
            "type": "image",
            "format": "base64.png",
            "content": base64png,
        },
    ]

    interpreter.llm.supports_vision = False
    interpreter.llm.model = "gpt-4o-mini"
    interpreter.llm.supports_functions = True
    interpreter.llm.context_window = 110000
    interpreter.llm.max_tokens = 4096
    interpreter.loop = True

    interpreter.chat(messages)

    interpreter.loop = False
    import time

    time.sleep(10)


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_point():
    """Manual display smoke: mouse.move finds common macOS icons by description."""

    interpreter.computer.mouse.move(icon="gear")
    interpreter.computer.mouse.move(icon="refresh")
    interpreter.computer.mouse.move(icon="play")
    interpreter.computer.mouse.move(icon="magnifying glass")
    interpreter.computer.mouse.move("Spaces:")
    assert False


@pytest.mark.skip(reason="Aifs not ready")
def test_skills():
    """Manual skills.search integration (Python 3.11 only; skipped on 3.12)."""

    import sys

    if sys.version_info[:2] == (3, 12):
        print(
            "skills.search is only for python 3.11 for now, because it depends on unstructured. skipping this test."
        )
        return

    import json

    interpreter.llm.model = "gpt-4o-mini"

    messages = ["USER: Hey can you search the web for me?\nAI: Sure!"]

    combined_messages = "\\n".join(json.dumps(x) for x in messages[-3:])
    query_msg = interpreter.chat(
        f"This is the conversation so far: {combined_messages}. What is a hypothetical python function that might help resolve the user's query? Respond with nothing but the hypothetical function name exactly."
    )
    query = query_msg[0]["content"]
    # skills_path = '/01OS/server/skills'
    # interpreter.computer.skills.path = skills_path
    print(interpreter.computer.skills.path)
    if os.path.exists(interpreter.computer.skills.path):
        for file in os.listdir(interpreter.computer.skills.path):
            os.remove(os.path.join(interpreter.computer.skills.path, file))
    print("Path: ", interpreter.computer.skills.path)
    print("Files in the path: ")
    interpreter.computer.run("python", "def testing_skilsl():\n    print('hi')")
    for file in os.listdir(interpreter.computer.skills.path):
        print(file)
    interpreter.computer.run("python", "def testing_skill():\n    print('hi')")
    print("Files in the path: ")
    for file in os.listdir(interpreter.computer.skills.path):
        print(file)

    try:
        skills = interpreter.computer.skills.search(query)
    except ImportError:
        print("Attempting to install unstructured[all-docs]")
        import subprocess

        subprocess.run(["pip", "install", "unstructured[all-docs]"], check=True)
        skills = interpreter.computer.skills.search(query)

    lowercase_skills = [skill[0].lower() + skill[1:] for skill in skills]
    output = "\\n".join(lowercase_skills)
    assert "testing_skilsl" in str(output)


@pytest.mark.skip(reason="Local only")
def test_browser():
    """Manual local smoke: browser.search issues a query against a local API base."""

    interpreter.computer.api_base = "http://0.0.0.0:80/v0"
    print(
        interpreter.computer.browser.search("When's the next Dune showing in Seattle?")
    )
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_display_api():
    """Manual display smoke: mouse.move locates many on-screen icons/text labels."""

    start = time.time()

    # interpreter.computer.display.find_text("submit")
    # assert False

    def say(icon_name):
        import subprocess

        subprocess.run(["say", "-v", "Fred", icon_name])

    icons = [
        "Submit",
        "Yes",
        "Profile picture icon",
        "Left arrow",
        "Magnifying glass",
        "star",
        "record icon icon",
        "age text",
        "call icon icon",
        "account text",
        "home icon",
        "settings text",
        "form text",
        "gear icon icon",
        "trash icon",
        "new folder icon",
        "phone icon icon",
        "home button",
        "trash button icon",
        "folder icon icon",
        "black heart icon icon",
        "white heart icon icon",
        "image icon",
        "test@mail.com text",
    ]

    # from random import shuffle
    # shuffle(icons)

    say("The test will begin in 3")
    time.sleep(1)
    say("2")
    time.sleep(1)
    say("1")
    time.sleep(1)

    import pyautogui

    pyautogui.mouseDown()

    for icon in icons:
        if icon.endswith("icon icon"):
            say("click the " + icon)
            interpreter.computer.mouse.move(icon=icon.replace("icon icon", "icon"))
        elif icon.endswith("icon"):
            say("click the " + icon)
            interpreter.computer.mouse.move(icon=icon.replace(" icon", ""))
        elif icon.endswith("text"):
            say("click " + icon)
            interpreter.computer.mouse.move(icon.replace(" text", ""))
        else:
            say("click " + icon)
            interpreter.computer.mouse.move(icon=icon)

    # interpreter.computer.mouse.move(icon="caution")
    # interpreter.computer.mouse.move(icon="bluetooth")
    # interpreter.computer.mouse.move(icon="gear")
    # interpreter.computer.mouse.move(icon="play button")
    # interpreter.computer.mouse.move(icon="code icon with '>_' in it")
    print(time.time() - start)
    assert False


@pytest.mark.skip(reason="Server is not a stable feature")
def test_websocket_server():
    """Manual smoke: legacy interpreter.server WebSocket accepts chat messages."""

    server_thread = threading.Thread(target=interpreter.server)
    server_thread.start()

    # Give the server a moment to start
    time.sleep(3)

    # Connect to the server
    ws = create_connection("ws://127.0.0.1:8000/")

    # Send the first message
    ws.send(
        "Hello, interpreter! What operating system are you on? Also, what time is it in Seattle?"
    )
    # Wait for a moment before sending the second message
    time.sleep(1)
    ws.send("Actually, nevermind. Thank you!")

    # Receive the responses
    responses = []
    while True:
        response = ws.recv()
        print(response)
        responses.append(response)

    # Check the responses
    assert responses  # Check that some responses were received

    ws.close()


@pytest.mark.skip(reason="Server is not a stable feature")
def test_i():
    """Manual smoke: HTTP POST to interpreter.server streams a non-empty response."""

    import requests

    url = "http://127.0.0.1:8000/"
    data = "Hello, interpreter! What operating system are you on? Also, what time is it in Seattle?"
    headers = {"Content-Type": "text/plain"}

    import threading

    server_thread = threading.Thread(target=interpreter.server)
    server_thread.start()

    import time

    time.sleep(3)

    response = requests.post(url, data=data, headers=headers, stream=True)

    full_response = ""

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            print(decoded_line, end="", flush=True)
            full_response += decoded_line

    assert full_response != ""


@pytest.mark.integration
def test_async():
    """Non-blocking chat returns immediately; wait() collects the final result."""

    interpreter.chat("Hello!", blocking=False)
    print(interpreter.wait())


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_find_text_api():
    """Manual display smoke: mouse.move with a long natural-language target string."""

    start = time.time()
    interpreter.computer.mouse.move(
        "Left Arrow Left Arrow and a bunch of hallucinated text? or was it..."
    )
    # Left Arrow Left Arrow
    # and a bunch of hallucinated text? or was it...
    print(time.time() - start)
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_getActiveWindow():
    """Manual smoke: pywinctl reports the currently active window."""

    import pywinctl

    print(pywinctl.getActiveWindow())
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_notify():
    """Manual smoke: os.notify shows a desktop notification."""

    interpreter.computer.os.notify("Hello")
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_get_text():
    """Manual display smoke: get_text_as_list_of_lists OCRs the screen."""

    print(interpreter.computer.display.get_text_as_list_of_lists())
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_keyboard():
    """Manual smoke: keyboard.write types a long multi-line string."""

    time.sleep(2)
    interpreter.computer.keyboard.write("Hello " * 50 + "\n" + "hi" * 50)
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_get_selected_text():
    """Manual smoke: os.get_selected_text reads the current text selection."""

    print("Getting selected text")
    time.sleep(1)
    text = interpreter.computer.os.get_selected_text()
    print(text)
    assert False


@pytest.mark.skip(reason="Computer with display only + no way to fail test")
def test_display_verbose():
    """Manual verbose smoke: mouse.move logs extra detail when verbose is on."""

    interpreter.computer.verbose = True
    interpreter.verbose = True
    interpreter.computer.mouse.move(x=500, y=500)
    assert False


# this function will run before each test
# we're clearing out the messages Array so we can start fresh and reduce token usage
def setup_function():
    interpreter.reset()
    interpreter.llm.temperature = 0
    interpreter.auto_run = True
    interpreter.llm.model = "gpt-4o-mini"
    interpreter.llm.context_window = 123000
    interpreter.llm.max_tokens = 4096
    interpreter.llm.supports_functions = True
    interpreter.verbose = False


@pytest.mark.skip(
    reason="Not working consistently, I think GPT related changes? It worked recently"
)
def test_long_message():
    """Integration: a very long user message is handled within a tiny context window.

    The model should still recall the four repeated characters from the prompt
    despite aggressive context_window truncation."""

    messages = [
        {
            "role": "user",
            "type": "message",
            "content": "ALKI" * 20000
            + "\nwhat are the four characters I just sent you? don't run ANY code, just tell me the characters. DO NOT RUN CODE. DO NOT PLAN. JUST TELL ME THE CHARACTERS RIGHT NOW. ONLY respond with the 4 characters, NOTHING else. The first 4 characters of your response should be the 4 characters I sent you.",
        }
    ]
    interpreter.llm.context_window = 300
    interpreter.chat(messages)
    assert len(interpreter.messages) > 1
    assert "A" in interpreter.messages[-1]["content"]


# Pause after OpenAI integration tests to reduce API rate-limit errors.
@pytest.fixture(autouse=True)
def _rate_limit_openai_after_integration(request):
    yield
    if request.node.get_closest_marker("integration"):
        time.sleep(4)


@pytest.mark.skip(reason="Mac only — manual harness; no assertion; not for CI")
def test_spotlight():
    """Manual Mac smoke: command+space opens Spotlight.

    Not suitable for CI: drives the GUI with no pass/fail check."""

    interpreter.computer.keyboard.hotkey("command", "space")


@pytest.mark.integration
def test_files(tmp_path):
    """Integration: LLM checks whether a user-attached file path exists on disk."""

    require_bash_compatible_shell()
    # Main used a hardcoded /Users/Killian/... path that only existed on one machine.
    # tmp_path creates a real file on any OS so the LLM can answer the existence question.
    image_file = tmp_path / "image.png"
    image_file.write_bytes(b"fake png content")
    messages = [
        {"role": "user", "type": "message", "content": "Does this file exist?"},
        {
            "role": "user",
            "type": "file",
            "format": "path",
            "content": str(image_file),
        },
    ]
    interpreter.chat(messages)


@pytest.mark.skip(reason="Only 100 vision calls allowed / day!")
def test_vision():
    """Manual vision integration: describe a base64 PNG with supports_vision enabled."""

    base64png = "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"
    messages = [
        {"role": "user", "type": "message", "content": "describe this image"},
        {
            "role": "user",
            "type": "image",
            "format": "base64.png",
            "content": base64png,
        },
    ]

    interpreter.llm.supports_vision = True
    interpreter.llm.model = "gpt-4o-mini"
    interpreter.system_message += "\nThe user will show you an image of the code you write. You can view images directly.\n\nFor HTML: This will be run STATELESSLY. You may NEVER write '<!-- previous code here... --!>' or `<!-- header will go here -->` or anything like that. It is CRITICAL TO NEVER WRITE PLACEHOLDERS. Placeholders will BREAK it. You must write the FULL HTML CODE EVERY TIME. Therefore you cannot write HTML piecemeal—write all the HTML, CSS, and possibly Javascript **in one step, in one code block**. The user will help you review it visually.\nIf the user submits a filepath, you will also see the image. The filepath and user image will both be in the user's message.\n\nIf you use `plt.show()`, the resulting image will be sent to you. However, if you use `PIL.Image.show()`, the resulting image will NOT be sent to you."
    interpreter.llm.supports_functions = True
    interpreter.llm.context_window = 110000
    interpreter.llm.max_tokens = 4096
    interpreter.loop = True

    interpreter.chat(messages)

    interpreter.loop = False


def test_multiple_instances():
    """Each OpenInterpreter instance keeps its own system_message."""

    interpreter.system_message = "i"
    agent_1 = OpenInterpreter()
    agent_1.system_message = "<3"
    agent_2 = OpenInterpreter()
    agent_2.system_message = "u"

    assert interpreter.system_message == "i"
    assert agent_1.system_message == "<3"
    assert agent_2.system_message == "u"


@pytest.mark.integration
def test_hello_world():
    """Integration: LLM replies with exactly 'Hello, World!' and no code execution."""

    hello_world_response = "Hello, World!"

    hello_world_message = f"Please reply with just the words {hello_world_response} and nothing else. Do not run code. No confirmation just the text."

    messages = interpreter.chat(hello_world_message)

    assert messages == [
        {"role": "assistant", "type": "message", "content": hello_world_response}
    ]


@pytest.mark.integration
def test_math():
    """Integration: LLM computes a random order-of-operations expression correctly."""

    min_number = randint(1, 99)
    max_number = randint(1001, 9999)

    n1 = randint(min_number, max_number)
    n2 = randint(min_number, max_number)

    test_result = n1 + n2 * (n1 - n2) / (n2 + n1)

    order_of_operations_message = f"""
    Please perform the calculation `{n1} + {n2} * ({n1} - {n2}) / ({n2} + {n1})` then reply with just the answer, nothing else. No confirmation. No explanation. No words. Do not use commas. Do not show your work. Just return the result of the calculation. Do not introduce the results with a phrase like \"The result of the calculation is...\" or \"The answer is...\"

    Round to 2 decimal places.
    """.strip()

    print("loading")
    messages = interpreter.chat(order_of_operations_message)
    print("done")

    assert str(round(test_result, 2)) in messages[-1]["content"]


@pytest.mark.timeout(120)
def test_break_execution():
    """
    Breaking from the generator while it's executing should halt the operation.
    """

    code = r"""print("starting")
import time
import os

# Always create a fresh file
open('numbers.txt', 'w').close()

# Open the file in append mode
with open('numbers.txt', 'a+') as f:
    # Loop through the numbers 1 to 5
    for i in [1,2,3,4,5]:
        # Print the number
        print("adding", i, "to file")
        # Append the number to the file
        f.write(str(i) + '\n')
        # Wait for 0.5 second
        print("starting to sleep")
        time.sleep(1)
        # # Read the file to make sure the number is in there
        # # Move the seek pointer to the start of the file
        # f.seek(0)
        # # Read the file content
        # content = f.read()
        # print("Current file content:", content)
        # # Check if the current number is in the file content
        # assert str(i) in content
        # Move the seek pointer to the end of the file for the next append operation
        f.seek(0, os.SEEK_END)
        """
    print("starting to code")
    for chunk in interpreter.computer.run("python", code, stream=True, display=True):
        print(chunk)
        if "format" in chunk and chunk["format"] == "output":
            if "adding 3 to file" in chunk["content"]:
                print("BREAKING")
                break

    time.sleep(3)

    # Open the file and read its content
    with open("numbers.txt", "r") as f:
        content = f.read()

    # Check if '1' and '5' are in the content
    assert "1" in content
    assert "5" not in content


@pytest.mark.integration
def test_delayed_exec():
    """Integration: LLM writes and runs code with a delay between print statements."""

    require_bash_compatible_shell()
    interpreter.chat(
        """Can you write a single block of code and execute it that prints something, then delays 1 second, then prints something else? No talk just code, execute the code. Thanks!"""
    )


# Python multiline + a trivial bash one-liner. Nested shell loops with echo
# variables often produce broken quoting and hang subprocess_language; echo is enough
# to verify the LLM can emit and run shell on Linux CI.
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_nested_loops_and_multiple_newlines():
    """Integration: LLM runs spaced Python loops then a single bash echo line."""

    require_bash_compatible_shell()
    messages = interpreter.chat(
        """Can you write a nested for loop in python and run it? Put 1-3 newlines between each line in the python code.

Then run exactly one line of bash (not python): echo shell_ok

Only generate and execute the code. Execute instantly. No explanations. Thanks!"""
    )
    combined = " ".join(
        str(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
    assert "shell_ok" in combined


@pytest.mark.integration
def test_write_to_file(monkeypatch, tmp_path):
    """Integration: LLM writes a file, then reads it back in a follow-up turn."""

    require_bash_compatible_shell()
    # Run in a temp directory so the LLM-generated file.txt does not land in the
    # repo root or the developer's home folder. monkeypatch.chdir restores cwd after
    # the test regardless of pass/fail.
    monkeypatch.chdir(tmp_path)
    interpreter.chat(
        """Write the word 'Washington' to a .txt file called file.txt. Instantly run the code! Save the file!"""
    )
    assert (tmp_path / "file.txt").exists()
    interpreter.messages = []  # Just reset message history, nothing else for this test
    messages = interpreter.chat(
        """Read file.txt in the current directory and tell me what's in it."""
    )
    assert "Washington" in messages[-1]["content"]


@pytest.mark.integration
def test_markdown():
    """Integration: LLM emits assorted markdown features in a single reply."""

    interpreter.chat(
        """Hi, can you test out a bunch of markdown features? Try writing a fenced code block, a table, headers, everything. DO NOT write the markdown inside a markdown code block, just write it raw."""
    )


def test_reset():
    """setup_function leaves messages empty after interpreter.reset()."""

    assert interpreter.messages == []


def test_token_counter():
    """count_tokens and count_messages_tokens agree for system and user prompts."""

    system_tokens = count_tokens(
        text=interpreter.system_message, model=interpreter.llm.model
    )

    prompt = "How many tokens is this?"

    prompt_tokens = count_tokens(text=prompt, model=interpreter.llm.model)

    messages = [
        {"role": "system", "message": interpreter.system_message}
    ] + interpreter.messages

    system_token_test = count_messages_tokens(
        messages=messages, model=interpreter.llm.model
    )

    system_tokens_ok = system_tokens == system_token_test[0]

    messages.append({"role": "user", "message": prompt})

    prompt_token_test = count_messages_tokens(
        messages=messages, model=interpreter.llm.model
    )

    prompt_tokens_ok = system_tokens + prompt_tokens == prompt_token_test[0]

    assert system_tokens_ok and prompt_tokens_ok
