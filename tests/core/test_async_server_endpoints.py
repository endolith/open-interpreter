import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from interpreter.core.async_core import AsyncInterpreter, Server


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Hide INTERPRETER_API_KEY from the auth middleware for every test here.

    The middleware resolves `server.authenticate` per request and
    `authenticate_function` re-reads the environment each time, so an exported
    key would answer 403 to every request below and fail them for a reason
    unrelated to what they check. No test in this file sets the key itself, so
    clearing it cannot mask an assertion about it.
    """
    monkeypatch.delenv("INTERPRETER_API_KEY", raising=False)


@pytest.fixture
def server_pair():
    """Build a (TestClient, AsyncInterpreter) pair for a fresh server."""
    interpreter = AsyncInterpreter()
    return TestClient(Server(interpreter).app), interpreter


@pytest.fixture
def interpreter():
    """A fresh AsyncInterpreter for building ad-hoc test servers."""
    return AsyncInterpreter()


@pytest.fixture
def client(server_pair):
    return server_pair[0]


@pytest.fixture
def client_no_raise(interpreter):
    """TestClient that converts endpoint exceptions into 500 responses."""
    return TestClient(Server(interpreter).app, raise_server_exceptions=False)


@pytest.fixture
def insecure_pair(monkeypatch):
    """(TestClient, AsyncInterpreter) pair with the insecure routes registered.

    create_router() reads INTERPRETER_INSECURE_ROUTES at router build time, so
    the env var must be set before Server() is constructed.
    """
    monkeypatch.setenv("INTERPRETER_INSECURE_ROUTES", "true")
    interpreter = AsyncInterpreter()
    return TestClient(Server(interpreter).app), interpreter


def test_heartbeat_endpoint(client):
    """GET /heartbeat returns a simple aliveness payload."""
    response = client.get("/heartbeat")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_home_endpoint_serves_html(client):
    """GET / serves the HTML chat page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text


def test_post_input_returns_success(client):
    """POST / accepts an LMC chunk and reports success."""
    response = client.post("/", json={"role": "user", "type": "message", "start": True})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_post_input_propagates_error(client, server_pair):
    """POST / swallows input() failures and reports success.

    KNOWN BUG: the handler calls async_interpreter.input(payload) without
    await, so the coroutine is never awaited and its exception is never
    raised. It responds 200 "success" and the error path is dead code. This
    test documents the current (wrong) behavior.
    """
    _, interpreter = server_pair
    interpreter.input = mock.AsyncMock(side_effect=ValueError("boom"))
    response = client.post("/", json={"role": "user", "type": "message", "start": True})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_get_setting_returns_serialized_value(client):
    """GET /settings/{name} returns the serialized interpreter attribute."""
    response = client.get("/settings/auto_run")
    assert response.status_code == 200
    assert json.loads(json.loads(response.text)) == {"auto_run": False}


def test_get_setting_unknown_name_returns_200_with_tuple_body(client):
    """GET /settings/{name} for an unknown name returns 200 with an array body.

    KNOWN BUG: the handler returns a Flask-style (content, status) tuple
    which FastAPI does not interpret as a status code. The response is HTTP
    200 with a JSON array [..., 404] instead of a 404 response.
    """
    response = client.get("/settings/no_such_setting")
    assert response.status_code == 200
    assert response.json() == [json.dumps({"error": "Setting not found"}), 404]


def test_post_settings_unknown_llm_subsetting_returns_200_with_tuple_body(client):
    """POST /settings with an unknown llm sub-key returns 200 with an array body.

    KNOWN BUG: same Flask-style tuple issue; should be a 404 response but is
    HTTP 200 with a JSON array [..., 404].
    """
    response = client.post("/settings", json={"llm": {"no_such_subkey": True}})
    assert response.status_code == 200
    assert response.json() == [
        {"error": "Sub-setting no_such_subkey not found in llm"},
        404,
    ]


def test_post_settings_unknown_top_level_key_returns_200_with_tuple_body(client):
    """POST /settings with an unknown top-level key returns 200 with an array body.

    KNOWN BUG: same Flask-style tuple issue; should be a 404 response but is
    HTTP 200 with a JSON array [..., 404].
    """
    response = client.post("/settings", json={"no_such_setting": True})
    assert response.status_code == 200
    assert response.json() == [{"error": "Setting no_such_setting not found"}, 404]


def test_chat_completion_rejects_non_user_last_message(client_no_raise):
    """The OpenAI-compatible endpoint requires the last message to be from the user."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert response.status_code == 500


def test_chat_completion_stop_token(client, server_pair):
    """The {STOP} sentinel sets then clears the stop event without a response."""
    _, interpreter = server_pair
    with mock.patch("interpreter.core.async_core.time.sleep"):
        response = client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": "{STOP}"}]},
        )
    assert response.status_code == 200
    assert not interpreter.stop_event.is_set()


def test_chat_completion_context_mode_on(client, server_pair):
    """{CONTEXT_MODE_ON} and {REQUIRE_START_ON} enable context mode."""
    _, interpreter = server_pair
    for token in ["{CONTEXT_MODE_ON}", "{REQUIRE_START_ON}"]:
        client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": token}]},
        )
        assert interpreter.context_mode is True


def test_chat_completion_context_mode_off(client, server_pair):
    """{CONTEXT_MODE_OFF} and {REQUIRE_START_OFF} disable context mode."""
    _, interpreter = server_pair
    interpreter.context_mode = True
    for token in ["{CONTEXT_MODE_OFF}", "{REQUIRE_START_OFF}"]:
        client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": token}]},
        )
        assert interpreter.context_mode is False


def test_chat_completion_auto_run_toggle(client, server_pair):
    """{AUTO_RUN_ON} / {AUTO_RUN_OFF} flip the auto_run flag."""
    _, interpreter = server_pair
    client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "{AUTO_RUN_ON}"}]},
    )
    assert interpreter.auto_run is True
    client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "{AUTO_RUN_OFF}"}]},
    )
    assert interpreter.auto_run is False


def test_chat_completion_text_message_returns_assistant_reply(client, server_pair):
    """A plain text user message is appended and answered via chat()."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "Hello there"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello there"
    assert any(
        m.get("content") == "hello" for m in interpreter.messages
    ), "the user message should be stored"


def test_chat_completion_list_text_content(client, server_pair):
    """A content list with a text part is appended as a user message."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]
        },
    )
    assert response.status_code == 200
    assert any(
        m.get("type") == "message" for m in interpreter.messages
    ), "the text part should become a message"


def test_chat_completion_list_base64_image(client, server_pair):
    """A content list with a base64 image becomes an image message."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "content": "ok"}]
    )
    url = "data:image/png;base64,iVBORw0KGgo="
    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": url}}],
                }
            ]
        },
    )
    assert response.status_code == 200
    image_messages = [m for m in interpreter.messages if m.get("type") == "image"]
    assert len(image_messages) == 1
    assert image_messages[0]["format"] == "base64.png"
    assert image_messages[0]["content"] == "iVBORw0KGgo="


def test_chat_completion_image_url_without_url_raises(client_no_raise):
    """An image_url part missing the url field is rejected."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
            ]
        },
    )
    assert response.status_code == 500


def test_chat_completion_image_url_without_base64_raises(client_no_raise):
    """An image_url that is not a base64 data URI is rejected."""
    response = client_no_raise.post(
        "/openai/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            ]
        },
    )
    assert response.status_code == 500


def test_chat_completion_stream_run_code(client, server_pair):
    """Streaming a 'yes' after a code message streams chunks from _respond_and_store."""
    _, interpreter = server_pair
    interpreter.messages = [
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
        }
    ]
    interpreter._respond_and_store = mock.MagicMock(
        return_value=iter(
            [
                {"role": "assistant", "type": "message", "content": "hi"},
                {"role": "assistant", "type": "code", "start": True, "format": "python"},
                {"role": "assistant", "type": "code", "end": True, "format": "python"},
            ]
        )
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )
    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text
    assert "hi" in response.text


def test_chat_completion_stream_confirmation_breaks(client, server_pair):
    """A confirmation chunk with auto_run off asks and stops streaming."""
    _, interpreter = server_pair
    interpreter.auto_run = False
    interpreter._respond_and_store = mock.MagicMock(
        return_value=iter(
            [
                {
                    "role": "computer",
                    "type": "confirmation",
                    "content": {"format": "python", "content": "print(1)"},
                }
            ]
        )
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )
    assert response.status_code == 200
    assert "Do you want to run this code?" in response.text


def test_chat_completion_stream_message_via_chat(client, server_pair):
    """Streaming a normal message streams chunks produced by chat()."""
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert response.status_code == 200
    assert "hi" in response.text


def test_insecure_routes_not_registered_by_default(monkeypatch):
    """The /run route is absent unless INTERPRETER_INSECURE_ROUTES is enabled."""
    monkeypatch.delenv("INTERPRETER_INSECURE_ROUTES", raising=False)
    interpreter = AsyncInterpreter()
    client = TestClient(Server(interpreter).app)
    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 404


def test_insecure_run_route(insecure_pair):
    """With INTERPRETER_INSECURE_ROUTES=true, /run executes code via the computer."""
    client, interpreter = insecure_pair
    interpreter.computer.run = mock.MagicMock(return_value="42")

    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 200
    assert response.json() == {"output": "42"}


def test_insecure_run_route_requires_language_and_code(insecure_pair):
    """The /run route reports 200 with an array body when code is missing.

    KNOWN BUG: Flask-style (content, status) tuple return; FastAPI does not
    interpret the second element as a status code, so the response is HTTP
    200 with a JSON array [..., 400] instead of a 400.
    """
    client, _ = insecure_pair

    response = client.post("/run", json={"language": "python"})
    assert response.status_code == 200
    assert response.json() == [
        {"error": "Both 'language' and 'code' are required."},
        400,
    ]


def test_insecure_run_route_propagates_error(insecure_pair):
    """The /run route reports 200 with an array body when computer.run raises.

    KNOWN BUG: same Flask-style tuple issue; should be a 500 response but is
    HTTP 200 with a JSON array [..., 500].
    """
    client, interpreter = insecure_pair
    interpreter.computer.run = mock.MagicMock(side_effect=RuntimeError("oops"))

    response = client.post("/run", json={"language": "python", "code": "1+1"})
    assert response.status_code == 200
    assert response.json() == [{"error": "oops"}, 500]


def test_insecure_upload_route(insecure_pair, tmp_path):
    """The /upload route writes the uploaded file to the requested path."""
    client, _ = insecure_pair
    target = tmp_path / "out.txt"

    response = client.post(
        "/upload",
        files={"file": ("in.txt", b"payload")},
        data={"path": str(target)},
    )
    assert response.status_code == 200
    assert target.read_text() == "payload"


def test_insecure_upload_route_propagates_error(insecure_pair):
    """The /upload route reports 200 with an array body when it cannot write.

    KNOWN BUG: same Flask-style tuple issue; should be a 500 response but is
    HTTP 200 with a JSON array [..., 500].
    """
    client, _ = insecure_pair

    response = client.post(
        "/upload",
        files={"file": ("in.txt", b"payload")},
        data={"path": "/no/such/dir/out.txt"},
    )
    assert response.status_code == 200
    assert response.json() == [
        {"error": "[Errno 2] No such file or directory: '/no/such/dir/out.txt'"},
        500,
    ]


def test_insecure_download_route_with_slashes_not_registered(insecure_pair):
    """The /download route does not match paths containing slashes.

    KNOWN BUG: the route is declared as /download/{filename}, which matches a
    single path segment. Any path with a slash (including the absolute paths
    the endpoint is meant to serve) falls through to 404. Documenting current
    behavior; a fix would use {filename:path}.
    """
    client, _ = insecure_pair
    response = client.get("/download/some/dir/file.bin")
    assert response.status_code == 404


def test_insecure_download_route_missing_file_reports_200_with_array_body(
    insecure_pair,
):
    """A missing file on /download reports 200 with an array body.

    KNOWN BUG: same Flask-style tuple issue; should be a 500 response but is
    HTTP 200 with a JSON array [..., 500].
    """
    client, _ = insecure_pair

    response = client.get("/download/nope.bin")
    assert response.status_code == 200
    assert response.json() == [
        {"error": "[Errno 2] No such file or directory: 'nope.bin'"},
        500,
    ]


def _deltas(body):
    """Extract the streamed delta content strings from an SSE response body."""
    deltas = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[len("data: ") :])
        deltas.append(payload["choices"][0]["delta"]["content"])
    return deltas


def _frames(body):
    """Extract every decoded SSE frame from a response body."""
    frames = []
    for line in body.splitlines():
        if line.startswith("data: "):
            frames.append(json.loads(line[len("data: ") :]))
    return frames


def _code_stream_pair(server_pair, chunks):
    """Point a server's interpreter at a canned run-code chunk stream."""
    _, interpreter = server_pair
    interpreter.messages = [
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "print(1)",
        }
    ]
    interpreter._respond_and_store = mock.MagicMock(return_value=iter(chunks))
    return interpreter


def test_stream_wraps_generated_code_in_a_markdown_fence(client, server_pair):
    """Code start/end chunks bracket the code as a fenced block.

    An OpenAI-compatible client renders the deltas verbatim, so without the
    fences a caller would display bare source with no code block around it.
    """
    _code_stream_pair(
        server_pair,
        [
            {"role": "assistant", "type": "code", "start": True, "format": "python"},
            {"role": "assistant", "type": "code", "content": "print(1)", "format": "python"},
            {"role": "assistant", "type": "code", "end": True, "format": "python"},
        ],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert _deltas(response.text) == ["```python\n", "print(1)", "\n```\n"]


def test_stream_fence_names_the_chunk_language(client, server_pair):
    """The opening fence carries the chunk's own format, not a fixed language.

    Code streamed as javascript has to be labelled as such or a client will
    highlight Python that is not Python.
    """
    _code_stream_pair(
        server_pair,
        [
            {"role": "assistant", "type": "code", "start": True, "format": "javascript"},
            {"role": "assistant", "type": "code", "end": True, "format": "javascript"},
        ],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert _deltas(response.text)[0] == "```javascript\n"


def test_stream_skips_chunks_with_nothing_to_render(client, server_pair):
    """A chunk with no content, no fence marker and no message role emits no frame.

    Carriage such as active_line updates and bare end-of-stream markers would
    otherwise reach the client as empty deltas and blank out the rendered message.
    """
    _code_stream_pair(
        server_pair,
        [
            {"role": "assistant", "type": "message", "content": "visible"},
            {
                "role": "output",
                "type": "console",
                "format": "active_line",
                "content": "ls -la",
            },
            {"role": "assistant", "type": "message"},
        ],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert _deltas(response.text) == ["visible"]


def test_stream_breaks_on_a_chunk_with_no_type(client, server_pair):
    """A chunk missing `type` raises inside the generator and truncates the stream.

    The generator reads chunk["type"] unguarded on every chunk, so a producer that
    omits the field kills the response mid-flight. Headers are already sent by then,
    so the caller sees a truncated 200 rather than an error. Pinned as-is: this
    campaign is characterization only, and the fix belongs with the producer that
    should always set `type`.
    """
    _code_stream_pair(
        server_pair,
        [
            {"role": "assistant", "type": "message", "content": "visible"},
            {"role": "output", "format": "active_line", "content": "ls -la"},
        ],
    )

    with pytest.raises(KeyError):
        client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
        )


def test_stream_frames_carry_the_openai_chunk_envelope(client, server_pair):
    """Each frame identifies itself as a chat.completion.chunk from open-interpreter.

    Clients switch on `object` to tell a chunk from a terminal response, and some
    display the `model` field, so both have to be present and correct.
    """
    _code_stream_pair(
        server_pair,
        [{"role": "assistant", "type": "message", "content": "hi"}],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    frame = _frames(response.text)[0]
    assert frame["object"] == "chat.completion.chunk"
    assert frame["model"] == "open-interpreter"
    assert isinstance(frame["created"], (int, float))


def test_stream_frame_ids_increase_with_each_chunk(client, server_pair):
    """Frame ids follow the source chunk index so a client can detect gaps.

    A repeated or reset id would make a client believe a chunk was retransmitted
    and duplicate content it has already rendered.
    """
    _code_stream_pair(
        server_pair,
        [
            {"role": "assistant", "type": "message", "content": "one"},
            {"role": "assistant", "type": "message", "content": "two"},
        ],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert [frame["id"] for frame in _frames(response.text)] == [0, 1]


def test_stream_is_served_as_ndjson(client, server_pair):
    """The streaming response is typed application/x-ndjson."""
    _code_stream_pair(
        server_pair,
        [{"role": "assistant", "type": "message", "content": "hi"}],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert response.headers["content-type"].startswith("application/x-ndjson")


def test_stream_ends_without_a_done_sentinel(client, server_pair):
    """The stream closes by ending, with no `data: [DONE]` frame.

    KNOWN GAP: the OpenAI streaming protocol terminates with a `[DONE]` sentinel,
    and strict clients wait for it rather than treating EOF as completion. Pinned
    as-is because this campaign is characterization only.
    """
    _code_stream_pair(
        server_pair,
        [{"role": "assistant", "type": "message", "content": "hi"}],
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert "[DONE]" not in response.text


def test_non_stream_envelope_echoes_the_requested_model(client, server_pair):
    """A non-streaming reply reports the model the caller asked for.

    Clients route on the model name, so answering with a hardcoded one would make
    every request look like it came from the same backend.
    """
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-4o-mini",
        },
    )

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o-mini"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hi"}


def test_non_stream_model_defaults_to_default_model(client, server_pair):
    """Omitting the model yields the documented `default-model` placeholder.

    A different default would be echoed straight back to the caller, so the
    placeholder is part of the request contract.
    """
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.json()["model"] == "default-model"


def test_stream_model_is_fixed_regardless_of_the_request(client, server_pair):
    """Streaming frames report `open-interpreter` even when a model was requested.

    This is inconsistent with the non-streaming path, which echoes the request.
    Pinned deliberately so the divergence is visible rather than accidental.
    """
    _code_stream_pair(
        server_pair,
        [{"role": "assistant", "type": "message", "content": "hi"}],
    )

    response = client.post(
        "/openai/chat/completions",
        json={
            "messages": [{"role": "user", "content": "yes"}],
            "model": "gpt-4o-mini",
            "stream": True,
        },
    )

    assert _frames(response.text)[0]["model"] == "open-interpreter"


def test_confirmation_does_not_ask_when_auto_run_is_on(client, server_pair):
    """With auto_run enabled a confirmation is not turned into a prompt.

    The code is already authorised to run, so asking "do you want to run this
    code?" would stall an unattended caller on a question it cannot answer.
    """
    _, interpreter = server_pair
    interpreter.auto_run = True
    interpreter.chat = mock.MagicMock(
        return_value=[
            {
                "role": "computer",
                "type": "confirmation",
                "content": {"format": "python", "content": "print(1)"},
            }
        ]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "yes"}], "stream": True},
    )

    assert "Do you want to run this code?" not in response.text


def test_stream_stops_when_the_stop_flag_is_set_mid_response(client, server_pair):
    """A stop raised after the first chunk ends the stream there.

    The chunk already in hand is dropped rather than forwarded, so an interrupted
    turn does not leak a partial delta the caller never cancelled.
    """
    _, interpreter = server_pair
    interpreter.auto_run = False

    def chat_with_stop(*args, **kwargs):
        """Yield one message chunk, then raise the stop flag before the next."""
        yield {"role": "assistant", "type": "message", "content": "first"}
        interpreter.stop_event.set()
        yield {"role": "assistant", "type": "message", "content": "dropped"}

    interpreter.chat = chat_with_stop

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert _deltas(response.text) == ["first"]


def test_silent_model_is_nudged_with_escalating_prompts(client, server_pair):
    """A model that returns nothing is retried with progressively blunter prompts.

    This is the recovery path for a silent provider, so both the number of
    attempts and their distinctness matter: repeating one prompt would make the
    retries useless.
    """
    _, interpreter = server_pair
    interpreter.auto_run = False
    interpreter.chat = mock.MagicMock(return_value=[])

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    attempted = [call.kwargs["message"] for call in interpreter.chat.call_args_list]
    assert len(attempted) == 6
    assert len(set(attempted)) == 6
    assert attempted[0] == "."


def test_nudge_loop_stops_as_soon_as_a_reply_arrives(client, server_pair):
    """A reply on the first attempt means no further nudges are sent.

    Otherwise every answered request would pay for the whole retry ladder and
    leave six spurious turns in the transcript.
    """
    _, interpreter = server_pair
    interpreter.auto_run = False
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert "hi" in response.text
    assert interpreter.chat.call_count == 1


def test_context_mode_does_not_suppress_a_plain_message(client, server_pair):
    """With context mode enabled, an ordinary message is still answered.

    KNOWN GAP: context mode is documented as accumulating context without
    replying until a {START} arrives, but the only code that reads
    `context_mode` sits in a branch entered solely when message content is
    neither str nor list — which the request schema cannot produce. So the flag is
    stored and reported as set, yet gates nothing reachable. Pinned as-is; the fix
    is to move the check into the path a real request actually takes.
    """
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )
    client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "{CONTEXT_MODE_ON}"}]},
    )
    assert interpreter.context_mode is True

    response = client.post(
        "/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert interpreter.chat.call_count == 1


def test_content_that_is_neither_text_nor_a_list_is_rejected(client, server_pair):
    """Only str and list message content is accepted, closing off the other branch.

    Every other JSON type fails validation, which is what makes the context-mode
    branch below the str/list dispatch unreachable. Pinned so that if the schema
    ever widens, this test is the one that notices the branch became reachable.
    """
    _, interpreter = server_pair
    interpreter.chat = mock.MagicMock(
        return_value=[{"role": "assistant", "type": "message", "content": "hi"}]
    )

    for content in (123, 4.5, True, None, {"text": "hi"}):
        response = client.post(
            "/openai/chat/completions",
            json={"messages": [{"role": "user", "content": content}]},
        )
        assert response.status_code == 422, content

    assert interpreter.chat.call_count == 0
