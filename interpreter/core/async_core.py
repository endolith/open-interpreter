import asyncio
import json
import os
import shutil
import socket
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import shortuuid
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from .core import OpenInterpreter

last_start_time = 0

# fastapi / uvicorn / janus are required only for AsyncInterpreter + Server. Import
# failures must leave defined names so this module loads; Server.__init__ raises clearly.
try:
    import janus
    import uvicorn
    from fastapi import (
        APIRouter,
        FastAPI,
        File,
        Form,
        HTTPException,
        Request,
        UploadFile,
        WebSocket,
    )
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from starlette.status import HTTP_403_FORBIDDEN
except ImportError:
    janus = None
    uvicorn = None
    APIRouter = None
    FastAPI = None
    File = Form = HTTPException = Request = UploadFile = WebSocket = None
    JSONResponse = PlainTextResponse = StreamingResponse = None
    HTTP_403_FORBIDDEN = 403


complete_message = {"role": "server", "type": "status", "content": "complete"}


class AsyncInterpreter(OpenInterpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.respond_thread = None
        self.stop_event = threading.Event()
        self.output_queue = None
        self.unsent_messages = deque()
        self.id = os.getenv("INTERPRETER_ID", datetime.now().timestamp())
        self.print = False  # Will print output

        self.require_acknowledge = (
            os.getenv("INTERPRETER_REQUIRE_ACKNOWLEDGE", "False").lower() == "true"
        )
        self.acknowledged_outputs = []
        self._server_request_system = None
        self._server_awaiting_code_approval = False

        try:
            self.server = Server(self)
        except ImportError:
            self.server = None

        # For the 01. This lets the OAI compatible server accumulate context before responding.
        self.context_mode = False

        # No terminal to answer view_image prompts; websocket clients typically do not reply.
        # OpenInterpreter() in CLI uses terminal_interface for approval instead.
        self._view_image_approval = "y"

    async def input(self, chunk):
        """
        Accumulates LMC chunks onto interpreter.messages.
        When it hits an "end" flag, calls interpreter.respond().
        """

        if "start" in chunk:
            # If the user is starting something, the interpreter should stop.
            if self.respond_thread is not None and self.respond_thread.is_alive():
                self.stop_event.set()
                self.respond_thread.join()
            self.accumulate(chunk)
        elif "content" in chunk:
            self.accumulate(chunk)
        elif "end" in chunk:
            # If the user is done talking, the interpreter should respond.

            run_code = None  # Will later default to auto_run unless the user makes a command here

            # But first, process any commands.
            if self.messages[-1].get("type") == "command":
                command = self.messages[-1]["content"]
                self.messages = self.messages[:-1]

                if command == "stop":
                    # Any start flag would have stopped it a moment ago, but to be sure:
                    self.stop_event.set()
                    self.respond_thread.join()
                    return
                if command == "go":
                    # This is to approve code.
                    run_code = True
                    pass

            self.stop_event.clear()
            self.respond_thread = threading.Thread(
                target=self.respond, args=(run_code,)
            )
            self.respond_thread.start()

    async def output(self):
        if self.output_queue == None:
            self.output_queue = janus.Queue()
        return await self.output_queue.async_q.get()

    def respond(self, run_code=None):
        for attempt in range(5):  # 5 attempts
            try:
                if run_code == None:
                    run_code = self.auto_run

                sent_chunks = False

                for chunk_og in self._respond_and_store():
                    chunk = (
                        chunk_og.copy()
                    )  # This fixes weird double token chunks. Probably a deeper problem?

                    if chunk["type"] == "confirmation":
                        if run_code:
                            run_code = False
                            continue
                        else:
                            break

                    if self.stop_event.is_set():
                        return

                    if self.print:
                        if "start" in chunk:
                            print("\n")
                        if chunk["type"] in ["code", "console"] and "format" in chunk:
                            if "start" in chunk:
                                print(
                                    "\n------------\n\n```" + chunk["format"],
                                    flush=True,
                                )
                            if "end" in chunk:
                                print("\n```\n\n------------\n\n", flush=True)
                        if chunk.get("format") != "active_line":
                            if "format" in chunk and "base64" in chunk["format"]:
                                print("\n[An image was produced]")
                            else:
                                content = chunk.get("content", "")
                                content = (
                                    str(content)
                                    .encode("ascii", "ignore")
                                    .decode("ascii")
                                )
                                print(content, end="", flush=True)

                    if self.debug:
                        print("Interpreter produced this chunk:", chunk)

                    self.output_queue.sync_q.put(chunk)
                    sent_chunks = True

                if not sent_chunks:
                    print("ERROR. NO CHUNKS SENT. TRYING AGAIN.")
                    print("Messages:", self.messages)
                    messages = [
                        "Hello? Answer please.",
                        "Just say something, anything.",
                        "Are you there?",
                        "Can you respond?",
                        "Please reply.",
                    ]
                    self.messages.append(
                        {
                            "role": "user",
                            "type": "message",
                            "content": messages[attempt % len(messages)],
                        }
                    )
                    time.sleep(1)
                else:
                    self.output_queue.sync_q.put(complete_message)
                    if self.debug:
                        print("\nServer response complete.\n")
                    return

            except Exception as e:
                error = traceback.format_exc() + "\n" + str(e)
                error_message = {
                    "role": "server",
                    "type": "error",
                    "content": traceback.format_exc() + "\n" + str(e),
                }
                self.output_queue.sync_q.put(error_message)
                self.output_queue.sync_q.put(complete_message)
                print("\n\n--- SENT ERROR: ---\n\n")
                print(error)
                print("\n\n--- (ERROR ABOVE WAS SENT) ---\n\n")
                return

        error_message = {
            "role": "server",
            "type": "error",
            "content": "No chunks sent or unknown error.",
        }
        self.output_queue.sync_q.put(error_message)
        self.output_queue.sync_q.put(complete_message)
        raise Exception("No chunks sent or unknown error.")

    def accumulate(self, chunk):
        """
        Accumulates LMC chunks onto interpreter.messages.
        """
        if type(chunk) == str:
            chunk = json.loads(chunk)

        if type(chunk) == dict:
            if chunk.get("format") == "active_line":
                # We don't do anything with these.
                pass

            elif "content" in chunk and not (
                len(self.messages) > 0
                and (
                    (
                        "type" in self.messages[-1]
                        and chunk.get("type") != self.messages[-1].get("type")
                    )
                    or (
                        "format" in self.messages[-1]
                        and chunk.get("format") != self.messages[-1].get("format")
                    )
                )
            ):
                if len(self.messages) == 0:
                    raise Exception(
                        "You must send a 'start: True' chunk first to create this message."
                    )
                # Append to an existing message
                if (
                    "type" not in self.messages[-1]
                ):  # It was created with a type-less start message
                    self.messages[-1]["type"] = chunk["type"]
                if (
                    chunk.get("format") and "format" not in self.messages[-1]
                ):  # It was created with a type-less start message
                    self.messages[-1]["format"] = chunk["format"]
                if "content" not in self.messages[-1]:
                    self.messages[-1]["content"] = chunk["content"]
                else:
                    self.messages[-1]["content"] += chunk["content"]

            # elif "content" in chunk and (len(self.messages) > 0 and self.messages[-1] == {'role': 'user', 'start': True}):
            #     # Last message was {'role': 'user', 'start': True}. Just populate that with this chunk
            #     self.messages[-1] = chunk.copy()

            elif "start" in chunk or (
                len(self.messages) > 0
                and (
                    chunk.get("type") != self.messages[-1].get("type")
                    or chunk.get("format") != self.messages[-1].get("format")
                )
            ):
                # Create a new message
                chunk_copy = (
                    chunk.copy()
                )  # So we don't modify the original chunk, which feels wrong.
                if "start" in chunk_copy:
                    chunk_copy.pop("start")
                if "content" not in chunk_copy:
                    chunk_copy["content"] = ""
                self.messages.append(chunk_copy)

        elif type(chunk) == bytes:
            if self.messages[-1]["content"] == "":  # We initialize as an empty string ^
                self.messages[-1]["content"] = b""  # But it actually should be bytes
            self.messages[-1]["content"] += chunk


def authenticate_function(key):
    """
    This function checks if the provided key is valid for authentication.

    Returns True if the key is valid, False otherwise.
    """
    # Fetch the API key from the environment variables. If it's not set, return True.
    api_key = os.getenv("INTERPRETER_API_KEY", None)

    # If the API key is not set in the environment variables, return True.
    # Otherwise, check if the provided key matches the fetched API key.
    # Return True if they match, False otherwise.
    if api_key is None:
        return True
    else:
        return key == api_key


# Blank line before the closing --- so Chatbox does not treat it as a setext heading.
OPENAI_CODE_APPROVAL_PROMPT = (
    "\n\n---\n"
    "**[Open Interpreter]** Execution is paused. "
    'Reply with exactly **yes** to run this code or **no** to skip.\n'
    "\n---\n"
)

OPENAI_CODE_APPROVAL_DECLINED = (
    "\n\n---\n**[Open Interpreter]** Okay, I won't run that code.\n\n---\n"
)

OPENAI_CODE_APPROVAL_INVALID_TEMPLATE = (
    "\n\n---\n**[Open Interpreter]** There is code waiting for your approval. "
    'Reply with exactly **yes** or **no** (you sent: "{reply}").\n'
    "\n---\n"
)

OPENAI_SHELL_OUTPUT_NOTE = "Note: Shell command output will be shown after completion."


def _normalize_openai_code_approval_reply(content):
    """Return ``yes``, ``no``, or ``None`` for a pending-code approval turn."""
    if not isinstance(content, str):
        return None
    normalized = content.lower().strip().strip(".!?,")
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    return None


def _openai_server_has_pending_code(async_interpreter):
    return bool(
        async_interpreter.messages
        and async_interpreter.messages[-1].get("type") == "code"
        and not async_interpreter.auto_run
    )


def _openai_server_awaiting_approval(async_interpreter):
    if async_interpreter.auto_run:
        return False
    if getattr(async_interpreter, "_server_awaiting_code_approval", False):
        return True
    return _openai_server_has_pending_code(async_interpreter)


def _clear_openai_code_approval_wait(async_interpreter):
    async_interpreter._server_awaiting_code_approval = False


def _cancel_pending_code(async_interpreter):
    _clear_openai_code_approval_wait(async_interpreter)
    if async_interpreter.messages and async_interpreter.messages[-1].get("type") == "code":
        async_interpreter.messages.append(
            {
                "role": "user",
                "type": "message",
                "content": "[User moved on without approving the pending code.]",
                "source": "server",
            }
        )


def _format_openai_console_output(content, language="bash"):
    if not content or not str(content).strip():
        return None
    text = str(content)
    if OPENAI_SHELL_OUTPUT_NOTE in text and text.strip() == OPENAI_SHELL_OUTPUT_NOTE.strip():
        return None
    text = text.replace(OPENAI_SHELL_OUTPUT_NOTE, "").strip()
    if not text:
        return None
    fence_lang = (language or "text").lower()
    if fence_lang == "shell":
        fence_lang = "bash"
    return f"\n\n```{fence_lang}\n{text}\n```\n\n"


def _openai_apply_request_messages(async_interpreter, request, last_message):
    """Keep server-side LMC state; Chatbox replays plain-text history each POST."""
    _, client_system = _openai_messages_to_lmc(request.messages)
    async_interpreter._server_request_system = client_system

    if _openai_server_awaiting_approval(async_interpreter):
        return

    if not async_interpreter.messages:
        async_interpreter.messages, _ = _openai_messages_to_lmc(request.messages)
        return

    if isinstance(last_message.content, str):
        user_msg = {
            "role": "user",
            "type": "message",
            "content": last_message.content,
        }
        if (
            async_interpreter.messages
            and async_interpreter.messages[-1].get("role") == "user"
            and async_interpreter.messages[-1].get("content") == user_msg["content"]
        ):
            return
        async_interpreter.messages.append(user_msg)
    elif isinstance(last_message.content, list):
        for part in last_message.content:
            if part.get("type") == "text":
                user_msg = {
                    "role": "user",
                    "type": "message",
                    "content": part.get("text", ""),
                }
                if (
                    async_interpreter.messages
                    and async_interpreter.messages[-1].get("role") == "user"
                    and async_interpreter.messages[-1].get("content")
                    == user_msg["content"]
                ):
                    continue
                async_interpreter.messages.append(user_msg)


def _is_openai_auxiliary_title_request(content):
    """Open WebUI and similar UIs send title prompts as normal chat completions."""
    if not isinstance(content, str):
        return False
    lower = content.lower()
    markers = (
        "give this conversation a name",
        "name this conversation",
        "conversation a name",
        "short title for this conversation",
        "generating a short title",
    )
    return any(marker in lower for marker in markers)


def _openai_messages_to_lmc(openai_messages):
    """Map an OpenAI-style messages array to LMC user/assistant turns.

    Client ``system`` messages are returned separately; ``respond()`` already
    prepends Open Interpreter's system message and ``llm.run`` forbids a second.
    """
    lmc = []
    client_system_parts = []
    for msg in openai_messages:
        role = msg.role
        content = msg.content
        if role == "system":
            if isinstance(content, str) and content.strip():
                client_system_parts.append(content.strip())
            continue
        if role == "assistant":
            text = content if isinstance(content, str) else str(content)
            if text:
                lmc.append({"role": "assistant", "type": "message", "content": text})
        elif role == "user":
            if isinstance(content, str):
                lmc.append({"role": "user", "type": "message", "content": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        lmc.append(
                            {
                                "role": "user",
                                "type": "message",
                                "content": part.get("text", ""),
                            }
                        )
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if "base64," not in url:
                            raise ValueError(
                                'Image must be "data:image/jpeg;base64,{data}"'
                            )
                        data = url.split("base64,")[1]
                        fmt = "base64." + url.split(";")[0].split("/")[1]
                        lmc.append(
                            {
                                "role": "user",
                                "type": "image",
                                "format": fmt,
                                "content": data,
                            }
                        )
    client_system = (
        "\n\n".join(client_system_parts) if client_system_parts else None
    )
    return lmc, client_system


def _new_openai_completion_id():
    return f"chatcmpl-{shortuuid.uuid()}"


def _openai_sse_chunk(
    completion_id,
    created,
    delta_content=None,
    finish_reason=None,
    *,
    role=None,
):
    delta = {}
    if role is not None:
        delta["role"] = role
    if delta_content is not None:
        delta["content"] = delta_content
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "open-interpreter",
                "choices": [choice],
            }
        )
        + "\n\n"
    )


def _lmc_chunk_to_openai_delta(chunk, auto_run, *, pending_code_language=None):
    if chunk.get("format") == "reasoning":
        return None
    if chunk.get("type") == "confirmation" and auto_run is False:
        return OPENAI_CODE_APPROVAL_PROMPT
    if chunk.get("type") == "message" and "content" in chunk:
        return chunk["content"]
    if chunk.get("type") == "code" and "start" in chunk:
        return "\n\n```" + chunk["format"] + "\n"
    if chunk.get("type") == "code" and "content" in chunk:
        return chunk["content"]
    if chunk.get("type") == "code" and "end" in chunk:
        return "\n```\n\n"
    if (
        chunk.get("role") == "computer"
        and chunk.get("type") == "console"
        and chunk.get("format") == "output"
        and chunk.get("content")
    ):
        return _format_openai_console_output(
            chunk["content"],
            language=pending_code_language or "bash",
        )
    return None


def _pending_code_language(async_interpreter):
    for message in reversed(async_interpreter.messages):
        if message.get("type") == "code":
            fmt = message.get("format") or "bash"
            if fmt == "shell":
                return "bash"
            return fmt
    return "bash"


def create_router(async_interpreter):
    router = APIRouter()

    @router.get("/heartbeat")
    async def heartbeat():
        return {"status": "alive"}

    @router.get("/")
    async def home():
        return PlainTextResponse(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chat</title>
            </head>
            <body>
                <form action="" onsubmit="sendMessage(event)">
                    <textarea id="messageInput" rows="10" cols="50" autocomplete="off"></textarea>
                    <button>Send</button>
                </form>
                <button id="approveCodeButton">Approve Code</button>
                <button id="authButton">Send Auth</button>
                <div id="messages"></div>
                <script>
                    var ws = new WebSocket("ws://"""
            + async_interpreter.server.host
            + ":"
            + str(async_interpreter.server.port)
            + """/");
                    var lastMessageElement = null;

                    ws.onmessage = function(event) {

                        var eventData = JSON.parse(event.data);

                        """
            + (
                """

                        // Acknowledge receipt
                        var acknowledge_message = {
                            "ack": eventData.id
                        };
                        ws.send(JSON.stringify(acknowledge_message));

                        """
                if async_interpreter.require_acknowledge
                else ""
            )
            + """

                        if (lastMessageElement == null) {
                            lastMessageElement = document.createElement('p');
                            document.getElementById('messages').appendChild(lastMessageElement);
                            lastMessageElement.innerHTML = "<br>"
                        }

                        if ((eventData.role == "assistant" && eventData.type == "message" && eventData.content) ||
                            (eventData.role == "computer" && eventData.type == "console" && eventData.format == "output" && eventData.content) ||
                            (eventData.role == "assistant" && eventData.type == "code" && eventData.content)) {
                            lastMessageElement.innerHTML += eventData.content;
                        } else {
                            lastMessageElement.innerHTML += "<br><br>" + JSON.stringify(eventData) + "<br><br>";
                        }
                    };
                    function sendMessage(event) {
                        event.preventDefault();
                        var input = document.getElementById("messageInput");
                        var message = input.value;
                        if (message.startsWith('{') && message.endsWith('}')) {
                            message = JSON.stringify(JSON.parse(message));
                            ws.send(message);
                        } else {
                            var startMessageBlock = {
                                "role": "user",
                                //"type": "message",
                                "start": true
                            };
                            ws.send(JSON.stringify(startMessageBlock));

                            var messageBlock = {
                                "role": "user",
                                "type": "message",
                                "content": message
                            };
                            ws.send(JSON.stringify(messageBlock));

                            var endMessageBlock = {
                                "role": "user",
                                //"type": "message",
                                "end": true
                            };
                            ws.send(JSON.stringify(endMessageBlock));
                        }
                        var userMessageElement = document.createElement('p');
                        userMessageElement.innerHTML = '<b>' + input.value + '</b><br>';
                        document.getElementById('messages').appendChild(userMessageElement);
                        lastMessageElement = document.createElement('p');
                        document.getElementById('messages').appendChild(lastMessageElement);
                        input.value = '';
                    }
                function approveCode() {
                    var startCommandBlock = {
                        "role": "user",
                        "type": "command",
                        "start": true
                    };
                    ws.send(JSON.stringify(startCommandBlock));

                    var commandBlock = {
                        "role": "user",
                        "type": "command",
                        "content": "go"
                    };
                    ws.send(JSON.stringify(commandBlock));

                    var endCommandBlock = {
                        "role": "user",
                        "type": "command",
                        "end": true
                    };
                    ws.send(JSON.stringify(endCommandBlock));
                }
                function authenticate() {
                    var authBlock = {
                        "auth": "dummy-api-key"
                    };
                    ws.send(JSON.stringify(authBlock));
                }

                document.getElementById("approveCodeButton").addEventListener("click", approveCode);
                document.getElementById("authButton").addEventListener("click", authenticate);
                </script>
            </body>
            </html>
            """,
            media_type="text/html",
        )

    @router.websocket("/")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        try:  # solving it ;)/ # killian super wrote this

            async def receive_input():
                authenticated = False
                while True:
                    try:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            return
                        data = await websocket.receive()

                        if (
                            not authenticated
                            and os.getenv("INTERPRETER_REQUIRE_AUTH") != "False"
                        ):
                            if "text" in data:
                                data = json.loads(data["text"])
                                if "auth" in data:
                                    if async_interpreter.server.authenticate(
                                        data["auth"]
                                    ):
                                        authenticated = True
                                        await websocket.send_text(
                                            json.dumps({"auth": True})
                                        )
                            if not authenticated:
                                await websocket.send_text(json.dumps({"auth": False}))
                            continue

                        if data.get("type") == "websocket.receive":
                            if "text" in data:
                                data = json.loads(data["text"])
                                if (
                                    async_interpreter.require_acknowledge
                                    and "ack" in data
                                ):
                                    async_interpreter.acknowledged_outputs.append(
                                        data["ack"]
                                    )
                                    continue
                            elif "bytes" in data:
                                data = data["bytes"]
                            await async_interpreter.input(data)
                        elif data.get("type") == "websocket.disconnect":
                            print("Client wants to disconnect, that's fine..")
                            return
                        else:
                            print("Invalid data:", data)
                            continue

                    except Exception as e:
                        error = traceback.format_exc() + "\n" + str(e)
                        error_message = {
                            "role": "server",
                            "type": "error",
                            "content": traceback.format_exc() + "\n" + str(e),
                        }
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_text(json.dumps(error_message))
                            await websocket.send_text(json.dumps(complete_message))
                            print("\n\n--- SENT ERROR: ---\n\n")
                        else:
                            print(
                                "\n\n--- ERROR (not sent due to disconnected state): ---\n\n"
                            )
                        print(error)
                        print("\n\n--- (ERROR ABOVE) ---\n\n")

            async def send_output():
                while True:
                    if websocket.client_state != WebSocketState.CONNECTED:
                        return
                    try:
                        # First, try to send any unsent messages
                        while async_interpreter.unsent_messages:
                            output = async_interpreter.unsent_messages[0]
                            if async_interpreter.debug:
                                print("This was unsent, sending it again:", output)

                            success = await send_message(output)
                            if success:
                                async_interpreter.unsent_messages.popleft()

                        # If we've sent all unsent messages, get a new output
                        if not async_interpreter.unsent_messages:
                            output = await async_interpreter.output()
                            success = await send_message(output)
                            if not success:
                                async_interpreter.unsent_messages.append(output)
                                if async_interpreter.debug:
                                    print(
                                        f"Added message to unsent_messages queue after failed attempts: {output}"
                                    )

                    except Exception as e:
                        error = traceback.format_exc() + "\n" + str(e)
                        error_message = {
                            "role": "server",
                            "type": "error",
                            "content": error,
                        }
                        async_interpreter.unsent_messages.append(error_message)
                        async_interpreter.unsent_messages.append(complete_message)
                        print("\n\n--- ERROR (will be sent when possible): ---\n\n")
                        print(error)
                        print(
                            "\n\n--- (ERROR ABOVE WILL BE SENT WHEN POSSIBLE) ---\n\n"
                        )

            async def send_message(output):
                if isinstance(output, dict) and "id" in output:
                    id = output["id"]
                else:
                    id = shortuuid.uuid()
                    if (
                        isinstance(output, dict)
                        and async_interpreter.require_acknowledge
                    ):
                        output["id"] = id

                for attempt in range(20):
                    # time.sleep(0.5)

                    if websocket.client_state != WebSocketState.CONNECTED:
                        return False

                    try:
                        # print("sending:", output)

                        if isinstance(output, bytes):
                            await websocket.send_bytes(output)
                            return True  # Haven't set up ack for this
                        else:
                            if async_interpreter.require_acknowledge:
                                output["id"] = id
                            if async_interpreter.debug:
                                print("Sending this over the websocket:", output)
                            await websocket.send_text(json.dumps(output))

                        if async_interpreter.require_acknowledge:
                            acknowledged = False
                            for _ in range(100):
                                if id in async_interpreter.acknowledged_outputs:
                                    async_interpreter.acknowledged_outputs.remove(id)
                                    acknowledged = True
                                    if async_interpreter.debug:
                                        print("This output was acknowledged:", output)
                                    break
                                await asyncio.sleep(0.0001)

                            if acknowledged:
                                return True
                            else:
                                if async_interpreter.debug:
                                    print("Acknowledgement not received for:", output)
                                return False
                        else:
                            return True

                    except Exception as e:
                        print(
                            f"Failed to send output on attempt number: {attempt + 1}. Output was: {output}"
                        )
                        print(f"Error: {str(e)}")
                        traceback.print_exc()
                        await asyncio.sleep(0.01)

                # If we've reached this point, we've failed to send after 100 attempts
                if output not in async_interpreter.unsent_messages:
                    print("Failed to send message:", output)
                else:
                    print(
                        "Failed to send message, also it was already in unsent queue???:",
                        output,
                    )

                return False

            await asyncio.gather(receive_input(), send_output())

        except Exception as e:
            error = traceback.format_exc() + "\n" + str(e)
            error_message = {
                "role": "server",
                "type": "error",
                "content": error,
            }
            async_interpreter.unsent_messages.append(error_message)
            async_interpreter.unsent_messages.append(complete_message)
            print("\n\n--- ERROR (will be sent when possible): ---\n\n")
            print(error)
            print("\n\n--- (ERROR ABOVE WILL BE SENT WHEN POSSIBLE) ---\n\n")

    # TODO
    @router.post("/")
    async def post_input(payload: Dict[str, Any]):
        try:
            async_interpreter.input(payload)
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}, 500

    @router.post("/settings")
    async def set_settings(payload: Dict[str, Any]):
        for key, value in payload.items():
            print("Updating settings...")
            # print(f"Updating settings: {key} = {value}")
            if key in ["llm", "toolbox"] and isinstance(value, dict):
                if key == "auto_run":
                    return {
                        "error": f"The setting {key} is not modifiable through the server due to security constraints."
                    }, 403
                if hasattr(async_interpreter, key):
                    for sub_key, sub_value in value.items():
                        if hasattr(getattr(async_interpreter, key), sub_key):
                            setattr(getattr(async_interpreter, key), sub_key, sub_value)
                        else:
                            return {
                                "error": f"Sub-setting {sub_key} not found in {key}"
                            }, 404
                else:
                    return {"error": f"Setting {key} not found"}, 404
            elif hasattr(async_interpreter, key):
                setattr(async_interpreter, key, value)
            else:
                return {"error": f"Setting {key} not found"}, 404

        return {"status": "success"}

    @router.get("/settings/{setting}")
    async def get_setting(setting: str):
        if hasattr(async_interpreter, setting):
            setting_value = getattr(async_interpreter, setting)
            try:
                return json.dumps({setting: setting_value})
            except TypeError:
                return {"error": "Failed to serialize the setting value"}, 500
        else:
            return json.dumps({"error": "Setting not found"}), 404

    if os.getenv("INTERPRETER_INSECURE_ROUTES", "").lower() == "true":

        @router.post("/run")
        async def run_code(payload: Dict[str, Any]):
            language, code = payload.get("language"), payload.get("code")
            if not (language and code):
                return {"error": "Both 'language' and 'code' are required."}, 400
            try:
                print(f"Running {language}:", code)
                output = async_interpreter.terminal.run(language, code)
                print("Output:", output)
                return {"output": output}
            except Exception as e:
                return {"error": str(e)}, 500

        @router.post("/upload")
        async def upload_file(file: UploadFile = File(...), path: str = Form(...)):
            try:
                with open(path, "wb") as output_file:
                    shutil.copyfileobj(file.file, output_file)
                return {"status": "success"}
            except Exception as e:
                return {"error": str(e)}, 500

        @router.get("/download/{filename}")
        async def download_file(filename: str):
            try:
                return StreamingResponse(
                    open(filename, "rb"), media_type="application/octet-stream"
                )
            except Exception as e:
                return {"error": str(e)}, 500

    ### OPENAI COMPATIBLE ENDPOINT

    class ChatMessage(BaseModel):
        role: str
        content: Union[str, List[Dict[str, Any]]]

    class ChatCompletionRequest(BaseModel):
        model: str = "default-model"
        messages: List[ChatMessage]
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        stream: Optional[bool] = False

    async def _stream_openai_assistant_text(text):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        yield _openai_sse_chunk(
            completion_id, created, delta_content=text, role="assistant"
        )
        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    def _openai_assistant_text_response(text, model):
        return {
            "id": _new_openai_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}}
            ],
        }

    async def _stream_openai_title(content_str):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        title_messages = [
            {
                "role": "system",
                "type": "message",
                "content": (
                    "Output only a short conversation title (max 10 words). "
                    "No quotes, no explanation."
                ),
            },
            {"role": "user", "type": "message", "content": content_str},
        ]
        for chunk in async_interpreter.llm.run(
            title_messages, auxiliary_title_request=True
        ):
            if chunk.get("format") == "reasoning":
                continue
            if chunk.get("type") == "message" and chunk.get("content"):
                await asyncio.sleep(0)
                yield _openai_sse_chunk(
                    completion_id, created, delta_content=chunk["content"]
                )
        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    async def openai_compatible_generator(run_code):
        completion_id = _new_openai_completion_id()
        created = int(time.time())
        sent_role = False

        async def emit_delta(text):
            nonlocal sent_role
            role = "assistant" if not sent_role else None
            if role:
                sent_role = True
            await asyncio.sleep(0)
            return _openai_sse_chunk(
                completion_id, created, delta_content=text, role=role
            )

        pending_lang = _pending_code_language(async_interpreter)

        if run_code:
            print("Running code.\n")
            chunk_iter = async_interpreter._respond_and_store()
        elif async_interpreter.context_mode:
            # 01 / context-mode clients: legacy nudge loop when the model stays silent.
            made_chunk = False
            for message in [
                ".",
                "Just say something, anything.",
                "Hello? Answer please.",
                "Are you there?",
                "Can you respond?",
                "Please reply.",
            ]:
                for chunk in async_interpreter.chat(
                    message=message, stream=True, display=False
                ):
                    await asyncio.sleep(0)
                    made_chunk = True
                    output_content = _lmc_chunk_to_openai_delta(
                        chunk,
                        async_interpreter.auto_run,
                        pending_code_language=pending_lang,
                    )
                    if output_content:
                        yield await emit_delta(output_content)
                    if async_interpreter.stop_event.is_set():
                        break
                if made_chunk:
                    break
            yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return
        else:
            async_interpreter.last_messages_count = len(async_interpreter.messages)
            chunk_iter = async_interpreter._respond_and_store()

        for chunk in chunk_iter:
            if run_code and "content" in chunk:
                print(chunk.get("content", ""), end="")
            if run_code and "start" in chunk:
                print("\n")

            if chunk.get("type") == "confirmation":
                # "yes" approves only the one pending block; further code in this
                # response must prompt again (same as terminal y/n per block).
                if run_code:
                    run_code = False
                    continue
                output_content = _lmc_chunk_to_openai_delta(
                    chunk,
                    async_interpreter.auto_run,
                    pending_code_language=pending_lang,
                )
                if output_content:
                    yield await emit_delta(output_content)
                if not async_interpreter.auto_run:
                    async_interpreter._server_awaiting_code_approval = True
                    break
                continue

            output_content = _lmc_chunk_to_openai_delta(
                chunk,
                async_interpreter.auto_run,
                pending_code_language=pending_lang,
            )
            if output_content:
                yield await emit_delta(output_content)

            if async_interpreter.stop_event.is_set():
                break

        if not _openai_server_has_pending_code(async_interpreter):
            _clear_openai_code_approval_wait(async_interpreter)

        yield _openai_sse_chunk(completion_id, created, finish_reason="stop")
        yield "data: [DONE]\n\n"

    @router.post("/openai/chat/completions")
    async def chat_completion(request: ChatCompletionRequest):
        global last_start_time

        # Convert to LMC
        last_message = request.messages[-1]

        if last_message.role != "user":
            raise ValueError("Last message must be from the user.")

        if last_message.content == "{STOP}":
            # Handle special STOP token
            async_interpreter.stop_event.set()
            time.sleep(5)
            async_interpreter.stop_event.clear()
            return

        if last_message.content in ["{CONTEXT_MODE_ON}", "{REQUIRE_START_ON}"]:
            async_interpreter.context_mode = True
            return

        if last_message.content in ["{CONTEXT_MODE_OFF}", "{REQUIRE_START_OFF}"]:
            async_interpreter.context_mode = False
            return

        if last_message.content == "{AUTO_RUN_ON}":
            async_interpreter.auto_run = True
            return

        if last_message.content == "{AUTO_RUN_OFF}":
            async_interpreter.auto_run = False
            return

        content_str = (
            last_message.content
            if isinstance(last_message.content, str)
            else None
        )

        if content_str and _is_openai_auxiliary_title_request(content_str):
            if request.stream:
                return StreamingResponse(
                    _stream_openai_title(content_str),
                    media_type="text/event-stream",
                )
            title_messages = [
                {
                    "role": "system",
                    "type": "message",
                    "content": (
                        "Output only a short conversation title (max 10 words). "
                        "No quotes, no explanation."
                    ),
                },
                {"role": "user", "type": "message", "content": content_str},
            ]
            content = ""
            for chunk in async_interpreter.llm.run(
                title_messages, auxiliary_title_request=True
            ):
                if chunk.get("format") == "reasoning":
                    continue
                if chunk.get("type") == "message" and chunk.get("content"):
                    content += chunk["content"]
            completion_id = _new_openai_completion_id()
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}}
                ],
            }

        run_code = False
        skip_message_replace = False
        if _openai_server_awaiting_approval(async_interpreter) and content_str is not None:
            approval = _normalize_openai_code_approval_reply(content_str)
            if approval == "yes":
                run_code = True
                skip_message_replace = True
                print(">", content_str, "(approve code)")
            elif approval == "no":
                _clear_openai_code_approval_wait(async_interpreter)
                async_interpreter.messages.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": "[User declined to run this code.]",
                        "source": "server",
                    }
                )
                print(">", content_str, "(decline code)")
                if request.stream:
                    return StreamingResponse(
                        _stream_openai_assistant_text(OPENAI_CODE_APPROVAL_DECLINED),
                        media_type="text/event-stream",
                    )
                return _openai_assistant_text_response(
                    OPENAI_CODE_APPROVAL_DECLINED, request.model
                )
            else:
                retry_text = OPENAI_CODE_APPROVAL_INVALID_TEMPLATE.format(
                    reply=content_str.strip()[:200]
                )
                print(">", content_str, "(invalid approval reply)")
                if request.stream:
                    return StreamingResponse(
                        _stream_openai_assistant_text(retry_text),
                        media_type="text/event-stream",
                    )
                return _openai_assistant_text_response(retry_text, request.model)
        if not skip_message_replace:
            if isinstance(last_message.content, (str, list)):
                _openai_apply_request_messages(async_interpreter, request, last_message)
                print(">", content_str or last_message.content)
            elif async_interpreter.context_mode:
                if last_message.content == "{START}":
                    if async_interpreter.messages[-1]["content"] == "{START}":
                        async_interpreter.messages = async_interpreter.messages[:-1]
                    last_start_time = time.time()
                    if (
                        async_interpreter.messages
                        and async_interpreter.messages[-1].get("role") != "user"
                    ):
                        return
                else:
                    current_time = time.time()
                    if current_time - last_start_time > 6:
                        return
            elif last_message.content == "{START}":
                async_interpreter.messages = async_interpreter.messages[:-1]
                return

        async_interpreter.stop_event.set()
        time.sleep(0.1)
        async_interpreter.stop_event.clear()

        if request.stream:
            return StreamingResponse(
                openai_compatible_generator(run_code),
                media_type="text/event-stream",
            )
        else:
            async_interpreter.last_messages_count = len(async_interpreter.messages)
            content = ""
            pending_lang = _pending_code_language(async_interpreter)
            for chunk in async_interpreter._respond_and_store():
                if chunk.get("type") == "confirmation" and run_code:
                    continue
                delta = _lmc_chunk_to_openai_delta(
                    chunk,
                    async_interpreter.auto_run,
                    pending_code_language=pending_lang,
                )
                if delta:
                    content += delta
                if (
                    chunk.get("type") == "confirmation"
                    and not run_code
                    and not async_interpreter.auto_run
                ):
                    break
            completion_id = _new_openai_completion_id()
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}}
                ],
            }

    return router


class Server:
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000

    def __init__(self, async_interpreter, host=None, port=None):
        if FastAPI is None or uvicorn is None or janus is None:
            raise ImportError(
                "Server mode requires fastapi, uvicorn, and janus. "
                "Install with: pip install fastapi uvicorn janus"
            )
        self.app = FastAPI()
        router = create_router(async_interpreter)
        self.authenticate = authenticate_function

        # Add authentication middleware
        @self.app.middleware("http")
        async def validate_api_key(request: Request, call_next):
            # Ignore authentication for the /heartbeat route
            if request.url.path == "/heartbeat":
                return await call_next(request)

            api_key = request.headers.get("X-API-KEY")
            if self.authenticate(api_key):
                response = await call_next(request)
                return response
            else:
                return JSONResponse(
                    status_code=HTTP_403_FORBIDDEN,
                    content={"detail": "Authentication failed"},
                )

        self.app.include_router(router)
        h = host or os.getenv("INTERPRETER_HOST", Server.DEFAULT_HOST)
        p = port or int(os.getenv("INTERPRETER_PORT", Server.DEFAULT_PORT))
        self.config = uvicorn.Config(app=self.app, host=h, port=p)
        self.uvicorn_server = uvicorn.Server(self.config)

    @property
    def host(self):
        return self.config.host

    @host.setter
    def host(self, value):
        self.config.host = value
        self.uvicorn_server = uvicorn.Server(self.config)

    @property
    def port(self):
        return self.config.port

    @port.setter
    def port(self, value):
        self.config.port = value
        self.uvicorn_server = uvicorn.Server(self.config)

    def run(self, host=None, port=None, retries=5):
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port

        # Print server information
        if self.host == "0.0.0.0":
            print(
                "Warning: Using host `0.0.0.0` will expose Open Interpreter over your local network."
            )
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            print(f"Server will run at http://{s.getsockname()[0]}:{self.port}")
            s.close()
        else:
            print(f"Server will run at http://{self.host}:{self.port}")

        self.uvicorn_server.run()

        # for _ in range(retries):
        #     try:
        #         self.uvicorn_server.run()
        #         break
        #     except KeyboardInterrupt:
        #         break
        #     except ImportError as e:
        #         if _ == 4:  # If this is the last attempt
        #             raise ImportError(
        #                 str(e)
        #                 + """\n\nPlease ensure you have run `pip install "open-interpreter[server]"` to install server dependencies."""
        #             )
        #     except:
        #         print("An unexpected error occurred:", traceback.format_exc())
        #         print("Server restarting.")
