import json
import os
import re
from types import SimpleNamespace

from .utils.merge_deltas import merge_deltas
from .utils.parse_partial_json import parse_partial_json

tool_schema = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Executes code on the user's machine **in the users local environment** and returns the output",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The programming language (required parameter to the `execute` function)",
                    "enum": [
                        # This will be filled dynamically with the languages OI has access to.
                    ],
                },
                "code": {
                    "type": "string",
                    "description": "The code to execute (required)",
                },
            },
            "required": ["language", "code"],
        },
    },
}


def _tool_call_entry_function(entry):
    """Return the function payload of a tool call entry, object or dict."""
    if isinstance(entry, dict):
        return entry.get("function")
    return getattr(entry, "function", None)


def _function_name_and_arguments(function):
    """Return (name, arguments) of a function payload, object or dict."""
    if isinstance(function, dict):
        return function.get("name"), function.get("arguments")
    return getattr(function, "name", None), getattr(function, "arguments", None)


def _tool_call_entry_key(entry, position):
    """Return a stable identity for a tool call entry across chunks.

    Streaming providers attach an index to every chunk of a call but the
    id only to its first chunk, so the index leads when present. Entries
    with neither (non-streamed shapes) fall back to their position in the
    chunk — which preserves today's behavior exactly for single-call flows.
    """
    if isinstance(entry, dict):
        if "index" in entry:
            return ("index", entry["index"])
        return entry.get("id") or ("pos", position)
    index = getattr(entry, "index", None)
    if index is not None:
        return ("index", index)
    return getattr(entry, "id", None) or ("pos", position)


def _queue_extra_tool_call(llm, entry, key):
    """Stash one extra parallel tool call for a later turn.

    The pipeline below executes a single call per turn, so entries beyond
    the first of a parallel response wait on the interpreter instead of
    being silently dropped. Fragments sharing a key are concatenated, so
    calls split across streamed chunks still arrive whole.
    """
    queue = getattr(llm.interpreter, "_pending_tool_calls", None)
    if queue is None:
        queue = llm.interpreter._pending_tool_calls = []
    function = _tool_call_entry_function(entry)
    if not function:
        return
    name, arguments = _function_name_and_arguments(function)
    call_id = str(key)
    for queued in queue:
        if queued["id"] == call_id:
            queued["arguments"] += arguments or ""
            return
    queue.append({"id": call_id, "name": name, "arguments": arguments or ""})


def _queued_tool_call_chunk(queued):
    """Wrap a queued call as a stream chunk the loop below already handles."""
    entry = SimpleNamespace(
        id=queued["id"],
        function=SimpleNamespace(name=queued["name"], arguments=queued["arguments"]),
    )
    return {"choices": [{"delta": {"tool_calls": [entry]}}]}


def process_messages(messages):
    processed_messages = []
    last_tool_id = 0

    i = 0
    while i < len(messages):
        message = messages[i]

        if message.get("function_call"):
            last_tool_id += 1
            tool_id = f"toolu_{last_tool_id}"

            # Convert function_call to tool_calls
            function = message.pop("function_call")
            message["tool_calls"] = [
                {"id": tool_id, "type": "function", "function": function}
            ]
            processed_messages.append(message)

            # Process the next message if it's a function response
            if i + 1 < len(messages) and messages[i + 1].get("role") == "function":
                next_message = messages[i + 1].copy()
                next_message["role"] = "tool"
                next_message["tool_call_id"] = tool_id
                processed_messages.append(next_message)
                i += 1  # Skip the next message as we've already processed it
            else:
                # Add an empty tool response if there isn't one
                processed_messages.append(
                    {"role": "tool", "tool_call_id": tool_id, "content": ""}
                )

        elif message.get("role") == "function":
            # This handles orphaned function responses
            last_tool_id += 1
            tool_id = f"toolu_{last_tool_id}"

            # Add a tool call before this orphaned tool response
            processed_messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": "# Automated tool call to fetch more output, triggered by the user.",
                            },
                        }
                    ],
                }
            )

            # Process the function response
            message["role"] = "tool"
            message["tool_call_id"] = tool_id
            processed_messages.append(message)

        else:
            # For non-tool-related messages, just add them as is
            processed_messages.append(message)

        i += 1

    return processed_messages


def run_tool_calling_llm(llm, request_params):
    ## Setup

    # Add languages OI has access to
    tool_schema["function"]["parameters"]["properties"]["language"]["enum"] = [
        i.name.lower() for i in llm.interpreter.computer.terminal.languages
    ]
    request_params["tools"] = [tool_schema]

    request_params["messages"] = process_messages(request_params["messages"])

    # # This makes any role: tool have the ID of the last tool call
    # last_tool_id = 0
    # for i, message in enumerate(request_params["messages"]):
    #     if "function_call" in message:
    #         last_tool_id += 1
    #         function = message.pop("function_call")
    #         message["tool_calls"] = [
    #             {
    #                 "id": "toolu_" + str(last_tool_id),
    #                 "type": "function",
    #                 "function": function,
    #             }
    #         ]
    #     if message["role"] == "function":
    #         if i != 0 and request_params["messages"][i - 1]["role"] == "tool":
    #             request_params["messages"][i]["content"] += message["content"]
    #             message = None
    #         else:
    #             message["role"] = "tool"
    #             message["tool_call_id"] = "toolu_" + str(last_tool_id)
    # request_params["messages"] = [m for m in request_params["messages"] if m != None]

    # This adds an empty tool response for any tool call without a tool response
    # new_messages = []
    # for i, message in enumerate(request_params["messages"]):
    #     new_messages.append(message)
    #     if "tool_calls" in message:
    #         tool_call_id = message["tool_calls"][0]["id"]
    #         if not any(
    #             m
    #             for m in request_params["messages"]
    #             if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id
    #         ):
    #             new_messages.append(
    #                 {"role": "tool", "tool_call_id": tool_call_id, "content": ""}
    #             )
    # request_params["messages"] = new_messages

    # messages = request_params["messages"]
    # for i in range(len(messages)):
    #     if messages[i]["role"] == "user" and isinstance(messages[i]["content"], list):
    #         # Found an image from the user
    #         image_message = messages[i]
    #         j = i + 1
    #         while j < len(messages) and messages[j]["role"] == "tool":
    #             # Move the image down until it's after all the role: tools
    #             j += 1
    #         messages.insert(j, image_message)
    #         del messages[i]
    # request_params["messages"] = messages

    # Add OpenAI's recommended function message
    # request_params["messages"][0][
    #     "content"
    # ] += "\nUse ONLY the function you have been provided with — 'execute(language, code)'."

    ## Convert output to LMC format

    accumulated_deltas = {}
    language = None
    code = ""
    function_call_detected = False
    # Identity of the call executed this turn; entries for any other call
    # in the same response are queued for later turns (see below).
    primary_call_key = None
    accumulated_review = ""
    review_category = None
    buffer = ""

    pending_calls = getattr(llm.interpreter, "_pending_tool_calls", None)
    if pending_calls:
        # A previous turn received more tool calls than the pipeline can
        # execute at once: serve the next queued call now, without spending
        # an LLM call, so every call runs in turn order.
        stream = [_queued_tool_call_chunk(pending_calls.pop(0))]
    else:
        stream = llm.completions(**request_params)

    for chunk in stream:
        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        delta = chunk["choices"][0]["delta"]

        # Convert tool call into function call, which we have great parsing logic for below
        if "tool_calls" in delta and delta["tool_calls"]:
            function_call_detected = True

            # import pdb; pdb.set_trace()
            entries = delta["tool_calls"]
            if primary_call_key is None:
                primary_call_key = _tool_call_entry_key(entries[0], 0)
            keyed = [
                (_tool_call_entry_key(entry, i), entry)
                for i, entry in enumerate(entries)
            ]
            primary_entries = [
                entry for key, entry in keyed if key == primary_call_key
            ]
            first_function = _tool_call_entry_function(
                primary_entries[0] if primary_entries else entries[0]
            )
            if len(entries) > 0 and first_function and primary_entries:
                function_name, function_arguments = _function_name_and_arguments(
                    first_function
                )
                delta = {
                    # "id": primary_entries[0],
                    "function_call": {
                        "name": function_name,
                        "arguments": function_arguments,
                    }
                }
            # The pipeline executes one call per turn: queue any entries for
            # other calls so they run on later turns instead of being
            # silently dropped.
            for key, extra_entry in keyed:
                if key != primary_call_key:
                    _queue_extra_tool_call(llm, extra_entry, key)
            if not primary_entries:
                # This chunk carries only queued calls: keep it out of the
                # accumulator (raw entries are not mergeable) while
                # preserving any message content it may hold.
                content = delta["content"] if "content" in delta else None
                delta = {"content": content} if content else {}

        # Accumulate deltas
        accumulated_deltas = merge_deltas(accumulated_deltas, delta)

        if "content" in delta and delta["content"]:
            if function_call_detected:
                # More content after a code block? This is a code review by a judge layer.

                # print("Code safety review:", delta["content"])

                if review_category == None:
                    accumulated_review += delta["content"]

                    if "<unsafe>" in accumulated_review:
                        review_category = "unsafe"
                    if "<warning>" in accumulated_review:
                        review_category = "warning"
                    if "<safe>" in accumulated_review:
                        review_category = "safe"

                if review_category != None:
                    for tag in [
                        "<safe>",
                        "</safe>",
                        "<warning>",
                        "</warning>",
                        "<unsafe>",
                        "</unsafe>",
                    ]:
                        delta["content"] = delta["content"].replace(tag, "")

                    if re.search("</.*>$", accumulated_review):
                        buffer += delta["content"]
                        continue
                    elif buffer:
                        yield {
                            "type": "review",
                            "format": review_category,
                            "content": buffer + delta["content"],
                        }
                        buffer = ""
                    else:
                        yield {
                            "type": "review",
                            "format": review_category,
                            "content": delta["content"],
                        }
                        buffer = ""

            else:
                yield {"type": "message", "content": delta["content"]}

        if (
            accumulated_deltas.get("function_call")
            and "name" in accumulated_deltas["function_call"]
            and (
                accumulated_deltas["function_call"]["name"] == "python"
                or accumulated_deltas["function_call"]["name"] == "functions"
            )
        ):
            if language is None:
                language = "python"

            # Pull the code string straight out of the "arguments" string
            code_delta = accumulated_deltas["function_call"]["arguments"][len(code) :]
            # Update the code
            code = accumulated_deltas["function_call"]["arguments"]
            # Yield the delta
            if code_delta:
                yield {
                    "type": "code",
                    "format": language,
                    "content": code_delta,
                }

        if (
            accumulated_deltas.get("function_call")
            and "arguments" in accumulated_deltas["function_call"]
            and accumulated_deltas["function_call"]["arguments"]
        ):
            if "arguments" in accumulated_deltas["function_call"]:
                arguments = accumulated_deltas["function_call"]["arguments"]
                arguments = parse_partial_json(arguments)

                if arguments:
                    if (
                        language is None
                        and "language" in arguments
                        and "code"
                        in arguments  # <- This ensures we're *finished* typing language, as opposed to partially done
                        and arguments["language"]
                    ):
                        language = arguments["language"]

                    if language is not None and "code" in arguments:
                        # Calculate the delta (new characters only)
                        code_delta = arguments["code"][len(code) :]
                        # Update the code
                        code = arguments["code"]
                        # Yield the delta
                        if code_delta:
                            yield {
                                "type": "code",
                                "format": language,
                                "content": code_delta,
                            }
                else:
                    if llm.interpreter.verbose:
                        print("Arguments not a dict.")

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
