import json
import os
import re

from .utils.merge_deltas import merge_deltas
from .utils.normalize_double_escaped_arguments import (
    normalize_double_escaped_arguments,
)
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


def process_messages(messages):
    processed_messages = []
    # Every call id already present in the history has to be reserved before
    # generating new ones. An assistant turn can declare calls with
    # provider-assigned ids without touching the legacy counter, so a bare
    # `toolu_1` would collide with it and declare the same call twice: a provider
    # treats an id as the identity of exactly one call.
    used_tool_ids = {
        call_id
        for message in messages
        for call_id in [message.get("tool_call_id")]
        + [call.get("id") for call in message.get("tool_calls") or []]
        if call_id
    }
    last_tool_id = 0

    # Ids any assistant message has already declared. A tool response may answer
    # one of them, but a *second* response to the same id cannot declare it
    # again: the provider treats an id as the identity of exactly one call.
    # Such a response gets a fresh id written back onto it instead.
    declared_tool_ids = {
        call.get("id")
        for message in messages
        for call in message.get("tool_calls") or []
        if call.get("id")
    }

    # `tool_call_id`s declared by the most recent assistant message. A single
    # turn may declare several calls, so the answers are not necessarily
    # adjacent to their declaration; the pending set tells the second answer
    # apart from a genuinely orphaned one.
    pending_tool_ids = []
    # The assistant message those ids belong to. An unmatchable answer that
    # arrives in the middle of that turn's answers has to be declared by the
    # same turn; a separate assistant message between the answers would leave
    # the later answer without the turn that declared its call.
    pending_assistant = None

    def next_tool_id():
        nonlocal last_tool_id
        while True:
            last_tool_id += 1
            candidate = f"toolu_{last_tool_id}"
            if candidate not in used_tool_ids:
                used_tool_ids.add(candidate)
                return candidate

    def placeholder_tool_call(tool_id):
        return {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": "execute",
                "arguments": json.dumps(
                    {
                        "language": "python",
                        "code": "# Placeholder for an unsupported or invalid tool call; the provider's error was returned as the tool response.",
                    }
                ),
            },
        }

    i = 0
    while i < len(messages):
        message = messages[i]

        if message.get("function_call"):
            tool_id = next_tool_id()

            # Convert function_call to tool_calls
            function = message.pop("function_call")
            message["tool_calls"] = [
                {"id": tool_id, "type": "function", "function": function}
            ]
            declared_tool_ids.add(tool_id)
            processed_messages.append(message)
            pending_tool_ids = [tool_id]
            pending_assistant = message

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
            pending_tool_ids = []

        elif message.get("role") == "function":
            # This handles orphaned function responses
            tool_id = next_tool_id()

            # Add a tool call before this orphaned tool response.
            # Keep a handle to the assistant message that declares it: a later
            # answer with no declaration must attach here, because the previous
            # turn's call was already answered and anything attached to it would
            # separate that answer from its own declaration.
            orphan_assistant = {
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
            processed_messages.append(orphan_assistant)
            declared_tool_ids.add(tool_id)
            pending_assistant = orphan_assistant

            # Process the function response
            message["role"] = "tool"
            message["tool_call_id"] = tool_id
            processed_messages.append(message)

        elif message.get("role") == "tool":
            # A tool message must follow an assistant message carrying the matching
            # tool_call_id (OpenAI, OpenRouter, Alibaba). A provider that rejects an
            # unsupported or malformed tool call still produces a tool response, but
            # no assistant message with tool_calls is stored for it, leaving the pair
            # unpaired and the next request rejected. Declare the call on the assistant
            # message that owns this turn so the two messages stay paired.
            # A declared call that has not been answered yet is not orphaned:
            # one assistant turn can declare several calls and their answers
            # arrive in sequence, so checking only the previous message
            # mispaired the second answer and declared its call twice.
            tool_id = message.get("tool_call_id")
            if tool_id and tool_id in pending_tool_ids:
                pending_tool_ids.remove(tool_id)
            else:
                if not tool_id or tool_id in declared_tool_ids:
                    # A missing id, or a second answer to an id that is already
                    # declared: neither may declare the same call twice, so the
                    # message gets its own id, written back onto it.
                    tool_id = next_tool_id()
                    message["tool_call_id"] = tool_id
                declared_tool_ids.add(tool_id)
                call = placeholder_tool_call(tool_id)
                if pending_assistant is not None:
                    # Attach the call to the turn that is still waiting for answers
                    # instead of inserting a new assistant message between them.
                    pending_assistant.setdefault("tool_calls", []).append(call)
                else:
                    processed_messages.append(
                        {"role": "assistant", "tool_calls": [call]}
                    )
            processed_messages.append(message)

        else:
            # For non-tool-related messages, just add them as is
            processed_messages.append(message)
            if message.get("tool_calls"):
                pending_tool_ids = [
                    call.get("id") for call in message["tool_calls"] if call.get("id")
                ]
                pending_assistant = message
            else:
                # This message separates any earlier declarations from whatever
                # follows, so a tool response after it cannot be paired with them.
                pending_tool_ids = []
                pending_assistant = None

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
    accumulated_review = ""
    review_category = None
    buffer = ""

    for chunk in llm.completions(**request_params):
        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        delta = chunk["choices"][0]["delta"]

        # Thinking-model reasoning must reach history even when this delta
        # carries no usable content (a reasoning-only delta, or one the
        # tool-call transform below replaces). Capture it before the
        # transform drops it.
        reasoning = delta.get("reasoning_content") or ""

        # Content in a delta of its own after a tool call is a judge layer's
        # review of the code (see below). Content packed into the *same* delta
        # as the tool call carries no such ordering: providers that collapse a
        # turn into a single message (litellm's ollama_chat and Gemini
        # builders) put the assistant's own text there. Position cannot tell
        # the two apart once the message is collapsed, so the review tags do.
        content_is_review = function_call_detected

        # Convert tool call into function call, which we have great parsing logic for below
        if "tool_calls" in delta and delta["tool_calls"]:
            # Content packed into the *same* delta as a tool call carries no
            # ordering, so the verdict tags classify it — independently for
            # every such delta, not just the first. A later delta mixing the
            # assistant's narration with a tool-call continuation must not
            # inherit review status from the earlier call. Deltas arrive as
            # dicts or objects depending on the provider/litellm version,
            # hence both.
            if isinstance(delta, dict):
                content = delta.get("content")
            else:
                content = getattr(delta, "content", None)

            if content:
                # A review already opened by an earlier content-only delta must
                # stay a review even when the body arrives with a later
                # tool-call fragment and contains no opening tag itself.
                content_is_review = review_category is not None or any(
                    tag in content for tag in ["<safe>", "<warning>", "<unsafe>"]
                )

            # import pdb; pdb.set_trace()
            converted_call = None
            for tool_call in delta["tool_calls"]:
                # Entries arrive as objects (with .function) or plain dicts
                # depending on the provider/litellm version.
                if isinstance(tool_call, dict):
                    function = tool_call.get("function")
                else:
                    function = getattr(tool_call, "function", None)
                if not function:
                    continue
                # A nameless continuation still belongs to a call in
                # progress: merge_deltas skips its None name and appends
                # the arguments to the accumulated call.
                if isinstance(function, dict):
                    name = function.get("name")
                    arguments = function.get("arguments")
                else:
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                if not name and not arguments:
                    # An entry with neither a name nor arguments (including
                    # empty strings) carries nothing to merge; keep scanning
                    # the list for a usable entry rather than selecting this
                    # one.
                    continue
                converted_call = {"name": name, "arguments": arguments}
                break
            if converted_call is not None:
                function_call_detected = True
                delta = {
                    # "id": delta["tool_calls"][0],
                    # Carry the content across: rebuilding the delta out of the
                    # tool call alone threw away text sent in the same message.
                    "content": content,
                    "function_call": converted_call,
                }
            else:
                # No entry carried a usable function, and the raw tool_calls
                # list is nothing the merge step can consume (merge_deltas
                # cannot dict() it), so drop the key and let the chunk pass
                # through empty rather than crashing (#254). dict() works on
                # plain dicts and litellm Delta objects alike; .items() only
                # works on the former (#267).
                delta = dict(delta)
                delta.pop("tool_calls", None)

        # Accumulate deltas
        accumulated_deltas = merge_deltas(accumulated_deltas, delta)

        if reasoning and not (
            "content" in delta and delta["content"] and not function_call_detected
        ):
            # No message chunk below will carry this reasoning (a
            # reasoning-only delta, a delta replaced by the tool-call
            # transform, or judge-review content), so emit it on its own for
            # history accumulation.
            yield {
                "type": "message",
                "content": "",
                "reasoning_content": reasoning,
            }

        if "content" in delta and delta["content"]:
            if content_is_review:
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
                message_chunk = {"type": "message", "content": delta["content"]}
                if delta.get("reasoning_content"):
                    message_chunk["reasoning_content"] = delta[
                        "reasoning_content"
                    ]
                yield message_chunk

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
                if isinstance(arguments, dict):
                    # Some providers escape the JSON a second time; without
                    # this the `language`/`code` lookups below miss and the
                    # call is dropped for the cost of a turn (#566).
                    unescaped = normalize_double_escaped_arguments(arguments)
                    if unescaped is not arguments and llm.interpreter.verbose:
                        print(
                            "Tool-call arguments arrived double-escaped; unescaped one level."
                        )
                    arguments = unescaped

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

    if buffer and review_category is not None:
        # A verdict that arrived whole in one chunk was buffered waiting for a
        # following chunk that never comes; flush it instead of dropping it
        # (#219).
        yield {
            "type": "review",
            "format": review_category,
            "content": buffer,
        }
        buffer = ""

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
