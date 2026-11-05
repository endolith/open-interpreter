import json
import os
import re
from types import SimpleNamespace

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


def _tool_call_entry_identity(entry):
    """Return (id, index) of a tool-call entry, object or dict."""
    if isinstance(entry, dict):
        return entry.get("id"), entry.get("index")
    return getattr(entry, "id", None), getattr(entry, "index", None)


def _tool_call_entry_key(entry, position):
    """Return a stable identity for a tool call entry across chunks.

    Streaming providers attach an index to every chunk of a call but the
    id only to its first chunk, so the index leads when present. Entries
    with neither (non-streamed shapes) fall back to their position in the
    chunk — which preserves today's behavior exactly for single-call flows.
    """
    if isinstance(entry, dict):
        if entry.get("index") is not None:
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
    being silently dropped. Fragments of one queued call are concatenated,
    so calls split across streamed chunks still arrive whole.

    Identity matches the primary path: an id is authoritative, an index is a
    fallback, and an index-only fragment whose call opened without an index
    attaches to the most recent id-only queued call.
    """
    queue = getattr(llm.interpreter, "_pending_tool_calls", None)
    if queue is None:
        queue = llm.interpreter._pending_tool_calls = []
    function = _tool_call_entry_function(entry)
    if not function:
        return
    name, arguments = _function_name_and_arguments(function)
    if name is None and arguments is None:
        # A present-but-empty function payload carries nothing to queue.
        return

    entry_id, entry_index = _tool_call_entry_identity(entry)
    target = None
    if entry_id is not None:
        for queued in queue:
            if entry_id in queued.get("ids", ()):
                target = queued
                break
    if target is None and entry_index is not None:
        for queued in queue:
            # An index is not an identity: providers restamp indexes per
            # chunk, so two calls can share one. Never attach a fragment
            # carrying a new id to a call that already has a different one.
            if queued.get("index") == entry_index and (
                entry_id is None or not queued.get("ids")
            ):
                target = queued
    if target is None and entry_index is not None and entry_id is None:
        # The call opened with an id but no index; this is its first
        # index-bearing fragment.
        for queued in reversed(queue):
            if queued.get("ids") and queued.get("index") is None:
                target = queued
                break
    if target is None and entry_id is None and entry_index is None and queue:
        target = queue[-1]

    if target is None:
        target = {
            "id": str(key),
            "ids": [entry_id] if entry_id is not None else [],
            "index": entry_index,
            "name": name,
            "arguments": arguments or "",
        }
        queue.append(target)
        return

    if entry_id is not None and entry_id not in target.setdefault("ids", []):
        target["ids"].append(entry_id)
    if entry_index is not None:
        target["index"] = entry_index
    if name and not target.get("name"):
        target["name"] = name
    target["arguments"] = (target.get("arguments") or "") + (arguments or "")


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
    # in the same response are queued for later turns (see below). The id is
    # the identity that holds; the index is a fallback and may arrive on a
    # later fragment, so both are tracked.
    primary_call_key = None
    primary_call_ids = set()
    primary_call_index = None
    accumulated_review = ""
    review_category = None
    buffer = ""

    pending_calls = getattr(llm.interpreter, "_pending_tool_calls", None)
    if pending_calls:
        queued_token = pending_calls[0].get("messages")
        current = list(request_params["messages"] or [{}])
        raw_messages = getattr(llm.interpreter, "messages", None) or []
        raw_last_role = raw_messages[-1].get("role") if raw_messages else None
        if raw_last_role == "user" and (
            queued_token is None or current != queued_token
        ):
            # A new user turn arrived after these calls were queued (or the
            # entry predates queue provenance): drop them instead of
            # replaying stale calls without an LLM request. An unchanged
            # retry still serves the queue, and continuations whose last
            # message is assistant/computer output are unaffected.
            pending_calls.clear()
    if pending_calls:
        # A previous turn received more tool calls than the pipeline can
        # execute at once: serve the next queued call now, without spending
        # an LLM call, so every call runs in turn order.
        queued = pending_calls.pop(0)
        if queued.get("review"):
            # The original response passed the judge layer; carry its
            # verdict so the replayed call passes the check below too.
            accumulated_review = queued["review"]
        stream = [_queued_tool_call_chunk(queued)]
    else:
        stream = llm.completions(**request_params)

    for chunk in stream:
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
            entries = delta["tool_calls"]
            if primary_call_key is None:
                # The first entry that carries a usable function becomes the
                # call this turn runs. A leading malformed entry must not hide
                # a later valid one (#254).
                chosen = None
                for i, entry in enumerate(entries):
                    function = _tool_call_entry_function(entry)
                    if function is None:
                        continue
                    name, arguments = _function_name_and_arguments(function)
                    if name is None and arguments is None:
                        continue
                    chosen = i
                    break
                if chosen is None:
                    chosen = 0
                primary_call_key = _tool_call_entry_key(entries[chosen], chosen)
                entry_id, entry_index = _tool_call_entry_identity(entries[chosen])
                if entry_id is not None:
                    primary_call_ids.add(entry_id)
                if entry_index is not None:
                    primary_call_index = entry_index
                primary_entries = [(primary_call_key, entries[chosen])]
                extras = [
                    (_tool_call_entry_key(entry, i), entry)
                    for i, entry in enumerate(entries)
                    if i != chosen
                ]
            else:
                # Match fragments the way merge_deltas does: the id is the
                # identity that holds, the index is a fallback, and it may
                # arrive after the call opened. A fresh per-chunk key would
                # give the opening fragment an id key and its index-only
                # continuation a different index key, queuing the rest of the
                # call as a second call with malformed arguments.
                primary_entries = []
                extras = []
                for i, entry in enumerate(entries):
                    key = _tool_call_entry_key(entry, i)
                    entry_id, entry_index = _tool_call_entry_identity(entry)
                    belongs = False
                    if entry_id is not None:
                        if entry_id in primary_call_ids:
                            belongs = True
                        elif not primary_call_ids and (
                            entry_index is None
                            or primary_call_index is None
                            or entry_index == primary_call_index
                        ):
                            # The call opened without an id; it may still
                            # receive one with a later fragment.
                            belongs = True
                    elif entry_index is not None:
                        belongs = (
                            primary_call_index is None
                            or entry_index == primary_call_index
                        )
                    else:
                        # A lone identity-less entry is a continuation, not a
                        # new call.
                        belongs = len(entries) == 1
                    if belongs:
                        primary_entries.append((key, entry))
                        if entry_id is not None:
                            primary_call_ids.add(entry_id)
                        if entry_index is not None and primary_call_index is None:
                            primary_call_index = entry_index
                    else:
                        extras.append((key, entry))
            first_function = None
            for _key, entry in primary_entries:
                function = _tool_call_entry_function(entry)
                if function is None:
                    continue
                name, arguments = _function_name_and_arguments(function)
                # A nameless continuation still belongs to a call in
                # progress: merge_deltas skips its None name and appends
                # the arguments to the accumulated call. An entry with
                # neither a name nor arguments (including empty strings)
                # carries nothing to merge; keep scanning the list for a
                # usable entry rather than selecting this one.
                if not name and not arguments:
                    continue
                first_function = function
                break
            if first_function is not None:
                function_call_detected = True
                function_name, function_arguments = _function_name_and_arguments(
                    first_function
                )
                delta = {
                    # "id": primary_entries[0],
                    # Carry the content across: rebuilding the delta out of the
                    # tool call alone threw away text sent in the same message.
                    "content": content,
                    "function_call": {
                        "name": function_name,
                        "arguments": function_arguments,
                    },
                }
            else:
                # No entry carried a usable function. The raw tool_calls list is
                # nothing the merge step can consume (merge_deltas cannot
                # dict() it), so drop the key and keep any message content,
                # rather than crashing (#254). dict() works on plain dicts and
                # litellm Delta objects alike; .items() only works on the
                # former (#267).
                content = delta["content"] if "content" in delta else None
                delta = {"content": content} if content else {}
            # The pipeline executes one call per turn: queue any entries for
            # other calls so they run on later turns instead of being
            # silently dropped.
            for key, extra_entry in extras:
                _queue_extra_tool_call(llm, extra_entry, key)

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

    for queued in getattr(llm.interpreter, "_pending_tool_calls", []) or []:
        # Stamp this response's judge-layer verdict on the calls it queued,
        # so replaying them on a later turn carries the original verdict.
        # Entries left over from an earlier response keep their own. Also
        # snapshot the messages they were queued for, so a later user turn
        # is recognized as new input (stale) rather than a retry (serve).
        queued.setdefault("review", accumulated_review)
        queued.setdefault("messages", list(request_params["messages"]))

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
