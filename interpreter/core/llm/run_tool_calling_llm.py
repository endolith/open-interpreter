import json
import os
import re

from .utils.merge_deltas import merge_deltas, normalize_delta_to_dict
from .utils.parse_partial_json import parse_partial_json

tool_schema = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Executes code on the user's machine **in the user's local environment** and returns the output",
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

view_image_tool_schema = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": "Show an image to the model so it can see it. Use when you need to look at an image file. Path must be absolute (e.g. on Windows: r'C:\\Users\\...', on Mac/Linux: '/home/...'). The model has no access to the current working directory of Python or the shell.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the image file (e.g. PNG, JPEG). Must be an absolute path.",
                },
            },
            "required": ["path"],
        },
    },
}


def generate_tool_id(tool_id_num, model=None):
    """
    Generate a tool call ID. For Mistral models, uses 9-character alphanumeric format.
    For other models, uses the original format.

    Mistral requires tool call IDs to match: ^[a-zA-Z0-9]{9}$
    See: https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L309
    """
    # Check if this is a Mistral model
    is_mistral = model and ("mistral" in model.lower() or "devstral" in model.lower())

    if is_mistral:
        # Mistral requires exactly 9 alphanumeric characters
        import string
        # Base36: 0-9, a-z (36 characters total)
        base36_chars = string.digits + string.ascii_lowercase
        num = tool_id_num
        suffix = ""
        for _ in range(5):  # 5 digits to make total 9 chars (4 for "tool" + 5 for number)
            suffix = base36_chars[num % 36] + suffix
            num //= 36
        return f"tool{suffix}"
    else:
        # Original format for other models
        return f"toolu_{tool_id_num}"


def process_messages(messages, model=None):
    processed_messages = []
    last_tool_id = 0

    i = 0
    while i < len(messages):
        message = messages[i]

        if message.get("function_call"):
            last_tool_id += 1
            tool_id = generate_tool_id(last_tool_id, model)

            # Convert function_call to tool_calls
            function = message.pop("function_call")
            # Some providers (e.g. Alibaba via OpenRouter) require function.arguments to be valid JSON string.
            args = function.get("arguments")
            if isinstance(args, dict):
                function = {**function, "arguments": json.dumps(args)}
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
            tool_id = generate_tool_id(last_tool_id, model)

            # Add a tool call before this orphaned tool response. Providers like Alibaba require
            # function.arguments to be valid JSON; use execute-shaped payload to avoid API errors.
            processed_messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": "execute",
                                "arguments": json.dumps({
                                    "language": "python",
                                    "code": "# Automated tool call to fetch more output, triggered by the user.",
                                }),
                            },
                        }
                    ],
                }
            )

            # Process the function response
            message["role"] = "tool"
            message["tool_call_id"] = tool_id
            processed_messages.append(message)

        elif message.get("role") == "tool":
            # Tool message must follow an assistant message with tool_calls (OpenRouter/Alibaba etc.).
            # Unsupported/invalid tool calls yield a tool response but we never store an assistant
            # with tool_calls for that call, so we can get assistant (content) then tool (error).
            # Insert a synthetic assistant with tool_calls using this message's tool_call_id.
            prev = processed_messages[-1] if processed_messages else None
            if not prev or "tool_calls" not in prev or not prev.get("tool_calls"):
                tool_id = message.get("tool_call_id") or generate_tool_id(last_tool_id + 1, model)
                processed_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": json.dumps({
                                        "language": "python",
                                        "code": "# Placeholder for unsupported/invalid tool call (error was returned as tool response).",
                                    }),
                                },
                            }
                        ],
                    }
                )
            processed_messages.append(message)

        else:
            # For non-tool-related messages, just add them as is
            processed_messages.append(message)

        i += 1

    return processed_messages


def run_tool_calling_llm(llm, request_params):
    ## Setup

    # Check verbose flag for debug output
    verbose = llm.interpreter.verbose or llm.interpreter.debug

    if verbose:
        print(f"[DEBUG] run_tool_calling_llm called with model: {llm.model}", flush=True)
        if "reasoning" in request_params:
            print(f"[DEBUG] reasoning parameter: {request_params['reasoning']}", flush=True)

    # Add languages OI has access to
    tool_schema["function"]["parameters"]["properties"]["language"]["enum"] = [
        i.name.lower() for i in llm.interpreter.terminal.languages
    ]
    tools = [tool_schema]
    if getattr(llm, "supports_vision", None) is True:
        tools.append(view_image_tool_schema)
    request_params["tools"] = tools

    request_params["messages"] = process_messages(request_params["messages"], model=llm.model)

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
    content_yielded_during_streaming = False  # Track if content was already yielded
    has_reasoning_content = False  # Track if we have reasoning_content (to delay content output)
    reasoning_buffer = ""  # Incomplete line when streaming reasoning_content
    reasoning_streamed = False  # True if we yielded reasoning during the stream (so post-stream only yields remainder)
    reasoning_replace_yielded = False  # True after we yield the replace chunk (must happen before first content chunk)

    for chunk in llm.completions(**request_params):
        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        raw_delta = chunk["choices"][0]["delta"]

        # Normalize delta to dict immediately - LiteLLM may return Pydantic objects
        # This ensures all code paths work with plain dicts consistently
        delta = normalize_delta_to_dict(raw_delta)

        # Mark if we see tool_calls (but don't try to parse incomplete streaming data)
        if "tool_calls" in delta and delta["tool_calls"]:
            function_call_detected = True

        # Accumulate deltas
        # Note: merge_deltas now handles lists (like tool_calls) properly
        # We accumulate everything during streaming, but only parse after stream completes
        accumulated_deltas = merge_deltas(accumulated_deltas, delta)

        # Track if we have reasoning_content (even if incomplete) - this will delay content output
        if "reasoning_content" in accumulated_deltas and accumulated_deltas.get("reasoning_content"):
            has_reasoning_content = True

        # Stream reasoning_content as it arrives (raw, no blockquote). When the block is complete we
        # yield a replace chunk so the display and stored message get blockquote formatting.
        if "reasoning_content" in delta and delta["reasoning_content"]:
            new_chunk = delta["reasoning_content"]
            if isinstance(new_chunk, str):
                reasoning_buffer += new_chunk
                lines = reasoning_buffer.split("\n")
                reasoning_buffer = lines[-1]
                for line in lines[:-1]:
                    yield {"role": "assistant", "type": "message", "format": "reasoning", "content": line + "\n"}
                reasoning_streamed = True

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

                # If we have review tags, process as review
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
                    # function_call_detected is True but no review tags found
                    # This might be regular content, not a review - yield it as message
                    # But only if we don't have actual tool_calls (might be false positive)
                    if not accumulated_deltas.get("tool_calls") and not accumulated_deltas.get("function_call"):
                        # Yield replace chunk before first content so reasoning block is closed with blockquotes before the response.
                        if has_reasoning_content and reasoning_streamed and not reasoning_replace_yielded:
                            full_raw = accumulated_deltas.get("reasoning_content") or ""
                            if isinstance(full_raw, str) and full_raw.strip():
                                yield {"role": "assistant", "type": "message", "format": "reasoning", "content": full_raw.rstrip() + "\n\n", "replace": True}
                            reasoning_replace_yielded = True
                        # No actual tool calls, so this is just regular content. Stream it;
                        # reasoning (if any) was already streamed first by the provider.
                        yield {"role": "assistant", "type": "message", "content": delta["content"]}
                        content_yielded_during_streaming = True

            else:
                # Yield replace chunk before first content so reasoning block is closed with blockquotes before the response.
                if has_reasoning_content and reasoning_streamed and not reasoning_replace_yielded:
                    full_raw = accumulated_deltas.get("reasoning_content") or ""
                    if isinstance(full_raw, str) and full_raw.strip():
                        yield {"role": "assistant", "type": "message", "format": "reasoning", "content": full_raw.rstrip() + "\n\n", "replace": True}
                    reasoning_replace_yielded = True
                # Stream content as it arrives; reasoning (if any) already streamed first.
                yield {"role": "assistant", "type": "message", "content": delta["content"]}
                content_yielded_during_streaming = True

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
            arguments_str = accumulated_deltas["function_call"]["arguments"]
            # Ensure arguments is a string before slicing
            if isinstance(arguments_str, str):
                code_delta = arguments_str[len(code) :]
                # Update the code
                code = arguments_str
                # Yield the delta
                if code_delta:
                    yield {
                        "role": "assistant",
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

                # Ensure arguments is a dictionary, not a string or None
                if not isinstance(arguments, dict):
                    arguments = None

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
                        # Ensure code is a string (some models may return other types)
                        code_value = arguments["code"]
                        if not isinstance(code_value, str):
                            # If code is not a string, skip this chunk
                            continue
                        # Calculate the delta (new characters only)
                        code_delta = code_value[len(code) :]
                        # Update the code
                        code = code_value
                        # Yield the delta
                        if code_delta:
                            yield {
                                "role": "assistant",
                                "type": "code",
                                "format": language,
                                "content": code_delta,
                            }
                else:
                    if llm.interpreter.verbose:
                        print("Arguments not a dict.")

    # After stream completes, convert tool_calls to function_call format if needed
    # Don't try to parse incomplete tool_calls during streaming

    # Debug: Always check what we have after stream
    has_tool_calls = "tool_calls" in accumulated_deltas and accumulated_deltas["tool_calls"]
    has_function_call = bool(accumulated_deltas.get("function_call"))
    has_content = "content" in accumulated_deltas and accumulated_deltas.get("content")

    # NOTE: llm.interpreter.verbose sometimes returns False even when --verbose flag is passed.
    # As a workaround, we check the debug attribute which is typically set alongside verbose.
    # If debug is True, we also enable verbose output for consistency.
    # This appears to be related to interpreter instance handling during profile loading.
    verbose = llm.interpreter.verbose or llm.interpreter.debug

    # Debug info only in verbose mode
    if verbose:
        print(f"[DEBUG] After stream - has_tool_calls: {has_tool_calls}, has_function_call: {has_function_call}, has_content: {bool(has_content)}", flush=True)
        print(f"[DEBUG] accumulated_deltas keys: {list(accumulated_deltas.keys())}", flush=True)
        # NOTE: Provider detection removed - OpenRouter routes to different providers (DeepInfra, Together)
        # but this information is only available in OpenRouter's API response metadata, not in LiteLLM chunks.
        # DeepInfra returns reasoning_content as separate field, Together mixes it into content.
        if has_tool_calls:
            print(f"[DEBUG] tool_calls type: {type(accumulated_deltas['tool_calls'])}, value: {json.dumps(accumulated_deltas['tool_calls'], default=str)[:1000]}", flush=True)
        if has_function_call:
            print(f"[DEBUG] function_call: {json.dumps(accumulated_deltas['function_call'], default=str)[:500]}", flush=True)
        if has_content:
            content_preview = str(accumulated_deltas.get("content", ""))[:200]
            print(f"[DEBUG] content preview: {repr(content_preview)}", flush=True)
        if "reasoning_content" in accumulated_deltas:
            reasoning_preview = str(accumulated_deltas.get("reasoning_content", ""))[:200]
            print(f"[DEBUG] reasoning_content preview: {repr(reasoning_preview)}", flush=True)

    # POST-STREAM PROCESSING: Yield in order: reasoning → content → code
    # This ensures the model's thought process is shown before actions

    # 1. REASONING: Yield reasoning_content if present (in block quotes)
    if has_reasoning_content and "reasoning_content" in accumulated_deltas and accumulated_deltas["reasoning_content"]:
        if reasoning_streamed:
            # Replace was already yielded before first content chunk; only yield if we never got any content (no content delta in stream).
            if not reasoning_replace_yielded:
                full_raw = accumulated_deltas.get("reasoning_content") or ""
                if isinstance(full_raw, str) and full_raw.strip():
                    yield {"role": "assistant", "type": "message", "format": "reasoning", "content": full_raw.rstrip() + "\n\n", "replace": True}
        else:
            # Provider sent reasoning only at end (e.g. no per-chunk reasoning_content); yield full block
            reasoning_content = accumulated_deltas["reasoning_content"]
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                if verbose:
                    print(f"[DEBUG] reasoning_content length: {len(reasoning_content)}, preview: {repr(reasoning_content[:200])}", flush=True)
                    if "content" in accumulated_deltas:
                        print(f"[DEBUG] content length: {len(accumulated_deltas['content'])}, preview: {repr(accumulated_deltas['content'][:200])}", flush=True)

                yield {"role": "assistant", "type": "message", "format": "reasoning", "content": reasoning_content.rstrip() + "\n\n"}

    # 2. CONTENT: Yield accumulated_review or regular content
    if accumulated_review and review_category == None:
        if accumulated_review.strip():
            yield {"role": "assistant", "type": "message", "content": accumulated_review}
    elif "content" in accumulated_deltas and accumulated_deltas["content"]:
        content = accumulated_deltas["content"]
        if not accumulated_deltas.get("function_call") and not accumulated_deltas.get("tool_calls"):
            if content.strip() and not content_yielded_during_streaming:
                yield {"role": "assistant", "type": "message", "content": content}

    # 3. Finally, process and yield code blocks (function_call/tool_calls)
    tool_call_id_for_error = None  # Store tool_call_id in case we need to yield error as tool response
    if "tool_calls" in accumulated_deltas and accumulated_deltas["tool_calls"]:
        if not accumulated_deltas.get("function_call"):
            # Try to convert tool_calls to function_call format now that stream is complete
            tool_calls = accumulated_deltas["tool_calls"]

            # Debug: log what we received (only in verbose mode)
            if llm.interpreter.verbose:
                print(f"[DEBUG] Converting tool_calls after stream. tool_calls type: {type(tool_calls)}, value: {json.dumps(tool_calls, default=str)[:500]}", flush=True)

            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                # Extract tool_call_id for potential error response
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call_id_for_error = tool_call["id"]
                elif hasattr(tool_call, "id"):
                    tool_call_id_for_error = tool_call.id

                converted = False
                if isinstance(tool_call, dict) and "function" in tool_call:
                    if isinstance(tool_call["function"], dict):
                        accumulated_deltas["function_call"] = {
                            "name": tool_call["function"].get("name"),
                            "arguments": tool_call["function"].get("arguments"),
                        }
                        function_call_detected = True
                        converted = True
                        if llm.interpreter.verbose:
                            print(f"[DEBUG] Converted tool_call to function_call: name={accumulated_deltas['function_call']['name']}", flush=True)
                elif hasattr(tool_call, "function"):
                    accumulated_deltas["function_call"] = {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    function_call_detected = True
                    converted = True
                    if llm.interpreter.verbose:
                        print(f"[DEBUG] Converted tool_call (object) to function_call: name={accumulated_deltas['function_call']['name']}", flush=True)

                # If we still couldn't convert, raise an error with details
                if not converted:
                    raise Exception(
                        f"Unsupported tool_call format. Type: {type(tool_call)}, "
                        f"Has 'function' attr: {hasattr(tool_call, 'function')}, "
                        f"Is dict: {isinstance(tool_call, dict)}, "
                        f"Dict keys if dict: {list(tool_call.keys()) if isinstance(tool_call, dict) else 'N/A'}"
                    )

    # Process the converted function_call (if any) to yield code
    if accumulated_deltas.get("function_call"):
        function_call = accumulated_deltas["function_call"]
        function_name = function_call.get("name", "")

        # If we don't have tool_call_id yet, try to extract it from the last assistant message
        if not tool_call_id_for_error:
            # Look at the last message in the conversation to find tool_call_id
            messages = request_params.get("messages", [])
            for message in reversed(messages):
                if message.get("role") == "assistant" and "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    if isinstance(tool_calls, list) and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        if isinstance(tool_call, dict) and "id" in tool_call:
                            tool_call_id_for_error = tool_call["id"]
                            break
                        elif hasattr(tool_call, "id"):
                            tool_call_id_for_error = tool_call.id
                            break

        # Ensure tool_call_id is a non-empty string if we have it
        if tool_call_id_for_error and not isinstance(tool_call_id_for_error, str):
            tool_call_id_for_error = str(tool_call_id_for_error)
        if tool_call_id_for_error == "":
            tool_call_id_for_error = None

        # Only "execute" is supported as a direct tool call
        # Other functions (like toolbox.web.brave) must be called from within Python code
        if function_name == "execute":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)

            # Validate arguments and yield code, or yield error as tool response
            if isinstance(arguments, dict):
                if language is None and "language" in arguments and "code" in arguments and arguments["language"]:
                    language = arguments["language"]

                if language is not None and "code" in arguments:
                    code_value = arguments["code"]
                    if isinstance(code_value, str):
                        # Yield the full code (since we converted after stream, code variable is empty)
                        if code_value:
                            yield {
                                "role": "assistant",
                                "type": "code",
                                "format": language,
                                "content": code_value,
                            }
                        else:
                            # Empty code - yield error as tool response
                            error_msg = "Invalid execute call: code is empty"
                            if tool_call_id_for_error and isinstance(tool_call_id_for_error, str) and tool_call_id_for_error.strip():
                                yield {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id_for_error,
                                    "type": "message",
                                    "content": error_msg
                                }
                            elif verbose:
                                print(f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}", flush=True)
                            if verbose:
                                print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                    else:
                        # Code is not a string - yield error as tool response
                        error_msg = f"Invalid execute call: code must be a string, got {type(code_value).__name__}"
                        if tool_call_id_for_error and isinstance(tool_call_id_for_error, str) and tool_call_id_for_error.strip():
                            yield {
                                "role": "tool",
                                "tool_call_id": tool_call_id_for_error,
                                "type": "message",
                                "content": error_msg
                            }
                        elif verbose:
                            print(f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}", flush=True)
                        if verbose:
                            print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                else:
                    # Missing language or code - yield error as tool response
                    error_msg = f"Invalid execute call: missing required fields. Got: {list(arguments.keys())}"
                    if verbose:
                        print(f"[ERROR] {error_msg}. Arguments: {json.dumps(arguments, default=str)}", flush=True)
                        print(f"[ERROR] tool_call_id_for_error: {repr(tool_call_id_for_error)}, type: {type(tool_call_id_for_error)}", flush=True)

                    if tool_call_id_for_error and isinstance(tool_call_id_for_error, str) and tool_call_id_for_error.strip():
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call_id_for_error,
                            "type": "message",
                            "content": error_msg
                        }
                        if verbose:
                            print(f"[ERROR] Yielding tool response: {json.dumps(tool_response, default=str)}", flush=True)
                        yield tool_response
                    else:
                        # No tool_call_id available - this should not happen, but log it
                        if verbose:
                            print(f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}", flush=True)
                            print(f"[ERROR] tool_call_id_for_error value: {repr(tool_call_id_for_error)}", flush=True)
                        # Still yield as assistant message so user sees the error
                        yield {
                            "role": "assistant",
                            "type": "message",
                            "content": f"**Error:** {error_msg}"
                        }
            else:
                # Arguments is not a dict - yield error as tool response
                error_msg = f"Invalid execute call: arguments must be a dict, got {type(arguments).__name__}"
                if tool_call_id_for_error and isinstance(tool_call_id_for_error, str) and tool_call_id_for_error.strip():
                    yield {
                        "role": "tool",
                        "tool_call_id": tool_call_id_for_error,
                        "type": "message",
                        "content": error_msg
                    }
                elif verbose:
                    print(f"[ERROR] Cannot yield tool response: missing tool_call_id. Error: {error_msg}", flush=True)
                if verbose:
                    print(f"[ERROR] {error_msg}. Function call: {json.dumps(function_call, default=str)}", flush=True)
        elif function_name == "view_image":
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_partial_json(arguments)
            path = isinstance(arguments, dict) and arguments.get("path")
            if not path or not isinstance(path, str):
                content = "view_image: path is required and must be a string."
            elif not os.path.isabs(path):
                content = "view_image: path must be absolute (e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac)."
            elif not os.path.exists(path):
                content = f"view_image: file not found: {path}"
            else:
                llm.interpreter._pending_view_image_path = path
                content = "Image added; you will see it when you continue."
            if tool_call_id_for_error:
                yield {
                    "role": "tool",
                    "tool_call_id": tool_call_id_for_error,
                    "type": "message",
                    "content": content,
                }
            else:
                yield {"role": "assistant", "type": "message", "content": content}
        elif function_name:
            # Unsupported function call - yield error as tool response to maintain proper message ordering
            # The API expects: assistant (with tool_call) → tool (response) → user
            error_msg = (
                f"Unsupported function call: '{function_name}'. "
                f"Only 'execute' and 'view_image' (vision models only) are supported as direct tool calls. "
                f"To use '{function_name}', call it from within Python code using the execute function. "
                f"For example: `toolbox.web.brave(query='...')`"
            )

            # Yield error as tool response so the model sees it and message ordering stays correct (assistant → tool → …).
            # Any assistant message content the model sent before this tool call is already yielded above with role "assistant".
            if tool_call_id_for_error:
                yield {
                    "role": "tool",
                    "tool_call_id": tool_call_id_for_error,
                    "type": "message",
                    "content": error_msg
                }
            else:
                yield {
                    "role": "assistant",
                    "type": "message",
                    "content": f"**Error:** {error_msg}"
                }

            if verbose:
                print(f"[ERROR] {error_msg}", flush=True)
                print(f"[ERROR] Function call details: {json.dumps(function_call, default=str)}", flush=True)
                if tool_call_id_for_error:
                    print(f"[ERROR] Yielding error as tool response with tool_call_id: {tool_call_id_for_error}", flush=True)

    if os.getenv("INTERPRETER_REQUIRE_AUTHENTICATION", "False").lower() == "true":
        print("function_call_detected", function_call_detected)
        print("accumulated_review", accumulated_review)
        if function_call_detected and not accumulated_review:
            print("WTF!!!!!!!!!")
            # import pdb
            # pdb.set_trace()
            raise Exception("Judge layer required but did not run.")
