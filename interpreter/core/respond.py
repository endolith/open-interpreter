import json
import os
import re
import sys
import time
import traceback

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import html2text
import litellm
import openai
from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from ..terminal_interface.utils.display_markdown_message import display_markdown_message
from .render_message import render_message
from .tools.file_edit import run_edit
from .toolbox.web.web import WebToolboxError, ApiKeyError
from .utils.prompt_choice import prompt_choice

_LITELLM_OPTIONAL_API_EXCEPTIONS = tuple(
    getattr(litellm.exceptions, name)
    for name in ("ServiceUnavailableError", "InternalServerError")
    if hasattr(litellm.exceptions, name)
)


def _html_error_to_renderable(error_str):
    """
    If error_str contains an HTML error body (e.g. a provider 502 page),
    convert it to a Rich Markdown renderable and return it. Otherwise return
    None.

    The exception string has a plain-text prefix before the HTML (e.g.
    "litellm.APIError: OpenrouterException - <!DOCTYPE html>..."), so we slice
    from the first HTML tag before handing off to html2text, which converts it
    to Markdown for Rich to render.
    """
    # Quick bail-out: not HTML at all
    lower = error_str.lower()
    doctype_idx = lower.find("<!doctype")
    html_idx = lower.find("<html")
    if doctype_idx == -1 and html_idx == -1:
        return None

    # Slice from whichever HTML marker appears first
    indices = [i for i in (doctype_idx, html_idx) if i >= 0]
    html_part = error_str[min(indices):]

    try:
        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # Let Rich handle reflowing
        md = h2t.handle(html_part).strip()
    except Exception:
        return None

    if not md:
        return None

    # Leading newline so the first line of content is not clipped by the panel title bar
    return Markdown("\n" + md)


def _stdin_is_interactive():
    """False for uvicorn workers, pytest, and other environments with no real TTY."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _is_temporary_provider_error(error):
    error_message = str(error).lower()
    temporary_markers = (
        "temporarily",
        "temporary",
        "retry shortly",
        "try again later",
        "rate-limit",
        "rate limited",
        "429",
        "503",
        "504",
        "timeout",
        "timed out",
        "overloaded",
        "unavailable",
    )
    return any(marker in error_message for marker in temporary_markers)


def _temporary_error_signature(error):
    """
    Normalize temporary provider errors so repeated retries with the same
    upstream failure don't spam duplicate full error panels.
    """
    return re.sub(r"\s+", " ", str(error)).strip()


def _render_temporary_retry_status(retry_count):
    """
    Update a single in-place status line for temporary upstream retries.
    """
    dots = "." * ((retry_count - 1) % 3 + 1)
    sys.stdout.write(f"\r  ▌ Temporary upstream provider error; retrying{dots}")
    sys.stdout.flush()


def respond(interpreter):
    """
    Yields chunks.
    Responds until it decides not to run any more code or say anything else.
    """

    last_unsupported_code = ""
    insert_loop_message = False
    always_retry_provider_errors = False
    temporary_provider_error_retries = 0
    last_temporary_provider_error_signature = None

    while True:
        ## RENDER SYSTEM MESSAGE ##

        system_message = interpreter.system_message

        # Add language-specific system messages
        for language in interpreter.terminal.languages:
            if hasattr(language, "system_message"):
                system_message += "\n\n" + language.system_message

        # Add custom instructions
        if interpreter.custom_instructions:
            system_message += "\n\n## User's Custom Instructions\n\n" + interpreter.custom_instructions

        # OpenAI-compatible server: client system prompt(s) from the HTTP request body
        server_request_system = getattr(interpreter, "_server_request_system", None)
        if server_request_system:
            system_message += (
                "\n\n## Client system prompt\n\n" + server_request_system
            )

        # Add toolbox API system message
        if interpreter.toolbox.import_toolbox_api:
            if interpreter.toolbox.system_message not in system_message:
                system_message = (
                    system_message + "\n\n" + interpreter.toolbox.system_message
                )

        # Storing the messages so they're accessible in the interpreter's toolbox
        # no... this is a huge time sink.....
        # if interpreter.sync_computer:
        #     output = interpreter.toolbox.run(
        #         "python", f"messages={interpreter.messages}"
        #     )

        ## Rendering ↓
        rendered_system_message = render_message(interpreter, system_message)
        ## Rendering ↑

        # Store the actual rendered system message for %info command (before converting to dict)
        interpreter._last_rendered_system_message = rendered_system_message

        rendered_system_message = {
            "role": "system",
            "type": "message",
            "content": rendered_system_message,
        }

        # Create the version of messages that we'll send to the LLM
        messages_for_llm = [
            m for m in interpreter.messages.copy() if m.get("role") != "system"
        ]
        messages_for_llm = [rendered_system_message] + messages_for_llm

        if insert_loop_message:
            messages_for_llm.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": loop_message,
                }
            )
            # Yield two newlines to separate the LLMs reply from previous messages.
            yield {"role": "assistant", "type": "message", "content": "\n\n"}
            insert_loop_message = False

        ### RUN THE LLM ###

        assert (
            len(interpreter.messages) > 0
        ), "User message was not passed in. You need to pass in at least one message."

        if interpreter.messages[-1]["type"] not in ("code", "edit"):  # If it is, we run below
            try:
                for chunk in interpreter.llm.run(messages_for_llm):
                    yield {"role": "assistant", **chunk}

            except litellm.exceptions.BudgetExceededError:
                interpreter.display_message(
                    f"""> Max budget exceeded

                    **Session spend:** ${litellm._current_cost}
                    **Max budget:** ${interpreter.max_budget}

                    Press CTRL-C then run `interpreter --max_budget [higher USD amount]` to proceed.
                """
                )
                break

            except Exception as e:
                # Continue with existing error handling
                error_message = str(e).lower()

                # Check for API errors - display them in a panel without tracebacks
                # Also check for our formatted errors (containing |||)
                error_str = str(e)
                # Normalize provider/API errors (LiteLLM + OpenAI) so we can render
                # them consistently using Rich, regardless of which exception class
                # LiteLLM chose for the underlying provider (e.g. OpenRouter 5xx).
                if isinstance(e, (
                    # LiteLLM exception variants
                    getattr(litellm, "APIError", Exception),
                    getattr(litellm, "OpenAIError", Exception),
                    litellm.exceptions.APIError,
                    litellm.exceptions.OpenAIError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.BadRequestError,
                    litellm.exceptions.RateLimitError,
                    litellm.exceptions.AuthenticationError,
                    *_LITELLM_OPTIONAL_API_EXCEPTIONS,
                    # OpenAI Python client variants (defensive, in case they leak through)
                    getattr(openai, "APIError", Exception),
                    getattr(openai, "OpenAIError", Exception),
                )):
                    is_temporary_error = _is_temporary_provider_error(e)
                    panel_border_style = "yellow" if is_temporary_error else "red"
                    panel_title = "Warning" if is_temporary_error else "Error"
                    temporary_error_signature = (
                        _temporary_error_signature(e) if is_temporary_error else None
                    )
                    if (
                        is_temporary_error
                        and temporary_error_signature
                        == last_temporary_provider_error_signature
                    ):
                        temporary_provider_error_retries += 1
                        _render_temporary_retry_status(temporary_provider_error_retries)
                        time.sleep(2)
                        continue
                    # Format with Rich Panel with red border for errors
                    # Check if this is an error with JSON structure that can be parsed
                    if "{" in error_str and "}" in error_str:
                        # Parse the JSON structure and format it nicely
                        try:
                            # Extract JSON from error string
                            json_start = error_str.find("{")
                            json_end = error_str.rfind("}") + 1
                            error_data = json.loads(error_str[json_start:json_end])

                            # Build text content with nested structure
                            lines = ["[bold]OpenRouterException:[/bold]"]
                            lines.append("")

                            def format_value(val, indent=0):
                                """Format a value with proper indentation"""
                                prefix = "  " * indent
                                if isinstance(val, dict):
                                    result = []
                                    for key, value in val.items():
                                        if isinstance(value, dict):
                                            result.append(f"{prefix}• {key}:")
                                            result.extend(format_value(value, indent + 1))
                                        elif isinstance(value, list):
                                            result.append(f"{prefix}• {key}:")
                                            for item in value:
                                                result.extend(format_value(item, indent + 1))
                                        else:
                                            # For certain fields, show the value directly if it's already a helpful message
                                            if key == "raw" or (key == "message" and isinstance(value, str) and len(value) < 100):
                                                result.append(f"{prefix}• {key}: {value}")
                                            else:
                                                result.append(f"{prefix}• {key}: {value}")
                                    return result
                                else:
                                    return [f"{prefix}• {val}"]

                            lines.extend(format_value(error_data))

                            formatted_error = "\n".join(lines)
                            panel = Panel(
                                formatted_error,
                                border_style=panel_border_style,
                                title=panel_title,
                                title_align="left"
                            )
                            # Yield a special chunk to stop Live display before printing error panel
                            # This prevents Live display from overwriting the error panel
                            yield {"type": "stop_live_display"}
                            print("")  # Newline so panel top border is not cut off
                            rich_print(panel)
                            print("")  # Add space after error
                        except Exception:
                            # Fallback if JSON parsing fails (e.g. body is HTML)
                            display = _html_error_to_renderable(error_str) or error_str
                            yield {"type": "stop_live_display"}
                            print("")  # Newline so panel top border is not cut off
                            panel = Panel(
                                display,
                                border_style=panel_border_style,
                                title=panel_title,
                                title_align="left"
                            )
                            rich_print(panel)
                            print("")
                    else:
                        # Format all other API errors in a Panel. If the body is HTML (e.g. provider 502 page),
                        # convert with html2text and render as Rich Markdown.
                        display = _html_error_to_renderable(error_str) or error_str
                        yield {"type": "stop_live_display"}
                        print("")  # Newline so panel top border is not cut off by previous output
                        panel = Panel(
                            display,
                            border_style=panel_border_style,
                            title=panel_title,
                            title_align="left"
                        )
                        rich_print(panel)
                        print("")  # Add space after error

                    # Temporary provider errors (including upstream rate limits)
                    # are retried automatically to avoid blocking on user input.
                    if is_temporary_error:
                        last_temporary_provider_error_signature = temporary_error_signature
                        temporary_provider_error_retries += 1
                        _render_temporary_retry_status(temporary_provider_error_retries)
                        time.sleep(2)
                        continue

                    # For non-temporary provider errors, offer manual retry.
                    if always_retry_provider_errors:
                        print("")
                        interpreter.display_message("> Retrying...")
                        time.sleep(2)
                        continue

                    if _stdin_is_interactive():
                        retry_choice = prompt_choice(
                            "  Retry? (y = retry once, a = keep retrying, n = stop)\n\n  ",
                            ("y", "a", "n"),
                        )

                        if retry_choice == "a":
                            always_retry_provider_errors = True
                            interpreter.display_message("> Retrying...")
                            time.sleep(2)
                            continue
                        if retry_choice == "y":
                            interpreter.display_message("> Retrying...")
                            time.sleep(2)
                            continue

                        interpreter._stopped_retrying = True
                        return

                    interpreter._stopped_retrying = True
                    return

                if (
                    interpreter.offline == False
                    and ("auth" in error_message or
                         "api key" in error_message)
                ):
                    # Provide extra information on how to change API keys, if
                    # we encounter that error (Many people writing GitHub
                    # issues were struggling with this)
                    output = traceback.format_exc()

                    # Generic hint: a hard-coded llm.api_key can conflict with CLI-provided model/provider
                    api_key_in_config = bool(getattr(interpreter.llm, "api_key", None))
                    provider_hint = ""
                    if api_key_in_config:
                        provider_hint = (
                            "\n\nHint: You have `llm.api_key` set in your profile/config. "
                            "If you pass `--model` (or `--api_key`) on the command line and they don't match the same provider, you'll get 401 Unauthorized. "
                            "Either remove `llm.api_key` from your profile/default.yaml so your command-line selection takes effect, "
                            "or pass both `--model` and `--api_key` together on the command line to ensure they match."
                        )

                    raise Exception(
                        f"{output}\n\nThere might be an issue with your API key(s).{provider_hint}\n\n"
                        "To reset your API key (we'll use OPENAI_API_KEY for this example, but you may need to reset your ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY, etc):\n        Mac/Linux: 'export OPENAI_API_KEY=your-key-here'. Update your ~/.zshrc on MacOS or ~/.bashrc on Linux with the new key if it has already been persisted there.,\n        Windows: 'setx OPENAI_API_KEY your-key-here' then restart terminal.\n\n"
                    )
                elif (
                    isinstance(e, litellm.exceptions.RateLimitError)
                    and ("exceeded" in str(e).lower() or
                         "insufficient_quota" in str(e).lower())
                ):
                    display_markdown_message(
                        f""" > You ran out of current quota for OpenAI's API, please check your plan and billing details. You can either wait for the quota to reset or upgrade your plan.

                        To check your current usage and billing details, visit the [OpenAI billing page](https://platform.openai.com/settings/organization/billing/overview).

                        You can also use `interpreter --max_budget [higher USD amount]` to set a budget for your sessions.
                        """
                    )

                elif (
                    interpreter.offline == False and "not have access" in str(e).lower()
                ):
                    # Check for invalid model in error message and then fallback.
                    if (
                        "invalid model" in error_message
                        or "model does not exist" in error_message
                    ):
                        provider_message = f"\n\nThe model '{interpreter.llm.model}' does not exist or is invalid. Please check the model name and try again.\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n  "
                    elif "groq" in error_message:
                        provider_message = f"\n\nYou do not have access to {interpreter.llm.model}. Please check with Groq for more details.\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n  "
                    else:
                        provider_message = f"\n\nYou do not have access to {interpreter.llm.model}. If you are using an OpenAI model, you may need to add a payment method and purchase credits for the OpenAI API billing page (this is different from ChatGPT Plus).\n\nhttps://platform.openai.com/account/billing/overview\n\nWould you like to try Open Interpreter's hosted `i` model instead? (y/n)\n\n"

                    print(provider_message)

                    response = prompt_choice("  ", ("y", "n"))

                    if response == "y":
                        interpreter.llm.model = "i"
                        interpreter.display_message(f"> Model set to `i`")
                        interpreter.display_message(
                            "***Note:*** *Conversations with this model will be used to train our open-source model.*\n"
                        )

                    else:
                        raise
                elif interpreter.offline and not interpreter.os:
                    raise
                else:
                    raise

            else:
                if temporary_provider_error_retries > 0:
                    print("")
                temporary_provider_error_retries = 0
                last_temporary_provider_error_signature = None

        # Inject image from view_image tool call (tool appends result first, then we add user image)
        pending_path = getattr(interpreter, "_pending_view_image_path", None)
        if pending_path is not None:
            delattr(interpreter, "_pending_view_image_path")
            pending_shrink = getattr(interpreter, "_pending_view_image_shrink", None)
            if pending_shrink is not None:
                delattr(interpreter, "_pending_view_image_shrink")
            img_msg = {
                "role": "user",
                "type": "image",
                "format": "path",
                "content": pending_path,
            }
            if pending_shrink is not None:
                img_msg["shrink"] = pending_shrink
            interpreter.messages.append(img_msg)

        ### RUN FILE EDIT (if it's there) ###

        if interpreter.messages[-1]["type"] == "edit":
            edit_msg = interpreter.messages[-1]
            language = edit_msg["format"].lower().strip()
            code = edit_msg["content"]
            target = edit_msg.get("target", "")
            tool_call_id = edit_msg.get("tool_call_id")

            if interpreter.verbose:
                print("Running edit:", edit_msg)

            try:
                # Yield confirmation so the terminal can prompt y/n (respects auto_run).
                # format: "edit" distinguishes this from a code execution confirmation.
                try:
                    yield {
                        "role": "computer",
                        "type": "confirmation",
                        "format": "edit",
                        "content": {
                            "format": language,
                            "content": code,
                            "target": target,
                        },
                    }
                except GeneratorExit:
                    break

                # Re-read in case the user edited target/content (unlikely for edit, but consistent)
                edit_msg = [m for m in interpreter.messages if m["type"] == "edit"][-1]
                language = edit_msg["format"].lower().strip()
                code = edit_msg["content"]
                target = edit_msg.get("target", target)

                output = run_edit(language, code, target)

            except KeyboardInterrupt:
                break
            except Exception as e:
                output = traceback.format_exc() if interpreter.debug else str(e)

            if tool_call_id and isinstance(tool_call_id, str) and tool_call_id.strip():
                yield {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "type": "message",
                    "content": output,
                }
            else:
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": output,
                }
            continue

        ### RUN CODE (if it's there) ###

        if interpreter.messages[-1]["type"] == "code":
            if interpreter.verbose:
                print("Running code:", interpreter.messages[-1])

            try:
                # What language/code do you want to run?
                language = interpreter.messages[-1]["format"].lower().strip()
                code = interpreter.messages[-1]["content"]

                if code.startswith("`\n"):
                    code = code[2:].strip()
                    if interpreter.verbose:
                        print("Removing `\n")
                    interpreter.messages[-1]["content"] = code  # So the LLM can see it.

                # A common hallucination
                if code.startswith("functions.execute("):
                    edited_code = code.replace("functions.execute(", "").rstrip(")")
                    try:
                        code_dict = json.loads(edited_code)
                        language = code_dict.get("language", language)
                        code = code_dict.get("code", code)
                        interpreter.messages[-1][
                            "content"
                        ] = code  # So the LLM can see it.
                        interpreter.messages[-1][
                            "format"
                        ] = language  # So the LLM can see it.
                    except:
                        pass

                # print(code)
                # print("---")
                # time.sleep(2)

                if code.strip().endswith("executeexecute"):
                    code = code.replace("executeexecute", "")
                    try:
                        interpreter.messages[-1][
                            "content"
                        ] = code  # So the LLM can see it.
                    except:
                        pass

                if code.replace("\n", "").replace(" ", "").startswith('{"language":'):
                    try:
                        code_dict = json.loads(code)
                        if set(code_dict.keys()) == {"language", "code"}:
                            language = code_dict["language"]
                            code = code_dict["code"]
                            interpreter.messages[-1][
                                "content"
                            ] = code  # So the LLM can see it.
                            interpreter.messages[-1][
                                "format"
                            ] = language  # So the LLM can see it.
                    except:
                        pass

                if code.replace("\n", "").replace(" ", "").startswith("{language:"):
                    try:
                        code = code.replace("language: ", '"language": ').replace(
                            "code: ", '"code": '
                        )
                        code_dict = json.loads(code)
                        if set(code_dict.keys()) == {"language", "code"}:
                            language = code_dict["language"]
                            code = code_dict["code"]
                            interpreter.messages[-1][
                                "content"
                            ] = code  # So the LLM can see it.
                            interpreter.messages[-1][
                                "format"
                            ] = language  # So the LLM can see it.
                    except:
                        pass

                if (
                    language == "text"
                    or language == "markdown"
                    or language == "plaintext"
                ):
                    # It does this sometimes just to take notes. Let it, it's useful.
                    # In the future we should probably not detect this behavior as code at all.
                    real_content = interpreter.messages[-1]["content"]
                    interpreter.messages[-1] = {
                        "role": "assistant",
                        "type": "message",
                        "content": f"```\n{real_content}\n```",
                    }
                    continue

                # Is this language enabled/supported?
                if interpreter.terminal.get_language(language) is None:
                    output = f"`{language}` disabled or not supported."

                    yield {
                        "role": "computer",
                        "type": "console",
                        "format": "output",
                        "content": output,
                    }

                    # Let the response continue so it can deal with the unsupported code in another way. Also prevent looping on the same piece of code.
                    if code != last_unsupported_code:
                        last_unsupported_code = code
                        continue
                    else:
                        break

                # Is there any code at all?
                if code.strip() == "":
                    yield {
                        "role": "computer",
                        "type": "console",
                        "format": "output",
                        "content": "Code block was empty. Please try again, be sure to write code before executing.",
                    }
                    continue

                # Yield a message, such that the user can stop code execution if they want to
                try:
                    yield {
                        "role": "computer",
                        "type": "confirmation",
                        "format": "execution",
                        "content": {
                            "type": "code",
                            "format": language,
                            "content": code,
                        },
                    }
                except GeneratorExit:
                    # The user might exit here.
                    # We need to tell python what we (the generator) should do if they exit
                    break

                # They may have edited the code! Grab it again
                code = [m for m in interpreter.messages if m["type"] == "code"][-1][
                    "content"
                ]

                # don't let it import toolbox — we handle that!
                if interpreter.toolbox.import_toolbox_api and language == "python":
                    # Check for nested imports like "from toolbox.ai2 import client"
                    nested_import_match = re.search(r"from toolbox\.(\w+) import (.+)", code)
                    if nested_import_match:
                        module = nested_import_match.group(1)
                        items = [item.strip() for item in nested_import_match.group(2).split(",")]
                        first_item = items[0]
                        raise ValueError(
                            f"Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                            f"Instead of: `from toolbox.{module} import {', '.join(items)}`\n"
                            f"Use directly: `toolbox.{module}.{first_item}` (and similarly for other items)\n"
                            f"For example, instead of `from toolbox.ai2 import client`, use `toolbox.ai2.client` directly.\n"
                            f"Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available."
                        )

                    # Check for direct imports from toolbox
                    direct_import_match = re.search(r"from toolbox import (.+)", code)
                    if direct_import_match:
                        items = [item.strip() for item in direct_import_match.group(1).split(",")]
                        first_item = items[0]
                        raise ValueError(
                            f"Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                            f"Instead of: `from toolbox import {', '.join(items)}`\n"
                            f"Use directly: `toolbox.{first_item}` (and similarly for other items)\n"
                            f"Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available."
                        )

                    # Check for simple import statements
                    if re.search(r"^import toolbox\b", code, re.MULTILINE):
                        raise ValueError(
                            "Cannot import `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                            "Do NOT import `toolbox`. It is already available as a variable named `toolbox`.\n"
                            "Use `toolbox` directly without any import statement."
                        )

                    # Check for import toolbox.something
                    if re.search(r"^import toolbox\.\w+", code, re.MULTILINE):
                        raise ValueError(
                            "Cannot import from `toolbox`. The `toolbox` object is already available as a variable in your namespace.\n"
                            "Do NOT import `toolbox` or try to import any of its sub-modules. The `toolbox` object is already available.\n"
                            "Use `toolbox` directly without any import statement."
                        )
                    # If it does this it sees the screenshot twice (which is expected jupyter behavior)
                    if any(
                        code.strip().split("\n")[-1].startswith(text)
                        for text in [
                            "toolbox.display.view",
                            "toolbox.display.screenshot",
                            "toolbox.view",
                            "toolbox.screenshot",
                        ]
                    ):
                        code = code + "\npass"

                # sync up some things (is this how we want to do this?)
                interpreter.toolbox.verbose = interpreter.verbose
                interpreter.toolbox.debug = interpreter.debug
                interpreter.toolbox.emit_images = interpreter.llm.supports_vision
                interpreter.toolbox.max_output = interpreter.max_output

                # sync up the interpreter's toolbox with your toolbox
                try:
                    if interpreter.sync_computer and language == "python":
                        toolbox_dict = interpreter.toolbox.to_dict()
                        if "_hashes" in toolbox_dict:
                            toolbox_dict.pop("_hashes")
                        if "system_message" in toolbox_dict:
                            toolbox_dict.pop("system_message")
                        toolbox_json = json.dumps(toolbox_dict)
                        sync_code = f"""import json\ntoolbox.load_dict(json.loads('''{toolbox_json}'''))"""
                        interpreter.terminal.run("python", sync_code)
                except Exception as e:
                    if interpreter.debug:
                        raise
                    print(str(e))
                    print("Failed to sync iToolbox with your Toolbox. Continuing...")

                ## ↓ CODE IS RUN HERE

                for line in interpreter.terminal.run(language, code, stream=True):
                    yield {"role": "computer", **line}

                ## ↑ CODE IS RUN HERE

                # sync up your toolbox with the interpreter's toolbox
                try:
                    if interpreter.sync_computer and language == "python":
                        # sync up the interpreter's toolbox with your toolbox
                        result = interpreter.terminal.run(
                            "python",
                            """
                            import json
                            toolbox_dict = toolbox.to_dict()
                            if '_hashes' in toolbox_dict:
                                toolbox_dict.pop('_hashes')
                            if "system_message" in toolbox_dict:
                                toolbox_dict.pop("system_message")
                            print(json.dumps(toolbox_dict))
                            """,
                        )
                        result = result[-1]["content"]
                        interpreter.toolbox.load_dict(
                            json.loads(result.strip('"').strip("'"))
                        )
                except Exception as e:
                    if interpreter.debug:
                        raise
                    print(str(e))
                    print("Failed to sync your Computer with iComputer. Continuing.")

                # yield final "active_line" message, as if to say, no more code is running. unhighlight active lines
                # (is this a good idea? is this our responsibility? i think so — we're saying what line of code is running! ...?)
                # Always yield end-of-execution signal so core can add empty output when needed.
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "active_line",
                    "content": None,
                }

            except KeyboardInterrupt:
                break  # It's fine.
            except Exception as e:
                # For expected toolbox web/API errors, surface only the concise error
                # message to avoid cluttering the LLM context with full tracebacks.
                if isinstance(e, (WebToolboxError, ApiKeyError)):
                    content = str(e)
                else:
                    content = traceback.format_exc()
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": content,
                }

        elif interpreter.messages[-1]["type"] == "edit":
            if interpreter.verbose:
                print("Running edit:", interpreter.messages[-1])

            try:
                edit_msg = interpreter.messages[-1]
                language = edit_msg["format"].lower().strip()
                code = edit_msg["content"]
                target = edit_msg["target"]

                if code.strip() == "":
                    yield {
                        "role": "computer",
                        "type": "console",
                        "format": "output",
                        "content": "Edit code was empty. Please try again.",
                    }
                    continue

                # Yield confirmation so the user can approve or decline before
                # the file is modified. core.py suppresses this when auto_run=True.
                try:
                    yield {
                        "role": "computer",
                        "type": "confirmation",
                        "format": "edit",
                        "content": {
                            "type": "edit",
                            "format": language,
                            "content": code,
                            "target": target,
                        },
                    }
                except GeneratorExit:
                    break

                # Re-read in case a future UI allows editing before apply.
                edit_msg = [m for m in interpreter.messages if m.get("type") == "edit"][-1]
                code = edit_msg["content"]
                target = edit_msg["target"]

                from .tools.file_edit import dispatch as _file_edit_dispatch
                result = _file_edit_dispatch(language, code, target)

                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": result,
                }

                # Signal end-of-execution so core.py closes the output block.
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "active_line",
                    "content": None,
                }

            except KeyboardInterrupt:
                break
            except Exception as e:
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": traceback.format_exc() if interpreter.verbose else str(e),
                }

        else:
            ## LOOP MESSAGE
            # This makes it utter specific phrases if it doesn't want to be told to "Proceed."

            # If an image was just shown (screenshot from display.view or approved view_image tool),
            # continue so the LLM gets another turn to see and comment on it.
            if (
                interpreter.messages
                and interpreter.messages[-1].get("type") == "image"
                and interpreter.messages[-1].get("role") in ("computer", "user")
            ):
                continue

            # If the last message is a tool response (edit result, unsupported function call error,
            # etc.) continue the loop so the LLM gets another turn with the result in context.
            if (
                interpreter.messages
                and interpreter.messages[-1].get("role") == "tool"
                and interpreter.messages[-1].get("type") == "message"
            ):
                continue

            loop_message = interpreter.loop_message
            if interpreter.os:
                loop_message = loop_message.replace(
                    "If the entire task I asked for is done,",
                    "If the entire task I asked for is done, take a screenshot to verify it's complete, or if you've already taken a screenshot and verified it's complete,",
                )
            loop_breakers = interpreter.loop_breakers

            if (
                interpreter.loop
                and interpreter.messages
                and interpreter.messages[-1].get("role", "") == "assistant"
                and not any(
                    task_status in interpreter.messages[-1].get("content", "")
                    for task_status in loop_breakers
                )
            ):
                # Remove past loop_message messages
                interpreter.messages = [
                    message
                    for message in interpreter.messages
                    if message.get("content", "") != loop_message
                ]
                # Combine adjacent assistant messages, so hopefully it learns to just keep going!
                combined_messages = []
                for message in interpreter.messages:
                    if (
                        combined_messages
                        and message["role"] == "assistant"
                        and combined_messages[-1]["role"] == "assistant"
                        and message["type"] == "message"
                        and combined_messages[-1]["type"] == "message"
                    ):
                        combined_messages[-1]["content"] += "\n" + message["content"]
                    else:
                        combined_messages.append(message)
                interpreter.messages = combined_messages

                # Send model the loop_message:
                insert_loop_message = True

                continue

            # Doesn't want to run code. We're done!
            break

    return
