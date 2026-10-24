import json
import os
import re
import time
import traceback

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import html2text
import litellm
from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from .render_message import render_message


def _error_body(error_str):
    """Render a provider's error text, unwrapping an HTML body if that is what it is.

    Returns a Rich-renderable. Gateways (OpenRouter, Cloudflare, nginx) often
    answer a failed request with an HTML error page rather than JSON, and the
    useful part is buried under <!DOCTYPE html>...
    """
    lower = error_str.lower()
    markers = [i for i in (lower.find("<!doctype"), lower.find("<html")) if i != -1]
    if not markers:
        return error_str

    try:
        converter = html2text.HTML2Text()
        converter.body_width = 0  # Let Rich handle reflowing
        start = min(markers)
        # The text before the markup names the exception class and provider,
        # which is the context this panel exists to show. Keep it.
        prefix = error_str[:start].strip()
        body = converter.handle(error_str[start:]).strip()
        text = f"{prefix}\n\n{body}".strip() if prefix else body
    except Exception:
        return error_str

    return text or error_str


def display_provider_error(error):
    """Show the provider's own error message, boxed, instead of paraphrasing it.

    The provider knows what went wrong and has usually already said it well.
    Substituting our own prose loses the detail that makes the error
    actionable (which parameter, which model, which account) and, when the
    guess is wrong, actively misleads.
    """
    error_str = str(error)
    if not error_str.strip():
        # Some providers raise with no message at all. Naming the type is more
        # use than an empty box.
        error_str = type(error).__name__

    rich_print(
        Panel(
            Markdown(_error_body(error_str)),
            border_style="red",
            title="Provider error",
            title_align="left",
        )
    )


def _provider_hint(error, interpreter):
    """Return a provider-agnostic hint for a recognized failure, or None.

    The panel carries the provider's own message; this only adds advice that is
    true across providers. It deliberately does not guess at a billing page or
    offer the removed hosted model.
    """
    message = str(error).lower()
    if (
        isinstance(error, litellm.exceptions.AuthenticationError)
        or "authenticationerror" in message
    ):
        hint = "Check the API key for this provider."
        if getattr(interpreter.llm, "api_key", None):
            hint += (
                " `llm.api_key` is set in your profile/config; if you pass a "
                "different `--model` or `--api_key` on the command line, they "
                "must match the same provider."
            )
        return hint
    if (
        isinstance(error, litellm.exceptions.RateLimitError)
        or any(
            marker in message
            for marker in ("rate limit", "rate-limit", "rate limited", "quota")
        )
        or re.search(r"\b429\b", message)
    ):
        return (
            "The provider rate-limited this request or your quota is exhausted; "
            "retry later or check the provider's billing."
        )
    if isinstance(error, litellm.exceptions.NotFoundError) or re.search(
        r"\bmodel\b.{0,80}\b(does not exist|not found|unknown model|no such model)\b",
        message,
    ):
        return "Check the model name."
    if "do not have access" in message or "does not have access" in message:
        return "Check the model name and that your account has access to it."
    return None


def display_provider_hint(error, interpreter):
    """Print a provider-agnostic hint after the provider's own error panel."""
    hint = _provider_hint(error, interpreter)
    if hint:
        interpreter.display_message(f"> **Hint:** {hint}")


def respond(interpreter):
    """
    Yields chunks.
    Responds until it decides not to run any more code or say anything else.
    """

    last_unsupported_code = ""
    insert_loop_message = False

    while True:
        ## RENDER SYSTEM MESSAGE ##

        system_message = interpreter.system_message

        # Add language-specific system messages
        for language in interpreter.computer.terminal.languages:
            if hasattr(language, "system_message"):
                system_message += "\n\n" + language.system_message

        # Add custom instructions
        if interpreter.custom_instructions:
            system_message += "\n\n" + interpreter.custom_instructions

        # Add computer API system message
        if interpreter.computer.import_computer_api:
            if interpreter.computer.system_message not in system_message:
                system_message = (
                    system_message + "\n\n" + interpreter.computer.system_message
                )

        # Storing the messages so they're accessible in the interpreter's computer
        # no... this is a huge time sink.....
        # if interpreter.sync_computer:
        #     output = interpreter.computer.run(
        #         "python", f"messages={interpreter.messages}"
        #     )

        ## Rendering ↓
        rendered_system_message = render_message(interpreter, system_message)
        ## Rendering ↑

        rendered_system_message = {
            "role": "system",
            "type": "message",
            "content": rendered_system_message,
        }

        # Create the version of messages that we'll send to the LLM
        messages_for_llm = interpreter.messages.copy()
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

        if (
            interpreter.messages[-1]["type"] != "code"
        ):  # If it is, we should run the code (we do below)
            try:
                for chunk in interpreter.llm.run(messages_for_llm):
                    yield {"role": "assistant", **chunk}

            except litellm.exceptions.BudgetExceededError:
                interpreter.display_message(
                    f"""> Max budget exceeded

                    **Session spend:** ${litellm._current_cost}
                    **Max budget:** ${interpreter.llm.max_budget}

                    Press CTRL-C then run `interpreter --max_budget [higher USD amount]` to proceed.
                """
                )
                break

            except Exception as e:
                display_provider_error(e)
                display_provider_hint(e, interpreter)
                try:
                    e._oi_provider_error_displayed = True
                except Exception:
                    pass
                raise

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
                if interpreter.computer.terminal.get_language(language) is None:
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
                    # Without breaking, the loop re-enters this branch forever because
                    # messages[-1] is still type "code" with empty content.
                    break

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

                # don't let it import computer — we handle that!
                if interpreter.computer.import_computer_api and language == "python":
                    code = code.replace("import computer\n", "pass\n")
                    code = re.sub(
                        r"import computer\.(\w+) as (\w+)", r"\2 = computer.\1", code
                    )
                    code = re.sub(
                        r"from computer import (.+)",
                        lambda m: "\n".join(
                            f"{x.strip()} = computer.{x.strip()}"
                            for x in m.group(1).split(", ")
                        ),
                        code,
                    )
                    code = re.sub(r"import computer\.\w+\n", "pass\n", code)
                    # If it does this it sees the screenshot twice (which is expected jupyter behavior)
                    if any(
                        code.strip().split("\n")[-1].startswith(text)
                        for text in [
                            "computer.display.view",
                            "computer.display.screenshot",
                            "computer.view",
                            "computer.screenshot",
                        ]
                    ):
                        code = code + "\npass"

                # sync up some things (is this how we want to do this?)
                interpreter.computer.verbose = interpreter.verbose
                interpreter.computer.debug = interpreter.debug
                interpreter.computer.emit_images = interpreter.llm.supports_vision
                interpreter.computer.max_output = interpreter.max_output

                # sync up the interpreter's computer with your computer
                try:
                    if interpreter.sync_computer and language == "python":
                        computer_dict = interpreter.computer.to_dict()
                        if "_hashes" in computer_dict:
                            computer_dict.pop("_hashes")
                        if "system_message" in computer_dict:
                            computer_dict.pop("system_message")
                        computer_json = json.dumps(computer_dict)
                        sync_code = f"""import json\ncomputer.load_dict(json.loads('''{computer_json}'''))"""
                        interpreter.computer.run("python", sync_code)
                except Exception as e:
                    if interpreter.debug:
                        raise
                    print(str(e))
                    print("Failed to sync iComputer with your Computer. Continuing...")

                ## ↓ CODE IS RUN HERE

                for line in interpreter.computer.run(language, code, stream=True):
                    yield {"role": "computer", **line}

                ## ↑ CODE IS RUN HERE

                # sync up your computer with the interpreter's computer
                try:
                    if interpreter.sync_computer and language == "python":
                        # sync up the interpreter's computer with your computer
                        result = interpreter.computer.run(
                            "python",
                            """
                            import json
                            computer_dict = computer.to_dict()
                            if '_hashes' in computer_dict:
                                computer_dict.pop('_hashes')
                            if "system_message" in computer_dict:
                                computer_dict.pop("system_message")
                            print(json.dumps(computer_dict))
                            """,
                        )
                        result = result[-1]["content"]
                        interpreter.computer.load_dict(
                            json.loads(result.strip('"').strip("'"))
                        )
                except Exception as e:
                    if interpreter.debug:
                        raise
                    print(str(e))
                    print("Failed to sync your Computer with iComputer. Continuing.")

                # yield final "active_line" message, as if to say, no more code is running. unhighlight active lines
                # (is this a good idea? is this our responsibility? i think so — we're saying what line of code is running! ...?)
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "active_line",
                    "content": None,
                }

            except KeyboardInterrupt:
                break  # It's fine.
            except:
                yield {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": traceback.format_exc(),
                }

        else:
            ## LOOP MESSAGE
            # This makes it utter specific phrases if it doesn't want to be told to "Proceed."

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
