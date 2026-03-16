"""
The terminal interface is just a view. Just handles the very top layer.
If you were to build a frontend this would be a way to do it.
"""

try:
    import readline
except ImportError:
    pass

import os
import platform
import random
import re
import subprocess
import tempfile
import time

from ..core.utils.prompt_choice import prompt_choice
from ..core.utils.scan_code import scan_code
from ..core.utils.system_debug_info import system_info
from ..core.utils.truncate_output import truncate_output
from .components.code_block import CodeBlock
from .components.message_block import MessageBlock
from .magic_commands import handle_magic_command
from .utils.check_for_package import check_for_package
from .utils.cli_input import cli_input
from .utils.display_output import display_output
from .utils.find_image_path import find_image_path

# Add examples to the readline history
examples = [
    "How many files are on my desktop?",
    "What time is it in Seattle?",
    "Make me a simple Pomodoro app.",
    "Open Chrome and go to YouTube.",
    "Can you set my system to light mode?",
]
random.shuffle(examples)
try:
    for example in examples:
        readline.add_history(example)
except:
    # If they don't have readline, that's fine
    pass


def terminal_interface(interpreter, message):
    # TEMP: Disable active-line instrumentation/highlighting globally while
    # investigating streaming markdown rendering issues.
    os.environ["INTERPRETER_ACTIVE_LINE_DETECTION"] = "false"

    # Auto run and offline (this.. this isn't right) don't display messages.
    # Probably worth abstracting this to something like "debug_cli" at some point.
    # If (len(interpreter.messages) == 1), they probably used the advanced "i {command}" entry, so no message should be displayed.
    if (
        not interpreter.auto_run
        and not interpreter.offline
        and not (len(interpreter.messages) == 1)
    ):
        interpreter_intro_message = [
            "**Open Interpreter** will require approval before running code."
        ]

        if interpreter.safe_mode == "ask" or interpreter.safe_mode == "auto":
            if not check_for_package("semgrep"):
                interpreter_intro_message.append(
                    f"**Safe Mode**: {interpreter.safe_mode}\n\n>Note: **Safe Mode** requires `semgrep` (`pip install semgrep`)"
                )
        else:
            interpreter_intro_message.append("Use `interpreter -y` to bypass this.")

        if (
            not interpreter.plain_text_display
        ):  # A proxy/heuristic for standard in mode, which isn't tracked (but prob should be)
            interpreter_intro_message.append("Press `CTRL-C` to exit.")

        interpreter.display_message("\n\n".join(interpreter_intro_message) + "\n")
        print()

    if message:
        interactive = False
    else:
        interactive = True

    active_block = None
    voice_subprocess = None

    while True:
        if interactive:
            if (
                len(interpreter.messages) == 1
                and interpreter.messages[-1]["role"] == "user"
                and interpreter.messages[-1]["type"] == "message"
            ):
                # They passed in a message already, probably via "i {command}"!
                message = interpreter.messages[-1]["content"]
                interpreter.messages = interpreter.messages[:-1]
            else:
                ### This is the primary input for Open Interpreter.
                try:
                    message = (
                        cli_input("> ").strip()
                        if interpreter.multi_line
                        else input("> ").strip()
                    )
                except (KeyboardInterrupt, EOFError):
                    # Treat Ctrl-D on an empty line the same as Ctrl-C by exiting gracefully
                    interpreter.display_message("\n\n`Exiting...`")
                    raise KeyboardInterrupt

            try:
                # This lets users hit the up arrow key for past messages
                readline.add_history(message)
            except:
                # If the user doesn't have readline (may be the case on windows), that's fine
                pass

        if isinstance(message, str):
            # This is for the terminal interface being used as a CLI — messages are strings.
            # This won't fire if they're in the python package, display=True, and they passed in an array of messages (for example).

            if message == "":
                # Ignore empty messages when user presses enter without typing anything
                continue

            if message.startswith("%") and interactive:
                handle_magic_command(interpreter, message)
                continue

            # Many users do this
            if message.strip() == "interpreter --local":
                print("Please exit this conversation, then run `interpreter --local`.")
                continue
            if message.strip() == "pip install --upgrade open-interpreter":
                print(
                    "Please exit this conversation, then run `pip install --upgrade open-interpreter`."
                )
                continue

            if (
                interpreter.llm.supports_vision
                or interpreter.llm.vision_renderer != None
            ):
                # Is the input a path to an image? Like they just dragged it into the terminal?
                image_paths = find_image_path(message)

                ## If we found images, ask for approval before uploading
                if image_paths:
                    # Ask for approval before uploading images
                    if len(image_paths) == 1:
                        prompt = f"  Found image: {image_paths[0]}\n  Upload this image to the vision model? (y/n)\n\n  "
                    else:
                        paths_list = "\n".join(f"  - {path}" for path in image_paths)
                        prompt = f"  Found {len(image_paths)} images:\n{paths_list}\n  Upload these images to the vision model? (y/n)\n\n  "

                    response = prompt_choice(prompt, ("y", "n"))

                    if response == "y":
                        # Add the text message to interpreter's message history
                        interpreter.messages.append(
                            {
                                "role": "user",
                                "type": "message",
                                "content": message,
                                "sent_at": time.time(),
                            }
                        )

                        # Add remaining image messages (first one will be added by chat() when we pass it)
                        for image_path in image_paths[1:]:
                            interpreter.messages.append(
                                {
                                    "role": "user",
                                    "type": "image",
                                    "format": "path",
                                    "content": image_path,
                                    "sent_at": time.time(),
                                }
                            )

                        # Pass the first image dict to chat to trigger processing
                        message = {
                            "role": "user",
                            "type": "image",
                            "format": "path",
                            "content": image_paths[0],
                            "sent_at": time.time(),
                        }
                    else:
                        # User declined, just process the text message normally
                        pass

        try:
            for chunk in interpreter.chat(message, display=False, stream=True):
                yield chunk

                # Is this for thine eyes?
                if "recipient" in chunk and chunk["recipient"] != "user":
                    continue

                if interpreter.verbose:
                    print("Chunk in `terminal_interface`:", chunk)

                # Comply with PyAutoGUI fail-safe for OS mode
                # so people can turn it off by moving their mouse to a corner
                if interpreter.os:
                    if (
                        chunk.get("format") == "output"
                        and "failsafeexception" in chunk["content"].lower()
                    ):
                        print("Fail-safe triggered (mouse in one of the four corners).")
                        break

                if chunk["type"] == "review" and chunk.get("content"):
                    # Specialized models can emit a code review.
                    print(chunk.get("content"), end="", flush=True)

                # Execution notice
                if chunk["type"] == "confirmation":
                    if not interpreter.auto_run:
                        # OI is about to execute code. The user wants to approve this

                        # End the active code block so you can run input() below it
                        if active_block and not interpreter.plain_text_display:
                            active_block.refresh(cursor=False)
                            active_block.end()
                            active_block = None

                        code_to_run = chunk["content"]
                        language = code_to_run["format"]
                        code = code_to_run["content"]

                        should_scan_code = False

                        if not interpreter.safe_mode == "off":
                            if interpreter.safe_mode == "auto":
                                should_scan_code = True
                            elif interpreter.safe_mode == "ask":
                                response = prompt_choice(
                                    "  Would you like to scan this code? (y/n)\n\n  ",
                                    ("y", "n"),
                                )
                                if response == "y":
                                    should_scan_code = True

                        if should_scan_code:
                            scan_code(code, language, interpreter)

                        run_prompt = (
                            "Would you like to run this code? (y/n/e = edit)\n\n"
                            if interpreter.plain_text_display
                            else "  Would you like to run this code? (y/n/e = edit)\n\n  "
                        )
                        response = prompt_choice(run_prompt, ("y", "n", "e"))

                        if response == "y":
                            # Create a new, identical block where the code will actually be run
                            # Conveniently, the chunk includes everything we need to do this:
                            active_block = CodeBlock(interpreter)
                            active_block.margin_top = False  # <- Aesthetic choice
                            active_block.language = language
                            active_block.code = code
                        elif response == "e":
                            # Edit

                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(
                                suffix=".tmp", delete=False
                            ) as tf:
                                tf.write(code.encode())
                                tf.flush()

                            # Open the temporary file with the default editor
                            default_editor = "notepad" if platform.system() == "Windows" else "vim"
                            editor = os.environ.get("EDITOR", default_editor)
                            subprocess.call([editor, tf.name])

                            # Read the modified code
                            with open(tf.name, "r") as tf:
                                code = tf.read()

                            interpreter.messages[-1]["content"] = code  # Give it code

                            # Delete the temporary file
                            os.unlink(tf.name)
                            active_block = CodeBlock()
                            active_block.margin_top = False  # <- Aesthetic choice
                            active_block.language = language
                            active_block.code = code
                        else:
                            # User declined to run code. source="terminal"
                            # marks this as UI-injected so %undo skips it
                            # when finding the last real user message.
                            interpreter.messages.append(
                                {
                                    "role": "user",
                                    "type": "message",
                                    "content": "I have declined to run this code.",
                                    "sent_at": time.time(),
                                    "source": "terminal",
                                }
                            )
                            break

                # Plain text mode
                if interpreter.plain_text_display:
                    if chunk.get("replace"):
                        continue  # Replace chunk only updates display/storage; raw content was already printed
                    if "start" in chunk or "end" in chunk:
                        print("")
                    if chunk["type"] in ["code", "console"] and "format" in chunk:
                        if "start" in chunk:
                            print("```" + chunk["format"], flush=True)
                        if "end" in chunk:
                            print("```", flush=True)
                    if chunk.get("format") != "active_line":
                        print(chunk.get("content", ""), end="", flush=True)
                    continue

                # Handle special chunk to stop Live display before error panel
                if chunk.get("type") == "stop_live_display":
                    if active_block and hasattr(active_block, 'live') and active_block.live.is_started:
                        active_block.live.stop()
                    continue

                if "end" in chunk and active_block:
                    # For message blocks, finalize before ending (skip refresh to avoid duplication)
                    if chunk["type"] == "message" and hasattr(active_block, 'finalize'):
                        active_block.finalize()
                    else:
                        active_block.refresh(cursor=False)

                    if chunk["type"] in [
                        "message",
                        "console",
                    ]:  # We don't stop on code's end — code + console output are actually one block.
                        active_block.end()
                        active_block = None

                # Assistant message blocks
                if chunk["type"] == "message":
                    if "start" in chunk:
                        active_block = MessageBlock()
                        # Enable debug mode if environment variable is set
                        active_block.debug = os.environ.get("OI_DEBUG_MARKDOWN", "").lower() in ("1", "true", "yes")
                        if chunk.get("format") == "reasoning":
                            active_block.reasoning_mode = True
                        render_cursor = True

                    if "content" in chunk:
                        if active_block:
                            if chunk.get("replace") and hasattr(active_block, "replace_content"):
                                active_block.replace_content(chunk["content"])
                            elif not chunk.get("replace"):
                                active_block.add_content(chunk["content"])

                    if "end" in chunk and interpreter.os:
                            last_message = interpreter.messages[-1]["content"]

                            # Remove markdown lists and the line above markdown lists
                            lines = last_message.split("\n")
                            i = 0
                            while i < len(lines):
                                # Match markdown lists starting with hyphen, asterisk or number
                                if re.match(r"^\s*([-*]|\d+\.)\s", lines[i]):
                                    del lines[i]
                                    if i > 0:
                                        del lines[i - 1]
                                        i -= 1
                                else:
                                    i += 1
                            message = "\n".join(lines)
                            # Replace newlines with spaces, escape double quotes and backslashes
                            sanitized_message = (
                                message.replace("\\", "\\\\")
                                .replace("\n", " ")
                                .replace('"', '\\"')
                            )

                            # Display notification in OS mode
                            interpreter.toolbox.os.notify(sanitized_message)

                            # Speak message aloud
                            if platform.system() == "Darwin" and interpreter.speak_messages:
                                if voice_subprocess:
                                    voice_subprocess.terminate()
                                voice_subprocess = subprocess.Popen(
                                    [
                                        "osascript",
                                        "-e",
                                        f'say "{sanitized_message}" using "Fred"',
                                    ]
                                )
                            else:
                                pass
                                # User isn't on a Mac, so we can't do this. You should tell them something about that when they first set this up.
                                # Or use a universal TTS library.

                # Assistant code blocks
                elif chunk["role"] == "assistant" and chunk["type"] == "code":
                    if "start" in chunk:
                        active_block = CodeBlock()
                        active_block.language = chunk["format"]
                        render_cursor = True

                    if "content" in chunk:
                        active_block.code += chunk["content"]

                # Toolbox can display visual types to user,
                # Which sometimes creates more toolbox output (e.g. HTML errors, eventually)
                if (
                    chunk["role"] == "computer"
                    and "content" in chunk
                    and (
                        chunk["type"] == "image"
                        or ("format" in chunk and chunk["format"] == "html")
                        or ("format" in chunk and chunk["format"] == "javascript")
                    )
                ):
                    if (interpreter.os == True) and (interpreter.verbose == False):
                        # We don't display things to the user in OS control mode, since we use vision to communicate the screen to the LLM so much.
                        # But if verbose is true, we do display it!
                        continue

                    # Never display HTML/JS in browser, just show as plain text
                    if ("format" in chunk and (chunk["format"] == "html" or chunk["format"] == "javascript")):
                        print(chunk["content"])
                        continue

                    assistant_code_blocks = [
                        m
                        for m in interpreter.messages
                        if m.get("role") == "assistant" and m.get("type") == "code"
                    ]
                    if assistant_code_blocks:
                        code = assistant_code_blocks[-1].get("content")
                        if any(
                            text in code
                            for text in [
                                "toolbox.display.view",
                                "toolbox.display.screenshot",
                                "toolbox.view",
                                "toolbox.screenshot",
                            ]
                        ):
                            # If the last line of the code is a toolbox.view command, don't display it.
                            # The LLM is going to see it, the user doesn't need to.
                            continue

                    # Display and give extra output back to the LLM
                    extra_computer_output = display_output(chunk)

                    # We're going to just add it to the messages directly, not changing `recipient` here.
                    # Mind you, the way we're doing this, this would make it appear to the user if they look at their conversation history,
                    # because we're not adding "recipient: assistant" to this block. But this is a good simple solution IMO.
                    # we just might want to change it in the future, once we're sure that a bunch of adjacent type:console blocks will be rendered normally to text-only LLMs
                    # and that if we made a new block here with "recipient: assistant" it wouldn't add new console outputs to that block (thus hiding them from the user)

                    if (
                        interpreter.messages[-1].get("format") != "output"
                        or interpreter.messages[-1]["role"] != "computer"
                        or interpreter.messages[-1]["type"] != "console"
                    ):
                        # If the last message isn't a console output, make a new block
                        interpreter.messages.append(
                            {
                                "role": "computer",
                                "type": "console",
                                "format": "output",
                                "content": extra_computer_output,
                            }
                        )
                    else:
                        # If the last message is a console output, simply append the extra output to it
                        interpreter.messages[-1]["content"] += (
                            "\n" + extra_computer_output
                        )
                        interpreter.messages[-1]["content"] = interpreter.messages[-1][
                            "content"
                        ].strip()

                # Console
                if chunk["type"] == "console":
                    render_cursor = False
                    if "format" in chunk and chunk["format"] == "output":
                        active_block.output += "\n" + chunk["content"]
                        active_block.output = (
                            active_block.output.strip()
                        )  # ^ Aesthetic choice

                        # Truncate output
                        active_block.output = truncate_output(
                            active_block.output,
                            interpreter.max_output,
                            add_scrollbars=False,
                        )  # ^ Notice that this doesn't add the "scrollbars" line, which I think is fine
                    if "format" in chunk and chunk["format"] == "active_line":
                        active_block.active_line = chunk["content"]

                        # Display action notifications if we're in OS mode
                        if interpreter.os and active_block.active_line != None:
                            action = ""

                            code_lines = active_block.code.split("\n")
                            if active_block.active_line < len(code_lines):
                                action = code_lines[active_block.active_line].strip()

                            if action.startswith("toolbox") or action.startswith("computer"):
                                description = None

                                # Extract arguments from the action
                                start_index = action.find("(")
                                end_index = action.rfind(")")
                                if start_index != -1 and end_index != -1:
                                    # (If we found both)
                                    arguments = action[start_index + 1 : end_index]
                                else:
                                    arguments = None

                                # NOTE: Do not put the text you're clicking on screen
                                # (unless we figure out how to do this AFTER taking the screenshot)
                                # otherwise it will try to click this notification!

                                if any(
                                    action.startswith(text)
                                    for text in [
                                        "toolbox.screenshot",
                                        "toolbox.display.screenshot",
                                        "toolbox.display.view",
                                        "toolbox.view",
                                    ]
                                ):
                                    description = "Viewing screen..."
                                elif action == "toolbox.mouse.click()":
                                    description = "Clicking..."
                                elif action.startswith("toolbox.mouse.click("):
                                    if "icon=" in arguments:
                                        text_or_icon = "icon"
                                    else:
                                        text_or_icon = "text"
                                    description = f"Clicking {text_or_icon}..."
                                elif action.startswith("toolbox.mouse.move("):
                                    if "icon=" in arguments:
                                        text_or_icon = "icon"
                                    else:
                                        text_or_icon = "text"
                                    if (
                                        "click" in active_block.code
                                    ):  # This could be better
                                        description = f"Clicking {text_or_icon}..."
                                    else:
                                        description = f"Mousing over {text_or_icon}..."
                                elif action.startswith("toolbox.keyboard.write("):
                                    description = f"Typing {arguments}."
                                elif action.startswith("toolbox.keyboard.hotkey("):
                                    description = f"Pressing {arguments}."
                                elif action.startswith("toolbox.keyboard.press("):
                                    description = f"Pressing {arguments}."
                                elif action == "toolbox.os.get_selected_text()":
                                    description = f"Getting selected text."

                                if description:
                                    interpreter.toolbox.os.notify(description)

                    if "start" in chunk:
                        # We need to make a code block if we pushed out an HTML block first, which would have closed our code block.
                        if not isinstance(active_block, CodeBlock):
                            if active_block:
                                active_block.end()
                            active_block = CodeBlock()

                if active_block and not isinstance(active_block, MessageBlock):
                    # MessageBlock handles its own refresh internally
                    active_block.refresh(cursor=render_cursor)

            # (Sometimes -- like if they CTRL-C quickly -- active_block is still None here)
            if "active_block" in locals():
                if active_block:
                    active_block.end()
                    active_block = None
                    time.sleep(0.1)

            # Only exit when the user chose "n" at the API retry prompt (not when
            # they declined to run code). respond() sets _stopped_retrying in that case.
            if (
                interactive
                and getattr(interpreter, "_stopped_retrying", False)
            ):
                interpreter._stopped_retrying = False
                if interpreter.messages and interpreter.messages[-1].get("role") == "user":
                    interpreter.messages.pop()
                interpreter.display_message("\n\n`Stopped retrying. Exiting...`")
                raise SystemExit(1)

            if not interactive:
                # Don't loop
                break

        except KeyboardInterrupt:
            # Exit gracefully
            if "active_block" in locals() and active_block:
                active_block.end()
                active_block = None

            if interactive:
                # (this cancels LLM, returns to the interactive "> " input)
                continue
            else:
                break
        except Exception as e:
            # For errors, show debug info if enabled and re-raise
            if interpreter.debug:
                system_info(interpreter)
            raise
