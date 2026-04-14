"""
This file handles conversations.
"""

import json
import os
import platform
import subprocess

import inquirer

from .render_past_conversation import render_past_conversation
from .utils.local_storage_path import get_storage_path


def conversation_navigator(interpreter):
    import time

    conversations_dir = get_storage_path("conversations")

    interpreter.display_message(
        f"""> Conversations are stored in "`{conversations_dir}`".

    Select a conversation to resume.
    """
    )

    # Check if conversations directory exists
    if not os.path.exists(conversations_dir):
        print(f"No conversations found in {conversations_dir}")
        return None

    # Get list of all JSON files in the directory and sort them by modification time, newest first
    json_files = sorted(
        [f for f in os.listdir(conversations_dir) if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(conversations_dir, x)),
        reverse=True,
    )

    # Make a dict that maps reformatted "First few words... (September 23rd)" -> "First_few_words__September_23rd.json" (original file name)
    readable_names_and_filenames = {}
    for filename in json_files:
        name = (
            filename.replace(".json", "")
            .replace(".JSON", "")
            .replace("__", "... (")
            .replace("_", " ")
            + ")"
        )
        readable_names_and_filenames[name] = filename

    # Add the option to open the folder or start a new conversation. These don't map to filenames, we'll catch them
    readable_names_and_filenames_list = list(readable_names_and_filenames.keys())
    readable_names_and_filenames_list = [
        "New Conversation →",
        "Open Folder →",
    ] + readable_names_and_filenames_list

    # Use inquirer to let the user select a file
    questions = [
        inquirer.List(
            "name",
            message="",
            choices=readable_names_and_filenames_list,
        ),
    ]
    answers = inquirer.prompt(questions)

    # User chose to exit
    if not answers:
        return

    # If the user selected to start a new conversation, do so
    if answers["name"] == "New Conversation →":
        interpreter.chat()
        return

    # If the user selected to open the folder, do so and return
    if answers["name"] == "Open Folder →":
        open_folder(conversations_dir)
        return

    selected_filename = readable_names_and_filenames[answers["name"]]

    # Open the selected file and load the JSON data
    with open(os.path.join(conversations_dir, selected_filename), "r") as f:
        messages = json.load(f)

    # Pass the data into render_past_conversation
    render_past_conversation(messages)

    # Set the interpreter's settings to the loaded messages
    interpreter.messages = messages

    # Drop any prior resume alerts from saved history so we never stack
    # duplicates across resume / undo / autosave cycles.
    interpreter.messages = [
        m
        for m in interpreter.messages
        if m.get("alert_kind") != "conversation_resumed"
    ]

    current_cwd = os.getcwd()

    alert_text = (
        "⚠️ **SYSTEM ALERT:** This conversation was just resumed from a saved state.\n\n"
        "**1. Python REPL Reset:** Your Python environment has been completely cleared. "
        "All previously imported modules, defined functions, and variables are GONE. "
        "You must redefine them if you need them.\n\n"
        f"**2. CWD Reset:** The Current Working Directory (CWD) has been reset to: `{current_cwd}`. "
        "You MUST `import os` and `os.chdir()` back to the directory you were actively working in "
        "before running any relative file operations."
    )

    # Mark this as terminal-injected UI context so command history features
    # can treat it differently from real user prompts.
    interpreter.messages.append(
        {
            "role": "user",
            "type": "message",
            "content": alert_text,
            "source": "terminal",
            "format": "system_alert",
            "alert_kind": "conversation_resumed",
        }
    )

    interpreter.display_message(f"---\n{alert_text}\n\n---")

    interpreter.conversation_filename = selected_filename

    # Start the chat
    interpreter.chat()


def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    else:
        # Assuming it's Linux
        subprocess.run(["xdg-open", path])
