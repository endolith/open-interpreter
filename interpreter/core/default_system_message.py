import getpass
import os
import platform

CWD_LINE_PREFIX = "Current working directory: "


def current_working_directory():
    """Return the directory, or None when the process no longer has one."""
    try:
        return os.getcwd()
    except OSError:
        # The process can outlive its working directory; importing or answering
        # must not fail just because the path cannot be read.
        return None


def current_working_directory_line():
    """Return the directory line, or None when the directory is unreadable."""
    directory = current_working_directory()
    return CWD_LINE_PREFIX + directory if directory is not None else None


def build_default_system_message():
    cwd_line = current_working_directory_line()
    cwd_suffix = "\n" + cwd_line if cwd_line else ""
    return f"""
You are Open Interpreter, a world-class programmer that can complete any goal by executing code.
For advanced requests, start by writing a plan.
When you execute code, it will be executed **on the user's machine**. The user has given you **full and complete permission** to execute any code necessary to complete the task. Execute the code.
You can access the internet. Run **any code** to achieve the goal, and if at first you don't succeed, try again and again.
You can install new packages.
When a user refers to a filename, they're likely referring to an existing file in the directory you're currently executing code in.
Write messages to the user in Markdown.
In general, try to **make plans** with as few steps as possible. As for actually executing code to carry out that plan, for *stateful* languages (like python, javascript, shell, but NOT for html which starts from 0 every time) **it's critical not to try to do everything in one code block.** You should try something, print information about it, then continue from there in tiny, informed steps. You will never get it on the first try, and attempting it in one go will often lead to errors you cant see.
You are capable of **any** task.

User's Name: {getpass.getuser()}
User's OS: {platform.system()}{cwd_suffix}""".strip()


def __getattr__(name):
    """Keep the old module-level constant importable without freezing it.

    `default_system_message` used to be a string computed at import time. It is
    now built per session so the directory is current, but it is still a public
    module name and third parties import it. Returning a freshly built message
    keeps those imports working instead of failing on upgrade.
    """
    if name == "default_system_message":
        return build_default_system_message()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
