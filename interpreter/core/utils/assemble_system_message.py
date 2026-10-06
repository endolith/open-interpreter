from ..render_message import render_message


def _environment_block(interpreter):
    """One-off environment facts for the system message.

    Shape follows the Codex runtime's system prompt, which reports the same
    things in an ``<env>`` block:

        Here is useful information about the environment you are running in:
        <env>
          Open Interpreter started in: /path
          Open Interpreter started at: 2026-10-05 19:57
        </env>

    Both values are read from attributes captured at startup, never recomputed,
    so this block is byte-identical on every turn and sits unchanged in the
    cached prefix. Reporting the live cwd here instead would churn the cache on
    every ``cd`` and could disagree with the per-command shell state line.

    Deliberately separate from that state line: this is where the session
    *started*, that is where the shell *is*. They agree until the first cd, and
    keeping both gives the model the anchor as well as the live position.

    Attributes are fetched defensively because the server, tests and embedders
    all build partial interpreters.
    """
    cwd = getattr(interpreter, "_launch_cwd", None)
    started = getattr(interpreter, "_launch_time", None)
    if cwd is None and started is None:
        return ""
    lines = []
    # Labelled as when Open Interpreter itself started, in the past tense, and
    # naming Open Interpreter rather than the conversation. Both matter:

    # - "Conversation started" would be false. These are captured in
    #   OpenInterpreter.__init__, which runs once per process, so every
    #   conversation in one session would report the same instant, and resuming
    #   a conversation opened days ago would claim it began today.
    # - It must not read as "now". Each message already carries its own
    #   timestamp, so a session that spans days has messages far from this one,
    #   and a current-sounding value here would contradict them.
    if cwd is not None:
        lines.append(f"  Open Interpreter started in: {cwd}")
    if started is not None:
        lines.append(
            f"  Open Interpreter started at: {started.strftime('%Y-%m-%d %H:%M')}"
        )
    if not lines:
        return ""
    return (
        "\n\nHere is useful information about the environment you are running in:\n"
        "<env>\n" + "\n".join(lines) + "\n</env>"
    )


def assemble_system_message(interpreter):
    """Build the rendered system prompt before tool/text-mode appendices (matches respond.py)."""
    system_message = interpreter.system_message

    for language in interpreter.terminal.languages:
        if hasattr(language, "system_message"):
            system_message += "\n\n" + language.system_message

    if interpreter.custom_instructions:
        system_message += (
            "\n\n## User's Custom Instructions\n\n" + interpreter.custom_instructions
        )

    server_request_system = getattr(interpreter, "_server_request_system", None)
    if server_request_system:
        system_message += "\n\n## Client system prompt\n\n" + server_request_system

    if interpreter.toolbox.import_toolbox_api:
        if interpreter.toolbox.system_message not in system_message:
            system_message = system_message + "\n\n" + interpreter.toolbox.system_message

    # Appended after render_message on purpose. render_message interpolates
    # {{...}} template variables, and a directory literally containing braces
    # would be substituted away -- a path of "/tmp/{{x}}/dir" reaches the model
    # as "/tmp/------------------". Rare, but silent and corrupting.
    return render_message(interpreter, system_message) + _environment_block(interpreter)
