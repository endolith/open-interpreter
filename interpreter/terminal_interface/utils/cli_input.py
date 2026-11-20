def _readline_backend(readline_module) -> str:
    """Identify which readline implementation was imported.

    Python 3.13+ sets ``readline.backend`` to ``"gnu"`` or ``"libedit"``.
    Earlier builds expose the backend only through the module docstring, so
    fall back to that and assume GNU Readline when neither is readable.
    """
    backend = getattr(readline_module, "backend", None)
    if backend in ("gnu", "libedit"):
        return backend

    docstring = getattr(readline_module, "__doc__", None) or ""
    if "libedit" in docstring:
        return "libedit"
    return "gnu"


def cli_input(prompt: str = "") -> str:
    start_marker = '"""'
    end_marker = '"""'
    libedit_readline = None
    try:
        # Pasted tabs are otherwise eaten by readline's Tab-completion binding
        # (notably when pasting spreadsheet TSV data). GNU Readline's bracketed
        # paste makes the terminal deliver pastes literally, preserving tabs.
        import readline

        if _readline_backend(readline) == "libedit":
            # libedit (often the macOS backend) ignores that GNU setting and
            # keeps Tab bound to completion, which consumes pasted tabs. Bind
            # Tab to literal insertion for the duration of this read instead
            # (#535); completion is restored below so consecutive prompts
            # keep behaving normally.
            readline.parse_and_bind("bind ^I ed-insert")
            libedit_readline = readline
        else:
            readline.parse_and_bind("set enable-bracketed-paste on")
    except ImportError:
        # No readline on this platform (e.g. Windows) — nothing to configure.
        pass

    try:
        message = input(prompt)

        # Multi-line input mode
        if start_marker in message:
            # The whole quoted block may already be here: with bracketed paste
            # the terminal delivers a multi-line paste (embedded newlines) in a
            # single input() call, and a one-line '"""..."""' was previously left
            # hanging forever waiting for another closing line.
            if message.count(start_marker) >= 2:
                return message
            lines = [message]
            while True:
                line = input()
                lines.append(line)
                if end_marker in line:
                    break
            return "\n".join(lines)

        # Single-line input mode
        return message
    finally:
        if libedit_readline is not None:
            # Runs on Ctrl-C as well, so an interrupted prompt does not leave
            # the user's shell with Tab bound to literal insertion.
            libedit_readline.parse_and_bind("bind ^I rl_complete")
