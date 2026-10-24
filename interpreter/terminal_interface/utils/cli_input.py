def cli_input(prompt: str = "") -> str:
    start_marker = '"""'
    end_marker = '"""'
    try:
        # Pasted tabs are otherwise eaten by readline's Tab-completion binding
        # (notably when pasting spreadsheet TSV data). Bracketed paste makes
        # the terminal deliver pastes literally, preserving tabs. Terminals
        # without support simply ignore it and pastes behave as before.
        import readline

        readline.parse_and_bind("set enable-bracketed-paste on")
    except ImportError:
        # No readline on this platform (e.g. Windows) — nothing to configure.
        pass
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
