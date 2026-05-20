# Human-readable labels for execute-tool metadata (one line per mode, not per language).
EXECUTION_MODE_LABELS = {
    "repl": "persistent REPL — variables/imports survive across code blocks",
    "per_block": "stateless — fresh process each block, no state persists",
    "display": "display only — renders to the user's UI, no code executed",
}


def format_execute_language_description(languages):
    """Build the execute tool's language parameter description from terminal language classes."""
    by_mode = {}
    for lang in languages:
        mode = getattr(lang, "execution_mode", "repl")
        by_mode.setdefault(mode, []).append(lang.name.lower())

    lines = ["The programming language to execute (use an enum value). Execution modes:"]
    for mode in ("repl", "per_block", "display"):
        if mode not in by_mode:
            continue
        label = EXECUTION_MODE_LABELS.get(mode, mode)
        names = ", ".join(sorted(by_mode[mode]))
        lines.append(f"  - {label}: {names}")
    return "\n".join(lines)


class BaseLanguage:
    """

    Attributes

    name = "baselanguage" # Name as it is seen by the LLM
    file_extension = "sh" # (OPTIONAL) File extension, used for safe_mode code scanning
    aliases = ["bash", "sh", "zsh"] # (OPTIONAL) Aliases that will also point to this language if the LLM runs them
    execution_mode = "repl" # (OPTIONAL) One of: "repl" (persistent session), "per_block" (fresh run each time), "display" (renders to UI, no execution)

    Methods

    run (Generator that yields a dictionary in LMC format)
    stop (Halts code execution, but does not terminate state)
    terminate (Terminates state)
    """

    execution_mode = "repl"

    def run(self, code):
        """
        Generator that yields a dictionary in LMC format:
        {"type": "console", "format": "output", "content": "a printed statement"}
        {"type": "console", "format": "active_line", "content": "1"}
        {"type": "image", "format": "base64", "content": "{base64}"}
        """
        return {"type": "console", "format": "output", "content": code}

    def stop(self):
        """
        Halts code execution, but does not terminate state.
        """
        pass

    def terminate(self):
        """
        Terminates state.
        """
        pass
