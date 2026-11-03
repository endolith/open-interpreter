# Human-readable labels for execute-tool metadata (one line per mode, not per language).
EXECUTION_MODE_LABELS = {
    "repl": "persistent REPL — variables/imports survive across code blocks; start from the live session and send only the next missing operation",
    "per_block": "stateless — fresh process each block; send a complete, self-contained block",
    "display": "display only — renders to the user's UI, no code executed; send only complete display source",
}


def format_language_selector_description(languages):
    """Build the clean selector description for the execute tool's `language` field.

    This text sits next to the enum value the model is about to emit, so it must
    not contain code fragments, per-language recipes, or "send/append" wording;
    that guidance belongs in the tool description, away from the selector.
    """
    return (
        "The programming language to execute. Must be one of the enum values; "
        "this is the language name, never the command or code."
    )


def format_execute_language_description(languages):
    """Build the per-language execution notes for the execute tool's description.

    These notes used to live on `language`, where their code examples primed the
    model to put the program in that enum field. They are tool documentation,
    not a value description.
    """
    by_mode = {}
    for lang in languages:
        mode = getattr(lang, "execution_mode", "repl")
        by_mode.setdefault(mode, []).append(lang.name.lower())

    lines = ["Execution modes:"]
    for mode in ("repl", "per_block", "display"):
        if mode not in by_mode:
            continue
        label = EXECUTION_MODE_LABELS.get(mode, mode)
        names = ", ".join(sorted(by_mode[mode]))
        lines.append(f"  - {label} ({names})")
    hints = []
    for lang in languages:
        hint = getattr(lang, "execute_tool_hint", None)
        if hint:
            hints.append(f"  - {lang.name.lower()}: {hint}")
    if hints:
        lines.append("Language notes:")
        lines.extend(hints)
    return "\n".join(lines)


class BaseLanguage:
    """

    Attributes

    name = "baselanguage" # Name as it is seen by the LLM (must match execute tool enum exactly)
    file_extension = "sh" # (OPTIONAL) File extension, used for temporary files and previews
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

    def strip_boilerplate(self, code):
        """Return (stripped_code, notice_or_None) after removing redundant boilerplate.

        Persistent REPL languages override this to drop habitual redundancy the
        user doesn't want to see (re-imported modules, cd to the current dir).
        ``notice`` is a short human-readable line shown with the confirmation
        prompt; None means nothing was removed.
        """
        return code, None
