"""Letting a profile name an environment variable instead of holding a secret.

An API key written into a profile is a secret sitting in a config file, which
then cannot be copied between machines, checked into a dotfiles repo, or left
world-readable without leaking. Naming the variable instead keeps the value in
a root-only file, a keyring, or a systemd EnvironmentFile, and leaves the
profile safe to read.

Two forms are recognised, and nothing else is touched:

    api_key: $OLLAMA_PASS            the whole value is one variable
    api_base: http://${HOST}:11434   braces, anywhere in the value

A bare ``$NAME`` only counts when it is the entire value. Prose settings such
as ``custom_instructions`` routinely contain a stray "$" — a price, a shell
snippet, an escaped ``$$`` in SQL — and silently rewriting part of a user's
system message would be far worse than asking for braces where substitution is
actually wanted.

``$(...)`` is deliberately NOT evaluated. A profile is data; anything that
reads one would become a code-execution path the moment it ran a shell.
"""

import os
import re

_WHOLE_VALUE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_BRACED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve(name, key):
    """The variable's value, or an error naming what is missing and where."""
    if name not in os.environ:
        raise ValueError(
            f"Profile setting '{key}' refers to the environment variable "
            f"{name}, which is not set.\n"
            f"Export it before starting, or replace it in the profile with a "
            f"literal value."
        )
    return os.environ[name]


def expand_env(value, key):
    """Substitute environment variables named by a profile value.

    An unset variable raises rather than passing the literal text through. A
    key sent as the string "$OLLAMA_PASS" comes back from the provider as a
    401, which reads like a wrong credential and sends the user looking in the
    wrong place; naming the missing variable points straight at it.
    """
    if not isinstance(value, str):
        return value

    whole = _WHOLE_VALUE.match(value)
    if whole:
        return _resolve(whole.group(1), key)

    return _BRACED.sub(lambda match: _resolve(match.group(1), key), value)
