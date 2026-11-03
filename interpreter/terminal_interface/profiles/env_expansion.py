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

The braced form is read only on settings that name a credential or an endpoint.
The reasoning above applies to it just as much: prose is exactly where
``${...}`` turns up as literal text — a SQL identifier, a shell parameter
expansion, a templating placeholder the user means to keep. Substituting there
rewrites the user's words, and because an unset variable raises rather than
passing through, a ``${schema}`` in a system message would stop the session
starting at all. So the whole-value form (unambiguous: the entire setting is
one variable) works on any setting, and inline substitution is confined to the
settings that exist to hold a secret or a host.

``$(...)`` is deliberately NOT evaluated. A profile is data; anything that
reads one would become a code-execution path the moment it ran a shell.
"""

import os
import re

_WHOLE_VALUE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_BRACED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Settings whose value is a credential or a machine address, where `${...}`
# inside a longer string is wanted (`api_base: http://${HOST}:11434`). Matched
# on the leaf key name, which is what apply_profile_to_object passes: a nested
# `llm: {api_key: ...}` arrives here as "api_key".
_INLINE_SUBSTITUTION_KEYS = frozenset({"api_key", "api_base", "api_version"})
_INLINE_SUBSTITUTION_SUFFIXES = (
    "_key",
    "_token",
    "_secret",
    "_password",
    "_base",
    "_url",
    "_host",
)


def _allows_inline_substitution(key):
    """Whether `${...}` inside a longer value is read on this setting.

    Deliberately a short list rather than "everything except prose": a new
    prose setting added later would silently inherit the footgun, whereas a new
    credential setting that wants inline substitution is one entry here.
    """
    name = str(key).lower()
    return name in _INLINE_SUBSTITUTION_KEYS or name.endswith(_INLINE_SUBSTITUTION_SUFFIXES)


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

    That raise is also why inline `${...}` is confined to credential and
    endpoint settings: on a prose setting it would turn a literal `${schema}`
    in a system message into a failure to start.
    """
    if not isinstance(value, str):
        return value

    whole = _WHOLE_VALUE.fullmatch(value)
    if whole:
        return _resolve(whole.group(1), key)

    if not _allows_inline_substitution(key):
        return value

    return _BRACED.sub(lambda match: _resolve(match.group(1), key), value)
