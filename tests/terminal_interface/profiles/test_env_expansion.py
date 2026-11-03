"""A profile setting may name an environment variable instead of holding a secret.

An API key written into a profile is a secret sitting in a config file. It
cannot then be copied between machines, checked into a dotfiles repo, or left
world-readable without leaking — and profiles are created world-readable, so
the leak is the default.
"""

from types import SimpleNamespace

import pytest

from interpreter.terminal_interface.profiles.env_expansion import expand_env
from interpreter.terminal_interface.profiles.profiles import apply_profile_to_object


def test_a_bare_variable_as_the_whole_value_is_substituted(monkeypatch):
    """`api_key: $OLLAMA_PASS` sends the variable's value, not its name."""
    monkeypatch.setenv("OLLAMA_PASS", "sk-from-the-environment")
    obj = SimpleNamespace(api_key=None)

    apply_profile_to_object(obj, {"api_key": "$OLLAMA_PASS"})

    assert obj.api_key == "sk-from-the-environment"


def test_braces_are_substituted_inside_a_longer_value(monkeypatch):
    """`${VAR}` is replaced wherever it appears, so hosts and paths compose."""
    monkeypatch.setenv("OLLAMA_HOST", "gpu.internal")
    obj = SimpleNamespace(api_base=None)

    apply_profile_to_object(obj, {"api_base": "http://${OLLAMA_HOST}:11434"})

    assert obj.api_base == "http://gpu.internal:11434"


def test_a_stray_dollar_in_prose_is_left_alone():
    """Only a whole-value $NAME or an explicit ${NAME} counts.

    Prose settings such as custom_instructions routinely contain a stray "$" —
    a price, an escaped $$ in SQL. Rewriting part of a user's system message
    because it mentioned one would be worse than requiring braces where
    substitution is actually wanted.
    """
    obj = SimpleNamespace(custom_instructions=None)
    text = "Quote prices like $5 and escape $$ in SQL"

    apply_profile_to_object(obj, {"custom_instructions": text})

    assert obj.custom_instructions == text


def test_command_substitution_is_not_evaluated(monkeypatch):
    """`$(...)` stays literal — a profile is data, never a shell script.

    Shelling out while reading a profile would make every profile a
    code-execution path for anything that loads one.
    """
    monkeypatch.setenv("OLLAMA_PASS", "sk-from-the-environment")
    obj = SimpleNamespace(api_key=None)

    apply_profile_to_object(obj, {"api_key": "$(echo $OLLAMA_PASS)"})

    assert obj.api_key == "$(echo $OLLAMA_PASS)"


def test_an_unset_variable_is_an_error_rather_than_a_literal(monkeypatch):
    """A missing variable stops startup and names itself.

    Passing "$OLLAMA_PASS" through as the key reaches the provider and comes
    back a 401, which reads like a wrong credential rather than a missing one
    and sends the user looking in the wrong place.
    """
    monkeypatch.delenv("OLLAMA_PASS", raising=False)
    obj = SimpleNamespace(api_key=None)

    with pytest.raises(ValueError, match="OLLAMA_PASS"):
        apply_profile_to_object(obj, {"api_key": "$OLLAMA_PASS"})


def test_nested_sections_are_expanded_too(monkeypatch):
    """The substitution reaches settings inside `llm:`, where api_key lives."""
    monkeypatch.setenv("OLLAMA_PASS", "sk-nested")
    obj = SimpleNamespace(llm=SimpleNamespace(api_key=None))

    apply_profile_to_object(obj, {"llm": {"api_key": "$OLLAMA_PASS"}})

    assert obj.llm.api_key == "sk-nested"


@pytest.mark.parametrize("value", [1.0, True, None, ["a"], {"k": "v"}])
def test_non_string_values_pass_through_unchanged(value):
    """Numbers, booleans and containers are returned as they were."""
    assert expand_env(value, "some_key") is value


def test_a_trailing_newline_keeps_the_value_literal(monkeypatch):
    """`$OLLAMA_PASS\\n` is not a whole-value reference and stays literal.

    `$` also matches just before a trailing newline, so `.match()` treated a
    YAML block scalar ending in "\\n" as a whole-value reference and silently
    dropped the newline. `fullmatch` requires the whole string to match, so the
    newline survives.
    """
    monkeypatch.setenv("OLLAMA_PASS", "sk-from-the-environment")
    obj = SimpleNamespace(api_key=None)

    apply_profile_to_object(obj, {"api_key": "$OLLAMA_PASS\n"})

    assert obj.api_key == "$OLLAMA_PASS\n"


def test_braces_in_prose_stay_literal_and_do_not_block_startup(monkeypatch):
    """`${...}` in a prose setting is text, not a reference.

    The whole-value rule already protected prose from a bare "$", but the
    braced form was substituted on every setting. Prose is exactly where
    `${...}` appears literally — a SQL identifier, a shell parameter
    expansion, a templating placeholder the user means to keep — so
    substituting there rewrites the user's own words.

    Worse, an unset variable raises rather than passing through, so a
    `${schema}` in a system message stopped the session starting at all. The
    assertion below is that one: an unset name in prose must be inert.
    """
    monkeypatch.delenv("schema", raising=False)
    obj = SimpleNamespace(custom_instructions=None, system_message=None)
    prose = "In SQL escape with $$ and name the table ${schema}.users"

    apply_profile_to_object(obj, {"custom_instructions": prose, "system_message": prose})

    assert obj.custom_instructions == prose
    assert obj.system_message == prose


def test_a_set_variable_in_prose_is_still_not_substituted(monkeypatch):
    """Prose is left alone whether or not the name happens to be exported.

    Keying the behaviour on whether the variable exists would make a profile's
    meaning depend on the machine's environment: the same system message would
    read differently on a box that happened to export `HOME` or `USER`.
    """
    monkeypatch.setenv("TABLE", "secrets")
    obj = SimpleNamespace(custom_instructions=None)
    prose = "Never read ${TABLE} without asking."

    apply_profile_to_object(obj, {"custom_instructions": prose})

    assert obj.custom_instructions == prose


@pytest.mark.parametrize("key", ["api_key", "api_base", "api_version", "ollama_token"])
def test_credential_and_endpoint_settings_still_take_braces(monkeypatch, key):
    """The settings the feature exists for keep inline substitution.

    `api_base: http://${HOST}:11434` is the documented use, and a key whose
    name ends in _key/_token/_secret/_password/_base/_url/_host is a credential
    or an address by convention, not prose.
    """
    monkeypatch.setenv("HOST", "gpu.internal")
    obj = SimpleNamespace(**{key: None})

    apply_profile_to_object(obj, {key: "http://${HOST}:11434"})

    assert getattr(obj, key) == "http://gpu.internal:11434"


def test_a_whole_value_reference_works_on_any_setting(monkeypatch):
    """A setting that is entirely `$NAME` is unambiguous, so it is read anywhere.

    There is no prose reading of a value that is nothing but one variable
    reference, so this form needs no allowlist — only the inline form does.
    """
    monkeypatch.setenv("SOME_PROSE", "be concise")
    obj = SimpleNamespace(custom_instructions=None)

    apply_profile_to_object(obj, {"custom_instructions": "$SOME_PROSE"})

    assert obj.custom_instructions == "be concise"
