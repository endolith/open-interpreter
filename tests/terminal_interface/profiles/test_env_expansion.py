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
