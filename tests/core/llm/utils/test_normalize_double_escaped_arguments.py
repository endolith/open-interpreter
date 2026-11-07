from interpreter.core.llm.utils.normalize_double_escaped_arguments import (
    normalize_double_escaped_arguments,
)


def test_double_escaped_key_and_value_are_decoded():
    """A key that arrives wrapped in quotes marks its pair as double-encoded.

    This is the shape from #566: OpenInference streams the key and value with
    an extra escape level, so after parsing the object holds `'"language"'`
    and every schema lookup misses, dropping the call.
    """
    arguments = {"code": "echo hi", '"language"': '"bash"'}

    assert normalize_double_escaped_arguments(arguments) == {
        "code": "echo hi",
        "language": "bash",
    }


def test_normal_arguments_are_returned_unchanged():
    """Well-formed arguments come back as the same object, untouched."""
    arguments = {"language": "python", "code": "print(1)"}

    assert normalize_double_escaped_arguments(arguments) is arguments


def test_every_double_escaped_key_is_decoded():
    """A wholly escaped object decodes key by key, its code value included."""
    arguments = {'"language"': '"python"', '"code"': '"print(1)"'}

    assert normalize_double_escaped_arguments(arguments) == {
        "language": "python",
        "code": "print(1)",
    }


def test_a_plain_key_keeps_its_value_verbatim():
    """Code is arbitrary text that may legitimately start and end with a quote."""
    arguments = {"language": "python", "code": '"print(1)"'}

    assert normalize_double_escaped_arguments(arguments) is arguments


def test_quoted_language_value_is_decoded_even_with_a_plain_key():
    """Values-only mangling still gets repaired for `language`, a bare token."""
    arguments = {"language": '"python"', "code": '"print(1)"'}
    repaired = normalize_double_escaped_arguments(arguments)

    assert repaired == {"language": "python", "code": '"print(1)"'}
    assert repaired is not arguments


def test_undecodable_values_survive_the_repair():
    """A value that merely looks like a string literal is not rewritten."""
    assert normalize_double_escaped_arguments({'"language"': "python"}) == {
        "language": "python"
    }
    assert normalize_double_escaped_arguments({'"language"': '"ba"sh"'}) == {
        "language": '"ba"sh"'
    }


def test_an_unescaped_copy_wins_over_the_mangled_key():
    """If both spellings arrived, the model's own key keeps its value."""
    arguments = {"language": "python", '"language"': '"bash"'}

    assert normalize_double_escaped_arguments(arguments) is arguments
    assert arguments["language"] == "python"


def test_non_dict_arguments_pass_through():
    """A string or None from a partial parse is handed back untouched."""
    assert normalize_double_escaped_arguments(None) is None
    assert normalize_double_escaped_arguments('{"language"') == '{"language"'
