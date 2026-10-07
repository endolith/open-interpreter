"""Regression tests for #404: unlisted output types must not vanish.

The display_data/execute_result mapping named five representations and
dropped everything else without a trace. Unknown types now leave an
"(unrendered ...)" console chunk so skipped output never looks like no
output; the named mappings are unchanged.
"""

from interpreter.core.computer.terminal.languages.jupyter_language import (
    _display_data_chunk,
)


def test_unknown_representation_leaves_a_record():
    """application/json output reports itself instead of disappearing."""
    chunk = _display_data_chunk({"application/json": '{"a": 1}'})

    assert chunk["type"] == "console"
    assert chunk["format"] == "output"
    assert "application/json" in chunk["content"]


def test_empty_data_dict_leaves_a_record():
    """An empty data dict still says something was skipped."""
    chunk = _display_data_chunk({})

    assert chunk["type"] == "console"
    assert "unrendered" in chunk["content"]


def test_named_representations_unchanged():
    """The five previously supported types map exactly as before."""
    assert _display_data_chunk({"image/png": "abc"}) == {
        "type": "image",
        "format": "base64.png",
        "content": "abc",
    }
    assert _display_data_chunk({"image/jpeg": "abc"})["format"] == "base64.jpeg"
    assert _display_data_chunk({"text/html": "<b>x</b>"}) == {
        "type": "code",
        "format": "html",
        "content": "<b>x</b>",
    }
    assert _display_data_chunk({"text/plain": "42"}) == {
        "type": "console",
        "format": "output",
        "content": "42",
    }
    assert _display_data_chunk({"application/javascript": "alert(1)"}) == {
        "type": "code",
        "format": "javascript",
        "content": "alert(1)",
    }
