from interpreter.core.llm.utils.merge_deltas import merge_deltas


def test_merge_string_deltas_concatenates():
    """String values for the same key are concatenated in place on the original delta dict."""
    original = {"content": "hello"}
    delta = {"content": " world"}
    result = merge_deltas(original, delta)
    assert result is original
    assert original["content"] == "hello world"


def test_merge_adds_new_string_key():
    """A string key present only in the incoming delta is copied onto the original."""
    original = {}
    delta = {"content": "new"}
    merge_deltas(original, delta)
    assert original["content"] == "new"


def test_merge_skips_none_values():
    """None values in the delta are ignored so existing string content is not overwritten."""
    original = {"content": "keep"}
    merge_deltas(original, {"content": None, "role": "assistant"})
    assert original == {"content": "keep", "role": "assistant"}


def test_merge_nested_dicts():
    """Nested dict values (e.g. function_call.arguments) are merged recursively across deltas."""
    original = {"function_call": {"arguments": "{"}}
    delta = {"function_call": {"arguments": '"code": "x"}'}}
    merge_deltas(original, delta)
    assert original["function_call"]["arguments"] == '{"code": "x"}'


def test_merge_empty_original():
    """Merging into an empty original dict populates it with all keys from the delta."""
    original = {}
    delta = {"role": "assistant", "content": "hi"}
    merge_deltas(original, delta)
    assert original == {"role": "assistant", "content": "hi"}


def test_merge_string_delta_onto_falsy_value():
    """A string delta concatenating onto a falsy stored value starts from empty.

    An empty-string delta stores "", so a later content delta re-enters the
    concatenation branch with a falsy left side; the "or ''" guard keeps the
    literal out of the result. None cannot reach this branch — merge_deltas
    skips None deltas and no caller pre-seeds the accumulator — so "" is the
    reachable shape of the guard.
    """
    original = {}
    merge_deltas(original, {"content": ""})
    merge_deltas(original, {"content": "hi"})
    assert original["content"] == "hi"


def test_merge_empty_string_delta_preserves_content():
    """An empty-string delta concatenates as an empty suffix, leaving existing content unchanged."""
    original = {"content": "x"}
    delta = {"content": ""}
    merge_deltas(original, delta)
    assert original["content"] == "x"


def test_merge_new_dict_key_copies_value():
    """A dict-valued key absent from the original is initialized with a shallow copy of the value.

    dict(value) copies the outer dict only, so later edits to the merged
    outer level leave the delta untouched while nested dicts stay shared.
    """
    original = {}
    delta = {"choices": {"delta": {"content": "x"}}}
    merge_deltas(original, delta)
    assert original["choices"] == {"delta": {"content": "x"}}
    original["choices"]["added"] = True
    assert "added" not in delta["choices"]
