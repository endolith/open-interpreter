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
def test_merge_string_delta_onto_none_value():
    """A string delta whose existing key holds None starts concatenation from empty, not from a fallback literal."""
    original = {"content": None}
    delta = {"content": "hi"}
    merge_deltas(original, delta)
    assert original["content"] == "hi"


def test_merge_empty_string_delta_preserves_content():
    """An empty-string delta concatenates as an empty suffix, leaving existing content unchanged."""
    original = {"content": "x"}
    delta = {"content": ""}
    merge_deltas(original, delta)
    assert original["content"] == "x"


def test_merge_new_dict_key_copies_value():
    """A dict-valued key absent from the original is deep-copied into it, not replaced with None."""
    original = {}
    delta = {"choices": {"delta": {"content": "x"}}}
    merge_deltas(original, delta)
    assert original["choices"] == {"delta": {"content": "x"}}
    delta["choices"]["delta"]["content"] = "mutated"
