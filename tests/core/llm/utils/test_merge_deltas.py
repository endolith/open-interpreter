from interpreter.core.llm.utils.merge_deltas import merge_deltas


def test_merge_string_deltas_concatenates():
    original = {"content": "hello"}
    delta = {"content": " world"}
    result = merge_deltas(original, delta)
    assert result is original
    assert original["content"] == "hello world"


def test_merge_adds_new_string_key():
    original = {}
    delta = {"content": "new"}
    merge_deltas(original, delta)
    assert original["content"] == "new"


def test_merge_skips_none_values():
    original = {"content": "keep"}
    merge_deltas(original, {"content": None, "role": "assistant"})
    assert original == {"content": "keep", "role": "assistant"}


def test_merge_nested_dicts():
    original = {"function_call": {"arguments": "{"}}
    delta = {"function_call": {"arguments": '"code": "x"}'}}
    merge_deltas(original, delta)
    assert original["function_call"]["arguments"] == '{"code": "x"}'


def test_merge_empty_original():
    original = {}
    delta = {"role": "assistant", "content": "hi"}
    merge_deltas(original, delta)
    assert original == {"role": "assistant", "content": "hi"}
