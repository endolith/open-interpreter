"""Reassembling a streamed completion from its deltas.

merge_deltas is applied to every chunk of every response. Its failure mode is
silent corruption: a tool call whose arguments are concatenated in the wrong
order or across the wrong index produces JSON that either fails to parse or,
worse, parses into a different call than the model made.
"""

import pytest

from interpreter.core.llm.utils.merge_deltas import merge_deltas, normalize_delta_to_dict


class PydanticV2Like:
    """Stands in for a LiteLLM Delta object (some providers return these, not dicts)."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, exclude_unset=False):
        return dict(self._payload)


class PydanticV1Like:
    def __init__(self, payload):
        self._payload = payload

    def dict(self, exclude_unset=False):
        return dict(self._payload)


def test_string_fields_accumulate_across_chunks():
    """Text content is concatenated, not replaced.

    This is the whole point: each chunk carries a fragment. Replacing would
    leave only the last token of every reply.
    """
    message = {}
    for fragment in ["Hel", "lo ", "world"]:
        merge_deltas(message, {"content": fragment})
    assert message["content"] == "Hello world"


def test_a_null_field_does_not_wipe_what_was_accumulated():
    """Providers send explicit nulls for fields they are not updating.

    Treating null as a value would blank the content on the final chunk, which
    is exactly the chunk most providers send with everything null.
    """
    message = {"content": "so far"}
    merge_deltas(message, {"content": None, "role": None})
    assert message["content"] == "so far"


def test_a_null_accumulator_is_treated_as_empty():
    """Appending to a field whose current value is None yields the new text.

    Some providers open the stream with {"content": null}; without the `or ""`
    the next chunk would raise TypeError mid-response.
    """
    message = {"content": None}
    merge_deltas(message, {"content": "text"})
    assert message["content"] == "text"


def test_nested_dicts_merge_rather_than_replace():
    """A nested object is merged key by key, so partial updates accumulate.

    function_call arrives as {"name": ...} then {"arguments": ...}. Replacing
    would drop the name.
    """
    message = {}
    merge_deltas(message, {"function_call": {"name": "run"}})
    merge_deltas(message, {"function_call": {"arguments": '{"a":'}})
    merge_deltas(message, {"function_call": {"arguments": " 1}"}})
    assert message["function_call"] == {"name": "run", "arguments": '{"a": 1}'}


def test_pydantic_deltas_are_converted_before_merging():
    """LiteLLM returns Pydantic Delta objects for some models; both versions work.

    Without the conversion the merge iterates attributes of an object instead
    of a mapping and produces an empty message — the symptom is a model that
    streams nothing at all.
    """
    message = {}
    merge_deltas(message, PydanticV2Like({"content": "from v2"}))
    assert message["content"] == "from v2"

    message = {}
    merge_deltas(message, PydanticV1Like({"content": "from v1"}))
    assert message["content"] == "from v1"


def test_an_unconvertible_delta_becomes_an_empty_dict():
    """A delta that is neither a mapping nor a Pydantic model is dropped, not fatal.

    One odd chunk from a provider should not end the response.
    """
    assert normalize_delta_to_dict(object()) == {}


def test_nested_pydantic_values_are_merged_as_dicts():
    """A Pydantic value inside a dict delta is converted before being merged.

    Providers mix the two representations within one chunk.
    """
    message = {"function_call": {"name": "run"}}
    merge_deltas(message, {"function_call": PydanticV2Like({"arguments": "{}"})})
    assert message["function_call"] == {"name": "run", "arguments": "{}"}


# --- tool calls -------------------------------------------------------------


def test_tool_call_arguments_accumulate_within_one_index():
    """Argument fragments for the same tool call are concatenated in order.

    Providers stream tool arguments a few characters at a time. Any other
    combination yields JSON the dispatcher cannot parse, and the tool call is
    lost.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run", "arguments": ""}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": '{"code"'}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": ': "ls"}'}}]})

    call = message["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "run"
    assert call["function"]["arguments"] == '{"code": "ls"}'


def test_parallel_tool_calls_stay_separate():
    """Two indices are two calls; their arguments never bleed into each other.

    Parallel tool calls interleave in the stream. Appending by position
    instead of by index would concatenate one call's arguments onto another's
    and produce a single malformed call.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "first", "arguments": "{"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "second", "arguments": "["}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 1, "function": {"arguments": "]"}}]})

    by_index = {call["index"]: call for call in message["tool_calls"]}
    assert by_index[0]["function"] == {"name": "first", "arguments": "{}"}
    assert by_index[1]["function"] == {"name": "second", "arguments": "[]"}


def test_a_tool_call_id_is_set_once_and_not_appended():
    """The id arrives on the first fragment only and must not accumulate.

    Concatenating it would produce "call_1call_1", which no provider will
    accept back in the tool result message.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]})
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["type"] == "function"


def test_a_tool_call_name_is_kept_from_the_first_fragment():
    """The function name is not re-appended when a later chunk repeats it."""
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"name": "run"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"name": "run", "arguments": "{}"}}]})
    assert message["tool_calls"][0]["function"]["name"] == "run"


def test_a_missing_index_defaults_to_the_first_call():
    """A tool-call fragment with no index is treated as index 0.

    Some providers omit it for single calls. Creating a separate entry would
    split one call into two half-formed ones.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]})
    merge_deltas(message, {"tool_calls": [{"function": {"arguments": "}"}}]})
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_two_calls_stamped_with_the_same_index_do_not_concatenate():
    """A second call sharing the first one's index starts its own entry.

    litellm's Delta.__init__ numbers index-less tool calls from zero *per
    chunk*, so a provider that streams two finished calls in two chunks — its
    native Ollama path does exactly this — hands both of them index 0. Merging
    on index alone appended the second call's arguments onto the first,
    producing {"language":"python",...}{"language":"bash",...} in one arguments
    string. That is the malformed call the model was then told it had sent, and
    the reason it could not correct itself: it had not sent it. The differing id
    is what says these are two calls.
    """
    message = {}
    merge_deltas(
        message,
        {
            "tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "execute", "arguments": '{"language": "python"}'}}
            ]
        },
    )
    merge_deltas(
        message,
        {"tool_calls": [{"index": 0, "id": "b", "function": {"name": "execute", "arguments": '{"language": "bash"}'}}]},
    )

    assert len(message["tool_calls"]) == 2, message["tool_calls"]
    assert message["tool_calls"][0]["function"]["arguments"] == '{"language": "python"}'
    assert message["tool_calls"][1]["function"]["arguments"] == '{"language": "bash"}'
    # The renumber keeps index usable as a lookup key for any later index-only delta.
    assert message["tool_calls"][0]["index"] != message["tool_calls"][1]["index"]


def test_a_late_arriving_id_still_joins_the_call_it_belongs_to():
    """An id that turns up after the call opened is adopted, not read as a new call.

    Some providers send the function name in the opening delta and the id with
    the first arguments fragment. Treating "an id we have not seen" as proof of
    a second call would split every such call in half.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "function": {"name": "execute", "arguments": "{"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1", "function": {"arguments": "}"}}]})

    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_a_repeated_id_continues_its_own_call_whatever_the_index_says():
    """Fragments are matched by id first, so a re-stamped index cannot mis-route them.

    A provider that repeats the id on every fragment but lets litellm re-stamp
    the index (0 in each chunk) would otherwise feed the second call's tail into
    the first.
    """
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "execute", "arguments": "1"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "b", "function": {"name": "execute", "arguments": "2"}}]})
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "b", "function": {"arguments": "3"}}]})

    by_id = {call["id"]: call["function"]["arguments"] for call in message["tool_calls"]}
    assert by_id == {"a": "1", "b": "23"}


def test_non_tool_call_lists_are_extended():
    """Ordinary list fields accumulate by extension.

    Only tool_calls needs index-aware merging; anything else (e.g. reasoning
    fragments) is a plain sequence.
    """
    message = {}
    merge_deltas(message, {"parts": ["a"]})
    merge_deltas(message, {"parts": ["b", "c"]})
    assert message["parts"] == ["a", "b", "c"]


def test_the_original_dict_is_mutated_and_returned():
    """merge_deltas both mutates and returns; callers rely on the mutation.

    The streaming loop keeps one message dict and merges into it. Returning a
    copy would leave it empty.
    """
    message = {}
    result = merge_deltas(message, {"content": "x"})
    assert result is message


# --- cases main had before the port -----------------------------------------


def test_an_unconvertible_non_mapping_delta_is_stored_as_is():
    """A non-dict value that master's old code would have copied is stored verbatim.

    The develop implementation only copies mapping-like values; anything else
    (a scalar, a sentinel object) keeps its identity so it can still be read
    back out of the accumulated message.
    """
    sentinel = object()
    message = {}
    merge_deltas(message, {"weird": sentinel})
    assert message["weird"] is sentinel


def test_empty_string_delta_preserves_content():
    """An empty-string delta concatenates as an empty suffix."""
    message = {"content": "x"}
    merge_deltas(message, {"content": ""})
    assert message["content"] == "x"


def test_a_list_onto_a_non_list_replaces_it():
    """A list delta overwrites a stale non-list value instead of extending it."""
    message = {"reasoning_details": "stale"}
    merge_deltas(message, {"reasoning_details": [{"type": "text"}]})
    assert message["reasoning_details"] == [{"type": "text"}]


def test_reasoning_details_lists_accumulate():
    """OpenRouter/DeepSeek reasoning_details lists grow across deltas.

    This is the shape that crashed main: a list of dicts, which the old
    ``dict(value)`` raised ValueError on and aborted the turn.
    """
    message = {}
    merge_deltas(message, {"reasoning_details": [{"type": "reasoning"}]})
    merge_deltas(message, {"reasoning_details": [{"type": "text"}]})
    assert message["reasoning_details"] == [
        {"type": "reasoning"},
        {"type": "text"},
    ]


def test_a_nested_pydantic_v1_value_is_converted():
    """A nested value exposing only ``.dict()`` (Pydantic v1) is converted too."""
    message = {}
    merge_deltas(message, {"function_call": PydanticV1Like({"name": "run"})})
    assert message["function_call"] == {"name": "run"}


def test_a_nested_pydantic_value_on_a_new_key_is_stored():
    """A Pydantic value for a key not yet present initialises that key."""
    message = {}
    merge_deltas(message, {"tool_result": PydanticV2Like({"ok": True})})
    assert message["tool_result"] == {"ok": True}


def test_a_later_delta_can_supply_the_function_name_type_and_container():
    """A call opened with only an id gains function/name/type from a later fragment."""
    message = {}
    merge_deltas(message, {"tool_calls": [{"index": 0, "id": "call_1"}]})
    merge_deltas(
        message,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                }
            ]
        },
    )

    call = message["tool_calls"][0]
    assert call["function"] == {"name": "execute", "arguments": "{}"}
    assert call["type"] == "function"


def test_a_non_dict_entry_in_tool_calls_is_appended():
    """A non-dict tool_calls entry is appended rather than merged.

    Some providers put opaque objects in the list; they must not crash the
    merge or be silently dropped.
    """
    message = {"tool_calls": [{"index": 0}]}
    merge_deltas(message, {"tool_calls": ["opaque"]})
    assert message["tool_calls"] == [{"index": 0}, "opaque"]


def test_a_mapping_like_value_is_converted_and_merged():
    """A non-dict mapping (e.g. MappingProxyType) goes through the dict() fallback."""
    from types import MappingProxyType

    message = {}
    merge_deltas(message, {"nested": MappingProxyType({"a": 1})})
    assert message["nested"] == {"a": 1}

    merge_deltas(message, {"nested": MappingProxyType({"b": 2})})
    assert message["nested"] == {"a": 1, "b": 2}
