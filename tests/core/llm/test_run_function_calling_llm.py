from types import SimpleNamespace

from interpreter.core.llm.run_function_calling_llm import run_function_calling_llm


class FakeLanguage:
    name = "Python"


def _make_llm(chunks, verbose=False):
    def completions(**params):
        for chunk in chunks:
            yield chunk

    terminal = SimpleNamespace(languages=[FakeLanguage()])
    computer = SimpleNamespace(terminal=terminal)
    return SimpleNamespace(
        completions=completions,
        interpreter=SimpleNamespace(computer=computer, verbose=verbose),
    )


def _chunk(delta):
    """Wrap a delta dict in the OpenAI streaming chunk shape the generator consumes."""
    return {"choices": [{"delta": delta}]}


def test_message_content_yielded():
    """Plain text content in streaming deltas is yielded as message-type chunks."""
    llm = _make_llm([{"choices": [{"delta": {"content": "Hi there"}}]}])
    result = list(run_function_calling_llm(llm, {"messages": []}))
    assert result == [{"type": "message", "content": "Hi there"}]


def test_streaming_function_call_yields_code():
    """A streamed execute function_call is reassembled across deltas into a single code chunk."""
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "function_call": {
                            "name": "execute",
                            "arguments": '{"language": "python", "code": "print(',
                        }
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"function_call": {"arguments": '1)"'}}}
            ]
        },
    ]
    llm = _make_llm(chunks)
    result = list(run_function_calling_llm(llm, {"messages": []}))
    code_chunks = [r for r in result if r["type"] == "code"]
    assert code_chunks
    assert code_chunks[0]["format"] == "python"
    assert "".join(c["content"] for c in code_chunks) == "print(1)"


def test_hallucinated_python_name_yields_code():
    """When the model names the function 'python' instead of 'execute', the arguments are still treated as code."""
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "function_call": {
                            "name": "python",
                            "arguments": "print(2)",
                        }
                    }
                }
            ]
        }
    ]
    llm = _make_llm(chunks)
    result = list(run_function_calling_llm(llm, {"messages": []}))
    assert result == [{"type": "code",
                       "format": "python",
                       "content": "print(2)"}]


def test_chunks_without_choices_are_skipped():
    """Chunks with no choices list (or an empty one) are ignored rather than failing."""
    llm = _make_llm([{"foo": "bar"}, {"choices": []}, _chunk({"content": "hi"})])
    assert list(run_function_calling_llm(llm, {"messages": []})) == [
        {"type": "message", "content": "hi"}
    ]


def test_review_layer_yields_safe_review_after_function_call():
    """Content following a function_call with a <safe> tag is streamed as a safe review."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "function_call": {
                        "name": "execute",
                        "arguments": '{"language": "python", "code": "print(1)"}',
                    }
                }
            ),
            _chunk({"content": "<safe>"}),
            _chunk({"content": "all good"}),
            _chunk({"content": "</safe>"}),
        ]
    )
    assert list(run_function_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "safe", "content": ""},
        {"type": "review", "format": "safe", "content": "all good"},
        {"type": "review", "format": "safe", "content": ""},
    ]


def test_review_layer_yields_unsafe_review_after_function_call():
    """Content following a function_call with an <unsafe> tag is streamed as an unsafe review."""
    llm = _make_llm(
        [
            _chunk(
                {
                    "function_call": {
                        "name": "execute",
                        "arguments": '{"language": "python", "code": "print(1)"}',
                    }
                }
            ),
            _chunk({"content": "<unsafe>DANGER"}),
        ]
    )
    assert list(run_function_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(1)"},
        {"type": "review", "format": "unsafe", "content": "DANGER"},
    ]


def test_unparseable_arguments_verbose_prints_warning(capsys):
    """In verbose mode, unparseable function arguments trigger the 'Arguments not a dict.' notice."""
    llm = _make_llm(
        [_chunk({"function_call": {"name": "execute", "arguments": "not json"}})],
        verbose=True,
    )
    assert list(run_function_calling_llm(llm, {"messages": []})) == []
    assert "Arguments not a dict." in capsys.readouterr().out


def test_hallucinated_python_name_verbose_prints_notice(capsys):
    """In verbose mode, a hallucinated 'python' function_call logs 'Got direct python call'."""
    llm = _make_llm(
        [_chunk({"function_call": {"name": "python", "arguments": "print(9)"}})],
        verbose=True,
    )
    assert list(run_function_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "print(9)"}
    ]
    assert "Got direct python call" in capsys.readouterr().out


def test_unknown_function_name_yields_name_and_stops():
    """An unrecognized function name is emitted as python code and the stream stops."""
    llm = _make_llm(
        [
            _chunk({"function_call": {"name": "whatever", "arguments": "x"}}),
            _chunk({"content": "after"}),
        ]
    )
    assert list(run_function_calling_llm(llm, {"messages": []})) == [
        {"type": "code", "format": "python", "content": "whatever"}
    ]
