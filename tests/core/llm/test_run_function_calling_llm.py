from types import SimpleNamespace

from interpreter.core.llm.run_function_calling_llm import run_function_calling_llm


class FakeLanguage:
    name = "Python"


def _make_llm(chunks):
    def completions(**params):
        for chunk in chunks:
            yield chunk

    terminal = SimpleNamespace(languages=[FakeLanguage()])
    computer = SimpleNamespace(terminal=terminal)
    return SimpleNamespace(
        completions=completions,
        interpreter=SimpleNamespace(computer=computer, verbose=False),
    )


def test_message_content_yielded():
    llm = _make_llm([{"choices": [{"delta": {"content": "Hi there"}}]}])
    result = list(run_function_calling_llm(llm, {"messages": []}))
    assert result == [{"type": "message", "content": "Hi there"}]


def test_streaming_function_call_yields_code():
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


def test_hallucinated_python_name_yields_code():
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
    assert result == [{"type": "code", "format": "python", "content": "print(2)"}]
