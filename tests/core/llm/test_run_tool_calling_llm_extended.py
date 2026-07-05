from types import SimpleNamespace

from interpreter.core.llm.run_tool_calling_llm import run_tool_calling_llm


class Lang:
    name = "Python"


def test_tool_calling_llm_sets_language_enum():
    """The execute tool's language enum lists every terminal language the computer supports."""
    captured = {}

    def completions(**params):
        captured.update(params)
        return iter([])

    llm = SimpleNamespace(
        completions=completions,
        interpreter=SimpleNamespace(
            computer=SimpleNamespace(
                terminal=SimpleNamespace(languages=[Lang()])
            ),
            verbose=False,
        ),
    )
    list(run_tool_calling_llm(llm, {"messages": []}))
    tool = captured["tools"][0]["function"]
    assert tool["name"] == "execute"
    enum = tool["parameters"]["properties"]["language"]["enum"]
    assert enum == ["python"]
