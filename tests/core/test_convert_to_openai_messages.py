from interpreter.core.llm.utils.convert_to_openai_messages import convert_to_openai_messages


def test_computer_role_description_image_becomes_user():
    messages = [
        {
            "role": "computer",
            "type": "image",
            "format": "description",
            "content": "A screenshot of bold text.",
            "recipient": "assistant",
        }
    ]
    out = convert_to_openai_messages(
        messages, function_calling=True, vision=False, interpreter=None
    )
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "screenshot" in out[0]["content"]


def test_html_skips_png_when_not_vision():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    html = interpreter.terminal.get_language("html")
    html_lang = html(interpreter)
    chunks = list(html_lang.run("<b>edited</b>"))
    image_chunks = [c for c in chunks if c.get("type") == "image"]
    assert image_chunks == []
    assistant_chunks = [
        c
        for c in chunks
        if c.get("recipient") == "assistant" and c.get("type") == "console"
    ]
    assert any("```html" in c.get("content", "") for c in assistant_chunks)


def test_react_incompatible_code_reports_error_without_crashing():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    chunks = list(react_lang.run("import React from 'react'"))
    assert chunks
    assert chunks[0]["type"] == "console"
    assert "React format not supported" in chunks[0]["content"]
    assert "require" in chunks[0]["content"] or "import" in chunks[0]["content"]


def test_react_compatible_code_yields_html_for_user():
    from interpreter import OpenInterpreter

    interpreter = OpenInterpreter()
    interpreter.llm.supports_vision = False
    react = interpreter.terminal.get_language("react")
    react_lang = react(interpreter)
    code = 'ReactDOM.render(<h1>Hello, edit!</h1>, document.getElementById("root"));'
    chunks = list(react_lang.run(code))
    user_html = [c for c in chunks if c.get("recipient") == "user" and c.get("format") == "html"]
    assert len(user_html) == 1
    assert "Hello, edit!" in user_html[0]["content"]
    assert "type=\"text/babel\"" in user_html[0]["content"]
