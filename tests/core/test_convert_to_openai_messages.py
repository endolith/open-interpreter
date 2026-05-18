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
