from tests.support.mock_openai_server import pick_reply, stream_reply_chunks


def test_pick_reply_hello_world():
    """pick_reply returns a plain greeting for the integration hello-world prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Please reply with just the words Hello, World! and nothing else.",
            }
        ]
    }
    assert pick_reply(body) == "Hello, World!"


def test_pick_reply_write_file_returns_python():
    """pick_reply returns a Python code block for the write-to-file integration prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Write the word 'Washington' to a .txt file called file.txt.",
            }
        ]
    }
    reply = pick_reply(body)
    assert reply.startswith("```python")
    assert "file.txt" in reply
    assert "Washington" in reply


def test_pick_reply_read_file_returns_content():
    """pick_reply returns file contents for a read-file follow-up prompt."""
    body = {
        "messages": [
            {"role": "user", "content": "Read file.txt in the current directory."}
        ]
    }
    assert pick_reply(body) == "Washington"


def test_stream_reply_chunks_splits_fenced_code():
    """stream_reply_chunks emits multiple deltas for fenced code so run_text_llm parses it."""
    chunks = stream_reply_chunks("```python\nprint(1)\n```")
    assert chunks == ["```", "python\nprint(1)\n", "```"]
