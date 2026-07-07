from interpreter.core.computer.ai.ai import chunk_responses, split_into_chunks


def test_split_into_chunks_with_tiktoken():
    """split_into_chunks splits long text into overlapping token-sized windows."""
    import tiktoken

    llm = type("Llm", (), {"model": "gpt-4"})()  # tiktoken encoding name, not API model
    text = "word " * 100
    tokens = 20
    overlap = 5
    chunks = split_into_chunks(text, tokens=tokens, llm=llm, overlap=overlap)
    encoding = tiktoken.encoding_for_model(llm.model)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(len(encoding.encode(chunk)) <= tokens for chunk in chunks)
    joined = " ".join(chunks)
    assert joined[:20] == text[:20]
    assert joined[-20:] == text[-20:]


def test_split_into_chunks_fallback_without_tiktoken():
    """Invalid model name forces character-based fallback when tiktoken fails."""
    llm = type("Llm", (), {"model": "totally-invalid-model-name-xyz"})()
    text = "abcdefghij" * 50
    chunks = split_into_chunks(text, tokens=10, llm=llm, overlap=2)
    assert len(chunks) >= 2
    assert chunks[0].startswith("abcd")


def test_chunk_responses_respects_token_limit():
    """Multiple responses under the token budget merge into one list element.

    Both strings together are well under 100 tokens, so chunk_responses joins
    them with a blank line separator. Happy path — no splitting required.
    """
    llm = type("Llm", (), {"model": "gpt-4"})()
    responses = ["short", "another short response"]
    result = chunk_responses(responses, tokens=100, llm=llm)
    assert result == ["short\n\nanother short response"]


def test_chunk_responses_oversized_single_response():
    """A single response larger than the token budget is returned unsplit."""
    llm = type("Llm", (), {"model": "gpt-4"})()
    big = "x" * 5000
    result = chunk_responses([big], tokens=50, llm=llm)
    assert result == [big]


def test_split_into_chunks_empty_text_returns_empty_list():
    """Empty input text produces no chunks."""
    llm = type("Llm", (), {"model": "gpt-4"})()
    assert split_into_chunks("", tokens=20, llm=llm, overlap=5) == []


def test_chunk_responses_empty_list_returns_empty():
    """An empty responses list produces an empty result list."""
    llm = type("Llm", (), {"model": "gpt-4"})()
    assert chunk_responses([], tokens=100, llm=llm) == []


def test_split_into_chunks_overlap_greater_than_tokens_returns_empty():
    """When overlap exceeds tokens the tiktoken step is negative and yields no chunks.

    This documents current behavior; callers should keep overlap < tokens.
    """
    llm = type("Llm", (), {"model": "gpt-4"})()
    assert split_into_chunks("abcdefghij", tokens=5, llm=llm, overlap=6) == []
