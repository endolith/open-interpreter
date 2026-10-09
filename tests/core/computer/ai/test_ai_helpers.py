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


def test_chunk_responses_flushes_the_open_chunk_when_the_budget_is_exceeded():
    """A response that overflows the budget starts a new chunk rather than merging.

    This is the path where a chunk is already open: the in-progress chunk is
    flushed and the overflowing response begins the next one. The happy-path and
    oversized-single tests never reach it, because they either never overflow or
    overflow with nothing open yet.
    """
    llm = type("Llm", (), {"model": "gpt-4"})()
    # 40 tokens total against a budget of 25: the first fits, the second cannot
    # join it, so there must be two chunks rather than one oversized or a merge.
    responses = ["word " * 20, "word " * 20]
    result = chunk_responses(responses, tokens=25, llm=llm)

    assert len(result) == 2
    assert result[0].rstrip() == ("word " * 20).rstrip()
    assert result[1].rstrip() == ("word " * 20).rstrip()


def test_chunk_responses_falls_back_to_characters_without_tiktoken():
    """An unknown model name switches to character-based chunking.

    chunk_responses has a second, entirely separate implementation behind the
    except branch, using a tokens*4 character budget and a blank-line separator.
    It was previously exercised only in split_into_chunks, so a regression here
    would not have been caught by the tiktoken tests.
    """
    llm = type("Llm", (), {"model": "totally-invalid-model-name-xyz"})()
    budget = 10 * 4  # the fallback measures characters, at tokens*4 per chunk
    # Each response is deliberately larger than the whole budget, so each is
    # appended standalone rather than merged — there is nothing to merge into.
    responses = ["a" * (budget * 3), "b" * (budget * 3)]
    result = chunk_responses(responses, tokens=10, llm=llm)

    assert result == responses

    # Responses that *do* fit are merged until the budget is reached, so the
    # fallback genuinely chunks rather than always emitting one chunk per input.
    small = ["a" * 10, "b" * 10, "c" * 10]
    merged = chunk_responses(small, tokens=10, llm=llm)
    assert len(merged) < len(small)
    assert all(len(chunk) <= budget for chunk in merged)


def test_chunk_responses_character_fallback_keeps_every_response():
    """The fallback preserves all input content rather than dropping responses.

    The character path is a rewrite of the token path, so "it ran without
    raising" is not enough — the concatenation must still contain both inputs.
    """
    llm = type("Llm", (), {"model": "totally-invalid-model-name-xyz"})()
    responses = ["alpha" * 50, "beta" * 50]
    result = chunk_responses(responses, tokens=10, llm=llm)

    joined = "".join(result)
    assert "alpha" in joined
    assert "beta" in joined


def test_chunk_responses_character_fallback_empty_list():
    """The fallback branch also returns an empty list for no input."""
    llm = type("Llm", (), {"model": "totally-invalid-model-name-xyz"})()
    assert chunk_responses([], tokens=10, llm=llm) == []
