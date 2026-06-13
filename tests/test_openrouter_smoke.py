import os

import pytest

from interpreter import OpenInterpreter

pytestmark = pytest.mark.openrouter

CI_MODEL = os.environ.get("OI_CI_MODEL", "openrouter/openrouter/free")


def test_openrouter_simple_chat():
    interpreter = OpenInterpreter()
    interpreter.llm.model = CI_MODEL
    interpreter.llm.supports_functions = False
    interpreter.llm._is_loaded = False

    messages = interpreter.chat(
        "Reply with exactly the word pong and nothing else. Do not run code."
    )

    assert "pong" in messages[-1]["content"].lower()
