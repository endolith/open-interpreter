"""
This is an Open Interpreter profile. It configures Open Interpreter to run a
model from your OpenCode Go subscription.

Get a key from https://opencode.ai/docs/go/ and set the OPENCODE_GO_API_KEY
environment variable to it. This is NOT your OpenAI key -- it is a different
credential, and Open Interpreter will refuse to fall back to it.

Open Interpreter sends the x-opencode-session header Go requires, using a stable
per-conversation id so prompt caching and provider routing stay warm.

Go serves each model over exactly one endpoint. The chat-completions families
below work; MiniMax and Qwen (Anthropic messages) and Grok/GPT-5.6-Luna (OpenAI
responses) are not reachable from Open Interpreter and raise an error instead.
See https://opencode.ai/docs/go/ for the full endpoint table.
"""

from interpreter import interpreter

# Pick any Go model served over /chat/completions.
interpreter.llm.model = "opencode_go/deepseek-v4-flash"

# Go's own catalog defines the context window; these are only here as overrides
# if the model is unknown to LiteLLM.
interpreter.llm.context_window = 128000
interpreter.llm.max_tokens = 4096
