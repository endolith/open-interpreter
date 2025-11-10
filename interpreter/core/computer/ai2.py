import os
from enum import Enum
from typing import Dict, List

import litellm
from openai import OpenAI
from pydantic import BaseModel, Field, create_model


class Ai2:
    """
    Lightweight helper for delegating small, *stateless* AI tasks to a hosted
    LLM using a hybrid approach: LiteLLM for basic text generation, OpenAI for
    structured outputs.

    Unlike the existing `computer.ai` which preserves the full Open Interpreter
    conversation and supports chunk-level map-reduce workflows, **Ai2** is
    intentionally minimal:

    • No conversation state is kept between calls – each helper builds its own
      messages and sends *one* request to the model.
    • Always calls remote APIs – no attempt is made to start or manage local
      models.
    • Provides a handful of strongly-typed convenience helpers so your scripts
      can *delegate* cognitive subtasks without having to worry about prompt
      engineering or token limits.
    • Uses hybrid approach: LiteLLM for basic responses, OpenAI for structured outputs

    ----------------------------------------------------------------------
    Supported Models and Methods
    ----------------------------------------------------------------------
    • single_response(): Works with ANY LiteLLM compatible model including:
      - OpenAI models: gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.
      - OpenRouter models: openrouter/qwen/qwen3-4b:free, etc.
      - Anthropic models: claude-3-haiku-20240307, etc.
      - Google models: gemini-pro, etc.
      - And many more providers supported by LiteLLM

    • boolean_query() and choice_query(): Require OpenAI models with structured
      output support (gpt-4o, gpt-4o-mini, etc.) due to OpenRouter limitations

    ----------------------------------------------------------------------
    API Key Configuration
    ----------------------------------------------------------------------
    Ai2 automatically detects API keys from environment variables:
    • OPENAI_API_KEY - for OpenAI models (required for structured outputs)
    • OPENROUTER_API_KEY - for OpenRouter models (get free key at https://openrouter.ai/)
    • ANTHROPIC_API_KEY - for Anthropic models
    • GOOGLE_API_KEY - for Google models
    • COHERE_API_KEY - for Cohere models
    • HUGGINGFACE_API_KEY - for Hugging Face models
    • LITELLM_API_KEY - for general LiteLLM usage

    ----------------------------------------------------------------------
    Public interface
    ----------------------------------------------------------------------
    Attributes
    ----------
    available_models : list[str]
        Cached list of model IDs returned from LiteLLM model list at
        instantiation time.  Use this to inspect which hosted models your API
        key has access to.

    default_model : str
        The model ID used when a helper call does not explicitly provide a
        ``model=`` argument.  Defaults to ``"gpt-5-nano"`` (or the value of
        the ``AI2_MODEL`` environment variable).

    Methods
    -------
    single_response(instruction, content, *, model=None, temperature=0.0)
        Send a single user message under a custom system prompt and return the
        raw text output. Works with ANY LiteLLM compatible model.

    boolean_query(instruction, content, *, model=None, temperature=0.0)
        Return a strict boolean result using OpenAI structured outputs.
        Requires OpenAI models (gpt-4o, gpt-4o-mini, etc.).

    choice_query(instruction, content, choices, *, model=None, temperature=0.0)
        Force the model to pick exactly one item from ``choices`` using OpenAI
        structured outputs. Requires OpenAI models (gpt-4o, gpt-4o-mini, etc.).

    Examples
    --------
    >>> from interpreter.core.computer.ai2 import ai2
    >>>
    >>> # Use OpenRouter free model for basic text generation
    >>> response = ai2.single_response(
    ...     instruction="Summarize this text",
    ...     content="Long text here...",
    ...     model="openrouter/qwen/qwen3-4b:free"
    ... )
    >>>
    >>> # Use OpenAI model for structured outputs
    >>> is_valid = ai2.boolean_query(
    ...     instruction="Is this text about AI?",
    ...     content="Machine learning is fascinating",
    ...     model="gpt-4o-mini"  # Must use OpenAI model
    ... )
    """

    def __init__(self, computer=None, default_model: str = None,
                 temperature: float = 0.0):
        # The parent computer reference is optional – it lets callers access the
        # global interpreter settings if they want, but nothing in Ai2 depends
        # on it.
        self.computer = computer
        # Prefer newest model capable of structured outputs
        self._default_model = default_model or os.getenv("AI2_MODEL",
                                                         "gpt-5-nano")
        self.temperature = temperature

        # Re-use the same API key the main interpreter is using (or env var)
        self.openai_api_key = None
        if computer and hasattr(computer.interpreter, "llm") and computer.interpreter.llm:
            self.openai_api_key = getattr(computer.interpreter.llm, "api_key", None) or None
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")

        # Set up OpenAI client for structured outputs (lazy initialization)
        self._client = None

        # Store all available API keys for different providers
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
            "litellm": os.getenv("LITELLM_API_KEY")
        }

        # Don't set litellm.api_key globally - pass it per request instead

        # ------------------------------------------------------------------
        # Fetch & cache model list
        # ------------------------------------------------------------------
        try:
            # Get available models from LiteLLM
            models_response = litellm.model_list()
            # Extract model IDs from the response
            self._available_models: List[str] = [model.get("id", model.get("model_name", "")) for model in models_response if model.get("id") or model.get("model_name")]
        except Exception:
            # Swallow errors (network issues, permissions) – callers can still
            # pass any valid model ID even if pre-fetch failed.
            self._available_models = []

    def _get_temperature_for_model(self, model: str, requested_temperature: float) -> float:
        """Get the appropriate temperature for a given model.

        Reasoning models (GPT-5, O1, O3, etc.) only support temperature=1.0,
        so we override any other value. For other models, we use the requested temperature.
        """
        # TODO: Verify this
        if model and (model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")):
            return 1.0
        return requested_temperature

    def _get_api_key_for_model(self, model: str) -> str:
        """Get the appropriate API key for a given model.

        Parameters
        ----------
        model : str
            The model identifier

        Returns
        -------
        str
            The API key for the model's provider, or None if not available
        """
        if not model:
            return None

        # Determine provider based on model name
        if model.startswith("openrouter/"):
            return self.api_keys["openrouter"]
        elif model.startswith("claude-") or model.startswith("anthropic/"):
            return self.api_keys["anthropic"]
        elif model.startswith("gemini-") or model.startswith("google/"):
            return self.api_keys["google"]
        elif model.startswith("command") or model.startswith("cohere/"):
            return self.api_keys["cohere"]
        elif model.startswith("gpt-") or model.startswith("openai/"):
            return self.api_keys["openai"]
        elif model.startswith("huggingface/") or model.startswith("hf/"):
            return self.api_keys["huggingface"]
        else:
            # For unknown models, try to find any available API key
            for key in self.api_keys.values():
                if key:
                    return key
            return None

    def _get_max_tokens_for_model(self, model: str) -> int:
        """Get appropriate max_tokens for a given model to avoid context window issues.

        Parameters
        ----------
        model : str
            The model identifier

        Returns
        -------
        int
            Maximum tokens to request for this model
        """
        if not model:
            return 1000

        # Conservative defaults for different model types
        if "qwen" in model.lower():
            return 500  # Qwen models often have smaller context windows
        elif "llama" in model.lower():
            return 800
        elif "claude" in model.lower():
            return 1000
        elif "gpt" in model.lower():
            return 1000
        elif "gemini" in model.lower():
            return 1000
        else:
            return 500  # Conservative default for unknown models

    @property
    def client(self):
        """Lazily initialize OpenAI client to avoid SSL errors at import time."""
        if self._client is None and self.openai_api_key:
            try:
                self._client = OpenAI(api_key=self.openai_api_key)
            except (FileNotFoundError, OSError) as e:
                # Handle SSL certificate file issues gracefully
                # If SSL_CERT_FILE is set but file doesn't exist, try without it
                ssl_cert_file = os.environ.get("SSL_CERT_FILE")
                if ssl_cert_file and not os.path.exists(ssl_cert_file):
                    # Temporarily unset SSL_CERT_FILE and retry
                    ssl_cert_file_backup = os.environ.pop("SSL_CERT_FILE", None)
                    try:
                        self._client = OpenAI(api_key=self.openai_api_key)
                    except Exception:
                        # Restore the env var if retry also fails
                        if ssl_cert_file_backup:
                            os.environ["SSL_CERT_FILE"] = ssl_cert_file_backup
                        raise
                    # Don't restore if successful - the env var was pointing to a non-existent file
                else:
                    raise
        return self._client

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def single_response(self, instruction: str, content: str, **kwargs) -> str:
        """Send a single user message under a custom system prompt.

        Parameters
        ----------
        instruction
            High-level instruction for the model (becomes the *system* message).
        content
            The payload to evaluate (becomes the *user* message).

        Returns
        -------
        str
            The response from the model.

        Example
        -------
        >>> summary = ai2.single_response(
        ...     instruction="Summarize the following text in four words.",
        ...     user_message="Open Interpreter lets you run natural-language commands as code."
        ... )
        >>> print(summary)
        "Natural language to code."

        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
        model = kwargs.get("model", self.default_model)
        temperature = self._get_temperature_for_model(
            model, kwargs.get("temperature", self.temperature)
        )

        # Use LiteLLM for basic text generation (supports any compatible model)
        try:
            # Get the appropriate API key and max_tokens for this model
            api_key = self._get_api_key_for_model(model)
            max_tokens = self._get_max_tokens_for_model(model)

            # Pass API key directly to avoid global interference
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if api_key:
                completion_kwargs["api_key"] = api_key

            response = litellm.completion(**completion_kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            # If LiteLLM fails, fall back to OpenAI if available
            if self.client and model.startswith("gpt-"):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self._get_max_tokens_for_model(model),
                    )
                    return response.choices[0].message.content.strip()
                except Exception as openai_error:
                    raise RuntimeError(f"Both LiteLLM and OpenAI failed. LiteLLM error: {e}, OpenAI error: {openai_error}") from e
            else:
                raise RuntimeError(f"LiteLLM error with model '{model}': {e}") from e

    def boolean_query(self, instruction: str, content: str, **kwargs) -> bool:
        """Call an LLM to return a strict boolean.

        Notes
        -----
        This uses OpenAI Structured Outputs internally to coerce the response
        to a boolean.

        Parameters
        ----------
        instruction
            High-level instruction for the model (becomes the *system* message).
        content
            The payload to evaluate (becomes the *user* message).

        Returns
        -------
        bool
            The boolean result of the query.

        Example
        -------
        >>> is_about_ai = ai2.boolean_query(
        ...     instruction="Does the following text mention an animal?",
        ...     content="Open Interpreter lets you run natural-language commands as code."
        ... )
        >>> print(is_about_ai)
        False
        """

        # Check if OpenAI client is available for structured outputs
        if not self.client:
            raise RuntimeError("OpenAI API key required for boolean_query. Please set OPENAI_API_KEY environment variable.")

        class _BoolResp(BaseModel):
            thoughts: str = Field(..., description="Reasoning")
            value: bool = Field(..., description="Query result")

        model = kwargs.get("model", self.default_model)
        temperature = self._get_temperature_for_model(
            model, kwargs.get("temperature", self.temperature)
        )

        try:
            response = self.client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
                text_format=_BoolResp,
                temperature=temperature,
            )
            return bool(response.output_parsed.value)
        except Exception as e:
            raise RuntimeError(f"OpenAI structured output error with model '{model}': {e}") from e

    # ------------------------------------------------------------------
    # Multiple-choice helper
    # ------------------------------------------------------------------
    def choice_query(
        self,
        instruction: str,
        content: str,
        choices: List[str],
        **kwargs,
    ) -> str:
        """Return one item from *choices* using Structured Outputs.

        The model must pick exactly one of the supplied *choices*. Internal
        reasoning is captured but not returned.

        Parameters
        ----------
        instruction
            High-level instruction for the model (becomes the *system* message).
        content
            The payload to evaluate (becomes the *user* message).
        choices
            The list of choices to choose from.

        Returns
        -------
        str
            The selected choice from the provided list.

        Example
        -------
        >>> animal = ai2.choice_query(
        ...     instruction="Which type of animal is the following text most likely about?",
        ...     content="She pounced on the mouse and caught it.",
        ...     choices=["Dog", "Cat", "Bird", "Fish"]
        ... )
        >>> print(animal)
        'Cat'
        """

        # Check if OpenAI client is available for structured outputs
        if not self.client:
            raise RuntimeError("OpenAI API key required for choice_query. Please set OPENAI_API_KEY environment variable.")

        # Dynamically create an Enum for pydantic
        _ChoiceEnum = Enum("AnswerEnum", {c: c for c in choices})

        _ChoiceResp = create_model(
            "ChoiceResp",
            thoughts=(str, Field(..., description="Reasoning")),
            answer=(_ChoiceEnum, Field(..., description="Selected choice")),
            __base__=BaseModel,
        )

        model = kwargs.get("model", self.default_model)
        temperature = self._get_temperature_for_model(
            model, kwargs.get("temperature", self.temperature)
        )

        try:
            response = self.client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": content},
                ],
                text_format=_ChoiceResp,
                temperature=temperature,
            )
            return str(response.output_parsed.answer.value)
        except Exception as e:
            raise RuntimeError(f"OpenAI structured output error with model '{model}': {e}") from e

    # ------------------------------------------------------------------
    # Read-only properties exposed for tool discovery
    # ------------------------------------------------------------------
    @property
    def available_models(self) -> List[str]:
        """List[str]: Cached list of model IDs returned when Ai2 was instantiated."""
        return self._available_models

    @property
    def default_model(self) -> str:
        """str: The model used when no `model=` override is provided to a helper."""
        return self._default_model


# Convenience singleton
ai2 = Ai2()
