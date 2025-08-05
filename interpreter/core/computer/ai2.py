import os
from enum import Enum
from typing import Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field, create_model


class Ai2:
    """
    Lightweight helper for delegating small, *stateless* AI tasks to a hosted
    LLM (OpenAI, OpenRouter, etc.) using OpenAI API.

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

    ----------------------------------------------------------------------
    Public interface
    ----------------------------------------------------------------------
    Attributes
    ----------
    available_models : list[str]
        Cached list of model IDs returned from ``client.models.list()`` at
        instantiation time.  Use this to inspect which hosted models your API
        key has access to.

    default_model : str
        The model ID used when a helper call does not explicitly provide a
        ``model=`` argument.  Defaults to ``"gpt-4.1-nano"`` (or the value of
        the ``AI2_MODEL`` environment variable).

    Methods
    -------
    single_response(instruction, content, \*, model=None, temperature=0.0)
        Send a single user message under a custom system prompt and return the
        raw text output.

    boolean_query(instruction, content, \*, model=None, temperature=0.0)
        Return a strict boolean result using OpenAI Structured Outputs. Helpful
        for yes/no validations where free-form text would be hard to parse.

    choice_query(instruction, content, choices, \*, model=None, temperature=0.0)
        Force the model to pick exactly one item from ``choices`` and return it
        as a string.
    """

    def __init__(self, computer=None, default_model: str = None,
                 temperature: float = 0.0):
        # The parent computer reference is optional – it lets callers access the
        # global interpreter settings if they want, but nothing in Ai2 depends
        # on it.
        self.computer = computer
        # Prefer newest model capable of structured outputs
        self.default_model = default_model or os.getenv("AI2_MODEL",
                                                        "gpt-4.1-nano")
        self.temperature = temperature

        # Re-use the same API key the main interpreter is using (or env var)
        self.api_key = None
        if computer and hasattr(computer.interpreter, "llm") and computer.interpreter.llm:
            self.api_key = getattr(computer.interpreter.llm, "api_key", None) or None
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=self.api_key)

        # ------------------------------------------------------------------
        # Fetch & cache model list
        # ------------------------------------------------------------------
        try:
            models_response = self.client.models.list()
            # Each item has an .id attribute – store as simple list[str]
            self.available_models: List[str] = [model.id for model in models_response.data]
        except Exception:
            # Swallow errors (network issues, permissions) – callers can still
            # pass any valid model ID even if pre-fetch failed.
            self.available_models = []

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
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.default_model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content.strip()

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

        class _BoolResp(BaseModel):
            thoughts: str = Field(..., description="Reasoning")
            value: bool = Field(..., description="Query result")

        response = self.client.responses.parse(
            model=kwargs.get("model", self.default_model),
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            text_format=_BoolResp,
            temperature=kwargs.get("temperature", self.temperature),
        )

        return bool(response.output_parsed.value)

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

        # Dynamically create an Enum for pydantic
        _ChoiceEnum = Enum("AnswerEnum", {c: c for c in choices})

        _ChoiceResp = create_model(
            "ChoiceResp",
            thoughts=(str, Field(..., description="Reasoning")),
            answer=(_ChoiceEnum, Field(..., description="Selected choice")),
            __base__=BaseModel,
        )

        response = self.client.responses.parse(
            model=kwargs.get("model", self.default_model),
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            text_format=_ChoiceResp,
            temperature=kwargs.get("temperature", self.temperature),
        )

        return str(response.output_parsed.answer.value)


# Convenience singleton
ai2 = Ai2()
