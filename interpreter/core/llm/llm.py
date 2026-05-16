import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import sys

# Note: litellm in DEV mode will load .env files from the current directory
# and all parent directories. This can lead to unexpected API keys being loaded
# if there are .env files in parent folders.
import litellm

litellm.suppress_debug_info = True
litellm.REPEATED_STREAMING_CHUNK_LIMIT = 99999999

import json
import logging
import re
import uuid

import requests
import tokentrim as tt
from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from .run_text_llm import run_text_llm

# from .run_function_calling_llm import run_function_calling_llm
from .run_tool_calling_llm import run_tool_calling_llm
from .utils.cache_aware_trim import cache_aware_trim
from .utils.convert_to_openai_messages import convert_to_openai_messages
from .utils.sanitize_secrets import sanitize_messages, should_sanitize_for_model


class FunctionCallingNotSupportedError(Exception):
    """Raised when a model doesn't support function calling"""
    pass


class ModelNotFoundError(Exception):
    """Raised when a model doesn't exist or isn't accessible"""
    pass


class AccessDeniedError(Exception):
    """Raised when access to a model is denied"""
    pass

# Create or get the logger
logger = logging.getLogger("LiteLLM")


class SuppressDebugFilter(logging.Filter):
    def filter(self, record):
        # Suppress only the specific message containing the keywords
        if "cost map" in record.getMessage():
            return False  # Suppress this log message
        return True  # Allow all other messages


class Llm:
    """
    A stateless LMC-style LLM with some helpful properties.
    """

    def __init__(self, interpreter):
        # Add the filter to the logger
        logger.addFilter(SuppressDebugFilter())

        # Store a reference to parent interpreter
        self.interpreter = interpreter

        # OpenAI-compatible chat completions "endpoint"
        self.completions = fixed_litellm_completions

        # Settings
        self.model = "gpt-4o-mini"
        self.temperature = 0

        self.supports_vision = None  # Will try to auto-detect
        self.vision_renderer = (
            self.interpreter.toolbox.vision.query
        )  # Will only use if supports_vision is False

        self.supports_functions = None  # Will try to auto-detect
        self.execution_instructions = "To execute code on the user's machine, write a markdown code block. Specify the language after the ```. You will receive the output. Use any programming language."  # If supports_functions is False, this will be added to the system message

        # Optional settings
        self.context_window = None
        self.max_tokens = None
        self.api_base = None
        self.api_key = None
        self.api_version = None
        self._is_loaded = False

        # Sanitize secrets (API keys, passwords) from messages before sending to API LLMs.
        # "auto" = sanitize for remote models, skip for local (ollama/local/jan). "on"/"off" override.
        self.sanitize_secrets = "auto"

        # Budget manager powered by LiteLLM
        self.max_budget = None

        # Cache-aware truncation: when set, history is dropped in chunks of this
        # many tokens instead of one turn at a time.  This keeps the prompt
        # prefix stable across multiple consecutive turns so the provider's KV
        # prefix cache stays warm, reducing per-token cost.  Set to e.g. 2000
        # for a 40k context window.  None (default) falls back to tokentrim's
        # sliding-window behaviour.
        # See: https://github.com/character-ai/prompt-poet#cache-aware-truncation-explained
        self.truncation_step = None

        # Reasoning settings: Use include_reasoning=True to request reasoning tokens
        # (delta.reasoning_content) from OpenRouter/LiteLLM. Use reasoning_effort
        # ("low", "medium", "high") to control the intensity of thinking for OpenAI o1/o3 models.
        self.include_reasoning = None  # None (auto), True, or False
        self.reasoning_effort = None   # "low", "medium", "high"

        # Filled from the final streaming chunk when the API sends usage (see stream_usage.record_stream_chunk_usage).
        self.last_completion_usage = None

    def run(self, messages, *, auxiliary_title_request=False):
        """
        We're responsible for formatting the call into the llm.completions object,
        starting with LMC messages in interpreter.messages, going to OpenAI compatible messages into the llm,
        respecting whether it's a vision or function model, respecting its context window and max tokens, etc.

        And then processing its output, whether it's a function or non function calling model, into LMC format.

        auxiliary_title_request: one-off naming completion — text path only, tight max_tokens,
        no reasoning request, LiteLLM timeout, and no code-execution system suffix on the prompt.
        """

        if not self._is_loaded:
            self.load()

        if (
            self.max_tokens is not None
            and self.context_window is not None
            and self.max_tokens > self.context_window
        ):
            print(
                "Warning: max_tokens is larger than context_window. Setting max_tokens to be 0.2 times the context_window."
            )
            self.max_tokens = int(0.2 * self.context_window)

        # Assertions
        assert (
            messages[0]["role"] == "system"
        ), "First message must have the role 'system'"
        for msg in messages[1:]:
            assert (
                msg["role"] != "system"
            ), "No message after the first can have the role 'system'"

        model = self.model
        if model in [
            "claude-3.5",
            "claude-3-5",
            "claude-3.5-sonnet",
            "claude-3-5-sonnet",
        ]:
            model = "claude-3-5-sonnet-20240620"
            self.model = "claude-3-5-sonnet-20240620"
        # Setup our model endpoint
        if model == "i":
            model = "openai/i"
            if not hasattr(self.interpreter, "conversation_id"):  # Only do this once
                self.context_window = 7000
                self.api_key = "x"
                self.max_tokens = 1000
                self.api_base = "https://api.openinterpreter.com/v0"
                self.interpreter.conversation_id = str(uuid.uuid4())

        # Detect function support
        if self.supports_functions == None:
            try:
                if litellm.supports_function_calling(model):
                    self.supports_functions = True
                else:
                    self.supports_functions = False
            except:
                self.supports_functions = False

        # Detect vision support
        if self.supports_vision == None:
            try:
                if litellm.supports_vision(model):
                    self.supports_vision = True
                else:
                    self.supports_vision = False
            except:
                self.supports_vision = False

        # Trim image messages if they're there
        image_messages = [msg for msg in messages if msg["type"] == "image"]
        if self.supports_vision:
            if self.interpreter.os:
                # Keep only the last two images if the interpreter is running in OS mode
                if len(image_messages) > 1:
                    for img_msg in image_messages[:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
            else:
                # Delete all the middle ones (leave only the first and last 2 images) from messages_for_llm
                if len(image_messages) > 3:
                    for img_msg in image_messages[1:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
                # Idea: we could set detail: low for the middle messages, instead of deleting them
        elif self.supports_vision == False and self.vision_renderer:
            for img_msg in image_messages:
                if img_msg["format"] != "description":
                    self.interpreter.display_message("\n  *Viewing image...*\n")

                    if img_msg["format"] == "path":
                        precursor = f"The image I'm referring to ({img_msg['content']}) contains the following: "
                        if self.interpreter.toolbox.import_toolbox_api:
                            postcursor = f"\nIf you want to ask questions about the image, run `toolbox.vision.query(path='{img_msg['content']}', query='(ask any question here)')` and a vision AI will answer it."
                        else:
                            postcursor = ""
                    else:
                        precursor = "Imagine I have just shown you an image with this description: "
                        postcursor = ""

                    try:
                        image_description = self.vision_renderer(lmc=img_msg)
                        ocr = self.interpreter.toolbox.vision.ocr(lmc=img_msg)

                        # It would be nice to format this as a message to the user and display it like: "I see: image_description"

                        img_msg["content"] = (
                            precursor
                            + image_description
                            + "\n---\nI've OCR'd the image, this is the result (this may or may not be relevant. If it's not relevant, ignore this): '''\n"
                            + ocr
                            + "\n'''"
                            + postcursor
                        )
                        img_msg["format"] = "description"

                    except ImportError:
                        # For toolbox-generated images, use a simple message instead of showing installation prompt
                        if img_msg.get("role") == "computer":
                            img_msg["format"] = "description"
                            img_msg["content"] = "An image was generated by the code execution."
                        else:
                            print(
                                "\nTo use local vision, run `pip install 'open-interpreter[local]'`.\n"
                            )
                            img_msg["format"] = "description"
                            img_msg["content"] = ""

        # Convert to OpenAI messages format
        messages = convert_to_openai_messages(
            messages,
            function_calling=self.supports_functions,
            vision=self.supports_vision,
            shrink_images=self.interpreter.shrink_images,
            interpreter=self.interpreter,
        )

        system_message = messages[0]["content"]
        messages = messages[1:]

        # Trim messages
        try:
            if self.truncation_step and self.context_window:
                # Cache-aware truncation: drop history in fixed-size chunks so
                # the prompt prefix stays stable across multiple turns.  This
                # keeps the provider's KV prefix cache warm between requests,
                # unlike a per-turn sliding window which invalidates the cache
                # on every call once the context fills up.
                token_limit = self.context_window - (self.max_tokens or 0) - 25
                messages = cache_aware_trim(
                    messages,
                    system_message=system_message,
                    token_limit=token_limit,
                    truncation_step=self.truncation_step,
                    model=model,
                )
            elif self.context_window and self.max_tokens:
                trim_to_be_this_many_tokens = (
                    self.context_window - self.max_tokens - 25
                )  # arbitrary buffer
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=trim_to_be_this_many_tokens,
                )
            elif self.context_window and not self.max_tokens:
                # Just trim to the context window if max_tokens not set
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=self.context_window,
                )
            else:
                try:
                    messages = tt.trim(
                        messages, system_message=system_message, model=model
                    )
                except:
                    if len(messages) == 1:
                        if self.interpreter.in_terminal_interface:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `interpreter --context_window {token limit} --max_tokens {max tokens per response}`.

Continuing...
                            """
                            )
                        else:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `self.context_window = {token limit}`.

Also please set `self.max_tokens = {max tokens per response}`.

Continuing...
                            """
                            )
                    messages = tt.trim(
                        messages, system_message=system_message, max_tokens=8000
                    )
        except:
            # If we're trimming messages, this won't work.
            # If we're trimming from a model we don't know, this won't work.
            # Better not to fail until `messages` is too big, just for frustrations sake, I suppose.

            # Reunite system message with messages
            messages = [{"role": "system", "content": system_message}] + messages

            pass

        # If there should be a system message, there should be a system message!
        # Empty system messages appear to be deleted :(
        if system_message == "":
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": system_message}] + messages

        if should_sanitize_for_model(model, self.sanitize_secrets):
            sanitize_messages(messages)

        ## Start forming the request

        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        # OpenAI-compatible: final stream chunk may include usage (prompt/completion/cached breakdown).
        stream_options = {"include_usage": True}

        # Reasoning tokens: Some models support separate reasoning content
        # Set to {"exclude": True} to disable, or remove this line to allow reasoning tokens
        # params["reasoning"] = {"exclude": True}

        # OpenRouter provider preferences: Ensure consistent behavior across providers
        # For OpenRouter models, require providers to support all parameters to avoid inconsistency
        if model.startswith("deepseek/"):
            # LiteLLM maps include_reasoning / reasoning_effort to DeepSeek's thinking API.
            if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(
                model=model or self.model
            ):
                if self.include_reasoning is not False:
                    params["include_reasoning"] = (
                        True if self.include_reasoning is None else self.include_reasoning
                    )
                    stream_options["include_reasoning"] = params["include_reasoning"]

        if model.startswith("openrouter/"):
            params["provider"] = {
                "require_parameters": True,  # Only use providers that support all request parameters
                "allow_fallbacks": False,     # Disable fallbacks to maintain consistency
            }
            # Request streaming reasoning when the model supports it, so LiteLLM forwards
            # delta.reasoning_content from OpenRouter. Requires LiteLLM v1.63.5+ (BerriAI/litellm#8631).
            # Only set for models that support reasoning to avoid 400 on non-reasoning OpenRouter models.
            if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(model=model or self.model):
                # We use extra_body to pass reasoning parameters for OpenRouter models.
                # This ensures the tokens are requested via both legacy and modern API formats.
                params["extra_body"] = params.get("extra_body", {})
                params["extra_body"]["include_reasoning"] = True
                params["extra_body"]["reasoning"] = {"enabled": True}
                
                # Also set standard params for LiteLLM to handle unified mapping
                params["include_reasoning"] = True
                stream_options["include_reasoning"] = True

        # Override reasoning settings if explicitly set on interpreter.llm
        if self.include_reasoning is not None:
            params["include_reasoning"] = self.include_reasoning
            stream_options["include_reasoning"] = self.include_reasoning
            if model.startswith("openrouter/"):
                params["extra_body"] = params.get("extra_body", {})
                params["extra_body"]["include_reasoning"] = self.include_reasoning
                params["extra_body"]["reasoning"] = {"enabled": self.include_reasoning}

        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            if model.startswith("openrouter/"):
                params["extra_body"] = params.get("extra_body", {})
                if "reasoning" not in params["extra_body"]:
                    params["extra_body"]["reasoning"] = {}
                params["extra_body"]["reasoning"]["effort"] = self.reasoning_effort

        params["stream_options"] = stream_options

        # Optional inputs
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if hasattr(self.interpreter, "conversation_id"):
            params["conversation_id"] = self.interpreter.conversation_id

        if auxiliary_title_request:
            # Avoid huge reasoning streams and hung streams with no wall clock (OpenRouter etc.).
            # Not a litellm re-import — this path still uses the same completions() generator.
            params["timeout"] = 120
            # OpenRouter (and other providers) thinking/reasoning models consume max_tokens
            # with internal reasoning tokens even when reasoning streaming is disabled.
            # With max_tokens=128, finish_reason='length' was observed with zero visible output
            # (all tokens consumed by internal thinking) for deepseek/deepseek-v4-flash.
            # litellm.supports_reasoning() does not detect all reasoning models reliably, so
            # we can't use it to branch here. A large cap is safe for all models: non-reasoning
            # models stop at "stop" well within this budget; reasoning models need the headroom
            # for thinking tokens before the title is produced.
            params["max_tokens"] = min(int(params.get("max_tokens") or 2048), 2048)
            params["include_reasoning"] = False
            params.pop("reasoning_effort", None)
            if "stream_options" in params:
                params["stream_options"].pop("include_reasoning", None)
            if model.startswith("openrouter/") and params.get("extra_body"):
                eb = params["extra_body"]
                eb.pop("include_reasoning", None)
                eb.pop("reasoning", None)
            params["skip_execution_instructions"] = True
            # ``run_text_llm`` treats ```...``` as executable code blocks and can yield zero
            # chunks when the model returns a one-line fenced title (no newline after the
            # fence). Plain streaming maps ``delta.content`` directly to messages for
            # ``%rename`` / auto-title only.
            params["conversation_title_plain_stream"] = True

        # Set some params directly on LiteLLM
        if self.max_budget:
            litellm.max_budget = self.max_budget
        # NOTE: We don't enable litellm.set_verbose here because it causes massive output
        # showing every single streaming chunk. Open Interpreter's verbose mode should
        # only show our own debug messages, not LiteLLM's internal logging.
        # if self.interpreter.verbose:
        #     litellm.set_verbose = True

        if (
            self.interpreter.debug == True and False  # DISABLED
        ):  # debug will equal "server" if we're debugging the server specifically
            print("\n\n\nOPENAI COMPATIBLE MESSAGES:\n\n\n")
            for message in messages:
                if len(str(message)) > 5000:
                    print(str(message)[:200] + "...")
                else:
                    print(message)
                print("\n")
            print("\n\n\n")

        if self.supports_functions and not auxiliary_title_request:
            # yield from run_function_calling_llm(self, params)
            try:
                yield from run_tool_calling_llm(self, params)
            except FunctionCallingNotSupportedError as e:
                # Model doesn't support function calling, fall back to text mode
                message = (
                    "Model doesn't support function calling, falling back to text mode.\n\n"
                    "Tip: Use `--no-llm_supports_functions` to skip function calling and avoid this message."
                )
                panel = Panel(
                    message,
                    border_style="yellow",
                    title="Warning",
                    title_align="left"
                )
                rich_print(panel)
                print("")  # Add space after message
                self.supports_functions = False
                # Re-convert messages for text mode
                messages = convert_to_openai_messages(
                    self.interpreter.messages,
                    function_calling=False,
                    vision=self.supports_vision,
                    shrink_images=self.interpreter.shrink_images,
                    interpreter=self.interpreter,
                )
                params["messages"] = messages
                # Remove tools parameter if present (it was added by run_tool_calling_llm)
                params.pop("tools", None)
                yield from run_text_llm(self, params)
        else:
            yield from run_text_llm(self, params)
        # Let ModelNotFoundError and AccessDeniedError bubble up to respond.py for proper handling

    # If you change model, set _is_loaded to false
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._is_loaded = False

    def load(self):
        if self._is_loaded:
            return

        # Route explicit DashScope models to DashScope defaults (OpenAI-compatible).
        # Prefixes avoid ambiguous auto-routing vs other providers (e.g. deepseek/*).
        # Slugs:
        # - dashscope-intl/<model> (Singapore ap-southeast-1)
        # - dashscope-us/<model> (Virginia, US us-east-1)
        model_lower = self.model.lower()
        dashscope_route = None
        if model_lower.startswith("dashscope-intl/"):
            dashscope_route = (
                self.model.split("/", 1)[1].lower(),
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        elif model_lower.startswith("dashscope-us/"):
            dashscope_route = (
                self.model.split("/", 1)[1].lower(),
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            )
        if dashscope_route is not None:
            model_name, _dashscope_default_base = dashscope_route
            if self.api_base is None:
                self.api_base = _dashscope_default_base
            if self.api_key is None:
                self.api_key = os.environ.get("DASHSCOPE_API_KEY")
            # Qwen3.5 is a unified vision-language architecture — there are no separate
            # VL variants because every model in the family natively supports image input.
            # LiteLLM's registry does not know this yet, so we set it explicitly.
            if model_name.startswith("qwen3.5") and self.supports_vision is None:
                self.supports_vision = True
            # Route through OpenAI-compatible formatting for DashScope's compatible endpoint.
            self.model = f"openai/{model_name}"

        # DeepSeek API (OpenAI-compatible). Keep deepseek/<model> for LiteLLM routing.
        if model_lower.startswith("deepseek/"):
            if self.api_base is None:
                self.api_base = os.environ.get(
                    "DEEPSEEK_API_BASE", "https://api.deepseek.com"
                )
            if self.api_key is None:
                self.api_key = os.environ.get("DEEPSEEK_API_KEY")

        if self.model.startswith("ollama/") and not ":" in self.model:
            self.model = self.model + ":latest"

        self._is_loaded = True

        if self.model.startswith("ollama/"):
            model_name = self.model.replace("ollama/", "")
            api_base = getattr(self, "api_base", None) or os.getenv(
                "OLLAMA_HOST", "http://localhost:11434"
            )
            names = []
            try:
                # List out all downloaded ollama models. Will fail if ollama isn't installed
                response = requests.get(f"{api_base}/api/tags")
                if response.ok:
                    data = response.json()
                    names = [
                        model["name"]
                        for model in data["models"]
                        if "name" in model and model["name"]
                    ]

            except Exception as e:
                print(str(e))
                self.interpreter.display_message(
                    f"> Ollama not found\n\nPlease download Ollama from [ollama.com](https://ollama.com/) to use `{model_name}`.\n"
                )
                exit()

            # Download model if not already installed
            if model_name not in names:
                self.interpreter.display_message(f"\nDownloading {model_name}...\n")
                requests.post(f"{api_base}/api/pull", json={"name": model_name})

            # Get context window if not set
            if self.context_window == None:
                response = requests.post(
                    f"{api_base}/api/show", json={"name": model_name}
                )
                model_info = response.json().get("model_info", {})
                context_length = None
                for key in model_info:
                    if "context_length" in key:
                        context_length = model_info[key]
                        break
                if context_length is not None:
                    self.context_window = context_length
            if self.max_tokens == None:
                if self.context_window != None:
                    self.max_tokens = int(self.context_window * 0.2)

            # Send a ping, which will actually load the model
            model_name = model_name.replace(":latest", "")
            print(f"Loading {model_name}...\n")

            old_max_tokens = self.max_tokens
            self.max_tokens = 1
            self.interpreter.toolbox.ai.chat("ping")
            self.max_tokens = old_max_tokens

            self.interpreter.display_message("*Model loaded.*\n")

        # Validate LLM should be moved here!!

        if self.context_window == None:
            try:
                model_info = litellm.get_model_info(model=self.model)
                self.context_window = model_info["max_input_tokens"]
                if self.max_tokens == None:
                    self.max_tokens = min(
                        int(self.context_window * 0.2), model_info["max_output_tokens"]
                    )
            except:
                pass


def fixed_litellm_completions(**params):
    """
    Just uses a dummy API key, since we use litellm without an API key sometimes.
    Hopefully they will fix this!
    """

    if "local" in params.get("model"):
        # Kinda hacky, but this helps sometimes
        params["stop"] = ["<|assistant|>", "<|end|>", "<|eot_id|>"]

    if params.get("model") == "i" and "conversation_id" in params:
        litellm.drop_params = (
            False  # If we don't do this, litellm will drop this param!
        )
    else:
        litellm.drop_params = True

    params["model"] = params["model"].replace(":latest", "")

    # Run completion
    # Keep provider retries explicit in respond.py so user choices ("n = stop")
    # are the single source of truth for retry behavior.
    params["num_retries"] = 0
    tried_dummy_key = False

    # DeepSeek models via OpenRouter have thinking mode enabled by default and require
    # reasoning_content on EVERY prior assistant message in the request — even turns
    # where the model chose not to think (empty string is accepted). Without this
    # the API returns 400 "reasoning_content must be passed back to the API".
    #
    # Primary fix: convert_to_openai_messages propagates reasoning_content to ALL
    # consecutive assistant messages in a turn (text preamble AND tool_calls message).
    # SiliconFlow (used as an OpenRouter backend) strictly enforces this — passing ""
    # on a tool_calls message triggers 400 when the model actually returned reasoning.
    #
    # This is a secondary/belt-and-suspenders pass that catches any remaining assistant
    # messages that slipped through (e.g. synthetic messages injected by process_messages).
    # We skip this only when reasoning was explicitly disabled by the caller.
    _model = params.get("model", "")
    _extra_body = params.get("extra_body", {})
    _reasoning_explicitly_disabled = (
        _extra_body.get("include_reasoning") is False
        or params.get("include_reasoning") is False
    )
    _uses_deepseek_reasoning_history = _model.startswith("deepseek/") or (
        _model.startswith("openrouter/") and "deepseek" in _model.lower()
    )
    if _uses_deepseek_reasoning_history and not _reasoning_explicitly_disabled:
        for msg in params.get("messages", []):
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                msg["reasoning_content"] = ""

    while True:
        try:
            yield from litellm.completion(**params)
            return  # If the completion is successful, exit the function
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        except Exception as e:
            # Check if this is a function-calling-not-supported error.
            # Only check for this if we're actually trying to use function calling.
            if "tools" in params:
                error_message = str(e).lower()
                if any(
                    phrase in error_message
                    for phrase in [
                        "no endpoints found that support tool use",
                        "tool use",
                        "function calling",
                        "tool calling",
                    ]
                ):
                    raise FunctionCallingNotSupportedError(str(e)) from e

            # LiteLLM sometimes requires an api_key parameter even when the backend
            # provider ignores it. Retry exactly once with a dummy key, then surface
            # the error so respond.py's retry prompt controls subsequent retries.
            if (
                isinstance(e, litellm.exceptions.AuthenticationError)
                and "api_key" not in params
                and not tried_dummy_key
            ):
                print(
                    "LiteLLM requires an API key. Trying again with a dummy API key. In the future, if this fixes it, please set a dummy API key to prevent this message. (e.g `interpreter --api_key x` or `self.api_key = 'x'`)"
                )
                params["api_key"] = "x"
                tried_dummy_key = True
                continue

            # Bubble up all provider errors to respond.py for user-facing handling.
            raise
