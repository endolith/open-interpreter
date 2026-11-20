"""Retry classification for provider errors in ``fixed_litellm_completions``.

A permanent provider failure (auth, missing model, malformed request, exhausted
quota) cannot succeed on a retry, so it must reach the user immediately instead
of consuming all four attempts. Transient failures -- throttling, timeouts, 5xx
-- keep the historical retry behaviour, and so does any error the classifier
cannot positively identify as permanent.
"""

from unittest import mock

import pytest

from interpreter.core.llm.llm import fixed_litellm_completions


class _ProviderError(Exception):
    """An exception carrying a structured HTTP status, like litellm's APIError."""

    def __init__(self, message, status_code=None, http_status=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if http_status is not None:
            self.http_status = http_status


def _fail_once_then_succeed(error):
    """Return a fake completion that raises ``error`` once, then finishes cleanly."""
    calls = {"n": 0}

    def completion(**params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise error
        return iter(())

    completion.calls = calls
    return completion


def _always_fail(error):
    """Return a fake completion that raises ``error`` on every attempt."""
    calls = {"n": 0}

    def completion(**params):
        calls["n"] += 1
        raise error

    completion.calls = calls
    return completion


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_4xx_raises_without_retrying(status):
    """Auth, forbidden, and missing-model 4xx responses are not retried.

    Retrying a request the provider has permanently rejected wastes three
    round-trips and delays the error the user needs to act on.
    """
    error = _ProviderError("invalid api key", status_code=status)
    completion = _always_fail(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with pytest.raises(_ProviderError, match="invalid api key"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 1


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503])
def test_transient_statuses_still_retry(status):
    """Timeouts, throttling, and 5xx responses are retried as before."""
    error = _ProviderError("upstream hiccup", status_code=status)
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


def test_quota_429_is_not_retried():
    """A quota-exhaustion 429 never clears, so the billing hint must show at once."""
    error = _ProviderError(
        "insufficient_quota: you exceeded your current quota", status_code=429
    )
    completion = _always_fail(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with pytest.raises(_ProviderError, match="insufficient_quota"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 1


def test_rate_limit_quota_message_is_retried():
    """A per-minute quota 429 is throttling, not exhausted billing quota.

    Providers describe rate limits with the word "quota" too
    ("Requests-per-minute quota exceeded; retry shortly"), and those clear on
    their own; treating every quota-related 429 as permanent would drop the
    retry that makes the client resilient.
    """
    error = _ProviderError(
        "Requests-per-minute quota exceeded; retry shortly", status_code=429
    )
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


@pytest.mark.parametrize(
    "status,message,retried",
    [
        # The status is authoritative; the message may be misleading.
        (503, "invalid api key", True),
        (400, "temporarily unavailable", False),
    ],
)
def test_status_number_wins_over_message_text(status, message, retried):
    """A structured status decides retryability regardless of the message wording.

    Providers often wrap a generic sentence around an upstream failure, so a
    5xx must count as transient even when its text says "invalid api key", and
    a 400 must count as permanent even when its text says "temporarily
    unavailable".
    """
    error = _ProviderError(message, status_code=status)
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        if retried:
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))
        else:
            with pytest.raises(_ProviderError, match=message):
                list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == (2 if retried else 1)


@pytest.mark.parametrize(
    "status,message,retried",
    [
        # A status outside the error ranges says nothing about retryability, so
        # the message heuristics decide: a permanent cause stops at once, an
        # unrecognised one keeps the historical retries.
        (200, "invalid api key", False),
        (200, "upstream hiccup", True),
    ],
)
def test_non_error_status_falls_back_to_message_text(status, message, retried):
    """A status code that is not an error code must not decide retryability.

    Some provider wrappers carry a success status on an exception raised for an
    unrelated failure. Reading that 200 as "not 4xx/5xx, therefore permanent"
    would skip the diagnostic text, so the classifier must ignore it and let
    the message heuristics decide.
    """
    error = _ProviderError(message, status_code=status)
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        if retried:
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))
        else:
            with pytest.raises(_ProviderError, match=message):
                list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == (2 if retried else 1)


@pytest.mark.parametrize("http_status,retried", [(503, True), (401, False)])
def test_usable_secondary_status_wins_over_nonsense_primary(http_status, retried):
    """A nonsense primary status must not hide a usable secondary one.

    A proxy can expose a success status in `status_code` while the real
    upstream failure is in `http_status`. Falling back to the message text on
    the first attribute would call a 503 "invalid api key" permanent, so the
    classifier must keep checking both attributes before using heuristics.
    """
    error = _ProviderError(
        "invalid api key", status_code=200, http_status=http_status
    )
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        if retried:
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))
        else:
            with pytest.raises(_ProviderError, match="invalid api key"):
                list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == (2 if retried else 1)


@pytest.mark.parametrize(
    "message",
    [
        "The model `foo` was not found",
        "model is unavailable",
        "The model `foo` is unavailable",
        "no such model: gpt-5",
    ],
)
def test_permanent_model_message_is_not_retried(message):
    """An unknown or retired model name is a typo, not an outage.

    A bare "unavailable" used to be treated as transient, which turned a
    misspelled model into four pointless attempts.
    """
    completion = _always_fail(Exception(message))
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with pytest.raises(Exception, match="model"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 1


def test_model_name_in_unrelated_error_is_retried():
    """Mentioning a model is not by itself a model rejection.

    "Unknown error while requesting model gpt-4o" describes an unidentified
    failure and happens to contain "model", "unknown", and the model name.
    Treating independent words as a rejection skipped the retry for a merely
    unrecognised error, which the classifier's fallback deliberately keeps.
    """
    error = Exception("Unknown error while requesting model gpt-4o")
    completion = _fail_once_then_succeed(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


def test_permanent_cause_wins_over_temporary_outage_text():
    """An auth failure is permanent even when the message also sounds transient.

    "invalid api key; service temporarily unavailable" names a cause a retry
    cannot fix, so the temporary-sounding remainder must not keep the loop
    running four rejected requests.
    """
    completion = _always_fail(
        Exception("invalid api key; service temporarily unavailable")
    )
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with pytest.raises(Exception, match="invalid api key"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 1


def test_temporary_model_outage_is_retried():
    """A model outage that says "temporarily" keeps its retry.

    "model temporarily unavailable" describes a transient condition, not a
    typo in the model name, so the model-rejection pattern must not claim it.
    """
    completion = _fail_once_then_succeed(Exception("model temporarily unavailable"))
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


@pytest.mark.parametrize(
    "message",
    [
        "Backend was not found while requesting model gpt-4o",
        "Request for model gpt-4o failed: endpoint not found",
    ],
)
def test_unrelated_missing_resource_is_retried(message):
    """A missing resource that is not the model keeps the retry.

    "was not found" describes any absent backend or endpoint, and the model
    rejection pattern must not treat intervening text as the model name, so a
    statusless message like these stays unrecognised and keeps its attempts.
    """
    completion = _fail_once_then_succeed(Exception(message))
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


@pytest.mark.parametrize(
    "message",
    [
        "The service is temporarily unavailable",
        "provider overloaded, please try again later",
        "Internal Server Error (request_id: abc)",
    ],
)
def test_transient_message_is_retried(message):
    """Phrases that describe a genuine outage keep their retry."""
    completion = _fail_once_then_succeed(Exception(message))
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        list(fixed_litellm_completions(model="gpt-4o", messages=[], api_key="x"))

    assert completion.calls["n"] == 2


def test_unrecognised_error_keeps_historical_retries():
    """An unclassified failure still gets every attempt, and the first error is raised.

    The classifier only short-circuits failures it can positively identify as
    permanent. Treating an unknown error as permanent would drop the retry for
    transient connection blips that carry no status and no recognisable text.
    """
    error = Exception("failure 0")
    completion = _always_fail(error)
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with pytest.raises(Exception, match="failure 0"):
            list(fixed_litellm_completions(model="gpt-4o", messages=[]))

    assert completion.calls["n"] == 4


def test_auth_without_api_key_retries_with_dummy_key_even_with_status():
    """The dummy-key retry is decided before the permanence check.

    A 401 caused by a missing key is our fault, not a permanently bad request,
    so it must still get the one retry that supplies the dummy key.
    """
    auth_error = type("AuthenticationError", (_ProviderError,), {})
    calls = []

    def completion(**params):
        calls.append(params)
        if len(calls) == 1:
            raise auth_error("no key", status_code=401)
        return iter(())

    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with mock.patch(
            "interpreter.core.llm.llm.litellm.exceptions.AuthenticationError", auth_error
        ):
            list(fixed_litellm_completions(model="gpt-4o", messages=[]))

    assert "api_key" not in calls[0]
    assert calls[1]["api_key"] == "x"
    assert len(calls) == 2


def test_auth_with_api_key_is_permanent():
    """A 401 when a key was already supplied is retried zero times."""
    auth_error = type("AuthenticationError", (_ProviderError,), {})
    completion = _always_fail(auth_error("invalid api key", status_code=401))
    with mock.patch("interpreter.core.llm.llm.litellm.completion", completion):
        with mock.patch(
            "interpreter.core.llm.llm.litellm.exceptions.AuthenticationError", auth_error
        ):
            with pytest.raises(auth_error, match="invalid api key"):
                list(
                    fixed_litellm_completions(
                        model="gpt-4o", messages=[], api_key="sk-real"
                    )
                )

    assert completion.calls["n"] == 1
