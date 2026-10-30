"""Ask a decision model what a code block would do, and turn the answers into a light.

The evaluator is display-only: it never blocks or approves execution. It is
built to fail soft — every error, timeout, missing key, or refusal becomes
``level: "unknown"`` so a broken provider can never be mistaken for "safe".
"""

from __future__ import annotations

import hashlib
import os
import time

import litellm

from ..llm.utils.sanitize_secrets import redact_text, should_sanitize_for_model
from .ontology import (
    ALLOW_IMPACT,
    DEFAULT_POLICY,
    FLAG_THRESHOLD,
    QUESTIONS,
    RED_FLAGS,
    YELLOW_FLAGS,
)

DEFAULT_MODEL = "typesafe/jev-latest"


class DecisionModelConfig:
    """Evaluator settings, shaped like ``interpreter.llm``'s model/api settings."""

    def __init__(self):
        self.model = DEFAULT_MODEL
        self.api_base = None
        self.api_key = None
        self.policy = DEFAULT_POLICY
        self.timeout = 5.0
        self.max_chars = 16000


def evaluate_code(interpreter, language, code):
    """Return the evaluation dict for ``code``, or ``None`` when there is nothing to judge.

    Never raises: a provider failure is reported as ``unknown`` so the caller can
    print it and carry on with the normal approval flow.
    """
    if not code or not code.strip():
        return None

    config = getattr(interpreter, "decision_model", None)
    if config is None:
        config = DecisionModelConfig()
    model = getattr(config, "model", None) or DEFAULT_MODEL
    policy = getattr(config, "policy", None) or DEFAULT_POLICY
    timeout = getattr(config, "timeout", 5.0)
    max_chars = getattr(config, "max_chars", 16000)

    code, redacted = _redact_for_provider(interpreter, model, code)
    code = _truncate(code, max_chars)

    cache = _cache(interpreter)
    key = _cache_key(model, language, code)
    if key in cache:
        return {**cache[key], "cached": True, "latency_ms": 0}

    state = {
        "language": language,
        "code": code,
        "cwd": _current_cwd(interpreter, language),
        "policy": policy,
    }

    started = time.monotonic()
    try:
        response = litellm.systemone(
            model=model,
            state=state,
            questions=QUESTIONS,
            api_base=getattr(config, "api_base", None),
            api_key=getattr(config, "api_key", None),
            timeout=timeout,
        )
    except Exception as exc:  # provider errors must not break the turn
        return _unknown(
            model,
            _reason(exc),
            _elapsed_ms(started),
            redacted=redacted,
        )

    result = _result_from_response(response, model, redacted, _elapsed_ms(started))
    cache[key] = result
    return result


def _result_from_response(response, model, redacted, latency_ms):
    answers = getattr(response, "answers", None) or {}
    probabilities = {}
    impact = None
    for name, answer in answers.items():
        value = getattr(answer, "noul", None)
        if isinstance(value, (int, float)):
            probabilities[name] = float(value)
        if name == "impact":
            score = getattr(answer, "score", None)
            if isinstance(score, (int, float)):
                impact = float(score)

    if not probabilities and impact is None:
        return _unknown(model, "no answer", latency_ms, redacted=redacted)

    fired_red = [
        name for name in RED_FLAGS if probabilities.get(name, 0.0) >= FLAG_THRESHOLD
    ]
    fired_yellow = [
        name for name in YELLOW_FLAGS if probabilities.get(name, 0.0) >= FLAG_THRESHOLD
    ]
    if fired_red:
        level = "red"
    elif fired_yellow:
        level = "yellow"
    else:
        level = "green"

    return {
        "level": level,
        "impact": impact,
        "probabilities": probabilities,
        "flags": sorted(
            fired_red + fired_yellow,
            key=lambda name: probabilities.get(name, 0.0),
            reverse=True,
        ),
        "would_allow": level == "green"
        and impact is not None
        and impact <= ALLOW_IMPACT,
        "model": getattr(response, "model", None) or model,
        "latency_ms": latency_ms,
        "cached": False,
        "redacted": redacted,
        "reason": None,
    }


def _unknown(model, reason, latency_ms=0, redacted=False):
    return {
        "level": "unknown",
        "impact": None,
        "probabilities": {},
        "flags": [],
        "would_allow": False,
        "model": model,
        "latency_ms": latency_ms,
        "cached": False,
        "redacted": redacted,
        "reason": reason,
    }


def _redact_for_provider(interpreter, model, code):
    """Remove literal secrets before the snippet leaves the machine.

    Reuses the same detectors and the same ``sanitize_secrets`` setting as the
    chat path (``auto`` means "unless the model is local"), so a user who turned
    redaction off there is not surprised by a second, invisible policy here.
    """
    llm = getattr(interpreter, "llm", None)
    setting = getattr(llm, "sanitize_secrets", "auto")
    if not should_sanitize_for_model(model, setting):
        return code, False
    redacted = redact_text(code, announce=False)
    return redacted, redacted != code


def _truncate(code, limit):
    if limit is None or len(code) <= limit:
        return code
    return code[:limit] + "\n[code truncated for evaluation]"


def _cache(interpreter):
    cache = getattr(interpreter, "_code_evaluation_cache", None)
    if cache is None:
        cache = {}
        try:
            interpreter._code_evaluation_cache = cache
        except AttributeError:
            pass
    return cache


def _cache_key(model, language, code):
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return (model, language, digest)


def _current_cwd(interpreter, language):
    """The shell's tracked cwd when it exists, else the process cwd.

    Python's kernel cwd can drift from this after ``os.chdir``; the state is a
    hint for the model, not a security boundary.
    """
    terminal = getattr(interpreter, "terminal", None)
    active = getattr(terminal, "_active_languages", None) or {}
    instance = active.get(language)
    return getattr(instance, "cwd", None) or os.getcwd()


def _elapsed_ms(started):
    return int((time.monotonic() - started) * 1000)


def _reason(exc):
    name = type(exc).__name__
    text = str(exc).lower()
    if "timeout" in name.lower() or "timed out" in text:
        return "timeout"
    if "authentication" in name.lower() or "api key" in text or "api_key" in text:
        return "no API key"
    return name
