"""Render a code evaluation as one line beside the approval prompt."""

from rich import print as rich_print

from .ontology import FLAG_LABELS, IMPACT_MAX

_LEVEL_COLORS = {"green": "green", "yellow": "yellow", "red": "red"}


def format_evaluation(result, verbose=False):
    """Return the plain one-line summary, or an empty string when there is nothing.

    Model name and latency are provenance, not part of the verdict; they are
    only shown with `--verbose`, while a failure `reason` is always shown.
    """
    if not result:
        return ""
    level = str(result.get("level", "unknown")).upper()
    parts = [f"Safety: {level}"]

    impact = result.get("impact")
    if impact is not None:
        parts.append(f"impact {impact:.1f}/{IMPACT_MAX}")

    probabilities = result.get("probabilities") or {}
    flags = result.get("flags") or []
    if flags:
        details = " · ".join(
            f"{FLAG_LABELS.get(name, name)} {probabilities.get(name, 0.0):.2f}"
            for name in flags[:3]
        )
        parts.append(details)

    tail = []
    if verbose:
        if result.get("cached"):
            tail.append("cached")
        if result.get("model"):
            tail.append(str(result["model"]))
        if result.get("latency_ms"):
            tail.append(f"{result['latency_ms']} ms")
    if result.get("reason"):
        tail.append(str(result["reason"]))
    if tail:
        parts.append("(" + ", ".join(tail) + ")")

    return "  " + "  ".join(parts)


def print_evaluation(result, interpreter=None):
    """Print the summary, colored unless the interpreter is in plain-text mode."""
    text = format_evaluation(result, verbose=getattr(interpreter, "verbose", False))
    if not text:
        return
    if getattr(interpreter, "plain_text_display", False):
        print(text, flush=True)
        return
    color = _LEVEL_COLORS.get(str(result.get("level", "")), "dim")
    rich_print(f"[{color}]{text}[/]", flush=True)
