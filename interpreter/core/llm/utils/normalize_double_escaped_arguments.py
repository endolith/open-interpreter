import json


def _unquote(value):
    """Decode a JSON string literal, or hand back whatever it was.

    Returns the input unchanged when it is not a quoted string or does not
    parse, so a repair can never damage text that was already correct.
    """
    if not isinstance(value, str) or len(value) < 2:
        return value
    if not (value.startswith('"') and value.endswith('"')):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded


def normalize_double_escaped_arguments(arguments):
    """Undo a provider that escapes JSON inside JSON.

    OpenInference-served models stream `tool_calls.function.arguments` with
    another level of escaping, so the parsed object comes back as
    `{"code": ..., '"language"': '"bash"'}`. OI's lookups for `language` and
    `code` then miss and the call is dropped, costing a turn (#566).

    A key wrapped in quotes marks its pair as double-encoded: decode such keys
    and the value beside them. A value whose key was *not* wrapped is left
    alone — an `execute` payload is arbitrary text that may legitimately start
    and end with a quote — except for `language`, whose value is a short token
    that never contains one. Nothing is changed unless a quoted key appears,
    and a value that fails to decode stays exactly as it arrived.

    Returns `arguments` unchanged when there is nothing to repair.
    """
    if not isinstance(arguments, dict):
        return arguments

    normalized = dict(arguments)
    repaired = False

    for key, value in arguments.items():
        if not (
            isinstance(key, str) and len(key) >= 2 and key.startswith('"') and key.endswith('"')
        ):
            continue
        decoded_key = _unquote(key)
        if not isinstance(decoded_key, str) or decoded_key == key:
            continue
        # An unescaped copy of this key already exists: the model's own
        # spelling wins over the mangled one.
        if decoded_key in arguments:
            continue
        del normalized[key]
        normalized[decoded_key] = _unquote(value)
        repaired = True

    if "language" in normalized:
        unquoted_language = _unquote(normalized["language"])
        if unquoted_language != normalized["language"]:
            normalized["language"] = unquoted_language
            repaired = True

    return normalized if repaired else arguments
