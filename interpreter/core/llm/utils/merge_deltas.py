def normalize_delta_to_dict(delta):
    """
    Normalize a delta object to a plain dict.

    LiteLLM may return Pydantic Delta objects for some models (e.g., GLM-4.6),
    but the codebase expects plain dicts. This function handles the conversion.

    Args:
        delta: Either a dict or a Pydantic Delta object

    Returns:
        A plain dict representation of the delta
    """
    if isinstance(delta, dict):
        return delta

    # Convert Pydantic Delta object to dict
    if hasattr(delta, "model_dump"):
        # Pydantic v2
        return delta.model_dump(exclude_unset=True)
    elif hasattr(delta, "dict"):
        # Pydantic v1
        return delta.dict(exclude_unset=True)
    else:
        # Try generic conversion
        try:
            return dict(delta)
        except (TypeError, ValueError):
            # If conversion fails, return empty dict
            return {}


def _tool_call_slot(accumulated_calls, new_tool_call):
    """Which accumulated call this delta continues, or None when it opens a new one.

    ``index`` is not an identity. litellm's ``Delta.__init__`` numbers index-less
    tool calls from zero *per chunk*, so a provider that streams two finished
    calls in two chunks — litellm's native Ollama path does exactly this — hands
    both of them index 0. Merging on that alone concatenated the second call's
    arguments onto the first, producing
    ``{"language":"python",...}{"language":"bash",...}`` inside one arguments
    string: the "malformed call" the model was then blamed for was built here.

    The call id is the identity that holds. Providers send it once, on the delta
    that opens a call, and never reuse it for another, so a delta whose id differs
    from the call sitting at its index has opened a second call no matter what the
    index says. An id is still allowed to arrive after the call has started (some
    providers send the function name in the opening delta and the id with the
    first arguments fragment), which is why a call that has no id yet still
    absorbs one rather than being treated as a different call.
    """
    new_id = new_tool_call.get("id")
    if new_id:
        for i, call in enumerate(accumulated_calls):
            if isinstance(call, dict) and call.get("id") == new_id:
                return i

    index = new_tool_call.get("index", 0)
    for i, call in enumerate(accumulated_calls):
        if not isinstance(call, dict) or call.get("index") != index:
            continue
        if new_id and call.get("id") and call["id"] != new_id:
            # Same stamped index, different call: keep looking rather than
            # appending this call's arguments onto that one.
            continue
        return i
    return None


def _append_tool_call(accumulated_calls, new_tool_call):
    """Start a new accumulated call, keeping ``index`` unique across the list.

    Two calls can arrive carrying the same stamped index (see _tool_call_slot).
    Storing both under it would leave every later index-only delta ambiguous, so
    a colliding newcomer is renumbered past the highest index in use. Nothing
    downstream reads this field — it is a streaming artefact, not part of the
    request shape — so renumbering costs nothing and keeps the lookup sound.
    """
    taken = {call.get("index") for call in accumulated_calls if isinstance(call, dict)}
    if new_tool_call.get("index", 0) in taken:
        new_tool_call = dict(new_tool_call)
        new_tool_call["index"] = max((i for i in taken if isinstance(i, int)), default=-1) + 1
    accumulated_calls.append(new_tool_call)


def _merge_tool_call(existing, new_tool_call):
    """Fold one delta into the call it continues."""
    # Merge id (use non-null value)
    if new_tool_call.get("id") and not existing.get("id"):
        existing["id"] = new_tool_call["id"]
    # Merge function
    if "function" in new_tool_call and isinstance(new_tool_call["function"], dict):
        if "function" not in existing:
            existing["function"] = {}
        func = existing["function"]
        new_func = new_tool_call["function"]
        # Merge name (use non-null value)
        if new_func.get("name") and not func.get("name"):
            func["name"] = new_func["name"]
        # Accumulate arguments (string concatenation)
        if "arguments" in new_func:
            if "arguments" in func:
                func["arguments"] += new_func.get("arguments", "")
            else:
                func["arguments"] = new_func.get("arguments", "")
    # Merge type
    if new_tool_call.get("type") and not existing.get("type"):
        existing["type"] = new_tool_call["type"]


def merge_deltas(original, delta):
    """
    Pushes the delta into the original and returns that.

    Great for reconstructing OpenAI streaming responses -> complete message objects.
    """

    # Normalize delta to dict first
    delta = normalize_delta_to_dict(delta)

    for key, value in dict(delta).items():
        if value != None:
            if isinstance(value, str):
                if key in original:
                    original[key] = (original[key] or "") + (value or "")
                else:
                    original[key] = value
            elif isinstance(value, dict):
                # Already a dict, use it directly
                if key not in original:
                    original[key] = value
                else:
                    merge_deltas(original[key], value)
            elif hasattr(value, "model_dump") or hasattr(value, "dict"):
                # Pydantic object - convert to dict first
                if hasattr(value, "model_dump"):
                    value = value.model_dump(exclude_unset=True)
                else:
                    value = value.dict(exclude_unset=True)
                if key not in original:
                    original[key] = value
                else:
                    merge_deltas(original[key], value)
            elif isinstance(value, list):
                # Handle lists (e.g., tool_calls, reasoning_content arrays)
                # Special handling for tool_calls: merge each delta into the call
                # it continues (identified by id, falling back to index), not extend
                if key == "tool_calls" and key in original and isinstance(original[key], list):
                    for new_tool_call in value:
                        if isinstance(new_tool_call, dict):
                            slot = _tool_call_slot(original[key], new_tool_call)
                            if slot is not None:
                                _merge_tool_call(original[key][slot], new_tool_call)
                            else:
                                _append_tool_call(original[key], new_tool_call)
                        else:
                            # Not a dict, just append
                            original[key].append(new_tool_call)
                else:
                    # For other lists, extend/append
                    if key not in original:
                        original[key] = value
                    else:
                        if isinstance(original[key], list):
                            original[key].extend(value)
                        else:
                            original[key] = value
            else:
                # Try to convert to dict, but handle cases where it can't be converted
                # (e.g., reasoning tokens, or other non-standard formats)
                try:
                    value_dict = dict(value)
                    if key not in original:
                        original[key] = value_dict
                    else:
                        merge_deltas(original[key], value_dict)
                except (ValueError, TypeError) as e:
                    # If conversion fails, skip this value or store it as-is
                    # This handles non-standard delta formats (e.g., reasoning tokens)
                    # that some models may output
                    if key not in original:
                        original[key] = value
                    # If key exists, we can't merge non-dict values, so skip
                    pass

    return original
