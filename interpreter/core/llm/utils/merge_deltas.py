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
    if hasattr(delta, 'model_dump'):
        # Pydantic v2
        return delta.model_dump(exclude_unset=True)
    elif hasattr(delta, 'dict'):
        # Pydantic v1
        return delta.dict(exclude_unset=True)
    else:
        # Try generic conversion
        try:
            return dict(delta)
        except (TypeError, ValueError):
            # If conversion fails, return empty dict
            return {}


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
            elif hasattr(value, 'model_dump') or hasattr(value, 'dict'):
                # Pydantic object - convert to dict first
                if hasattr(value, 'model_dump'):
                    value = value.model_dump(exclude_unset=True)
                else:
                    value = value.dict(exclude_unset=True)
                if key not in original:
                    original[key] = value
                else:
                    merge_deltas(original[key], value)
            elif isinstance(value, list):
                # Handle lists (e.g., tool_calls, reasoning_content arrays)
                # Special handling for tool_calls: merge by index, not extend
                if key == "tool_calls" and key in original and isinstance(original[key], list):
                    # Merge tool_calls by index
                    for new_tool_call in value:
                        if isinstance(new_tool_call, dict):
                            index = new_tool_call.get("index", 0)
                            # Find existing tool_call with same index
                            existing_tool_call = None
                            for i, tc in enumerate(original[key]):
                                if isinstance(tc, dict) and tc.get("index") == index:
                                    existing_tool_call = i
                                    break

                            if existing_tool_call is not None:
                                # Merge into existing tool_call
                                existing = original[key][existing_tool_call]
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
                            else:
                                # New tool_call, add it
                                original[key].append(new_tool_call)
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
