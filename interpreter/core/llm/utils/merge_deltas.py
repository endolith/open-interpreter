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
                # For lists, we typically want to extend/append, not replace
                if key not in original:
                    original[key] = value
                else:
                    # Extend the existing list with new items
                    if isinstance(original[key], list):
                        original[key].extend(value)
                    else:
                        # If original wasn't a list, replace it
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
