import base64
import io
import json
import sys
from datetime import datetime

from PIL import Image


def _user_ts(message, messages, *, _now=None):
    """Format sent_at for prepending to user message content. Concise: YYYY-MM-DD HH:MM."""
    sent_at = message.get("sent_at")
    if sent_at is not None:
        if isinstance(sent_at, (int, float)):
            return datetime.fromtimestamp(sent_at).strftime("%Y-%m-%d %H:%M")
        return datetime.fromisoformat(str(sent_at).replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M"
        )
    if _now is None:
        _now = datetime.now()
    last_user = [m for m in messages if m.get("role") == "user"]
    if last_user and message == last_user[-1]:
        return _now.strftime("%Y-%m-%d %H:%M")
    return None


def convert_to_openai_messages(
    messages,
    function_calling=True,
    vision=False,
    shrink_images=True,
    interpreter=None,
):
    """
    Converts LMC messages into OpenAI messages
    """
    new_messages = []

    # if function_calling == False:
    #     prev_message = None
    #     for message in messages:
    #         if message.get("type") == "code":
    #             if prev_message and prev_message.get("role") == "assistant":
    #                 prev_message["content"] += "\n```" + message.get("format", "") + "\n" + message.get("content").strip("\n`") + "\n```"
    #             else:
    #                 message["type"] = "message"
    #                 message["content"] = "```" + message.get("format", "") + "\n" + message.get("content").strip("\n`") + "\n```"
    #         prev_message = message

    #     messages = [message for message in messages if message.get("type") != "code"]

    for message in messages:
        # Is this for thine eyes?
        if "recipient" in message and message["recipient"] != "assistant":
            continue

        new_message = {}

        if message["type"] == "message":
            # Default to "assistant" for older saved messages that lack role (e.g. from tool-mode streams).
            role = message.get("role", "assistant")
            new_message["role"] = role

            if role == "user" and (
                message == [m for m in messages if m.get("role") == "user"][-1]
                or interpreter.always_apply_user_message_template
            ):
                # Only add the template for the last message?
                new_message["content"] = interpreter.user_message_template.replace(
                    "{content}", message["content"]
                )
            else:
                new_message["content"] = message["content"]

            ts = _user_ts(message, messages) if role == "user" else None
            if ts is not None:
                new_message["content"] = f"[{ts}] " + new_message["content"]

            # Preserve tool_call_id for tool role messages (required by OpenRouter and other APIs)
            if role == "tool" and "tool_call_id" in message:
                new_message["tool_call_id"] = message["tool_call_id"]

        elif message["type"] == "code":
            new_message["role"] = "assistant"
            if function_calling:
                new_message["function_call"] = {
                    "name": "execute",
                    "arguments": json.dumps(
                        {"language": message["format"], "code": message["content"]}
                    ),
                    # parsed_arguments isn't actually an OpenAI thing, it's an OI thing.
                    # but it's soo useful!
                    # "parsed_arguments": {
                    #     "language": message["format"],
                    #     "code": message["content"],
                    # },
                }
                # Add empty content to avoid error "openai.error.InvalidRequestError: 'content' is a required property - 'messages.*'"
                # especially for the OpenAI service hosted on Azure
                new_message["content"] = ""
            else:
                new_message[
                    "content"
                ] = f"""```{message["format"]}\n{message["content"]}\n```"""

        elif message["type"] == "console" and message["format"] == "output":
            if function_calling:
                new_message["role"] = "function"
                new_message["name"] = "execute"
                if "content" not in message:
                    print("What is this??", content)
                if type(message["content"]) != str:
                    if interpreter.debug:
                        print("\n\n\nStrange chunk found:", message, "\n\n\n")
                    message["content"] = str(message["content"])
                if message["content"].strip() == "":
                    new_message[
                        "content"
                    ] = "No output"  # I think it's best to be explicit, but we should test this.
                else:
                    new_message["content"] = message["content"]

            else:
                # This should be experimented with.
                if interpreter.code_output_sender == "user":
                    if message["content"].strip() == "":
                        content = interpreter.empty_code_output_template
                    else:
                        content = interpreter.code_output_template.replace(
                            "{content}", message["content"]
                        )

                    new_message["role"] = "user"
                    new_message["content"] = content
                elif interpreter.code_output_sender == "assistant":
                    new_message["role"] = "assistant"
                    new_message["content"] = (
                        "\n```output\n" + message["content"] + "\n```"
                    )

        elif message["type"] == "image":
            if message.get("format") == "description":
                # Convert computer role to user for Mistral models (Mistral only accepts: system, user, assistant, tool)
                role = message["role"]
                if role == "computer" and interpreter and interpreter.llm:
                    model = interpreter.llm.model
                    if model and ("mistral" in model.lower() or "devstral" in model.lower()):
                        role = "user"
                new_message["role"] = role
                new_message["content"] = message["content"]
                if role == "user":
                    ts = _user_ts(message, messages)
                    if ts is not None:
                        new_message["content"] = f"[{ts}] " + new_message["content"]
            else:
                if vision == False:
                    # If no vision, we only support the format of "description"
                    continue

                if "base64" in message["format"]:
                    # Extract the extension from the format, default to 'png' if not specified
                    if "." in message["format"]:
                        extension = message["format"].split(".")[-1]
                    else:
                        extension = "png"

                    encoded_string = message["content"]

                elif message["format"] == "path":
                    # Convert to base64
                    image_path = message["content"]
                    extension = image_path.split(".")[-1]

                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )

                else:
                    # Probably would be better to move this to a validation pass
                    # Near core, through the whole messages object
                    if "format" not in message:
                        raise Exception("Format of the image is not specified.")
                    else:
                        raise Exception(
                            f"Unrecognized image format: {message['format']}"
                        )

                content = f"data:image/{extension};base64,{encoded_string}"

                if shrink_images:
                    # Shrink to less than 5mb

                    # Calculate size
                    content_size_bytes = sys.getsizeof(str(content))

                    # Convert the size to MB
                    content_size_mb = content_size_bytes / (1024 * 1024)

                    # If the content size is greater than 5 MB, resize the image
                    if content_size_mb > 5:
                        # Decode the base64 image
                        img_data = base64.b64decode(encoded_string)
                        img = Image.open(io.BytesIO(img_data))

                        # Run in a loop to make SURE it's less than 5mb
                        for _ in range(10):
                            # Calculate the scale factor needed to reduce the image size to 4.9 MB
                            scale_factor = (4.9 / content_size_mb) ** 0.5

                            # Calculate the new dimensions
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)

                            # Resize the image
                            img = img.resize((new_width, new_height))

                            # Convert the image back to base64
                            buffered = io.BytesIO()
                            img.save(buffered, format=extension)
                            encoded_string = base64.b64encode(
                                buffered.getvalue()
                            ).decode("utf-8")

                            # Set the content
                            content = f"data:image/{extension};base64,{encoded_string}"

                            # Recalculate the size of the content in bytes
                            content_size_bytes = sys.getsizeof(str(content))

                            # Convert the size to MB
                            content_size_mb = content_size_bytes / (1024 * 1024)

                            if content_size_mb < 5:
                                break
                        else:
                            print(
                                "Attempted to shrink the image but failed. Sending to the LLM anyway."
                            )

                new_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": content, "detail": "low"},
                        }
                    ],
                }

                if message["role"] == "computer":
                    new_message["content"].append(
                        {
                            "type": "text",
                            "text": "This image is the result of the last tool output. What does it mean / are we done?",
                        }
                    )
                if message.get("format") == "path":
                    if any(
                        content.get("type") == "text"
                        for content in new_message["content"]
                    ):
                        for content in new_message["content"]:
                            if content.get("type") == "text":
                                content["text"] += (
                                    "\nThis image is at this path: "
                                    + message["content"]
                                )
                    else:
                        new_message["content"].append(
                            {
                                "type": "text",
                                "text": "This image is at this path: "
                                + message["content"],
                            }
                        )

                if message.get("role") == "user":
                    ts = _user_ts(message, messages)
                    if ts is not None:
                        new_message["content"].insert(
                            0, {"type": "text", "text": f"[{ts}] "}
                        )

        elif message["type"] == "file":
            ts = _user_ts(message, messages)
            content = message["content"]
            if ts is not None:
                content = f"[{ts}] " + content
            new_message = {"role": "user", "content": content}
        elif message["type"] == "error":
            print("Ignoring 'type' == 'error' messages.")
            continue
        else:
            raise Exception(f"Unable to convert this message type: {message}")

        if isinstance(new_message["content"], str):
            new_message["content"] = new_message["content"].strip()

        new_messages.append(new_message)

    if function_calling == False:
        combined_messages = []
        current_role = None
        current_content = []

        for message in new_messages:
            if isinstance(message["content"], str):
                if current_role is None:
                    current_role = message["role"]
                    current_content.append(message["content"])
                elif current_role == message["role"]:
                    current_content.append(message["content"])
                else:
                    combined_messages.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                    current_role = message["role"]
                    current_content = [message["content"]]
            else:
                if current_content:
                    combined_messages.append(
                        {"role": current_role, "content": "\n".join(current_content)}
                    )
                    current_content = []
                combined_messages.append(message)

        # Add the last message
        if current_content:
            combined_messages.append(
                {"role": current_role, "content": " ".join(current_content)}
            )

        new_messages = combined_messages

    return new_messages
