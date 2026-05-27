from ..render_message import render_message


def assemble_system_message(interpreter):
    """Build the rendered system prompt before tool/text-mode appendices (matches respond.py)."""
    system_message = interpreter.system_message

    for language in interpreter.terminal.languages:
        if hasattr(language, "system_message"):
            system_message += "\n\n" + language.system_message

    if interpreter.custom_instructions:
        system_message += (
            "\n\n## User's Custom Instructions\n\n" + interpreter.custom_instructions
        )

    server_request_system = getattr(interpreter, "_server_request_system", None)
    if server_request_system:
        system_message += "\n\n## Client system prompt\n\n" + server_request_system

    if interpreter.toolbox.import_toolbox_api:
        if interpreter.toolbox.system_message not in system_message:
            system_message = system_message + "\n\n" + interpreter.toolbox.system_message

    return render_message(interpreter, system_message)
