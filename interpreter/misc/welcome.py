from interpreter import __version__


def welcome_message():
    print(
        f"""
Open Interpreter {__version__}
Copyright (C) 2024 Open Interpreter Team
Licensed under GNU AGPL v3.0

A modern command-line assistant.

Documentation: docs.openinterpreter.com
Run 'interpreter --help' for all options
"""
    )
