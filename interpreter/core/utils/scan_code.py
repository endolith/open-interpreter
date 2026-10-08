import contextlib
import os
import shutil
import subprocess

from .temporary_file import cleanup_temporary_file, create_temporary_file

try:
    from yaspin import yaspin
    from yaspin.spinners import Spinners
except ImportError:
    yaspin = None
    Spinners = None


def scan_code(code, language, interpreter):
    """
    Scan code with semgrep
    """
    language_class = interpreter.computer.terminal.get_language(language)

    if shutil.which("semgrep") is None:
        # Missing dependency: say so instead of running nothing silently (#373).
        print(f"Could not scan {language} code. Have you installed 'semgrep'?")
        print("")  # <- Aesthetic choice
        return

    try:
        temp_file = create_temporary_file(
            code, language_class.file_extension, verbose=interpreter.verbose
        )
    except Exception as e:
        # The scan is optional: a temp-file failure skips it instead of
        # aborting the turn with the raw error (#374).
        if interpreter.verbose:
            print(f"Could not create temporary file for scanning: {e}")
        return

    if temp_file is None:
        # create_temporary_file swallows its own errors and returns None
        # instead of raising, so the except above never fires: skip the
        # optional scan the same way rather than crashing in dirname below.
        return

    temp_path = os.path.dirname(temp_file)
    file_name = os.path.basename(temp_file)

    if interpreter.verbose:
        print(f"Scanning {language} code in {file_name}")
        print("---")

    # Run semgrep
    try:
        # No shell: the command is an arg list run with cwd=, so a temp dir
        # containing spaces works, and a binary that vanishes mid-run raises
        # FileNotFoundError instead of failing silently inside a shell
        # (#375, #373).
        if yaspin is not None:
            spinner = yaspin(text="  Scanning code...").green.right.binary
        else:
            spinner = contextlib.nullcontext()
        with spinner:
            scan = subprocess.run(
                [
                    "semgrep",
                    "scan",
                    "--config",
                    "auto",
                    "--quiet",
                    "--error",
                    file_name,
                ],
                cwd=temp_path,
            )

        if scan.returncode == 0:
            language_name = language_class.name
            print(
                f"  {'Code Scanner: ' if interpreter.safe_mode == 'auto' else ''}No issues were found in this {language_name} code."
            )
            print("")

        # TODO: it would be great if we could capture any vulnerabilities identified by semgrep
        # and add them to the conversation history

    except Exception as e:
        print(f"Could not scan {language} code. Have you installed 'semgrep'?")
        print(e)
        print("")  # <- Aesthetic choice

    cleanup_temporary_file(temp_file, verbose=interpreter.verbose)
