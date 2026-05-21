import os
import platform
import shutil


def resolve_powershell_executable():
    """
    Return a path to PowerShell. Never falls back to another shell.

    Resolution order:
      1. INTERPRETER_POWERSHELL — full path (optional override)
      2. Windows: powershell.exe on PATH
      3. Unix: pwsh on PATH
    """
    env = os.environ.get("INTERPRETER_POWERSHELL", "").strip()
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"INTERPRETER_POWERSHELL is set but not a file: {env!r}")
        return env

    if platform.system() == "Windows":
        path = shutil.which("powershell") or shutil.which("powershell.exe")
        if path:
            return path
        raise FileNotFoundError(
            "PowerShell not found on Windows. Install PowerShell or set INTERPRETER_POWERSHELL."
        )

    path = shutil.which("pwsh")
    if path:
        return path

    raise FileNotFoundError(
        "PowerShell (pwsh) not found. Install PowerShell Core or set INTERPRETER_POWERSHELL."
    )
