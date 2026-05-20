import os
import platform
import shutil


def resolve_bash_executable():
    """
    Return a path to GNU bash. Never uses the user's login shell (fish/zsh).
    On Windows, searches INTERPRETER_BASH, PATH, then common Git Bash locations.
    """
    env = os.environ.get("INTERPRETER_BASH", "").strip()
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"INTERPRETER_BASH is set but not a file: {env!r}")
        return env

    which = shutil.which("bash")
    if which:
        return which

    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "bash not found on Windows. Install Git for Windows or WSL, add bash to PATH, "
            "or set INTERPRETER_BASH to the full path of bash.exe."
        )

    if os.path.isfile("/bin/bash"):
        return "/bin/bash"

    raise FileNotFoundError(
        "bash not found. Install bash or set INTERPRETER_BASH to its full path."
    )
