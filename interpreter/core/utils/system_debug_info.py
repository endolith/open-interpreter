import platform
import subprocess

from importlib.metadata import version, PackageNotFoundError
import psutil


def get_python_version():
    """Return the version string of the running Python interpreter."""
    return platform.python_version()


def get_pip_version():
    """Return the installed pip version, or the error message if it fails."""
    try:
        pip_version = subprocess.check_output(["pip", "--version"]).decode().split()[1]
    except Exception as e:
        pip_version = str(e)
    return pip_version


def get_oi_version():
    """Return the Open Interpreter version from `interpreter --version` and the
    installed package metadata as a (cmd_output, pkg_version) tuple."""
    try:
        oi_version_cmd = subprocess.check_output(
            ["interpreter", "--version"], text=True
        )
    except Exception as e:
        oi_version_cmd = str(e)
    try:
        pkg_ver = version("open-interpreter")
    except PackageNotFoundError:
        pkg_ver = None
    oi_version = oi_version_cmd, pkg_ver
    return oi_version


def get_os_version():
    """Return a human-readable OS name and version string."""
    return platform.platform()


def get_cpu_info():
    """Return the CPU model name reported by the platform."""
    return platform.processor()


def get_ram_info():
    """Return a human-readable summary of total, used, and free RAM in GB."""
    vm = psutil.virtual_memory()
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    total_ram_gb = vm.total / (1024**3)
    return f"{total_ram_gb:.2f} GB, used: {used_ram_gb:.2f}, free: {free_ram_gb:.2f}"


def interpreter_info(interpreter):
    """Render a debug summary of the interpreter's settings and messages."""
    try:
        if interpreter.offline and interpreter.llm.api_base:
            try:
                curl = subprocess.check_output(f"curl {interpreter.llm.api_base}")
            except Exception as e:
                curl = str(e)
        else:
            curl = "Not local"

        messages_to_display = []
        for message in interpreter.messages:
            message = str(message.copy())
            try:
                if len(message) > 2000:
                    message = message[:1000]
            except Exception as e:
                print(str(e), "for message:", message)
            messages_to_display.append(message)

        return f"""

        # Interpreter Info
        
        Vision: {interpreter.llm.supports_vision}
        Model: {interpreter.llm.model}
        Function calling: {interpreter.llm.supports_functions}
        Context window: {interpreter.llm.context_window}
        Max tokens: {interpreter.llm.max_tokens}
        Computer API: {interpreter.computer.import_computer_api}

        Auto run: {interpreter.auto_run}
        API base: {interpreter.llm.api_base}
        Offline: {interpreter.offline}

        Curl output: {curl}

        # Messages

        System Message: {interpreter.system_message}

        """ + "\n\n".join(
            [str(m) for m in messages_to_display]
        )
    except:
        return "Error, couldn't get interpreter info"


def system_info(interpreter):
    """Print a debug report of the environment and the given interpreter."""
    oi_version = get_oi_version()
    print(
        f"""
        Python Version: {get_python_version()}
        Pip Version: {get_pip_version()}
        Open-interpreter Version: cmd: {oi_version[0]}, pkg: {oi_version[1]}
        OS Version and Architecture: {get_os_version()}
        CPU Info: {get_cpu_info()}
        RAM Info: {get_ram_info()}
        {interpreter_info(interpreter)}
    """
    )
