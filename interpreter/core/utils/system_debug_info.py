import platform
import subprocess
from importlib.metadata import PackageNotFoundError, distributions, version

import psutil
import toml
from rich import print as rich_print
from rich.markdown import Markdown

from ..terminal.base_language import format_execute_language_description
from ..tools.file_edit import EDIT_LANGUAGES
from .assemble_system_message import assemble_system_message


def get_python_version():
    return platform.python_version()


def get_pip_version():
    try:
        pip_version = subprocess.check_output(
            ["pip", "--version"]).decode().split()[1]
    except Exception as e:
        pip_version = str(e)
    return pip_version


def get_oi_version():
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
    oi_version = oi_version_cmd.strip(), pkg_ver.strip()
    return oi_version


def get_os_version():
    return platform.platform()


def get_cpu_info():
    return platform.processor()


def get_ram_info():
    vm = psutil.virtual_memory()
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    total_ram_gb = vm.total / (1024**3)
    return f"{total_ram_gb:.2f} GB, used: {used_ram_gb:.2f}, free: {free_ram_gb:.2f}"


def get_package_mismatches(file_path="pyproject.toml"):
    with open(file_path, "r") as file:
        pyproject = toml.load(file)
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    dev_dependencies = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]
    dependencies.update(dev_dependencies)

    installed_packages = {
        dist.metadata["Name"].lower(): dist.version
        for dist in distributions()
    }
    mismatches = []
    for package, version_info in dependencies.items():
        if isinstance(version_info, dict):
            version_info = version_info["version"]
        installed_version = installed_packages.get(package)
        if installed_version and version_info.startswith("^"):
            expected_version = version_info[1:]
            if not installed_version.startswith(expected_version):
                mismatches.append(
                    f"\t  {package}: Mismatch, pyproject.toml={expected_version}, pip={installed_version}"
                )
        else:
            mismatches.append(f"\t  {package}: Not found in pip list")

    return "\n" + "\n".join(mismatches)


def _format_info_sections(sections):
    parts = []
    for title, body in sections:
        parts.append(f"### {title}\n\n----\n\n{body}\n\n----")
    return "\n\n".join(parts)


def _llm_prompt_sections_for_info(interpreter):
    """Full system prompt and tool metadata as sent (or would be sent) to the model."""
    from ..llm.run_tool_calling_llm import edit_tool_schema, view_image_tool_schema

    base = assemble_system_message(interpreter)
    sections = []

    if interpreter.llm.supports_functions:
        system_content = base
        if interpreter.llm.tool_calling_instructions:
            system_content += "\n" + interpreter.llm.tool_calling_instructions
        sections.append(
            ("System Message (tool-calling mode)", system_content or "(empty)")
        )

        languages = interpreter.terminal.languages
        execute_desc = format_execute_language_description(languages)
        lang_enum = ", ".join(sorted(lang.name.lower() for lang in languages))
        sections.append(
            (
                "Execute tool",
                f"**Languages (enum):** {lang_enum}\n\n{execute_desc}",
            )
        )

        edit_fn = edit_tool_schema["function"]
        edit_enum = ", ".join(sorted(EDIT_LANGUAGES))
        sections.append(
            (
                "Edit tool",
                f"**Languages (enum):** {edit_enum}\n\n{edit_fn['description']}",
            )
        )

        if interpreter.llm.supports_vision is True:
            view_fn = view_image_tool_schema["function"]
            sections.append(("View image tool", view_fn["description"]))
    else:
        system_content = base
        if interpreter.llm.execution_instructions:
            system_content += "\n" + interpreter.llm.execution_instructions
        sections.append(
            ("System Message (text / markdown mode)", system_content or "(empty)")
        )

    return sections


def interpreter_info(interpreter):
    try:
        if interpreter.offline and interpreter.llm.api_base:
            try:
                curl = subprocess.check_output(
                    f"curl {interpreter.llm.api_base}")
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

        prompt_sections = _format_info_sections(_llm_prompt_sections_for_info(interpreter))

        return f"""
## Interpreter Info

- Vision: {interpreter.llm.supports_vision}
- Model: {interpreter.llm.model}
- Function calling: {interpreter.llm.supports_functions}
- Context window: {interpreter.llm.context_window}
- Max tokens: {interpreter.llm.max_tokens}
- Toolbox API: {interpreter.toolbox.import_toolbox_api}

- Auto run: {interpreter.auto_run}
- API base: {interpreter.llm.api_base}
- Offline: {interpreter.offline}

- Curl output: {curl}

## LLM prompt (system + tools)

{prompt_sections}

## Conversation Messages

""" + "\n\n".join(
            [str(m) for m in messages_to_display]
        )
    except:
        return "Error, couldn't get interpreter info"


def system_info(interpreter):
    oi_version = get_oi_version()
    markdown_content = f"""
## System Debug Info

- Python Version: {get_python_version()}
- Pip Version: {get_pip_version()}
- Open-interpreter Version:
  - cmd: {oi_version[0]}
  - pkg: {oi_version[1]}
- OS Version and Architecture: {get_os_version()}
- CPU Info: {get_cpu_info()}
- RAM Info: {get_ram_info()}
{interpreter_info(interpreter)}
"""
    rich_print(Markdown(markdown_content))

    # Removed the following, as it causes `FileNotFoundError: [Errno 2] No such file or directory: 'pyproject.toml'`` on prod
    # (i think it works on dev, but on prod the pyproject.toml will not be in the cwd. might not be accessible at all)
    # Package Version Mismatches:
    # {get_package_mismatches()}
