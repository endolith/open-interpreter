import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "installers" / "oi-linux-installer.sh"


def _pip_install_line() -> str:
    """Return the installer line that runs the pyenv-wrapped pip install."""
    for line in INSTALLER.read_text(encoding="utf-8").splitlines():
        if "pip install open-interpreter" in line and "pyenv_root" in line:
            return line
    raise AssertionError("installer no longer contains a pyenv pip install line")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_linux_installer_pins_the_selected_python_for_pip(tmp_path):
    """The pip install must see the installer's python_version, not pyenv's default.

    `pyenv shell` only prints code to eval, so a script that calls it directly
    leaves the next `pyenv exec` on the global default interpreter. Running the
    real pip line with a stub pyenv reproduces that: the stub reports the version
    pyenv would use, which must be the 3.11.7 the script just installed.
    """
    fake_pyenv = tmp_path / "pyenv"
    fake_pyenv.write_text('#!/usr/bin/env bash\necho "${PYENV_VERSION:-unset}"\n', encoding="utf-8")
    fake_pyenv.chmod(0o755)
    line = _pip_install_line()
    command = f'pyenv_root={shlex.quote(str(fake_pyenv))}; python_version="3.11.7"; {line}'
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3.11.7"
