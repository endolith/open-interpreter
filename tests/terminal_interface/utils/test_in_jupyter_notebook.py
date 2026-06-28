import sys
from unittest import mock

from interpreter.terminal_interface.utils.in_jupyter_notebook import in_jupyter_notebook


def test_in_jupyter_notebook_false_without_ipython():
    with mock.patch.dict("sys.modules", {"IPython": None}):
        assert in_jupyter_notebook() is False


def test_in_jupyter_notebook_true_with_kernel_app():
    fake_ipython = mock.Mock()
    fake_ipython.return_value.config = {"IPKernelApp": {}}
    with mock.patch.dict(
        "sys.modules",
        {"IPython": mock.Mock(get_ipython=fake_ipython)},
    ):
        assert in_jupyter_notebook() is True
