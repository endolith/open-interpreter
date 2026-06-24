from interpreter.terminal_interface.utils.check_for_package import check_for_package


def test_check_for_package_returns_true_for_installed():
    assert check_for_package("json") is True


def test_check_for_package_returns_false_for_missing():
    assert check_for_package("definitely_not_a_real_package_xyz") is False
