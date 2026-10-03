from unittest import mock

import interpreter.terminal_interface.utils.check_for_update as check_for_update


def test_check_for_update_true_when_latest_is_newer(monkeypatch):
    """check_for_update returns True when PyPI reports a newer version."""

    class _Response:
        def json(self):
            return {"info": {"version": "99.0.0"}}

    monkeypatch.setattr(
        check_for_update.requests, "get", lambda *args, **kwargs: _Response()
    )

    assert check_for_update.check_for_update() is True


def test_check_for_update_false_when_latest_is_not_newer(monkeypatch):
    """check_for_update returns False when PyPI reports an older/same version."""

    class _Response:
        def json(self):
            return {"info": {"version": "0.0.1"}}

    monkeypatch.setattr(
        check_for_update.requests, "get", lambda *args, **kwargs: _Response()
    )

    assert check_for_update.check_for_update() is False


def test_check_for_update_queries_the_pypi_json_endpoint(monkeypatch):
    """check_for_update reads the project's PyPI JSON endpoint.

    The URL identifies the package to compare against; a wrong or wrapped URL
    would compare against the wrong project (or fail to parse) while the
    version comparison still ran on whatever came back.
    """
    calls = []

    class _Response:
        def json(self):
            return {"info": {"version": "99.0.0"}}

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        return _Response()

    monkeypatch.setattr(check_for_update.requests, "get", fake_get)

    check_for_update.check_for_update()

    assert calls == ["https://pypi.org/pypi/open-interpreter/json"]


def test_check_for_update_compares_against_the_installed_version(monkeypatch):
    """The current version is read via version("open-interpreter").

    If the installed version is looked up under the wrong name, the comparison
    is against a different (or missing) value; the spy records the exact name
    passed to importlib.metadata.version.
    """
    seen = {}

    def fake_version(name):
        seen["name"] = name
        return "0.5.0"

    monkeypatch.setattr(
        check_for_update.requests,
        "get",
        lambda *a, **k: type("R", (), {"json": lambda self: {"info": {"version": "1.0.0"}}})(),
    )
    monkeypatch.setattr(check_for_update, "version", fake_version)

    assert check_for_update.check_for_update() is True
    assert seen["name"] == "open-interpreter"
