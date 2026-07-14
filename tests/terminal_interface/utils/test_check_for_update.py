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
