import unittest


class TestGetDisplaysResolvesItsDependency(unittest.TestCase):
    """get_displays must reach screeninfo rather than dying on a missing name.

    screeninfo is imported lazily at module scope as `screeninfo`, but this
    function called a bare `get_monitors()`, which was never imported anywhere.
    That made it raise NameError on every call -- and it is reachable from
    view(), so `toolbox.display.view()` failed before it ever tried to open
    anything. The NameError also hid the real condition: on a machine with no
    display, screeninfo raises ScreenInfoError, and that is the error a caller
    actually needs to see.
    """

    def test_get_displays_does_not_raise_name_error(self):
        """No display required: the assertion is about which error can appear.

        With no monitor hardware the call legitimately fails inside screeninfo.
        What must never happen is a NameError for a name that was never
        imported, which is the regression this pins.
        """
        from interpreter.core.toolbox.display.display import get_displays

        try:
            monitors = get_displays()
        except NameError as exc:
            self.fail(f"get_displays() raised NameError: {exc}")
        except Exception:
            # Reaching screeninfo and failing there (no display in CI) is the
            # correct behaviour.
            return
        self.assertGreaterEqual(len(monitors), 0)
