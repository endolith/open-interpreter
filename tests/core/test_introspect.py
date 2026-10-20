import contextlib
import io
import unittest

from interpreter.core.utils.introspect import introspect


def _capture(*args, **kwargs):
    """Run introspect and return what it printed (it must return None)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = introspect(*args, **kwargs)
    assert result is None
    return buf.getvalue()


class _Base:
    def base_method(self, x, y=2):
        """Base method first line.

        Base method body paragraph.
        """


class _Child(_Base):
    def child_method(self, name, token="abc"):
        """Child method first line."""

    size = 5


def _module_level_function(x):
    """Module-level function first line."""
    return x


class TestIntrospectModule(unittest.TestCase):
    def test_module_members_show_signatures_and_provenance(self):
        """Members of a module render as name(sig) with [module] provenance."""
        import math

        out = _capture(math, "sin")
        self.assertIn("sin(", out)
        self.assertIn("[module]", out)
        self.assertIn("matched", out)

    def test_inherited_methods_marked_with_base(self):
        """Methods from a base class show [Inherited: Base], own show [Child]."""
        out = _capture(_Child, "")
        self.assertIn("[_Child]", out)
        self.assertIn("[Inherited: _Base]", out)
        self.assertIn("base_method(x, y=2)", out)

    def test_no_self_in_method_signatures(self):
        """Methods on a class render without the leading self parameter."""
        out = _capture(_Child, "child_method")
        self.assertIn("child_method(name,", out)
        self.assertNotIn("self", out)
        out = _capture([], "append")
        self.assertIn("append(", out)
        self.assertNotIn("self", out)

    def test_instance_attributes_shown_as_instance(self):
        """Instance __dict__ entries render with [Instance] provenance."""
        obj = _Child()
        obj.label = "hello"
        out = _capture(obj, "label")
        self.assertIn("label:", out)
        self.assertIn("[Instance]", out)

    def test_noncallable_shows_type(self):
        """A non-callable member renders as name: type [Owner]."""
        out = _capture(_Child, "size")
        self.assertIn("size: int", out)

    def test_pattern_list_matches_any(self):
        """A list of patterns matches members matching any entry."""
        out = _capture(_Child, ["child_method", "base_method"])
        self.assertIn("child_method(", out)
        self.assertIn("base_method(", out)
        self.assertNotIn("size", out)

    def test_pattern_is_a_regex(self):
        """Pattern is a regex: a dot is a wildcard, anchors are honoured."""
        out = _capture(_Child, "^base_method$")
        self.assertIn("base_method(", out)
        self.assertNotIn("child_method", out)

    def test_anchored_pattern_does_not_overmatch(self):
        """^name$ must not match a longer, unrelated member name."""
        out = _capture(_Child, "^child$")
        self.assertIn("no matches", out)

    def test_brief_appends_first_line_only(self):
        """brief=True appends the first docstring line, not the body."""
        out = _capture(_Child, "base_method", brief=True)
        self.assertIn("Base method first line.", out)
        self.assertNotIn("Base method body paragraph.", out)

    def test_doc_true_includes_full_docstring(self):
        """doc=True renders the whole docstring, body included."""
        out = _capture(_Child, "base_method", doc=True)
        self.assertIn("Base method body paragraph.", out)

    def test_function_src_prints_source(self):
        """src=True works on a plain function, not just class members."""
        out = _capture(_module_level_function, src=True)
        self.assertIn("def _module_level_function", out)

    def test_function_doc_true_includes_docstring(self):
        """doc=True works on a plain function (early-return path)."""
        out = _capture(_module_level_function, doc=True)
        self.assertIn("Module-level function first line.", out)

    def test_module_doc_shown_when_requested(self):
        """brief=True on a module surfaces the module's own docstring."""
        import math

        out = _capture(math, "sin", brief=True)
        self.assertIn("access to the mathematical functions", out)

    def test_src_multiple_matches_asks_to_narrow(self):
        """src=True with several matches refuses and asks for narrowing."""
        out = _capture(_Child, "", src=True)
        self.assertIn("needs exactly one match", out)

    def test_cap_truncates_with_count_and_advice(self):
        """Cap overflow drops tail lines and says how many, with a fix."""
        import os

        out = _capture(os, "", cap=300)
        self.assertIn("more not shown", out)
        self.assertIn("narrow the pattern", out)

    def test_secret_default_redacted(self):
        """Defaults of secret-named params render as *** instead of values."""

        def f(token="sk-live-abc123"):
            pass

        out = _capture(f)
        self.assertIn("token=***", out)
        self.assertNotIn("sk-live", out)

    def test_long_default_truncated(self):
        """Defaults with long reprs render as ... instead of the value."""

        def f(name="a-very-long-default-value-that-goes-on-and-on"):
            pass

        out = _capture(f)
        self.assertIn("name=...", out)
        self.assertNotIn("a-very-long", out)

    def test_no_matches_suggests_close_names(self):
        """Zero matches prints a no-matches message with a did-you-mean hint."""
        out = _capture(_Child, "chlid_method")
        self.assertIn("no matches", out)
        self.assertIn("child_method", out)

    def test_no_matches_invalid_regex_says_so_and_does_not_dump_all(self):
        """An invalid regex is reported as such, never as an empty match-all."""
        out = _capture(_Child, "(")
        self.assertIn("invalid regex", out)
        self.assertNotIn("child_method(", out)

    def test_no_matches_private_name_explains_hidden(self):
        """A hidden private/dunder name says it exists but is hidden."""

        class F:
            def __init__(self):
                self._secret = 1

        out = _capture(F(), "secret")
        self.assertIn("private/dunder", out)
        self.assertIn("_secret", out)
        self.assertIn("private=True", out)

    def test_private_flag_lists_dunders_and_privates(self):
        """private=True exposes _private/__dunder__ members, which are
        otherwise unreachable (e.g. __enter__/__iter__ on a context class)."""

        class Ctx:
            def __init__(self):
                self._hidden = 1

            def __enter__(self):
                return self

        out = _capture(Ctx(), "^__enter__$", private=True)
        self.assertIn("__enter__(", out)
        out = _capture(Ctx(), "_hidden", private=True)
        self.assertIn("_hidden", out)

    def test_private_hidden_message_is_plural_aware(self):
        """Multiple hidden matches use plural phrasing, not 'it exists'."""

        class C:
            def __init__(self):
                self._a = 1
                self._b = 2

        out = _capture(C(), "_a|_b")
        self.assertIn("they exist but are", out)
        self.assertIn("Reach them", out)

    def test_src_with_zero_matches_says_no_matches_not_narrow(self):
        """src=True with no match must not advise narrowing (nothing to
        narrow) — it reports no matches like a plain listing would."""
        out = _capture(_Child, "^nonexistent$", src=True)
        self.assertIn("no matches", out)
        self.assertNotIn("needs exactly one match", out)

    def test_descriptors_never_execute(self):
        """Reading members must not trigger properties or descriptors."""

        class Exploding:
            @property
            def boom(self):
                raise AssertionError("descriptor executed!")

        out = _capture(Exploding(), "")
        self.assertIn("boom", out)

    def test_never_raises_on_garbage(self):
        """Weird objects, patterns, and caps print a message, never raise."""
        _capture(object(), "")
        _capture(object(), "[unclosed")
        _capture(object(), "", cap="nonsense")


class TestIntrospectInstallCode(unittest.TestCase):
    def test_install_loads_by_path_without_importing_package(self):
        """The kernel snippet must load introspect.py by absolute path (the
        package __init__ constructs OpenInterpreter, so `import interpreter`
        would slow every kernel boot), and must bind the bare global."""
        import os

        from interpreter.core.terminal.languages.jupyter_language import (
            _build_introspect_install_code,
        )

        code = _build_introspect_install_code()
        self.assertIsNotNone(code)
        self.assertIn("spec_from_file_location", code)
        self.assertIn("introspect = _oi_m.introspect", code)
        self.assertNotIn("import interpreter", code)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.assertIn(
            repr(
                os.path.join(
                    repo_root, "interpreter", "core", "utils", "introspect.py"
                )
            ),
            code,
        )


if __name__ == "__main__":
    unittest.main()
