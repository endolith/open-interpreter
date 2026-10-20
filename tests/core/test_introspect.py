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

    def test_instance_attributes_shown_as_instance(self):
        """Instance __dict__ entries render with [Instance] provenance."""
        obj = _Child()
        obj.label = "hello"
        out = _capture(obj, "label")
        self.assertIn("label:", out)
        self.assertIn("[Instance]", out)

    def test_bare_function_renders_single_entry(self):
        """Passing a function itself renders one signature line."""
        out = _capture(len)
        self.assertIn("len(", out)
        self.assertIn("1 matched, 1 shown", out)

    def test_pattern_list_matches_any(self):
        """A list pattern matches members matching any entry."""
        out = _capture(_Child, ["child_method", "base_method"])
        self.assertIn("child_method(", out)
        self.assertIn("base_method(", out)
        self.assertNotIn("size", out)

    def test_doc_true_appends_first_line(self):
        """doc=True appends the first docstring line to each member line."""
        out = _capture(_Child, "child_method", doc=True)
        self.assertIn("Child method first line.", out)

    def test_doc_full_includes_body(self):
        """doc='full' includes the docstring body, not just the first line."""
        out = _capture(_Child, "base_method", doc="full")
        self.assertIn("Base method body paragraph.", out)

    def test_src_single_match_prints_source(self):
        """src=True with exactly one match prints that member's source."""
        out = _capture(_Child, "child_method", src=True)
        self.assertIn("def child_method", out)

    def test_src_multiple_matches_asks_to_narrow(self):
        """src=True with several matches refuses and asks for narrowing."""
        out = _capture(_Child, "", src=True)
        self.assertIn("needs exactly one match", out)

    def test_cap_truncates_with_notice(self):
        """Output over cap is cut and the footer says truncated with counts."""
        out = _capture(_Child, "", cap=60)
        self.assertIn("truncated", out)
        self.assertIn("matched", out)

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


if __name__ == "__main__":
    unittest.main()
