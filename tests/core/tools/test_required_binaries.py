import os
import unittest
from unittest import mock

from interpreter.core.tools import file_edit

# Languages whose external binary every CI runner must provide. These are the
# apt-available ones; yq (mikefarah build), comby, poke, R and pwsh each need a
# release download or an extra apt repository, so they stay optional and their
# tests skip.
REQUIRED = ("sed", "gawk", "patch", "jq")

# The override variable and candidate list each resolver is expected to hand to
# _resolve_binary. Pinned exactly rather than by prefix/substring: a prefix check
# passes on "INTERPRETER_PATCH_TYPO" and a substring check passes on
# "jq-does-not-exist", so both mutation checks were missed when written that way.
EXPECTED = {
    "sed": ("INTERPRETER_SED", ["sed"]),
    "gawk": ("INTERPRETER_GAWK", ["gawk", "awk"]),
    "patch": ("INTERPRETER_PATCH", ["patch"]),
    "jq": ("INTERPRETER_JQ", ["jq"]),
}

RESOLVERS = {
    "sed": file_edit._resolve_sed,
    "gawk": file_edit._resolve_gawk,
    "patch": file_edit._resolve_patch,
    "jq": file_edit._resolve_jq,
}


class TestRequiredEditBinariesResolve(unittest.TestCase):
    """Each required edit language's resolver must find a usable binary.

    The edit tests guard themselves with skipUnless(shutil.which(...)), so a
    resolver that stopped working -- a renamed env var, a bad candidate list --
    turned into tests that skip rather than fail. That is how the `patch` tests
    went untested for so long: nothing in the repo asserted the tool was
    reachable at all, only that it was optional.

    The resolution itself is what is exercised here, not the binary's presence:
    a missing binary raises FileNotFoundError, which is the correct behaviour
    and skips. What must never happen is a resolver that cannot find a binary
    that *is* on PATH.
    """

    def test_every_required_language_has_a_resolver(self):
        """A new required binary cannot be added without a resolver to check."""
        self.assertEqual(
            set(RESOLVERS), set(REQUIRED), "keep REQUIRED and RESOLVERS in step"
        )

    def _resolve_all(self):
        """Required binaries this machine can actually resolve.

        Driven off the resolvers rather than shutil.which, because they are not
        the same question: _resolve_gawk falls back to plain awk when gawk is
        absent, so `which("gawk")` being empty does not mean the tool is
        unreachable.
        """
        found = {}
        for name in REQUIRED:
            try:
                found[name] = RESOLVERS[name]()
            except FileNotFoundError:
                pass
        return found

    def test_resolver_returns_a_path_that_exists(self):
        """Where a required binary resolves, the path it returns is real."""
        found = self._resolve_all()
        self.assertTrue(
            found, f"none of {REQUIRED} could be resolved on this machine"
        )
        for name, resolved in found.items():
            with self.subTest(binary=name):
                self.assertTrue(
                    os.path.isfile(resolved),
                    f"{name} resolved to {resolved!r}, which is not a file",
                )

    def test_each_resolver_passes_its_own_override_env_var(self):
        """Each resolver hands _resolve_binary its own INTERPRETER_* name.

        Driven through a stub rather than by removing a binary, because the
        version that removed one asserted nothing on a machine where everything
        is installed -- which is every CI runner, so it would have been dead
        weight exactly where it was meant to help. Capturing the argument tests
        the wiring on any machine.

        This is the half the CI install step cannot see: the step proves a binary
        exists, while this proves the tool still knows how to reach it. Renaming
        an override silently stops a user's INTERPRETER_PATCH from working, and
        nothing else in the suite would notice.
        """
        for name in REQUIRED:
            with self.subTest(binary=name):
                captured = {}

                def _capture(env_var, candidates, _c=captured):
                    _c["env_var"] = env_var
                    _c["candidates"] = candidates
                    return "/nonexistent/for/test"

                with mock.patch.object(file_edit, "_resolve_binary", _capture):
                    RESOLVERS[name]()
                expected_env, expected_candidates = EXPECTED[name]
                self.assertEqual(
                    captured["env_var"],
                    expected_env,
                    f"{name} resolver's override variable was renamed or removed",
                )
                self.assertEqual(
                    captured["candidates"],
                    expected_candidates,
                    f"{name} resolver no longer looks for its own binary",
                )
