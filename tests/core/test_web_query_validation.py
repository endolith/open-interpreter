import unittest

from interpreter.core.toolbox.web.web import (
    Web,
    WebToolboxError,
    _reject_bad_country_code,
    _reject_bad_language_code,
    _reject_empty_query,
)


class TestEmptyQueryIsRejected(unittest.TestCase):
    """A blank query must fail on our side, before any request is spent.

    An empty query was passed through to the backend, which answered 400, and the
    failure surfaced as "check your API key and internet connection" -- sending
    the agent to fix credentials or its network when the fault was the argument
    it had just been handed. Verified before this change: `search('')` made a
    real request to Serper and returned that misleading error.
    """

    def _web(self):
        return Web(toolbox=None)

    def test_blank_query_is_rejected_without_a_request(self):
        """Empty and whitespace-only queries raise, naming the argument."""
        web = self._web()
        for bad in ("", "   ", "\t\n"):
            with self.subTest(query=bad):
                with self.assertRaises(WebToolboxError) as caught:
                    web.search(bad, backend="serper")
                self.assertIn("query is empty", str(caught.exception))

    def test_missing_query_is_rejected(self):
        """None is reported as a missing argument, not as an empty string."""
        with self.assertRaises(WebToolboxError) as caught:
            _reject_empty_query(None, "search")
        self.assertIn("query is required", str(caught.exception))

    def test_a_real_query_is_not_rejected(self):
        """The guard must not fire on ordinary input."""
        _reject_empty_query("how tall is the Eiffel tower", "search")

    def test_the_message_names_the_method_that_was_called(self):
        """answer() must not report a failure as coming from search()."""
        with self.assertRaises(WebToolboxError) as caught:
            _reject_empty_query("", "answer")
        self.assertIn("answer:", str(caught.exception))


class TestInvalidLocaleCodesAreRejected(unittest.TestCase):
    """A country or language code that is not real must be refused.

    "ZZ" is well-formed -- two letters -- so a shape check waves it through. The
    backend then either ignores it, returning localized results for somewhere
    the caller never asked about, or answers 400 and the agent is told its API
    key is wrong. Silently returning another country's results is the worse of
    the two, because nothing looks broken.
    """

    def test_fake_country_code_is_rejected(self):
        for bad in ("ZZ", "zz", "XX", "QQ"):
            with self.subTest(code=bad):
                with self.assertRaises(WebToolboxError) as caught:
                    _reject_bad_country_code(bad)
                self.assertIn("country_code", str(caught.exception))

    def test_real_country_codes_pass(self):
        for good in ("US", "gb", "FR", "DE", "JP", "BR", "IN"):
            with self.subTest(code=good):
                _reject_bad_country_code(good)

    def test_a_one_character_typo_offers_a_suggestion(self):
        """US -> U5 style typos are near misses worth naming."""
        with self.assertRaises(WebToolboxError) as caught:
            _reject_bad_country_code("U5")
        self.assertIn("did you mean", str(caught.exception))

    def test_full_locale_is_accepted_and_only_the_territory_is_checked(self):
        """en_GB should validate on GB, not fail as a malformed country."""
        _reject_bad_country_code("en_GB")

    def test_fake_language_code_is_rejected(self):
        for bad in ("xx", "zz", "qq"):
            with self.subTest(code=bad):
                with self.assertRaises(WebToolboxError) as caught:
                    _reject_bad_language_code(bad)
                self.assertIn("language_code", str(caught.exception))

    def test_real_language_codes_pass(self):
        for good in ("en", "es", "FR", "de", "pt", "ja"):
            with self.subTest(code=good):
                _reject_bad_language_code(good)

    def test_a_name_instead_of_a_code_is_rejected_clearly(self):
        """Passing 'United States' must not be read as a code."""
        with self.assertRaises(WebToolboxError) as caught:
            _reject_bad_country_code("United States")
        self.assertIn("two-letter", str(caught.exception))

    def test_empty_values_are_left_to_the_locale_defaults(self):
        """None and '' fall through to the system locale rather than erroring."""
        _reject_bad_country_code(None)
        _reject_bad_country_code("")
        _reject_bad_language_code(None)
        _reject_bad_language_code("")

    def test_search_rejects_a_bad_country_before_any_request(self):
        """End to end through search(), so the guard is actually wired up."""
        web = Web(toolbox=None)
        with self.assertRaises(WebToolboxError) as caught:
            web.search("test", country_code="ZZ", backend="serper")
        self.assertIn("country_code", str(caught.exception))


if __name__ == "__main__":
    unittest.main()