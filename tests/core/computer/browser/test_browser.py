"""Characterization tests for ``computer.browser``.

selenium and webdriver-manager are mocked so no browser or driver is ever
launched; the tests verify the driver lifecycle, API search calls, and page
interaction behavior.
"""

from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.browser import browser as browser_mod
from interpreter.core.computer.browser.browser import Browser


def _make_browser(computer=None):
    return Browser(
        computer
        or SimpleNamespace(
            api_base="http://api:8000",
            interpreter=SimpleNamespace(
                llm=SimpleNamespace(model="gpt-4"),
            ),
            ai=SimpleNamespace(chat=mock.Mock(return_value="analysis")),
        )
    )


def _driver_mock():
    driver = mock.Mock()
    driver.page_source = "<html><body>hi</body></html>"
    return driver


def test_driver_is_set_up_lazily():
    """Browser.driver spins up the driver on first access only."""
    browser = _make_browser()
    with mock.patch.object(
        browser, "setup", side_effect=lambda headless: setattr(browser, "_driver", "driver")
    ) as setup:
        assert browser.driver == "driver"
        assert browser.driver == "driver"
    setup.assert_called_once_with(False)


def test_driver_setter_stores_value():
    """Browser.driver can be assigned directly without triggering setup."""
    browser = _make_browser()
    driver = mock.Mock()
    browser.driver = driver
    assert browser.driver is driver


def test_search_queries_browser_api():
    """Browser.search() GETs /browser/search on the computer's API base."""
    browser = _make_browser()
    response = SimpleNamespace(json=lambda: {"result": "the answer"})
    with mock.patch.object(browser_mod.requests, "get", return_value=response) as get:
        result = browser.search("q")
    get.assert_called_once_with("http://api:8000/browser/search", params={"query": "q"})
    assert result == "the answer"


def test_fast_search_parallelizes_api_call_and_google():
    """Browser.fast_search() runs the API request and the Google search
    concurrently: the request must still be in flight when search_google runs,
    so a sequential implementation (which would let the request finish first)
    fails."""
    import threading

    browser = _make_browser()
    request_started = threading.Event()
    let_request_finish = threading.Event()
    request_released_by_google = threading.Event()
    response = SimpleNamespace(json=lambda: {"result": "the answer"})

    def fake_get(*args, **kwargs):
        request_started.set()
        # Only record a release when the Google search let us finish before the
        # timeout; a sequential implementation would time out here.
        if let_request_finish.wait(timeout=5):
            request_released_by_google.set()
        return response

    def fake_search_google(query, delays=False):
        assert request_started.wait(timeout=5)
        assert not let_request_finish.is_set()
        let_request_finish.set()

    with mock.patch.object(browser_mod.requests, "get", side_effect=fake_get):
        with mock.patch.object(browser, "search_google", side_effect=fake_search_google) as google:
            result = browser.fast_search("q")

    google.assert_called_once_with("q", delays=False)
    assert request_released_by_google.is_set()
    assert result == "the answer"


def test_setup_launches_headless_chrome():
    """Browser.setup(headless=True) builds a headless Chrome driver."""
    browser = _make_browser()
    options = mock.Mock()
    with mock.patch.object(browser_mod.ChromeDriverManager, "install", return_value="driverpath"):
        with mock.patch.object(browser_mod, "Service", return_value="service"):
            with mock.patch.object(browser_mod.webdriver, "ChromeOptions", return_value=options):
                with mock.patch.object(browser_mod.webdriver, "Chrome", return_value="driver") as chrome:
                    browser.setup(True)

    options.add_argument.assert_any_call("--headless")
    options.add_argument.assert_any_call("--disable-gpu")
    options.add_argument.assert_any_call("--no-sandbox")
    chrome.assert_called_once_with(service="service", options=options)
    assert browser._driver == "driver"


def test_setup_headless_false_skips_headless_flag():
    """Browser.setup(headless=False) adds no Chrome arguments at all."""
    browser = _make_browser()
    options = mock.Mock()
    with mock.patch.object(browser_mod.ChromeDriverManager, "install", return_value="driverpath"):
        with mock.patch.object(browser_mod, "Service", return_value="service"):
            with mock.patch.object(browser_mod.webdriver, "ChromeOptions", return_value=options):
                with mock.patch.object(browser_mod.webdriver, "Chrome", return_value="driver"):
                    browser.setup(False)
    options.add_argument.assert_not_called()


def test_setup_failure_leaves_no_driver():
    """Browser.setup() records a None driver when the driver can't be created."""
    browser = _make_browser()
    with mock.patch.object(
        browser_mod.ChromeDriverManager, "install", side_effect=Exception("boom")
    ):
        browser.setup(True)
    assert browser._driver is None


def test_go_to_url_navigates_and_waits():
    """Browser.go_to_url() calls driver.get() and pauses briefly."""
    browser = _make_browser()
    browser._driver = _driver_mock()
    with mock.patch.object(browser_mod.time, "sleep") as sleep:
        browser.go_to_url("https://example.com")
    browser._driver.get.assert_called_once_with("https://example.com")
    sleep.assert_called_once_with(1)


def test_search_google_types_query_into_perplexity():
    """Browser.search_google() opens Perplexity and types the query."""
    browser = _make_browser()
    driver = _driver_mock()
    body = mock.Mock()
    active = mock.Mock()
    driver.find_element.return_value = body
    driver.switch_to.active_element = active
    browser._driver = driver

    with mock.patch.object(browser_mod.time, "sleep") as sleep:
        browser.search_google("hello", delays=False)

    driver.get.assert_called_once_with("https://www.perplexity.ai")
    body.send_keys.assert_called_once_with(browser_mod.Keys.COMMAND + "k")
    active.send_keys.assert_any_call("hello")
    active.send_keys.assert_any_call(browser_mod.Keys.RETURN)
    sleep.assert_called_once_with(0.5)


def test_analyze_page_queries_ai_and_restores_model():
    """Browser.analyze_page() feeds the page text + elements to the computer's
    AI and restores the previous LLM model afterwards."""
    browser = _make_browser()
    browser._driver = _driver_mock()
    element_a = SimpleNamespace(text="A", get_attribute=lambda _: "<a>A</a>")
    element_b = SimpleNamespace(text="B", get_attribute=lambda _: "<button>B</button>")
    browser._driver.find_elements.return_value = [element_a, element_b]

    with mock.patch.object(browser_mod.html2text, "html2text", return_value="TEXT"):
        browser.analyze_page("find the price")

    assert browser.computer.interpreter.llm.model == "gpt-4"  # restored
    query = browser.computer.ai.chat.call_args[0][0]
    assert "find the price" in query
    assert "TEXT" in query
    assert "Interactive Elements" in query


def test_quit_closes_driver():
    """Browser.quit() calls driver.quit()."""
    browser = _make_browser()
    browser._driver = _driver_mock()
    browser.quit()
    browser._driver.quit.assert_called_once_with()
