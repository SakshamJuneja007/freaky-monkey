import time

from agent_control.text_observation import BrowserTextReader, TextObservation, UniversalTextReader
from agent_control.types import Verdict
from agent_control.verifiers import verify_requested_text


def obs(text, source="uia_text_pattern", **kwargs):
    return TextObservation(text=text, source=source, target={"kind": "window", "handle": 7}, **kwargs)


def test_uia_text_pattern_is_preferred_over_value_pattern():
    calls = []

    class Reader:
        def __init__(self, value, source):
            self.value, self.source = value, source
        def read(self, target):
            calls.append(self.source)
            return obs(self.value, self.source)

    result = UniversalTextReader(
        native_reader=Reader("document text", "uia_text_pattern"),
        accessibility_reader=Reader("value", "uia_value_pattern"),
    ).read({"kind": "window", "handle": 7})
    assert result.text == "document text"
    assert result.source == "uia_text_pattern"
    assert calls == ["uia_text_pattern"]


def test_value_pattern_reader_can_supply_single_line_value():
    result = UniversalTextReader(
        native_reader=type("R", (), {"read": lambda self, target: obs("search value", "uia_value_pattern")})()
    ).read({"kind": "window", "handle": 7})
    assert result.text == "search value"
    assert result.source == "uia_value_pattern"


def test_universal_text_reader_returns_structured_observation():
    result = UniversalTextReader(
        native_reader=type("R", (), {"read": lambda self, target: obs("hello")})()
    ).read({"kind": "window", "handle": 7})
    assert isinstance(result, TextObservation)
    assert result.target["handle"] == 7
    assert result.fresh is True
    assert result.observed_at <= time.time()


def test_browser_reader_normalizes_existing_browser_semantic_observation():
    class Browser:
        def page_text(self):
            return "Café hello world"

    result = UniversalTextReader(browser_reader=BrowserTextReader(Browser())).read({"kind": "browser", "target": "textbox"})
    assert result.source == "browser_dom"
    assert result.text == "Café hello world"


def test_exact_and_contains_multiline_matching():
    multiline = obs("hello\r\nworld")
    assert verify_requested_text("hello\nworld", multiline, exact=True).verdict is Verdict.PASS
    assert verify_requested_text("hello", multiline).verdict is Verdict.PASS


def test_unicode_is_normalized_without_collapsing_meaningful_spaces():
    assert verify_requested_text("café", obs("cafe\u0301"), exact=True).verdict is Verdict.PASS
    assert verify_requested_text("hello world", obs("hello  world"), exact=True).verdict is Verdict.FAIL
    assert verify_requested_text("hello world", obs("hello  world"), exact=True, normalize_whitespace=True).verdict is Verdict.PASS


def test_mismatch_is_fail():
    result = verify_requested_text("hello", obs("goodbye"))
    assert result.verdict is Verdict.FAIL


def test_unavailable_is_unknown():
    result = verify_requested_text("hello", TextObservation("", "unavailable", ok=False, error="not exposed"))
    assert result.verdict is Verdict.UNKNOWN


def test_stale_observation_cannot_pass():
    stale = obs("hello", fresh=True, observed_at=time.time() - 5)
    result = verify_requested_text("hello", stale, max_age_s=1.0)
    assert result.verdict is Verdict.UNKNOWN


def test_nonfresh_flag_cannot_pass_even_with_current_timestamp():
    stale = obs("hello", fresh=False)
    result = verify_requested_text("hello", stale)
    assert result.verdict is Verdict.UNKNOWN
