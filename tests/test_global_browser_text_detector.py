from agent_control.skills.browser.text_detector import GlobalBrowserTextDetector
from agent_control.text_observation import TextObservation


class FakeBackend:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def read_browser_text_observation(self):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        controls = [{"text": value, "role": "Edit", "process_id": 1} for value in self.values[index]]
        return TextObservation(
            text="\n".join(self.values[index]),
            source="fake_uia",
            target={"kind": "browser_global"},
            fresh=True,
            metadata={"controls": controls},
            ok=True,
        )


def test_detects_new_typed_text():
    detector = GlobalBrowserTextDetector(FakeBackend([[], ["hianime"]]))
    result = detector.detect_change()
    assert result.changed is True
    assert result.typed is True
    assert result.added == ("hianime",)


def test_detects_unchanged_text_without_claiming_typing():
    detector = GlobalBrowserTextDetector(FakeBackend([["hello"], ["hello"]]))
    result = detector.detect_change()
    assert result.changed is False
    assert result.typed is False


def test_missing_observation_is_unknown():
    class Broken:
        def read_browser_text_observation(self):
            raise RuntimeError("uia unavailable")

    result = GlobalBrowserTextDetector(Broken()).detect_change()
    assert result.changed is None
    assert result.typed is None


def test_removed_text_is_a_change_but_not_new_typing():
    detector = GlobalBrowserTextDetector(FakeBackend([["hello"], []]))
    result = detector.detect_change()
    assert result.changed is True
    assert result.typed is False
    assert result.removed == ("hello",)


def test_executor_result_is_not_used():
    class Backend:
        def read_browser_text_observation(self):
            return TextObservation(
                text="typed",
                source="fake_uia",
                target={"kind": "browser_global"},
                fresh=True,
                metadata={"controls": [{"text": "typed"}]},
                ok=True,
            )

    result = GlobalBrowserTextDetector(Backend()).observe()
    assert result.text == "typed"
