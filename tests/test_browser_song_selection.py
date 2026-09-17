from __future__ import annotations

import pytest

from agent_control.skills.browser.backend import (
    BrowserElement,
    BrowserSkillAdapter,
    BrowserSkillError,
)


def element(ref: str, name: str, role: str = "link", raw=None) -> BrowserElement:
    return BrowserElement(ref, role=role, name=name, raw=raw)


def test_song_ranking_prefers_lyrics_but_keeps_normal_videos_eligible():
    candidates = [
        element("@e1", "Magdalena Bay - Killshot (Lyrics)"),
        element("@e2", "Killshot slowed + reverb lyrics"),
        element("@e3", "Killshot Remix"),
        element("@e4", "Magdalena Bay - Killshot (Official Video)"),
        element("@e5", "Killshot sped up lyrics"),
        element("@e6", "Killshot - Shorts"),
    ]

    ranked = BrowserSkillAdapter._rank_song_candidates("Killshot", candidates)

    assert ranked
    assert ranked[0][1].ref == "@e1"
    assert all(item[1].ref != "@e6" for item in ranked)


def test_song_ranking_requires_a_lyrics_video_for_normal_song_playback():
    candidates = [
        element("@e1", "Killshot - Official Video"),
        element("@e2", "Killshot Remix"),
        element("@e3", "Killshot"),
    ]

    ranked = BrowserSkillAdapter._rank_song_candidates("Killshot", candidates)

    assert ranked == []


def test_song_ranking_rejects_short_by_semantic_url_even_when_title_says_lyrics():
    candidates = [
        element("@e1", "Killshot Lyrics", raw={"href": "https://www.youtube.com/shorts/abc123"}),
        element("@e2", "Killshot Lyrics", raw={"href": "https://www.youtube.com/watch?v=abc"}),
    ]

    ranked = BrowserSkillAdapter._rank_song_candidates("Killshot", candidates)

    assert [item[1].ref for item in ranked] == ["@e2"]


class SongCLI:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.watch = False
        self.observation_count = 0

    def run(self, args, *, session=None, timeout_s=None):
        args = list(args)
        self.calls.append(args)

        if args[:2] == ["session", "start"]:
            return {"session_id": "s1"}

        if args[:1] == ["navigate"]:
            return {"ok": True}

        if args[:1] == ["observe"]:
            self.observation_count += 1
            if self.watch:
                return {"text": '@e90 button "Pause"'}
            return {
                "text": (
                    '@e1 link "Killshot Remix"\n'
                    '@e2 link "Magdalena Bay - Killshot (Official Video)"\n'
                    '@e3 link "Killshot Lyrics"\n'
                    '@e4 link "Killshot sped up lyrics"\n'
                    '@e5 link "Killshot Shorts"'
                )
            }

        if args[:1] == ["click"]:
            self.watch = True
            return {"ok": True, "used_ref": "@e3"}

        if args[:2] == ["tab", "list"]:
            return {
                "tabs": [
                    {
                        "tab_id": 1,
                        "active": True,
                        "url": (
                            "https://www.youtube.com/watch?v=test"
                            if self.watch
                            else "https://www.youtube.com/results"
                        ),
                        "title": "Killshot Lyrics",
                    }
                ]
            }

        if args[:1] == ["wait-ms"]:
            return {"waited_ms": 1}

        return {"ok": True}


def test_play_song_searches_normally_and_clicks_lyrics_ref():
    cli = SongCLI()
    browser = BrowserSkillAdapter(cli)

    result = browser.play_song("Killshot", timeout_s=2)

    assert result["ok"] is True
    assert result["search_query"] == "Killshot"
    assert result["selected_ref"] == "@e3"
    assert "Lyrics" in result["selected_name"]
    assert result["playback"] == "verified"

    navigate_calls = [call for call in cli.calls if call[:1] == ["navigate"]]
    assert len(navigate_calls) == 1
    assert "Killshot" in navigate_calls[0][3]
    assert "lyrics" in navigate_calls[0][3].casefold()
    assert "-shorts" in navigate_calls[0][3].casefold()
    assert "official" not in navigate_calls[0][3].casefold()

    click_calls = [call for call in cli.calls if call[:1] == ["click"]]
    assert click_calls
    assert "@e3" in click_calls[0]


class NavigationCLI(SongCLI):
    def __init__(self, *, delayed=False, stale_once=False, never_navigates=False, dead=False):
        super().__init__()
        self.delayed = delayed
        self.stale_once = stale_once
        self.never_navigates = never_navigates
        self.dead = dead
        self.click_count = 0
        self.wait_count = 0

    def run(self, args, *, session=None, timeout_s=None):
        args = list(args)
        self.calls.append(args)

        if args[:2] == ["session", "start"]:
            return {"session_id": "s1"}
        if args[:1] == ["navigate"]:
            self.watch = False
            return {"ok": True}
        if args[:1] == ["observe"]:
            self.observation_count += 1
            if self.watch:
                return {"text": '@e90 button "Pause"'}
            ref = "@e1" if self.observation_count <= 1 else "@e7"
            return {"text": f'{ref} link "Killshot Lyrics"'}
        if args[:1] == ["click"]:
            self.click_count += 1
            if self.dead:
                raise BrowserSkillError("connection closed", code="browser_session_expired")
            if self.stale_once and self.click_count == 1:
                raise BrowserSkillError("stale target", code="browser_target_stale")
            if not self.never_navigates and not self.delayed:
                self.watch = True
            return {"ok": True}
        if args[:1] == ["wait-for-navigation"]:
            self.wait_count += 1
            if self.delayed and self.wait_count == 1:
                self.watch = True
                return {"ok": True}
            if self.never_navigates:
                return {"ok": False, "code": "navigation_timeout"}
            return {"ok": True}
        if args[:2] == ["tab", "list"]:
            return {
                "tabs": [{
                    "tab_id": 1,
                    "active": True,
                    "url": (
                        "https://www.youtube.com/watch?v=test"
                        if self.watch
                        else "https://www.youtube.com/results?search_query=killshot"
                    ),
                    "title": "Killshot Lyrics",
                }]
            }
        if args[:1] == ["wait-ms"]:
            return {"waited_ms": 1}
        return {"ok": True}


def test_play_song_accepts_delayed_navigation_after_state_aware_wait():
    cli = NavigationCLI(delayed=True)
    browser = BrowserSkillAdapter(cli)

    result = browser.play_song("Killshot", timeout_s=3)

    assert result["ok"] is True
    assert cli.wait_count == 1
    assert cli.click_count == 1


def test_play_song_reobserves_and_reresolves_after_stale_target():
    cli = NavigationCLI(stale_once=True)
    browser = BrowserSkillAdapter(cli)

    result = browser.play_song("Killshot", timeout_s=3)

    assert result["ok"] is True
    assert cli.click_count == 2
    click_calls = [call for call in cli.calls if call[:1] == ["click"]]
    assert "@e1" in click_calls[0]
    assert "@e7" in click_calls[1]


def test_play_song_classifies_live_search_without_navigation_as_expected_state_failure():
    cli = NavigationCLI(never_navigates=True)
    browser = BrowserSkillAdapter(cli)

    with pytest.raises(BrowserSkillError) as exc_info:
        browser.play_song("Killshot", timeout_s=3)

    exc = exc_info.value
    assert exc.code == "expected_state_not_reached"
    assert cli.click_count == 2


def test_play_song_distinguishes_dead_browser_session():
    cli = NavigationCLI(dead=True)
    browser = BrowserSkillAdapter(cli)

    with pytest.raises(BrowserSkillError) as exc_info:
        browser.play_song("Killshot", timeout_s=3)

    assert exc_info.value.code == "browser_session_dead"
    assert cli.click_count == 1


def test_play_song_navigation_recovery_is_bounded():
    cli = NavigationCLI(never_navigates=True)
    browser = BrowserSkillAdapter(cli)

    with pytest.raises(BrowserSkillError):
        browser.play_song("Killshot", timeout_s=3)

    assert cli.click_count == 2
    assert cli.click_count <= 2

class ShortsRedirectCLI(NavigationCLI):
    def __init__(self):
        super().__init__()
        self.location = "results"
        self.result_generation = 0

    def run(self, args, *, session=None, timeout_s=None):
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["session", "start"]:
            return {"session_id": "s1"}
        if args[:1] == ["navigate"]:
            self.location = "results"
            return {"ok": True}
        if args[:1] == ["observe"]:
            self.observation_count += 1
            if self.location == "watch":
                return {"text": '@e90 button "Pause"'}
            if self.location == "shorts":
                return {"text": '@e91 heading "Do I Wanna Know"'}
            return {
                "text": (
                    '@e1 link "Arctic Monkeys - Do I Wanna Know? (Lyrics Video)"\n'
                    '@e2 link "Do I Wanna Know - Arctic Monkeys (Lyrics)"\n'
                    '@e3 link "Do I Wanna Know - Arctic Monkeys with lyrics for status"'
                )
            }
        if args[:1] == ["click"]:
            self.click_count += 1
            if self.click_count == 1:
                self.location = "shorts"
            else:
                self.location = "watch"
            return {"ok": True}
        if args[:1] == ["navigate-back"]:
            self.location = "results"
            return {"ok": True}
        if args[:1] == ["wait-for-navigation"]:
            self.wait_count += 1
            return {"ok": True}
        if args[:1] == ["current-url"]:
            return {"url": {
                "results": "https://www.youtube.com/results?search_query=do+i+wanna+know",
                "shorts": "https://www.youtube.com/shorts/abc123",
                "watch": "https://www.youtube.com/watch?v=abc123",
            }[self.location]}
        if args[:2] == ["tab", "list"]:
            url = {
                "results": "https://www.youtube.com/results?search_query=do+i+wanna+know",
                "shorts": "https://www.youtube.com/shorts/abc123",
                "watch": "https://www.youtube.com/watch?v=abc123",
            }[self.location]
            return {"tabs": [{"tab_id": 1, "active": True, "url": url, "title": "Do I Wanna Know"}]}
        if args[:1] == ["wait-ms"]:
            return {"waited_ms": 1}
        return {"ok": True}


def test_play_song_rejects_shorts_destination_and_selects_another_result():
    cli = ShortsRedirectCLI()
    browser = BrowserSkillAdapter(cli)

    result = browser.play_song("Do I Wanna Know", timeout_s=4)

    assert result["ok"] is True
    assert result["selected_name"] == "Arctic Monkeys - Do I Wanna Know? (Lyrics Video)"
    assert cli.click_count == 2
    assert any(call[:1] == ["navigate-back"] for call in cli.calls)
    click_calls = [call for call in cli.calls if call[:1] == ["click"]]
    assert "@e2" in click_calls[0]
    assert "@e1" in click_calls[1]
