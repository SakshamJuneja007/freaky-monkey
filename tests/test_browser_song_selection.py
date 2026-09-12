from __future__ import annotations

import pytest

from agent_control.skills.browser.backend import (
    BrowserElement,
    BrowserSkillAdapter,
    BrowserSkillError,
)


def element(ref: str, name: str, role: str = "link") -> BrowserElement:
    return BrowserElement(ref, role=role, name=name)


def test_song_ranking_selects_only_a_lyrics_video():
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
    assert all("lyrics" in item[1].name.casefold() for item in ranked)
    assert all(
        item[1].ref not in {"@e2", "@e3", "@e4", "@e5", "@e6"}
        for item in ranked
    )


def test_song_ranking_rejects_normal_official_and_remix_results():
    candidates = [
        element("@e1", "Magdalena Bay - Killshot (Official Video)"),
        element("@e2", "Magdalena Bay - Killshot"),
        element("@e3", "Killshot Remix"),
    ]

    assert BrowserSkillAdapter._rank_song_candidates("Killshot", candidates) == []


def test_song_ranking_rejects_lyrics_speed_and_shorts_variants():
    candidates = [
        element("@e1", "Killshot sped up lyrics"),
        element("@e2", "Killshot slowed down lyrics"),
        element("@e3", "Killshot speed down lyrics"),
        element("@e4", "Killshot nightcore lyrics"),
        element("@e5", "Killshot Shorts lyrics"),
    ]

    assert BrowserSkillAdapter._rank_song_candidates("Killshot", candidates) == []


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
    assert "official" not in navigate_calls[0][3].casefold()

    click_calls = [call for call in cli.calls if call[:1] == ["click"]]
    assert click_calls
    assert "@e3" in click_calls[0]


def test_play_song_fails_when_no_lyrics_video_is_visible():
    class NoLyricsCLI(SongCLI):
        def run(self, args, *, session=None, timeout_s=None):
            args = list(args)
            if args[:1] == ["observe"] and not self.watch:
                self.observation_count += 1
                return {
                    "text": (
                        '@e1 link "Magdalena Bay - Killshot (Official Video)"\n'
                        '@e2 link "Killshot Remix"\n'
                        '@e3 link "Killshot"'
                    )
                }
            return super().run(args, session=session, timeout_s=timeout_s)

    browser = BrowserSkillAdapter(NoLyricsCLI())

    with pytest.raises(BrowserSkillError) as exc_info:
        browser.play_song("Killshot", timeout_s=0.2)

    assert exc_info.value.code == "semantic_song_result_not_ready"


def test_song_ranking_rejects_a_short_even_when_it_is_called_lyrics():
    candidates = [
        element(
            "@e1",
            "Killshot Lyrics",
            raw={"href": "https://www.youtube.com/shorts/abc123"},
        ),
        element(
            "@e2",
            "Killshot Lyrics",
            raw={"duration": "0:28"},
        ),
        element(
            "@e3",
            "Killshot Lyrics",
            raw={"duration": "3:42"},
        ),
    ]

    ranked = BrowserSkillAdapter._rank_song_candidates("Killshot", candidates)

    assert [item[1].ref for item in ranked] == ["@e3"]
