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


def test_song_ranking_falls_back_to_best_non_short_video():
    candidates = [
        element("@e1", "Killshot - Official Video"),
        element("@e2", "Killshot Remix"),
        element("@e3", "Killshot"),
    ]

    ranked = BrowserSkillAdapter._rank_song_candidates("Killshot", candidates)

    assert ranked
    assert all(item[1].ref in {"@e1", "@e2", "@e3"} for item in ranked)
    assert ranked[0][1].ref == "@e3"


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
    assert "official" not in navigate_calls[0][3].casefold()

    click_calls = [call for call in cli.calls if call[:1] == ["click"]]
    assert click_calls
    assert "@e3" in click_calls[0]

