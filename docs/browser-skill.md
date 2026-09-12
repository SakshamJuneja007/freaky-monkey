# DEIMOS BrowserSkill architecture

DEIMOS controls the real authenticated browser through Tencent BrowserSkill:

`BrowserSkill -> BrowserSkillAdapter -> bsk CLI -> BrowserSkill daemon -> extension -> Agent Window`

## Reference lifecycle

BrowserSkill `observe`/`snapshot` allocate fresh semantic references (`@eN`).
They are observation-scoped capabilities, not stable DOM ids. DEIMOS therefore
uses:

1. fresh observation;
2. semantic target resolution;
3. one action against the fresh ref;
4. invalidation of the local observation generation;
5. fresh observation for the next state decision;
6. independent verification of the intended postcondition.

`BrowserTarget` records the observation generation. Reusing it after a new
observation or browser mutation raises a stale-reference error.

## Readiness

Navigation is requested with BrowserSkill's lifecycle wait (`load`). That is
not treated as semantic readiness. High-level workflows then poll fresh
semantic observations with a bounded timeout until the requested target is
actually present. This avoids a global arbitrary sleep while handling SPAs and
pages whose accessibility tree appears after navigation.

## Target resolution

Resolution is deterministic and semantic. Exact accessible-name matches score
highest, followed by normalized/prefix/token matches, with role preferences
and explicit ambiguity rejection. High-level media search additionally
penalizes derivative forms such as `lyrics`, `remix`, `slowed`, `reverb`,
`cover`, `reaction`, `live`, `mix`, and `playlist` unless the user explicitly
requested such a form.

No YouTube-specific `@eN`, CSS id, coordinate, or video id is embedded in the
browser primitive layer.

## Verification

The browser verifier never uses the executor's success flag as proof. It
re-observes the browser independently. Verification is tri-state:

- `PASS`: the requested postcondition is independently established.
- `FAIL`: an observable contradiction or wrong state is established.
- `UNKNOWN`: the browser does not expose enough evidence to prove the claim.

For `browser_play_song`, verification requires a YouTube watch page containing
the requested query and independent playback evidence. Semantic media controls
are preferred; a narrowly scoped `evaluate` fallback may inspect only the
current page's HTML5 video playback state. It never reads credentials,
cookies, tokens, passwords, or other secrets.

## CLI compatibility

The adapter distinguishes daemon/session control commands from session-scoped
browser tools. In particular, `bsk status`, `bsk browsers`, `bsk doctor`,
`bsk wait-ms`, and `bsk session ...` are not given `--session`. Tab management
uses the current `bsk tab list/create/select/close/borrow/return` command tree.

The transport explicitly decodes UTF-8 and preserves useful stderr in the
structured result. It does not use `shell=True`, Playwright, CDP attachment,
or the user's normal Chrome profile.

## Human-only steps

CAPTCHA, login challenges, OTP, payment confirmation, and similar security or
human-only gates must use BrowserSkill's `request-help`/human-in-the-loop path.
The browser layer does not attempt to bypass them.
