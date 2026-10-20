---
description: One-time setup for a new project — reads this repository's complete history and source layout, derives (or confirms) .opencode/commit-style.md from it, and reports the toolchain and conventions this layer will now follow. Run once, right after installing the layer, before the first real request.
agent: manager
---

Make the layer compliant with this specific repository before anyone asks it to build anything.
This writes nothing beyond what the layer already allows itself — see "A layer, not a project
template" in `AGENTS.md` — it just does the first-run learning described in the `tdd` skill
up front and thoroughly, instead of leaving it to whichever commit happens to come first, and
tells the person what it found.

## 1. Check for existing work

!`test -f .opencode/commit-style.md && echo "exists:" && cat .opencode/commit-style.md || echo missing`

- **Exists** — show it to the person as-is. Ask whether to re-derive it; only worth it if it
  looks stale or wrong. A plain "looks right" ends the command here without touching anything.
  Re-deriving still goes through steps 2–5 below.
- **Missing** — continue.

## 2. Confirm there is history worth learning from

!`git rev-list --count HEAD 2>/dev/null || echo 0`

- **Fewer than five commits** — nothing to learn yet. Tell the person, then have the
  implementer write `.opencode/commit-style.md` bootstrapped as plain Conventional Commits per
  the `tdd` skill, noting why. Skip straight to step 4's toolchain/layout read, then report.
- **Five or more** — continue to a full pass, not the 30-commit sample a first commit settles
  for under time pressure.

## 3. Derive the commit style, in full

Delegate one call to the `implementer` agent: read the complete history —
`git log --format='%s%n%b%n---'` and `git log --oneline`, uncapped — and follow the `tdd`
skill's commit-style procedure exactly: work out subject style (length, case, mood, a
`type:`/`type(scope):` prefix, a ticket ID, an emoji), whether and how a body is used, and
whether a ticket ID or type is echoed from the branch name. For a repository large enough that
the full log is impractical to read in one pass, the most recent few hundred commits are
enough to confirm the pattern is consistent — say so in the report if the pass was cut short.
Write and commit `.opencode/commit-style.md` in the convention just derived, with one real
example lifted from the log, exactly as the `tdd` skill already describes for a first commit.

## 4. Read the ways of working, report-only

Same delegation, no further write beyond step 3:

- **Toolchain** — per `python-standards`: package manager (which lockfile, if any), linter and
  formatter, type checker, test runner — from `pyproject.toml`, a
  `Makefile`/`tox.ini`/`noxfile.py`, and CI config (`.github/workflows/`, `.gitlab-ci.yml`, …).
- **Layout** — source and test directory layout, src-layout vs. flat, naming already in place.
- **Collaboration conventions visible in the history** — branch naming, whether commits
  reference issues or PRs, direct-to-main vs. PR-and-merge, squashed vs. individual commits kept
  on merge.

None of this is written anywhere; it only shapes how the implementer and reviewer already
behave by reading the project themselves, every time, per `python-standards`.

## 5. Report back

One summary to the person: the commit-style rubric just written or confirmed, the detected
toolchain and layout, and anything about how this repository collaborates that is worth knowing
before the first real request. Say plainly if a repository had too little history to learn a
style from, or none at all. Nothing else is written — the next command should be a real request,
not more setup.
