---
description: One-time setup for a new project — reads this repository's complete history and source layout, derives (or confirms) .opencode/commit-style.md from it, and reports the toolchain and conventions this layer will now follow. Run once, right after installing the layer, before the first real request.
agent: manager
---

Make the layer compliant with this specific repository before anyone asks it to build anything.
This does the first-run learning described in the `tdd` skill up front and thoroughly, instead of
leaving it to whichever commit happens to come first, and tells the person what it found. Do not
create or re-derive `.opencode/commit-style.md` without the person's explicit confirmation, and
do not commit that file without a separate explicit confirmation. Starting this command is not
consent to either action.

## 1. Check for existing work and ask before changing it

!`test -f .opencode/commit-style.md && echo "exists:" && cat .opencode/commit-style.md || echo missing`

- **Exists** — show it to the person as-is. Ask whether they explicitly authorize re-deriving and
  replacing it; only worth doing if it looks stale or wrong. A plain "looks right" is not
  authorization and leaves the file untouched.
- **Missing** — tell the person it is missing and ask whether they explicitly authorize deriving
  and creating it. Do not treat running this command as permission.
- If the person declines, do not derive, write, or commit `.opencode/commit-style.md`. Skip steps
  2–3 and continue with the read-only findings in step 4 where practical.

## 2. Confirm there is history worth learning from

!`git rev-list --count HEAD 2>/dev/null || echo 0`

- **Fewer than five commits** — nothing to learn yet. Tell the person, then have the
  implementer prepare `.opencode/commit-style.md` bootstrapped as plain Conventional Commits per
  the `tdd` skill, noting why.
- **Five or more** — continue to a full pass, not the 30-commit sample a first commit settles
  for under time pressure.

## 3. Derive, write, then ask before committing

Only after the person explicitly authorizes creating or re-deriving the file, delegate one call to
the `implementer` agent: read the complete history —
`git log --format='%s%n%b%n---'` and `git log --oneline`, uncapped — and follow the `tdd`
skill's commit-style procedure exactly: work out subject style (length, case, mood, a
`type:`/`type(scope):` prefix, a ticket ID, an emoji), whether and how a body is used, and
whether a ticket ID or type is echoed from the branch name. For a repository large enough that
the full log is impractical to read in one pass, the most recent few hundred commits are
enough to confirm the pattern is consistent — say so in the report if the pass was cut short. Have
the implementer prepare `.opencode/commit-style.md` in the convention just derived, with one real
example lifted from the log. It must leave the file uncommitted: do not stage it, run a commit
command, or continue to another task before reporting the prepared file.

Show the person the resulting file (or the exact change to it) and ask for separate explicit
confirmation before committing it. Permission to create or re-derive the file does not authorize
its commit. Only after the person confirms, have the implementer commit the already-approved file
using `.opencode/commit-style.md`. If commit confirmation is declined, do not commit the file;
report that it remains written but uncommitted, then continue to step 4.

## 4. Read the ways of working, report-only

Same delegation, no further writes:

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

One summary to the person: the commit-style rubric just written, confirmed, or left untouched;
the detected toolchain and layout; and anything about how this repository collaborates that is
worth knowing before the first real request. Say plainly if a repository had too little history
to learn a style from, or none at all. If either confirmation was declined, state what was not
written or committed. Nothing else is written — the next command should be a real request, not
more setup.
