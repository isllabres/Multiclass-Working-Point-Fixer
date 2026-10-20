---
name: tdd
description: The red-green-refactor cycle this layer builds with, the commit convention it learns from this repository's own history on first use, and the rules that follow from both. Loaded by the implementer, and by the reviewer when it checks the implementer's history.
---

# Test-driven development, kept small

No spec files, no issue mirror, no generated tasks list: the manager describes the behaviour in
conversation, and the implementer builds it one behaviour at a time.

## The cycle

1. **Red** — write the test for one behaviour, with concrete inputs and the outcome you expect.
   Run it and watch it fail for the right reason (the behaviour is missing, not a typo in the
   test). Commit: `red: <the behaviour it expects>`.
2. **Green** — write the minimum code that makes it pass. Nothing the test doesn't ask for.
   Commit: `green: <the change>`.
3. **Refactor** — only when it genuinely improves the result, never while a test is red and
   never to add something no test asked for. Commit: `refactor: <what changed and why>`.

A behaviour that already exists is caught here, not guessed beforehand: if the test you just
wrote passes before any code, say so instead of committing it as red.

## Commit style, learned once per repository

This layer has no house commit convention of its own — it uses whatever the repository already
does. Before your first commit in a repository, check for `.opencode/commit-style.md`.

- **It exists:** read it and follow it. Nothing to derive.
- **It doesn't:** run `git log --format='%s%n%b%n---' -30` (and `git log --oneline -30` to see
  branch-name or ticket patterns echoed in subjects) and work out, from what's actually there:
  - subject style — length, case, mood (imperative vs. past tense), a `type:` or `type(scope):`
    prefix, a ticket ID in brackets or a trailing footer, an emoji;
  - whether a body is used, and for what (why vs. what, a `Review:`/footer line, issue
    references);
  - whether a ticket ID or type comes from the branch name and gets echoed into commits.

  Write what you found as a short rubric — the pattern plus one real example copied from the
  log, nothing this layer doesn't need — to `.opencode/commit-style.md`, then commit that file
  itself using the convention you just derived. On a repository with fewer than five commits to
  learn from, don't force a pattern out of noise: bootstrap with plain
  [Conventional Commits](https://www.conventionalcommits.org/) instead (`type: subject`,
  imperative, under 72 characters) and say so in the file.

  This first commit is the one exception to "red before green": it's tooling, not a behaviour.

Every commit after follows `.opencode/commit-style.md`, never a fresh guess. What never changes,
regardless of style:

## Commits

One logical step per commit, subject under 72 characters, imperative, naming the behaviour
("retry 5xx responses in `HttpClient.get`") not the mechanism ("fix bug"), formatted per
`.opencode/commit-style.md`. Files staged by name — never `git add -A`. A `green` commit never
touches a test; a test changed after its green cites the review finding that asked for it
(`Review: <finding>` in the body).

## What a test must do

One behaviour, one test. Arrange, Act, Assert map onto concrete inputs and a concrete expected
outcome — not "returns something", a value. No mock of the code under test. No `skip`, `xfail`,
or `sleep` to make a flaky test pass. Randomness seeded. An exact expected value is never loosened
to `is not None` to make the test pass more easily.

## What does not need a test

A pure refactor with no behaviour change needs a guard — a test that already passes and must
keep passing — not a new red/green pair. Tooling, formatting, and comment-only changes need
neither.
