---
name: implement-issue
description: Implements an existing GitHub issue's behaviour checklist on its own branch, using the same manager-implementer-reviewer loop as any other request, and ends at a local PR the person confirms before anything is pushed. The manager uses this when the person names an issue number, or just approved one from create-issue; see the manager's "Optional workflows" section.
---

# Implement an issue

Requires `git` and `gh`, signed in (`gh auth status`). This is the same delegate → implement →
review loop the manager always runs — see `AGENTS.md` — anchored to an issue and ending in a
pull request instead of only a report back to the person.

## Step 1 — Read the issue

```bash
gh issue view <n> --json number,title,body,labels,state
```

- **An `epic`**: not implemented directly. List its children and ask which to start.
- **`pending`**: it was filed for later. Confirm starting it now before going on.
- **Closed, or the behaviours read as already built**: say so with the evidence instead of
  reopening or rebuilding it.

## Step 2 — Branch

- **New**: from the up-to-date default branch, `git switch -c "<n>-<slug>"` (`<slug>` a short
  kebab-case version of the title).
- **Resuming** (the branch already exists): `git switch` to it and read its `git log` plus the
  issue's checklist to see which behaviours are already ticked off, done, or mid-review — carry
  on from there instead of starting over.

## Step 3 — Implement, exactly as usual

Hand the unticked behaviours to the `implementer`, the same way as any other request: it works
test-first (`tdd`), commits as it goes. As each behaviour is done, tick its box in the issue:

```bash
gh issue edit <n> --body-file <updated body>
```

## Step 4 — Review, same three-round cap

Hand the diff to the `reviewer`. `APPROVED` → Step 5. `CHANGES_REQUESTED` → relay to the
implementer, back to the reviewer, up to three rounds — exactly the round cap the manager
always uses. Before a fourth round, stop and tell the person what's still open instead of
looping.

## Step 5 — Local PR, and a stop

Show the diff against the default branch, the commits, and the reviewer's verdict (with any
disagreement, both positions). Nothing is pushed yet.

```markdown
## Summary
## Behaviours
- <behaviour>: <how it's verified>
## Review
- Verdict, and any disagreement (both positions)
Closes #<n>
```

## Step 6 — Only with the person's explicit confirmation

```bash
git push -u origin "<n>-<slug>"
gh pr create --title "<issue title>" --body-file <file>
```

Report the PR's URL. Never merge, approve, or close anything — merging is how the person
accepts the result, and GitHub closes the issue on merge.

## Principles

1. The issue's checklist is the source of truth for what's left; a behaviour already ticked is
   not redone without being asked.
2. Same round cap as every other review (`AGENTS.md`): three, then stop and ask.
3. Nothing reaches GitHub before Step 6, except ticking the checklist as work finishes.
