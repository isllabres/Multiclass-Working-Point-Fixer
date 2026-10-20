---
name: update-issue
description: Changes an issue's behaviour checklist or description after it was filed — the only way its scope changes once created. Applies the request, shows the person the difference, and edits the issue only with their approval. The manager uses this when a tracked issue's scope needs to move; see the manager's "Optional workflows" section.
---

# Update an issue

Requires `git` and `gh`.

## Step 1 — Read it as it stands

```bash
gh issue view <n> --json number,title,body,labels,state
```

## Step 2 — Apply the change

Edit the body: add or remove a behaviour (and its checkbox), adjust the summary, move something
between "Out of scope" and the behaviour list, update "Done means" if what counts as finished
changed. Keep already-ticked behaviours' checkmarks unless the request removes that behaviour
itself. If the request contradicts what's there, the request wins; if it's ambiguous, ask.

## Step 3 — What this means for work already done

If a branch for this issue exists, say what the change implies: a ticked behaviour being
removed (flag it — don't silently revert code, that's the person's call; loop in the
`implementer` if they want it undone), an added behaviour (the implementer picks it up next),
a changed "Done means" (the `reviewer` needs to know before its next round).

## Step 4 — The person's approval

Show the diff between the old and new body. Nothing is edited until they approve it. A
checklist tick made by `implement-issue` as it finishes a behaviour needs no separate approval
here — this gate is for changing what the issue asks for, not for recording progress on it.

## Step 5 — Edit it

```bash
gh issue edit <n> --body-file <file>
gh issue comment <n> --body "Updated: <what changed and why>."
```

Confirm the label set with the person — `pending` in particular — before changing it.

## Principles

1. The issue is the source of truth; nothing about it changes without the person seeing the
   diff first.
2. A change that only ticks progress needs no approval; a change to what's being asked for
   always does.
