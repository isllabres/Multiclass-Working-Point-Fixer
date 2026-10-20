---
name: create-issue
description: Turns an idea into a GitHub issue that states what to build and how "done" will be checked, after enough discovery that it won't need rewriting. The manager uses this only when the work is worth tracking before anyone builds it, and only after asking the person; see the manager's "Optional workflows" section.
---

# Create an issue

Requires `git` and `gh` (GitHub CLI), signed in: `gh auth status` must pass and `gh repo view`
must name a repository. Say so and stop if either fails.

## Step 1 — Light discovery

Four quick questions, without exploring the repository yet:

1. Bug, feature or refactor? A one-sentence summary.
2. By type — bug: current and expected behaviour, steps, severity; feature: the problem it
   solves and what success looks like from outside; refactor: what it touches, why, and which
   behaviour must not change.
3. Is there behaviour to verify at all? Some changes genuinely have none — a pure refactor, a
   dependency bump, tooling, docs. Say so in the issue if there is none, rather than inventing a
   test plan.
4. Build it from this issue right away, or file it for later?

## Step 1b — Interrogate, if it's worth it

If the idea is thin, high blast-radius, or touches a public interface, use the `grill-me` skill
now, on this idea, before writing anything. Skip it for anything small enough that light
discovery already settled it.

## Step 2 — Scope: one issue, or several

One issue carries one coherent piece of work. Independent deliverables, more than a day or two
of work, or real internal dependencies make an epic instead: a parent issue that lists its
children, one child issue per piece. **The person approves the split** before any issue is
created.

## Step 3 — Write the issue

A plain Markdown body, no external spec format:

```markdown
## Summary
<one paragraph: what and why>

## Behaviours
- [ ] <behaviour, one sentence, the kind of thing the implementer can write a test for>
- [ ] <behaviour>

## Out of scope
- <anything explicitly not covered, from grill-me if it ran>

## Open questions
- <anything left for implementation time, from grill-me if it ran, and why that's fine>

## Done means
<how the result will be checked: the behaviours above pass their tests, plus anything else
specific to this change>
```

Each `- [ ]` is a GitHub task-list item: it renders as a checkbox, and `implement-issue` ticks
it as the implementer finishes each behaviour.

## Step 4 — The person's approval

Show the body. Changes go back to Step 3. Nothing is created until they approve it explicitly.

## Step 5 — Create it

```bash
gh issue create --title "<imperative, one line>" --body-file <file>
```

Label it `pending` if it's filed for later rather than built now. A child of an epic names its
parent in the title or a label.

## Principles

1. No implementation here — that's `implement-issue`.
2. The person approves the issue, and any epic split, before it's created.
3. `grill-me` runs when Step 1 wasn't enough, never by default.
