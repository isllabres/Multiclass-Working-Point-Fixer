---
name: grill-me
description: A method for interrogating a plan, idea or design one question at a time, each with a recommendation, until every decision it depends on is pinned down. The manager performs this itself — there is no separate agent for it in this layer — and only after asking the person first; see the manager's "Optional workflows" section for when it's worth offering.
---

# Grill me

Your job is not to agree: find every decision the plan depends on and pin it down before
anyone builds anything.

## Method

1. **Build the decision tree first.** Break the plan into the decisions it depends on: scope,
   edge cases, data and state, errors, performance and security, integration points, what is
   left out. A decision is anything where a different answer changes what gets built. Ask what
   conditions other decisions first, so one answer prunes whole branches.
2. **Explore before asking.** If the repository can answer it — a convention, an existing
   function, a pattern already in use — find it yourself instead of asking. Ask only what needs
   a human: a trade-off, a product choice, a preference.
3. **One question at a time**, never a list. Later questions build on earlier answers.
4. **Always recommend.** Every question carries your own answer and its reason in one line, so
   the person can just say yes:

   > **Should a lookup with no match return `None`, or raise `LookupError`?**
   > My recommendation: raise — a silent `None` moves the failure far from its cause. Agreed?

5. **Resolve and move on.** A settled answer is not asked again. If a later answer contradicts
   an earlier one, say so and ask which prevails; never pick silently.
6. **Depth follows what is at stake**: blast radius, reversibility, and how precise the plan
   already is. A one-line tweak may need one question; a public API, fifteen. Every 4–6
   questions, say in a line how much is settled and what is still open.
7. **Stop** when every branch has an answer (including "out of scope" or "does not matter"),
   when the person says enough, or when only speculation is left. Do not pad the interview.

Walk the plan against these, skipping what clearly does not apply: functional scope; edge cases
and errors; data and state; non-functional constraints; integration points; what the person
sees on success and failure; how "done" will be verified (each behaviour becomes something the
implementer can write a test for); deployment and migration.

## Synthesis

```markdown
## Resolved
- <decision>: <the answer, one line>

## Explicitly out of scope
- <item>: <why>

## Still open (if any, and why that is fine)
- <item>: <why it can wait, e.g. "reversible, decided at implementation time">
```

Turn **Resolved** into the behaviours you hand the implementer, carry **Explicitly out of
scope** into your own report back so nobody re-raises it later, and leave **Still open** for the
implementer to decide at the point it matters rather than guessing now. If this ran ahead of
`create-issue`, the same three sections become the issue's behaviours, "Out of scope" and "Open
questions" sections directly.
