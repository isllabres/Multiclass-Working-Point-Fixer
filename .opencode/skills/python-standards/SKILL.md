---
name: python-standards
description: The shared standard for what good Python looks like in this project — structure, typing, language semantics and traps, error handling, data code (pandas, numpy), performance, security and tooling. Carried by the implementer, who writes the code and its tests, and the reviewer, who judges both, so both work from one rubric.
---

Pythonic means clear, explicit and consistent with the project, not clever. **The project's existing
conventions win over anything below**; where it has none, use this. Target Python 3.11+.

The implementer uses this as the way to write; the reviewer uses it as the list of things to
check. Each rule says why it matters, so a finding can name the consequence and not just cite a
rule.

## Structure

- One job per function and per module. Pure computation is separate from I/O. No work at import
  time; entry points sit behind `if __name__ == "__main__":`. No circular imports, no
  `from x import *`.
- Structured data has a type: a `dataclass` (`frozen=True, slots=True` for values), a `TypedDict`
  or a Pydantic model, not a dict with implicit keys. `Protocol` for what a function needs from its
  arguments; `Enum`, `StrEnum` or `Literal` for closed sets. *Why: a typo in a dict key fails at
  runtime, a typo in an attribute fails at the checker.*
- Configuration comes from the project's config file, never as literals in code.
  Paths are `pathlib.Path`. Diagnostics use `logging` with lazy arguments, not `print`.
- The public surface is deliberate: a leading underscore for internals, `__all__` where other
  modules import from this one. No dead code and no logic copied between modules.

## Types and docstrings

- Every public function and method is fully annotated, return type included. Modern syntax:
  `list[int]`, `X | None`, `Self`. Reach for `TypeVar`, `ParamSpec`, `Literal`, `TypeAlias` only
  where they remove an ambiguity.
- `# type: ignore` carries an error code and a reason. A gap in a third-party stub is closed
  narrowly, not by turning things into `Any`.
- The type checker is advisory in this project: run it on what you touch and report what it
  says; never silence it to make it green.
- Public functions and classes have a docstring: what it does, arguments, return, what it raises
  (Google style unless the project already uses another).

## Language semantics: the traps

- No mutable default arguments. No late-binding closure in a loop (bind with a default argument or
  `functools.partial`). `is` only for `None` and singletons. No shadowing of builtins (`list`,
  `id`, `type`, `input`, `filter`).
- `if df:` and `if array:` raise or mislead on pandas and numpy objects: test `.empty`, `len(...)`,
  `.any()` or `.all()` explicitly.
- Datetimes are timezone-aware: `datetime.utcnow()` is deprecated since 3.12, use
  `datetime.now(timezone.utc)`.
- Iterators and generators are consumed once. `zip(strict=True)` when the lengths must match.
  Never rely on set order or on dict order the reader cannot see.

## Errors and resources

- `with` for every resource: files, connections, locks, temporary directories, timers.
- Catch the narrowest exception you can actually handle. Never a bare `except:` and never
  `except Exception: pass`. A domain failure gets a domain exception, raised with its cause
  (`raise X(...) from err`). Validate input once at the boundary; inside, assume it is valid.
- Errors do not pass silently: a function that can fail raises, it does not return `None` and hope
  the caller checks.

## Idioms

- Comprehensions and generator expressions where they read better than the loop, and the loop where
  they do not (no comprehension written for its side effect, no triple nesting). `enumerate`,
  `itertools`, `functools.cache` (not on methods of long-lived objects, it keeps `self` alive),
  `contextlib`, `match` where it beats an if-chain, f-strings.
- The standard library and dependencies already in the project before a new dependency.

## Data code: pandas and numpy

- Vectorise. No `iterrows` or row-wise `apply` where a column operation exists. No `pd.concat` (or
  the removed `DataFrame.append`) inside a loop: collect, then concatenate once.
- No chained assignment: write with `.loc` and copy explicitly with `.copy()`. No `inplace=True`
  (it rarely saves memory and breaks chaining). No silent dtype coercion (`object` columns,
  a late `astype`).
- Randomness comes from a seeded `np.random.default_rng(seed)` (or the seed argument a library
  offers), the seed read from config. Never the global `np.random` or `random` state. *Why: an
  unseeded component makes a result unreproducible.*
- Memory: appropriate dtypes, chunk or stream what does not fit, generators for one-pass
  transformations, categoricals and `float32` where they are enough.

## Performance and concurrency

- Correct and readable first. Measure (`cProfile`, `timeit`) before optimising, and only where the
  criterion has a performance requirement. A better algorithm or vectorisation beats micro-tuning.
- Smells worth flagging without a profiler: `x in some_list` inside a loop (use a set), string `+=`
  in a loop (`"".join`), I/O or a query inside a loop, reading a whole file to use a slice.
- `asyncio` for I/O-bound concurrency (network and API calls), with timeouts and bounded concurrency
  (`asyncio.timeout`, `TaskGroup`, a semaphore) and no blocking call inside `async def`.
  `concurrent.futures` or multiprocessing for CPU-bound work.

## Security

- No `eval` or `exec` on external input. No `pickle` or `joblib.load` on files you do not control.
  `yaml.safe_load`, never `yaml.load`. `subprocess.run([...], check=True)` with an argument list,
  never `shell=True` with interpolated text. Parameterised SQL. No `verify=False`.
- Secrets come from the environment and never appear in code, notebooks or logs.

## Tooling

Use whatever this project already uses for packaging, linting, type-checking and testing —
don't assume `uv`/`ruff`/`mypy` if it's something else. Check, in order: `pyproject.toml`
(`[tool.*]` sections, `[project.scripts]`), a `Makefile`/`tox.ini`/`noxfile.py`, the CI config
(`.github/workflows/`, `.gitlab-ci.yml`, …), and the lockfile present (`uv.lock` → `uv run …`,
`poetry.lock` → `poetry run …`, `Pipfile.lock` → `pipenv run …`, none of those → plain
`python -m …` inside the project's own virtual environment). Typical shapes, once you know the
runner:

```bash
<runner> ruff check .                         # lint — or flake8 + isort if that's what's configured
<runner> ruff format --check <touched paths>  # format — or black --check
<runner> mypy <touched paths>                 # type check — or pyright; advisory either way
<runner> pytest -q                            # the suite — or python -m unittest
```

Dependencies are added the way the project already adds them (`uv add`, `poetry add`, pinning in
`requirements.txt`, …). Test coverage is not a target in itself: tests cover the behaviours
asked for.
