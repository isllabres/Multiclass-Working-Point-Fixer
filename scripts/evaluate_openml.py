"""Command-line entry point for reproducible OpenML evaluations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import openml
import numpy as np
from openml.tasks import TaskType


_CONFIG = Path(__file__).with_name("openml_config.json")


def filter_catalog(tasks: Iterable[object]) -> list[int]:
    """Retain only supervised tasks with precisely holdout procedures 5 or 6."""
    return sorted(
        task.task_id
        for task in tasks
        if task.estimation_procedure_id in (5, 6)
        and task.estimation_procedure["type"] == "holdout"
    )


def query_catalog() -> tuple[list[int], dict[str, str]]:
    """Fetch current classification tasks and verify each holdout's API procedure type."""
    listing = openml.tasks.list_tasks(
        task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
    )
    candidates = listing.loc[
        listing["estimation_procedure"].isin(("10% Holdout set", "33% Holdout set"))
    ]
    tasks = [openml.tasks.get_task(int(task_id)) for task_id in candidates.index]
    return filter_catalog(tasks), {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "filters": "task_type=supervised_classification(1); procedure_id in (5,6); procedure.type == holdout",
    }


def sample_tasks(candidates: list[int], n: int | None, seed: int) -> list[int]:
    """Choose IDs without replacement before attempting task eligibility checks.

    Raises:
        ValueError: If n is negative or exceeds the current candidate pool.
    """
    if n is not None and (n < 0 or n > len(candidates)):
        raise ValueError(f"--n must be between 0 and {len(candidates)}")
    if n is None:
        return list(candidates)
    return np.random.default_rng(seed).choice(candidates, size=n, replace=False).tolist()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation options; reject invalid types and negative sample sizes."""
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-type", choices=("binary", "multiclass", "both"), default="both"
    )
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--report", type=Path, default=Path("openml_report.html"))
    parser.add_argument("--audit", type=Path, default=Path("openml_audit.json"))
    args = parser.parse_args(argv)
    if args.n is not None and args.n < 0:
        parser.error("--n must not be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation specified on the command line."""
    parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
