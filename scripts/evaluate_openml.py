"""Command-line entry point for reproducible OpenML evaluations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import openml
import numpy as np
import pandas as pd
import requests
from openml.tasks import TaskType
from sklearn.metrics import f1_score


_CONFIG = Path(__file__).with_name("openml_config.json")


def filter_catalog(tasks: Iterable[object]) -> list[int]:
    """Retain only supervised tasks with precisely holdout procedures 5 or 6."""
    return sorted(
        task.task_id
        for task in tasks
        if task.estimation_procedure_id in (5, 6)
        and task.estimation_procedure["type"] == "holdout"
    )


def read_procedure(xml: str) -> tuple[int, str]:
    """Extract the actual estimation procedure ID and type from OpenML task XML.

    The OpenML Python SDK silently defaults `task.estimation_procedure_id` to 1
    when it reads an existing task; that attribute is not the server's ID.
    """
    root = ElementTree.fromstring(xml)
    namespace = {"oml": "http://openml.org/openml"}
    for entry in root.findall("oml:input", namespace):
        if entry.get("name") == "estimation_procedure":
            procedure = entry.find("oml:estimation_procedure", namespace)
            if procedure is not None:
                identifier = procedure.findtext("oml:id", namespaces=namespace)
                procedure_type = procedure.findtext("oml:type", namespaces=namespace)
                if identifier is not None and procedure_type is not None:
                    return int(identifier), procedure_type
    raise ValueError("task metadata has no estimation procedure ID/type")


def query_catalog() -> tuple[list[int], dict[str, str]]:
    """Fetch current classification tasks and verify each holdout's API procedure type."""
    listing = openml.tasks.list_tasks(
        task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
    )
    candidates = listing.loc[
        listing["estimation_procedure"].isin(("10% Holdout set", "33% Holdout set"))
    ]
    verified = []
    for task_id in candidates.index:
        try:
            verify_procedure(int(task_id))
        except ValueError:
            continue
        verified.append(int(task_id))
    return sorted(verified), {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "filters": "task_type=supervised_classification(1); listing procedure names 10%/33% Holdout set; XML procedure.id in (5,6) and procedure.type == holdout",
    }


def verify_procedure(task_id: int) -> None:
    """Check the actual task procedure metadata before attempting any run."""
    response = requests.get(f"{openml.config.server}/task/{task_id}", timeout=30)
    response.raise_for_status()
    identifier, procedure_type = read_procedure(response.text)
    if identifier not in (5, 6) or procedure_type != "holdout":
        raise ValueError(f"task {task_id} has incompatible procedure {identifier}/{procedure_type}")


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


def classify_target(target: pd.Series | pd.DataFrame) -> tuple[str, tuple[str, ...]]:
    """Infer the type and canonical class order from downloaded single-label targets.

    Raises:
        ValueError: If the target is missing, multilabel, or has fewer than two classes.
    """
    if not isinstance(target, pd.Series):
        raise ValueError("multilabel target: expected one target column")
    if target.isna().any():
        raise ValueError("missing target values")
    if target.map(lambda value: not np.isscalar(value)).any():
        raise ValueError("multilabel target values")
    labels = tuple(sorted(target.astype(str).unique()))
    if len(labels) < 2:
        raise ValueError("invalid target: fewer than two classes")
    return ("binary" if len(labels) == 2 else "multiclass", labels)


def discover_split(task: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read the sole holdout split as sorted OpenML dataset row IDs.

    Raises:
        ValueError: If there are multiple splits or overlapping partitions.
    """
    if task.get_split_dimensions() != (1, 1, 1):
        raise ValueError("expected exactly one train/test split")
    train, test = task.get_train_test_split_indices()
    train_ids, test_ids = tuple(sorted(map(int, train))), tuple(sorted(map(int, test)))
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("duplicate split row_id")
    if set(train_ids) & set(test_ids):
        raise ValueError("overlapping train/test row_id")
    return train_ids, test_ids


def rank_runs(
    runs: list[object], kind: str, labels: tuple[str, ...]
) -> list[tuple[object, float, str]]:
    """Rank candidate runs by comparable published or archived test F1.

    Selection on test outcomes is biased, including when F1 is calculated locally.
    OpenML's generic `f_measure` does not document the required averaging and is
    deliberately not used as a comparable score.
    """
    metric = "f1_binary" if kind == "binary" else "f1_macro"
    ranked = []
    for run in runs:
        if metric in run.evaluations:
            score = float(run.evaluations[metric])
            source = f"published {metric}"
        else:
            archived = run.predictions
            truth = archived["correct"].astype(str)
            prediction = archived["prediction"].astype(str)
            encoded_truth = pd.Categorical(truth, categories=labels).codes
            encoded_pred = pd.Categorical(prediction, categories=labels).codes
            if (encoded_truth < 0).any() or (encoded_pred < 0).any():
                raise ValueError("archived run has labels outside task classes")
            score = float(
                f1_score(
                    encoded_truth,
                    encoded_pred,
                    average="binary" if kind == "binary" else "macro",
                    labels=list(range(len(labels))),
                    zero_division=0,
                )
            )
            source = "archived test F1"
        ranked.append((run, score, source))
    return sorted(ranked, key=lambda item: (-item[1], item[0].id))


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
