"""Offline contract tests for the OpenML evaluation entry point."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_openml.py"


def test_entrypoint_exposes_modes_and_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for flag in ("--classification-type", "--n", "--seed", "--report", "--audit"):
        assert flag in result.stdout


def test_catalog_requires_exact_holdout_type_and_records_filter() -> None:
    from scripts.evaluate_openml import filter_catalog

    tasks = [
        SimpleNamespace(task_id=10, estimation_procedure_id=5, estimation_procedure={"type": "holdout"}),
        SimpleNamespace(task_id=11, estimation_procedure_id=6, estimation_procedure={"type": "holdout"}),
        SimpleNamespace(task_id=12, estimation_procedure_id=5, estimation_procedure={"type": "customholdout"}),
        SimpleNamespace(task_id=13, estimation_procedure_id=4, estimation_procedure={"type": "holdout"}),
    ]
    assert filter_catalog(tasks) == [10, 11]


def test_sample_selects_exactly_n_ids_before_any_eligibility_check() -> None:
    import numpy as np

    from scripts.evaluate_openml import sample_tasks

    candidates = list(range(100, 120))
    seed = 41
    expected = np.random.default_rng(seed).choice(candidates, size=4, replace=False).tolist()
    assert sample_tasks(candidates, n=4, seed=seed) == expected
    assert sample_tasks(candidates, n=4, seed=seed) == expected
    assert sample_tasks(candidates, n=None, seed=seed) == candidates
