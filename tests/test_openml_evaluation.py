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


def test_classify_target_rejects_multilabel_and_missing_values() -> None:
    import pandas as pd
    import pytest

    from scripts.evaluate_openml import classify_target

    assert classify_target(pd.Series(["no", "yes", "no"])) == ("binary", ("no", "yes"))
    assert classify_target(pd.Series(["b", "c", "a"])) == ("multiclass", ("a", "b", "c"))
    with pytest.raises(ValueError, match="missing target"):
        classify_target(pd.Series(["yes", None, "no"]))
    with pytest.raises(ValueError, match="multilabel"):
        classify_target(pd.DataFrame({"a": [0, 1], "b": [1, 0]}))


def test_discover_split_rejects_multiple_splits_and_aligns_by_row_id() -> None:
    import numpy as np
    import pytest

    from scripts.evaluate_openml import discover_split

    task = SimpleNamespace(
        get_split_dimensions=lambda: (1, 1, 1),
        get_train_test_split_indices=lambda: (np.array([3, 0]), np.array([2, 1])),
    )
    assert discover_split(task) == ((0, 3), (1, 2))
    task.get_split_dimensions = lambda: (1, 2, 1)
    with pytest.raises(ValueError, match="one train/test split"):
        discover_split(task)


def test_read_procedure_id_from_openml_xml_not_sdk_default() -> None:
    from scripts.evaluate_openml import read_procedure

    xml = '<oml:task xmlns:oml="http://openml.org/openml"><oml:input name="estimation_procedure"><oml:estimation_procedure><oml:id>6</oml:id><oml:type>holdout</oml:type></oml:estimation_procedure></oml:input></oml:task>'
    assert read_procedure(xml) == (6, "holdout")


def test_rank_runs_uses_only_comparable_published_f1() -> None:
    import pandas as pd

    from scripts.evaluate_openml import rank_runs

    archived = pd.DataFrame({"correct": ["a", "a", "b", "b"], "prediction": ["a", "b", "b", "b"]})
    runs = [
        SimpleNamespace(id=1, evaluations={"f_measure": 1.0}, predictions=archived),
        SimpleNamespace(id=2, evaluations={"f1_binary": 0.7}, predictions=archived),
    ]
    ranked = rank_runs(runs, "binary", ("a", "b"))
    assert [(run.id, score, source) for run, score, source in ranked] == [
        (1, 0.8, "archived test F1"),
        (2, 0.7, "published f1_binary"),
    ]
