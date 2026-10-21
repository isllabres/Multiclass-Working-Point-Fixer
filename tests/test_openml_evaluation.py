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


def test_archived_artifact_requires_split_rows_and_class_distributions() -> None:
    import pandas as pd
    import pytest

    from scripts.evaluate_openml import validate_artifact

    artifact = pd.DataFrame(
        {
            "row_id": [7, 2], "correct": ["b", "a"], "prediction": ["b", "a"],
            "confidence.a": [0.1, 0.8], "confidence.b": [0.9, 0.2],
        }
    )
    aligned = validate_artifact(artifact, (2, 7), ("a", "b"), {2: "a", 7: "b"})
    assert aligned["row_id"].tolist() == [2, 7]
    assert aligned[["confidence.a", "confidence.b"]].to_numpy().tolist() == [[0.8, 0.2], [0.1, 0.9]]
    with pytest.raises(ValueError, match="row_id"):
        validate_artifact(artifact, (2, 8), ("a", "b"), {2: "a", 8: "b"})
    artifact.loc[0, "confidence.b"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        validate_artifact(artifact, (2, 7), ("a", "b"), {2: "a", 7: "b"})


def test_adapter_dispatch_rejects_unsupported_models() -> None:
    import pytest

    from scripts.evaluate_openml import select_adapter

    assert select_adapter("fitted.pkl", "sklearn.pipeline.Pipeline") == "python-pickle"
    assert select_adapter("model.serialized", "weka.J48") == "weka"
    with pytest.raises(ValueError, match="unsupported"):
        select_adapter("model.onnx", "tensorflow.keras.Model")


def test_docker_restrictions_disallow_host_access_and_privilege() -> None:
    from scripts.evaluate_openml import docker_run_kwargs

    flags = docker_run_kwargs("evaluation:sha256-test", "/safe/input", "/safe/output")
    assert flags["network_mode"] == "none"
    assert flags["read_only"] is True
    assert flags["cap_drop"] == ["ALL"]
    assert flags["privileged"] is False
    assert flags["user"] == "10001:10001"
    assert flags["environment"] == {}
    assert flags["pids_limit"] > 0
    assert flags["mem_limit"]
    assert flags["nano_cpus"] > 0
    assert flags["volumes"]["/safe/input"]["mode"] == "ro"
    assert not any("docker.sock" in key for key in flags["volumes"])


def test_binary_orientation_matches_published_f1_or_rejects_mismatch() -> None:
    import pytest

    from scripts.evaluate_openml import orient_binary

    truth = ["a", "a", "b", "b"]
    predictions = ["a", "a", "a", "b"]
    assert orient_binary(truth, predictions, ("a", "b"), 2 / 3) == ("b", "a")
    assert orient_binary(truth, predictions, ("a", "b"), None) == ("b", "a")
    with pytest.raises(ValueError, match="published F1 mismatch"):
        orient_binary(truth, predictions, ("a", "b"), 0.01)


def test_binary_scanner_selects_on_train_and_applies_to_test() -> None:
    import numpy as np

    from scripts.evaluate_openml import compare_predictions

    train = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
    train_y = np.eye(2)[[0, 1, 1, 0]]
    test = np.array([[0.65, 0.35], [0.35, 0.65]])
    test_y = np.eye(2)[[1, 0]]
    result = compare_predictions("binary", train, train_y, test, test_y, np.array([0, 1]))
    assert result["method"] == "scanner"
    assert result["baseline"] == 0.0
    assert result["improved"] == 2 / 3
    assert 0.1 < result["threshold"] <= 0.35


def test_multiclass_fixer_is_fit_on_train_and_not_scanned() -> None:
    import numpy as np

    from scripts.evaluate_openml import compare_predictions

    train = np.tile(np.eye(3) * 0.8 + 0.2 / 3, (2, 1)).astype(np.float32)
    train_y = np.tile(np.eye(3), (2, 1)).astype(np.float32)
    test = (np.eye(3) * 0.8 + 0.2 / 3).astype(np.float32)
    test_y = np.eye(3).astype(np.float32)
    result = compare_predictions("multiclass", train, train_y, test, test_y, np.array([0, 1, 2]))
    assert result["method"] == "fixer"
    assert result["baseline"] == 1.0
    assert result["improved"] == 1.0
    assert result["threshold"] is None


def test_aggregate_keeps_negative_gains_and_equal_task_weights() -> None:
    from scripts.evaluate_openml import aggregate

    results = [
        {"kind": "binary", "baseline": 0.2, "improved": 0.5},
        {"kind": "binary", "baseline": 0.7, "improved": 0.5},
        {"kind": "multiclass", "baseline": 0.3, "improved": 0.3},
    ]
    summary = aggregate(results)
    assert round(summary["overall"]["mean_gain"], 6) == round(0.1 / 3, 6)
    assert round(summary["binary"]["mean_gain"], 6) == 0.05
    assert summary["overall"]["improved"] == 1
    assert summary["overall"]["decreased"] == 1
    assert summary["overall"]["unchanged"] == 1


def test_breakdown_preserves_per_class_old_and_new_f1() -> None:
    from scripts.evaluate_openml import class_breakdown

    breakdown = class_breakdown([0, 1, 2, 2], [0, 0, 1, 2], [0, 1, 1, 2], ("a", "b", "c"))
    assert breakdown == {"a": {"previous": 2 / 3, "new": 1.0}, "b": {"previous": 0.0, "new": 2 / 3}, "c": {"previous": 2 / 3, "new": 2 / 3}}


def test_html_report_is_standalone_when_every_task_is_excluded() -> None:
    from scripts.evaluate_openml import render_report

    html = render_report(
        [{"task_id": 7, "status": "excluded", "reason": "no fitted artifact"}],
        {"retrieved_at": "2026-10-21T00:00:00+00:00", "filters": "procedure.type == holdout"},
        {"n": 1, "seed": 41, "classification_type": "both"},
    )
    assert "plotly.js" in html.lower()
    assert "no fitted artifact" in html
    assert "Evaluated: 0" in html
    assert "selection bias" in html.lower()
    assert "Search tasks" in html


def test_docker_integration_is_opt_in() -> None:
    import os
    import pytest

    if os.environ.get("OPENML_DOCKER_INTEGRATION") != "1":
        pytest.skip("set OPENML_DOCKER_INTEGRATION=1 to test the local pinned image")
    from scripts.evaluate_openml import require_docker

    with require_docker() as container:
        result = container.exec_run(["python", "--version"], user="10001:10001")
        assert result.exit_code == 0


def test_sample_attempts_exact_n_tasks_and_audits_exclusions(monkeypatch, tmp_path) -> None:
    import json
    import pandas as pd
    import scripts.evaluate_openml as evaluation

    listing = pd.DataFrame({"estimation_procedure": ["33% Holdout set"] * 3}, index=[7, 8, 9])
    xml = '<oml:task xmlns:oml="http://openml.org/openml"><oml:input name="estimation_procedure"><oml:estimation_procedure><oml:id>6</oml:id><oml:type>holdout</oml:type></oml:estimation_procedure></oml:input></oml:task>'
    monkeypatch.setattr(evaluation.openml.tasks, "list_tasks", lambda **kwargs: listing)
    monkeypatch.setattr(evaluation.requests, "get", lambda *args, **kwargs: SimpleNamespace(text=xml, raise_for_status=lambda: None))
    monkeypatch.setattr(evaluation.openml.tasks, "get_task", lambda tid, **kwargs: SimpleNamespace(
        task_id=tid, dataset_id=12, target_name="class", get_split_dimensions=lambda: (1, 1, 1),
        get_train_test_split_indices=lambda: ([0, 1, 2], [3, 4, 5]),
        get_dataset=lambda: SimpleNamespace(name="toy", get_data=lambda **kwargs: (
            pd.DataFrame({"feature": range(6)}), pd.Series(["a", "b", "a", "b", "a", "b"]), [False], ["feature"])),
    ))
    monkeypatch.setattr(evaluation.openml.runs, "list_runs", lambda **kwargs: {})
    container = SimpleNamespace(stop=lambda **kwargs: None, remove=lambda **kwargs: None)
    client = SimpleNamespace(ping=lambda: True, images=SimpleNamespace(get=lambda image: SimpleNamespace(id="sha256:test")),
        containers=SimpleNamespace(run=lambda **kwargs: container), close=lambda: None)
    monkeypatch.setattr(evaluation.docker, "from_env", lambda: client)
    report = tmp_path / "sample.html"
    audit = tmp_path / "sample.json"
    assert evaluation.main(["--n", "1", "--seed", "41", "--report", str(report), "--audit", str(audit)]) == 0
    recorded = json.loads(audit.read_text())
    assert recorded["catalog_candidates"] == 3
    assert len(recorded["tasks"]) == 1
    assert recorded["tasks"][0]["status"] == "excluded"
    assert recorded["tasks"][0]["reason"] == "no candidate runs"
    assert "Evaluated: 0" in report.read_text()
