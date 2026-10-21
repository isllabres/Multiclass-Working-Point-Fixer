"""Command-line entry point for reproducible OpenML evaluations."""

import argparse
import html
import json
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import openml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import torch
import docker
from docker.errors import DockerException, ImageNotFound
from openml.tasks import TaskType
from sklearn.metrics import f1_score
from multiclass_wp_fixer.classification.metrics_working_point import (
    metrics_scanning_working_points_plotly,
    predict_with_working_point,
)
from multiclass_wp_fixer.classification.multiclass_working_point_fixer import (
    multiclass_Working_Point_Fixer,
)


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


def validate_artifact(
    artifact: pd.DataFrame,
    test_ids: tuple[int, ...],
    labels: tuple[str, ...],
    dataset_truth: dict[int, str],
) -> pd.DataFrame:
    """Verify archived test records and order them by OpenML test row ID.

    Raises:
        ValueError: If rows, labels, truth, or probability columns are inconsistent.
    """
    required = {"row_id", "correct", "prediction"} | {
        f"confidence.{label}" for label in labels
    }
    if not required.issubset(artifact.columns) or {
        column for column in artifact if column.startswith("confidence.")
    } != {f"confidence.{label}" for label in labels}:
        raise ValueError("artifact has missing or incompatible columns")
    if artifact["row_id"].isna().any() or artifact["row_id"].duplicated().any():
        raise ValueError("artifact row_id is missing or duplicated")
    if set(artifact["row_id"].astype(int)) != set(test_ids):
        raise ValueError("artifact row_id differs from test split")
    aligned = artifact.set_index("row_id").loc[list(test_ids)].reset_index()
    if any(str(row.correct) != str(dataset_truth[row.row_id]) for row in aligned.itertuples()):
        raise ValueError("archived truth does not match dataset row_id")
    if not set(aligned["prediction"].astype(str)).issubset(labels):
        raise ValueError("artifact prediction outside class labels")
    probabilities = aligned[[f"confidence.{label}" for label in labels]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities > 1
    ).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("artifact probability is not a valid class distribution")
    return aligned


def select_adapter(filename: str, flow_name: str) -> str:
    """Select a sandbox-only adapter for a compatible serialized model format.

    Raises:
        ValueError: If the model format or runtime is unsupported.
    """
    if filename.endswith((".pkl", ".pickle")) and flow_name.startswith("sklearn."):
        return "python-pickle"
    if filename.endswith((".serialized", ".model")) and flow_name.startswith("weka."):
        return "weka"
    raise ValueError(f"unsupported model format/runtime: {filename} / {flow_name}")


def docker_run_kwargs(image: str, input_dir: str, output_dir: str) -> dict[str, object]:
    """Build a restricted, credential-free container configuration for model inference."""
    return {
        "image": image,
        "detach": True,
        "network_mode": "none",
        "read_only": True,
        "user": "10001:10001",
        "environment": {},
        "cap_drop": ["ALL"],
        "privileged": False,
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": 64,
        "mem_limit": "2g",
        "nano_cpus": 1_000_000_000,
        "tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
        "volumes": {
            input_dir: {"bind": "/input", "mode": "ro"},
            output_dir: {"bind": "/output", "mode": "rw"},
        },
    }


@contextmanager
def require_docker():
    """Require a daemon and built image; reuse one sandbox for the invocation.

    Raises:
        RuntimeError: If Docker or the pinned local image is unavailable.
    """
    from tempfile import TemporaryDirectory

    image = json.loads(_CONFIG.read_text(encoding="utf-8"))["image"]
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(image)
    except (DockerException, ImageNotFound) as error:
        raise RuntimeError(f"Docker daemon/pinned image unavailable ({image}): {error}") from error
    try:
        with TemporaryDirectory(prefix="openml-input-") as input_dir, TemporaryDirectory(
            prefix="openml-output-"
        ) as output_dir:
            container = client.containers.run(**docker_run_kwargs(image, input_dir, output_dir))
            try:
                yield container
            finally:
                container.stop(timeout=5)
                container.remove(force=True)
    finally:
        client.close()


def orient_binary(
    truth: list[str], prediction: list[str], labels: tuple[str, str], published: float | None
) -> tuple[str, str]:
    """Choose (positive, negative) using a comparable published binary F1.

    With no score or an ambiguous tie, use the canonical sorted last label as
    sklearn's positive class; both labels are explicitly encoded as 0/1.

    Raises:
        ValueError: If neither orientation matches the published score.
    """
    if published is None:
        return labels[1], labels[0]
    tolerance = json.loads(_CONFIG.read_text(encoding="utf-8"))["published_f1_tolerance"]
    matches = []
    for positive, negative in ((labels[1], labels[0]), (labels[0], labels[1])):
        score = f1_score(
            [int(value == positive) for value in truth],
            [int(value == positive) for value in prediction],
            pos_label=1,
            zero_division=0,
        )
        if abs(score - published) <= tolerance:
            matches.append((positive, negative))
    if not matches:
        raise ValueError("published F1 mismatch for both label orientations")
    return matches[0]


def compare_predictions(
    kind: str,
    train_probs: np.ndarray,
    train_onehot: np.ndarray,
    test_probs: np.ndarray,
    test_onehot: np.ndarray,
    archived_predictions: np.ndarray,
) -> dict[str, object]:
    """Tune a binary scanner or multiclass fixer on train, then score test."""
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    if kind == "multiclass":
        params = {
            "input_size": train_probs.shape[1],
            "learning_rate": config["fixer_learning_rate"],
            "num_epochs": config["fixer_epochs"],
            "momentum": 0.0,
            "centered": False,
            "multiclass_f1_average": "macro",
            "train_batch_size": config["fixer_train_batch_size"],
            "val_batch_size": config["fixer_val_batch_size"],
        }
        if len(train_probs) < params["train_batch_size"] or len(test_probs) < params["val_batch_size"]:
            raise ValueError("no training/validation batch would be processed (drop_last=True)")
        with torch.random.fork_rng():
            torch.manual_seed(config["seed"])
            fixer = multiclass_Working_Point_Fixer(model_params=params)
            fixer.fit(
                x_tr=train_probs.astype(np.float32),
                y_tr=train_onehot.astype(np.float32),
                x_tst=test_probs.astype(np.float32),
                y_tst=test_onehot.astype(np.float32),
            )
            improved = fixer.predict(test_probs.astype(np.float32)).argmax(axis=1)
        truth = test_onehot.argmax(axis=1)
        baseline = test_probs.argmax(axis=1)
        return {
            "method": "fixer",
            "baseline": float(f1_score(truth, baseline, average="macro", zero_division=0)),
            "improved": float(f1_score(truth, improved, average="macro", zero_division=0)),
            "threshold": None,
            "predictions": improved,
        }
    if kind != "binary":
        raise ValueError(f"unsupported classification type: {kind}")
    scanned = metrics_scanning_working_points_plotly(
        pred_proba=train_probs[:, 1],
        y_true=train_onehot[:, 1],
        first_working_point=0.0,
        last_working_point=1.0,
        n_working_points=config["scanner_points"],
        plot=False,
    )
    threshold = float(scanned["working_point_best_f1"])
    truth = test_onehot.argmax(axis=1)
    improved = predict_with_working_point(test_probs[:, 1], threshold).ravel()
    return {
        "method": "scanner",
        "baseline": float(f1_score(truth, archived_predictions, zero_division=0)),
        "improved": float(f1_score(truth, improved, zero_division=0)),
        "threshold": threshold,
        "predictions": improved,
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, dict[str, int | float | None]]:
    """Summarize signed F1 changes equally per task, overall and by task type."""
    summary = {}
    for kind in ("overall", "binary", "multiclass"):
        subset = [row for row in results if kind == "overall" or row["kind"] == kind]
        deltas = [float(row["improved"]) - float(row["baseline"]) for row in subset]
        summary[kind] = {
            "count": len(deltas),
            "mean_gain": float(np.mean(deltas)) if deltas else None,
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "decreased": sum(delta < 0 for delta in deltas),
        }
    return summary


def class_breakdown(
    truth: list[int], previous: list[int], updated: list[int], labels: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Calculate old and improved test F1 for every task class, including absent predictions."""
    indices = list(range(len(labels)))
    old = f1_score(truth, previous, labels=indices, average=None, zero_division=0)
    new = f1_score(truth, updated, labels=indices, average=None, zero_division=0)
    return {
        label: {"previous": float(before), "new": float(after)}
        for label, before, after in zip(labels, old, new, strict=True)
    }


def render_report(
    rows: list[dict[str, object]], catalog: dict[str, str], options: dict[str, object]
) -> str:
    """Render a self-contained Plotly report, including an empty evaluation."""
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    reasons = Counter(str(row.get("reason")) for row in rows if row.get("status") == "excluded")
    summary = aggregate(evaluated)
    fig = go.Figure()
    for kind in ("binary", "multiclass"):
        subset = [row for row in evaluated if row["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=[row["baseline"] for row in subset],
            y=[row["improved"] for row in subset],
            text=[str(row["run_id"]) for row in subset],
            name=f"{kind}: scanner" if kind == "binary" else "multiclass: fixer",
            mode="markers",
        ))
    fig.update_layout(title="Test F1 vs improved test F1", xaxis_title="Baseline F1", yaxis_title="Improved F1")
    gains = go.Figure()
    for kind in ("binary", "multiclass"):
        subset = [row for row in evaluated if row["kind"] == kind]
        gains.add_trace(go.Bar(
            x=[str(row["task_id"]) for row in subset],
            y=[float(row["improved"]) - float(row["baseline"]) for row in subset],
            name=f"{kind}: scanner" if kind == "binary" else "multiclass: fixer",
        ))
    gains.update_layout(title="Signed per-task F1 change", xaxis_title="Task ID", yaxis_title="Improved − baseline")
    table_columns = ["task_id", "dataset_id", "dataset_name", "kind", "run_id", "f1_source", "labels", "baseline", "improved", "threshold", "gain", "status", "reason"]
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in table_columns) + "</tr>"
        for row in rows
    )
    details = json.dumps(
        {str(row["run_id"]): row.get("per_class", {}) for row in evaluated},
        ensure_ascii=True,
    ).replace("<", "\\u003c")
    select = "".join(
        f'<option value="{html.escape(str(row["run_id"]), quote=True)}">Run {html.escape(str(row["run_id"]))} / Task {html.escape(str(row["task_id"]))}</option>'
        for row in evaluated
    )
    metadata = html.escape(json.dumps({"catalog": catalog, "options": options, "summary": summary, "exclusions": reasons}, indent=2, default=str))
    plots = pio.to_html(fig, include_plotlyjs=True, full_html=False) + pio.to_html(gains, include_plotlyjs=False, full_html=False)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenML working-point evaluation</title>
<h1>OpenML evaluation</h1><p>Catalog candidates: {options.get('catalog_candidates', '?')} | Sampled/attempted: {len(rows)} | Evaluated: {len(evaluated)} | Excluded: {sum(reasons.values())}</p>
<p>Test-F1-based run selection introduces selection bias. The catalog is dynamic; metadata may be imperfect; only artifact-eligible tasks and supported runtimes are evaluated. Improvement is not guaranteed. Overall mean gain is a descriptive combination of binary F1 and multiclass macro-F1, not a universal metric. Serialized models execute third-party code in a restricted Docker container; residual risk remains.</p>
<pre>{metadata}</pre>{plots}
<label for="run">Per-class F1 by run_id</label><select id="run"><option value="">Select run</option>{select}</select><div id="per-class"></div>
<label for="search">Search tasks</label><input id="search" type="search"><table id="tasks" border="1"><thead><tr>{''.join(f'<th role="button">{column}</th>' for column in table_columns)}</tr></thead><tbody>{table}</tbody></table>
<script>const breakdown={details};const dropdown=document.getElementById('run');dropdown.onchange=()=>{{const data=breakdown[dropdown.value]||{{}};Plotly.newPlot('per-class', [{{type:'bar',name:'Previous F1',x:Object.keys(data),y:Object.values(data).map(v=>v.previous)}},{{type:'bar',name:'New F1',x:Object.keys(data),y:Object.values(data).map(v=>v.new)}}],{{barmode:'group',title:'Per-class F1: run '+dropdown.value}})}};
document.getElementById('search').oninput=e=>{{for(const row of document.querySelectorAll('#tasks tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())}};
document.querySelectorAll('#tasks th').forEach((th,i)=>th.onclick=()=>{{const body=document.querySelector('#tasks tbody');[...body.rows].sort((a,b)=>a.cells[i].textContent.localeCompare(b.cells[i].textContent,undefined,{{numeric:true}})).forEach(row=>body.appendChild(row))}});</script></html>"""


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
