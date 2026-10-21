"""Command-line entry point for reproducible OpenML evaluations."""

import argparse
import io
import html
import json
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable
from xml.etree import ElementTree

import openml
import arff
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
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Sandbox:
    """A single restricted Docker container with isolated per-task input/output roots."""

    container: object
    input_dir: Path
    output_dir: Path

    def exec_run(self, *args: object, **kwargs: object) -> object:
        """Start a fresh model-inference process in the existing container."""
        return self.container.exec_run(*args, **kwargs)

    @property
    def image(self) -> object:
        """Return the actual immutable image identity from Docker."""
        return self.container.image


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


def choose_run(
    runs: list[object],
    kind: str,
    labels: tuple[str, ...],
    test_ids: tuple[int, ...],
    truth: dict[int, str],
) -> tuple[object, pd.DataFrame, list[str]]:
    """Walk test-F1-ranked runs and retain the first artifact-eligible candidate.

    Raises:
        ValueError: If every archived candidate is ineligible.
    """
    rejected = []
    for run, _, _ in rank_runs(runs, kind, labels):
        try:
            if not run.output_files.get("model_serialized"):
                raise ValueError("no downloadable fitted model")
            extension = ".model" if run.flow_name.startswith("weka.") else ".pkl"
            select_adapter(f"model{extension}", run.flow_name)
            aligned = validate_artifact(run.predictions, test_ids, labels, truth)
            if kind == "binary":
                published = run.evaluations.get("f1_binary")
                orient_binary(aligned["correct"].astype(str).tolist(), aligned["prediction"].astype(str).tolist(), labels, published)
            return run, aligned, rejected
        except (KeyError, ValueError) as error:
            rejected.append(f"run {run.id}: {error}")
    raise ValueError("no eligible run: " + "; ".join(rejected))


def select_adapter(filename: str, flow_name: str) -> str:
    """Select a sandbox-only adapter for a compatible serialized model format.

    Raises:
        ValueError: If the model format or runtime is unsupported.
    """
    if filename.endswith((".pkl", ".pickle")) and flow_name.startswith("sklearn."):
        return "python-pickle"
    if filename.endswith((".serialized", ".model")) and flow_name.startswith("weka.J48"):
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
            Path(input_dir).chmod(0o755)
            Path(output_dir).chmod(0o755)
            container = client.containers.run(**docker_run_kwargs(image, input_dir, output_dir))
            try:
                yield Sandbox(container, Path(input_dir), Path(output_dir))
            finally:
                container.stop(timeout=5)
                container.remove(force=True)
    finally:
        client.close()


def infer_train(
    sandbox: Sandbox,
    model_bytes: bytes,
    adapter: str,
    features: pd.DataFrame,
    labels: tuple[str, ...],
    target_name: str,
    target_categories: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Infer train probabilities in one fresh container process per fitted model.

    Raises:
        ValueError: For unsupported schemas, runtime errors, or invalid output.
    """
    with TemporaryDirectory(dir=sandbox.input_dir) as input_dir, TemporaryDirectory(
        dir=sandbox.output_dir
    ) as output_dir:
        source = Path(input_dir)
        destination = Path(output_dir)
        source.chmod(0o755)
        destination.chmod(0o777)
        categorical = {
            column: features[column].cat.categories.astype(str).tolist()
            for column in features.columns if isinstance(features[column].dtype, pd.CategoricalDtype)
        }
        schema = {"columns": list(features.columns), "categorical": categorical, "labels": list(labels)}
        if adapter == "python-pickle":
            (source / "model.pkl").write_bytes(model_bytes)
            features.to_csv(source / "features.csv", index=False)
            (source / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
            command = ["python", "/opt/openml_inference.py", f"/input/{source.name}", f"/output/{destination.name}"]
        elif adapter == "weka":
            attributes = []
            for column in features.columns:
                if column in categorical:
                    attributes.append((column, categorical[column]))
                elif pd.api.types.is_numeric_dtype(features[column]):
                    attributes.append((column, "NUMERIC"))
                else:
                    raise ValueError(f"unsupported Weka feature schema: {column}")
            attributes.append((target_name, list(target_categories or labels)))
            data = [
                [None if pd.isna(value) else value for value in row] + [None]
                for row in features.itertuples(index=False, name=None)
            ]
            (source / "train.arff").write_text(arff.dumps({"relation": "train", "attributes": attributes, "data": data}), encoding="utf-8")
            (source / "model.model").write_bytes(model_bytes)
            command = ["java", "-cp", "/opt/weka.jar", "weka.classifiers.trees.J48", "-l", f"/input/{source.name}/model.model", "-T", f"/input/{source.name}/train.arff", "-p", "0", "-distribution"]
        else:
            raise ValueError(f"unsupported sandbox adapter: {adapter}")
        for file_path in source.iterdir():
            file_path.chmod(0o444)
        result = sandbox.exec_run(command, user="10001:10001", environment={}, workdir="/opt")
        if result.exit_code:
            raise ValueError(f"{adapter} sandbox inference failed: {result.output.decode(errors='replace')[:500]}")
        if adapter == "python-pickle":
            probabilities = np.load(destination / "probability.npy", allow_pickle=False)
        else:
            lines = [line.split() for line in result.output.decode(errors="replace").splitlines()]
            distributions = []
            for words in lines:
                if words and words[0].isdigit() and len(words) >= 4 and words[1].endswith(":?"):
                    distributions.append([float(value.replace("*", "")) for value in words[-1].split(",")])
            probabilities = np.asarray(distributions, dtype=float)
            order = tuple(target_categories or labels)
            if set(order) != set(labels):
                raise ValueError("Weka target schema differs from archived classes")
            if probabilities.shape == (len(features), len(order)):
                probabilities = probabilities[:, [order.index(label) for label in labels]]
        if probabilities.shape != (len(features), len(labels)) or not np.isfinite(probabilities).all() or (
            probabilities < 0
        ).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=0.02):
            raise ValueError("sandbox probabilities are not aligned class distributions")
        return probabilities


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
    args = parse_args(argv)
    catalog, filters = query_catalog()
    selected = sample_tasks(catalog, args.n, args.seed)
    options = {
        "classification_type": args.classification_type,
        "n": args.n,
        "seed": args.seed,
        "catalog_candidates": len(catalog),
        "selected_task_ids": selected,
        "settings": json.loads(_CONFIG.read_text(encoding="utf-8")),
        "versions": {package: version(package) for package in ("openml", "docker", "numpy", "scikit-learn", "torch", "plotly")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    rows: list[dict[str, object]] = []
    with require_docker() as container:
        options["image_id"] = getattr(getattr(container, "image", None), "id", "unknown")
        for index, task_id in enumerate(selected, start=1):
            _LOG.info("Attempting task %s (%s/%s)", task_id, index, len(selected))
            row: dict[str, object] = {"task_id": task_id}
            try:
                task = openml.tasks.get_task(task_id, download_splits=True)
                row["dataset_id"] = task.dataset_id
                dataset = task.get_dataset()
                row["dataset_name"] = dataset.name
                features, target, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
                kind, labels = classify_target(target)
                row["kind"] = kind
                row["labels"] = labels
                if args.classification_type != "both" and kind != args.classification_type:
                    raise ValueError("task type differs from requested classification type")
                train_ids, test_ids = discover_split(task)
                candidates = openml.runs.list_runs(task=[task_id], output_format="dict")
                if not candidates:
                    raise ValueError("no candidate runs")
                runs = [openml.runs.get_run(int(run_id)) for run_id in candidates]
                truth_by_id = {row_id: str(target.iloc[row_id]) for row_id in test_ids}
                run, aligned, rejected = choose_run(runs, kind, labels, test_ids, truth_by_id)
                ordered = labels
                if kind == "binary":
                    positive, negative = orient_binary(
                        aligned["correct"].astype(str).tolist(),
                        aligned["prediction"].astype(str).tolist(),
                        labels,
                        run.evaluations.get("f1_binary"),
                    )
                    ordered = (negative, positive)
                row["labels"] = ordered
                row["rejected_runs"] = rejected
                model_id = run.output_files["model_serialized"]
                model_response = requests.get(
                    f"https://www.openml.org/data/download/{model_id}", timeout=60
                )
                model_response.raise_for_status()
                adapter = select_adapter(
                    "model.model" if run.flow_name.startswith("weka.") else "model.pkl",
                    run.flow_name,
                )
                category_order = tuple(map(str, target.cat.categories)) if isinstance(
                    target.dtype, pd.CategoricalDtype
                ) else ordered
                train_probs = infer_train(
                    container,
                    model_response.content,
                    adapter,
                    features.iloc[list(train_ids)],
                    ordered,
                    task.target_name,
                    category_order,
                )
                test_probs = aligned[[f"confidence.{label}" for label in ordered]].to_numpy(dtype=float)
                train_y = pd.Categorical(target.iloc[list(train_ids)].astype(str), categories=ordered).codes
                test_y = pd.Categorical(aligned["correct"].astype(str), categories=ordered).codes
                archived_y = pd.Categorical(aligned["prediction"].astype(str), categories=ordered).codes
                if (train_y < 0).any() or (test_y < 0).any() or (archived_y < 0).any():
                    raise ValueError("target labels not aligned to the model class order")
                comparison = compare_predictions(
                    kind,
                    train_probs,
                    np.eye(len(ordered), dtype=np.float32)[train_y],
                    test_probs,
                    np.eye(len(ordered), dtype=np.float32)[test_y],
                    archived_y,
                )
                original = archived_y if kind == "binary" else test_probs.argmax(axis=1)
                row.update(
                    status="evaluated",
                    run_id=run.id,
                    adapter=adapter,
                    f1_source="published " + ("f1_binary" if kind == "binary" else "f1_macro")
                    if ("f1_binary" if kind == "binary" else "f1_macro") in run.evaluations
                    else "archived test F1 (biased selection)",
                    baseline=comparison["baseline"],
                    improved=comparison["improved"],
                    gain=float(comparison["improved"]) - float(comparison["baseline"]),
                    threshold=comparison["threshold"],
                    per_class=class_breakdown(
                        test_y.tolist(), original.tolist(), comparison["predictions"].tolist(), ordered
                    ),
                )
            except ValueError as error:
                row.update(status="excluded", reason=str(error))
            except (requests.RequestException, RuntimeError, OSError) as error:
                row.update(status="error", reason=str(error))
            rows.append(row)
            _LOG.info("Task %s: %s — %s", task_id, row["status"], row.get("reason", ""))
            args.audit.parent.mkdir(parents=True, exist_ok=True)
            args.audit.write_text(
                json.dumps({"catalog": filters, "catalog_candidates": len(catalog), "options": options, "tasks": rows}, indent=2, default=str),
                encoding="utf-8",
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(rows, filters, options), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
