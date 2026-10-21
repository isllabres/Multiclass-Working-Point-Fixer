"""Inference on untrusted serialized models; execute ONLY inside restricted Docker."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Load a Python classifier inside Docker and output only probabilities."""
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    features = pd.read_csv(input_dir / "features.csv")
    if list(features.columns) != schema["columns"]:
        raise ValueError("feature order differs from archived schema")
    for column, categories in schema["categorical"].items():
        features[column] = pd.Categorical(features[column], categories=categories)
    with (input_dir / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)  # untrusted code executes here, never on the host
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        raise ValueError("unsupported model: missing predict_proba or feature schema")
    if list(model.feature_names_in_) != schema["columns"]:
        raise ValueError("model feature order differs from dataset")
    class_order = list(map(str, model.classes_))
    if set(class_order) != set(schema["labels"]):
        raise ValueError("model classes differ from dataset target")
    probability = np.asarray(model.predict_proba(features), dtype=float)
    if probability.shape != (len(features), len(class_order)):
        raise ValueError("unexpected model probability dimensions")
    np.save(output_dir / "probability.npy", probability[:, [class_order.index(label) for label in schema["labels"]]])


if __name__ == "__main__":
    main()
