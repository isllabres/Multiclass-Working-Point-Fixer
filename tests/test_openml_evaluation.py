"""Offline contract tests for the OpenML evaluation entry point."""

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_openml.py"


def test_entrypoint_exposes_modes_and_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for flag in ("--classification-type", "--n", "--seed", "--report", "--audit"):
        assert flag in result.stdout
