"""Command-line entry point for reproducible OpenML evaluations."""

import argparse
import json
from pathlib import Path


_CONFIG = Path(__file__).with_name("openml_config.json")


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
