"""Run matched ModelNet40 stacking baselines on frozen expert posteriors."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


DATASET = "modelnet"
PLACEHOLDER_ROOT = Path("your path")


def project_root() -> Path:
    configured = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
    if configured.exists():
        return configured
    return next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )


def load_audit():
    code_root = project_root() / "Code"
    if not code_root.is_dir():
        raise FileNotFoundError(
            "Set DOME_X_PROJECT_ROOT or replace Path(\"your path\") with the project root."
        )
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return importlib.import_module("run_matched_stacking_audit")


def check_environment(audit) -> None:
    bundle = audit.LOADERS[DATASET]()
    audit.validate_bundle(bundle)
    prediction_path = audit.ADOPTED_PREDICTIONS[DATASET]
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing adopted predictions: {prediction_path}")
    print(
        f"ModelNet40 comparison check passed: development={len(bundle.labels)} "
        f"test={len(bundle.test_labels)} source={prediction_path.name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate dependencies and artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fit-contract",
        choices=("adopted", "equal_label_budget"),
        default="equal_label_budget",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_audit()
    if args.check:
        check_environment(audit)
        return
    output_root = args.output_root
    if output_root is None:
        output_root = project_root() / "Code" / "results" / "matched_stacking_audit"
    result = audit.run_dataset(DATASET, args.device, output_root, args.fit_contract)
    for row in result["rows"]:
        print(
            f"{row['model']}: Acc={100 * row['acc']:.2f} "
            f"Macro-F1={100 * row['f1']:.2f} NLL={row['nll']:.4f}"
        )


if __name__ == "__main__":
    main()
