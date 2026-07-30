from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER_ROOT = Path("your path")
PROJECT_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
CODE_ROOT = PROJECT_ROOT / "Code"
SCRIPTS_ROOT = CODE_ROOT / "UTD-MHAD" / "scripts"
for path in (CODE_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_same_path_context_stacking_audit as audit  # type: ignore[import-not-found]
from run_utd_canonical_rost_staged_rcf import (  # type: ignore[import-not-found]
    CONTINUATION_MANIFEST,
    POSTERIORS,
    load_canonical_utd,
)


CANONICAL_ROOT = CODE_ROOT / "UTD-MHAD/results/canonical_rost_paired_continuation_v1"
CHECKPOINT_ROOT = (
    CODE_ROOT
    / "UTD-MHAD/checkpoints/staged_rcf_v1/"
    "canonical_rost_paired_continuation_core5_s42-44_b5000"
)
LOG_ROOT = (
    CODE_ROOT
    / "UTD-MHAD/logs/staged_rcf_v1/"
    "canonical_rost_paired_continuation_core5_s42-44_b5000"
)
DEFAULT_OUTPUT = CANONICAL_ROOT / "same_path_context_stacking_adopted_contract_v1"
CANONICAL_LOADER_SCRIPT = SCRIPTS_ROOT / "run_utd_canonical_rost_staged_rcf.py"
SOURCE_AUDIT = CANONICAL_ROOT / "source_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate source artifacts")
    parser.add_argument(
        "--device", default="cuda:1" if audit.torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=5000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.check:
        required = (POSTERIORS, CONTINUATION_MANIFEST, SOURCE_AUDIT, CANONICAL_LOADER_SCRIPT)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing UTD-MHAD comparison artifacts: {missing}")
        bundle = load_canonical_utd()
        print(
            f"UTD-MHAD comparison check passed: development={len(bundle.labels)} "
            f"test={len(bundle.test_labels)}"
        )
        return

    spec = audit.DatasetSpec(
        key="utd_canonical",
        display_name="UTD-MHAD canonical ROST continuation",
        seeds=(42, 43, 44),
        checkpoint_root=CHECKPOINT_ROOT,
        log_root=LOG_ROOT,
        output_root=args.output_root,
        loader=load_canonical_utd,
    )
    result = audit.run_dataset(
        spec,
        args.device,
        args.bootstrap_rounds,
        output_override=args.output_root,
    )

    expected = {
        "acc": 0.827906976744186,
        "f1": 0.8262822309304907,
        "nll": 0.8384193778038025,
    }
    observed = result["metrics"]["DOME-X"]
    for metric, value in expected.items():
        if abs(observed[metric] - value) > 1e-12:
            raise RuntimeError(
                f"canonical DOME-X {metric} identity mismatch: "
                f"expected {value}, got {observed[metric]}"
            )
    for fit in result["fits"]:
        contract = fit["fit_contract"]
        if contract["final_fit_roles"] != ["fusion", "validation"]:
            raise RuntimeError("canonical same-signal readout used the wrong final-refit scope")
        if fit["selection_core"]["validation_metric_max_abs_error"] > 2e-7:
            raise RuntimeError("canonical pre-refit core replay failed")

    result["comparison_reference"] = {
        "name": "adopted canonical ROST staged Core5",
        "expected_metrics": expected,
        "role_seeds": [42, 43, 44],
        "final_refit_scope": "fusion plus validation (215 development rows)",
        "checkpoint_root": CHECKPOINT_ROOT,
    }
    result["historical_pipeline_exclusion"] = {
        "excluded_from_current_adopted_comparison": (
            "the older v7-posterior Core5 result at 83.95% accuracy"
        ),
        "reason": (
            "the paper now adopts the canonical paired-continuation posterior and "
            "its 82.79% staged Core5 result"
        ),
    }
    audit_file = getattr(audit, "__file__", None)
    if audit_file is None:
        raise RuntimeError("Unable to resolve the matched-stacking runner path")
    audit_path = Path(audit_file).resolve()
    result["canonical_provenance"] = {
        "posterior": POSTERIORS,
        "posterior_sha256": sha256_file(POSTERIORS),
        "continuation_manifest": CONTINUATION_MANIFEST,
        "continuation_manifest_sha256": sha256_file(CONTINUATION_MANIFEST),
        "source_identity_audit": SOURCE_AUDIT,
        "source_identity_audit_sha256": sha256_file(SOURCE_AUDIT),
        "canonical_loader": CANONICAL_LOADER_SCRIPT,
        "canonical_loader_sha256": sha256_file(CANONICAL_LOADER_SCRIPT),
        "adapter_script": SCRIPT_PATH,
        "adapter_script_sha256": sha256_file(SCRIPT_PATH),
        "same_signal_runner": audit_path,
        "same_signal_runner_sha256": sha256_file(audit_path),
    }
    result["source_artifacts"].extend(
        [
            {"path": SOURCE_AUDIT, "sha256": sha256_file(SOURCE_AUDIT)},
            {
                "path": CANONICAL_LOADER_SCRIPT,
                "sha256": sha256_file(CANONICAL_LOADER_SCRIPT),
            },
            {"path": SCRIPT_PATH, "sha256": sha256_file(SCRIPT_PATH)},
        ]
    )
    audit.save_json(result, args.output_root / "results.json")

    print("model | params | Acc | Macro-F1 | NLL | DOME-X-minus-model Acc CI")
    parameter_key = {
        "same_signal_logistic": "logistic",
        "same_signal_matched_mlp": "matched_mlp",
        "same_signal_contract_matched_mlp": "contract_matched_mlp",
    }
    for name in (
        "same_signal_logistic",
        "same_signal_matched_mlp",
        "same_signal_contract_matched_mlp",
    ):
        metric = result["metrics"][name]
        comparison = result["comparisons"][name]
        count = result["fits"][0][parameter_key[name]]["trainable_parameter_count"]
        interval = comparison["ci95"]["acc"]
        print(
            f"{name:24s} | {count:6d} | {100 * metric['acc']:5.2f} | "
            f"{100 * metric['f1']:8.2f} | {metric['nll']:.4f} | "
            f"{100 * comparison['observed']['acc']:+.2f} "
            f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}]"
        )
    print(
        f"canonical DOME-X: Acc={100 * observed['acc']:.2f}, "
        f"F1={100 * observed['f1']:.2f}, NLL={observed['nll']:.4f}"
    )
    print(f"results={args.output_root / 'results.json'}")


if __name__ == "__main__":
    main()
