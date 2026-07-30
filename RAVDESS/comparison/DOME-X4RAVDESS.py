"""Compare RAVDESS fusion rules on actor-disjoint OOF posteriors."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
PLACEHOLDER_ROOT = Path("your path")
PROJECT_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
CODE_ROOT = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_ROOT))
from dome_x_pure_rcf import (  # type: ignore[import-not-found]  # noqa: E402
    EPS,
    component_forward,
    estimate_structure,
    normalize_np,
    probability_metrics,
    predict_pure_rcf,
    train_pure_rcf,
)


NC = 8
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = CODE_ROOT / "RAVDESS"
SOURCE = ROOT / "checkpoints" / "strict_oof_v2" / "ce_rost_oof_posteriors.npz"
SOURCE_MANIFEST = ROOT / "checkpoints" / "strict_oof_v2" / "experiment_manifest.json"
OUT = ROOT / "checkpoints" / "pure_rcf_v2"
OUT.mkdir(parents=True, exist_ok=True)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False), encoding="utf-8")


def actor_partitions(actors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = sorted(np.unique(actors).tolist())
    # Assigned at actor level so OOF structure, fit, and validation never share speakers.
    structure_actors = set(unique[::3])
    fusion_actors = set(unique[1::3])
    validation_actors = set(unique[2::3])
    return (
        np.flatnonzero(np.isin(actors, sorted(structure_actors))),
        np.flatnonzero(np.isin(actors, sorted(fusion_actors))),
        np.flatnonzero(np.isin(actors, sorted(validation_actors))),
    )


def test_posterior(archive, regime: str) -> np.ndarray:
    audio_folds = normalize_np(archive[f"{regime}_test_folds_audio"])
    visual_folds = normalize_np(archive[f"{regime}_test_folds_visual"])
    # Canonical test rule: average experts first, then apply one frozen RCF.
    return np.stack([audio_folds.mean(axis=0), visual_folds.mean(axis=0)], axis=1)


def weighted_baseline(oof: np.ndarray, labels: np.ndarray, test: np.ndarray, structure: np.ndarray) -> np.ndarray:
    weights = estimate_structure(oof[structure], labels[structure])["reliability"]
    return normalize_np((test * weights[None]).sum(axis=1))


def net_correction(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    reference_correct = reference.argmax(1) == labels
    candidate_correct = candidate.argmax(1) == labels
    to_correct = int((~reference_correct & candidate_correct).sum())
    to_wrong = int((reference_correct & ~candidate_correct).sum())
    return {"wrong_to_correct": to_correct, "correct_to_wrong": to_wrong, "net_correction": to_correct - to_wrong}


def run_regime(archive, regime: str, components_only: bool = False) -> dict:
    labels = np.asarray(archive[f"{regime}_labels"], dtype=np.int64)
    actors = np.asarray(archive[f"{regime}_actors"], dtype=np.int64)
    test_labels = np.asarray(archive["test_labels"], dtype=np.int64)
    oof = np.stack([normalize_np(archive[f"{regime}_oof_audio"]), normalize_np(archive[f"{regime}_oof_visual"])], axis=1)
    test = test_posterior(archive, regime)
    structure, fusion, validation = actor_partitions(actors)
    model, fit = train_pure_rcf(oof, labels, structure, fusion, validation, seed=SEED, device=DEVICE)
    pure, diag = predict_pure_rcf(model, test, DEVICE)
    output = {
        "Pure RCF": pure,
        "w/o Reliability": component_forward(model, test, DEVICE, reliability=False),
        "w/o Calibration": component_forward(model, test, DEVICE, calibration=False),
        "w/o Bias Transport": component_forward(model, test, DEVICE, transport=False),
        "w/o Disagreement Refinement": component_forward(model, test, DEVICE, refinement=False),
    }
    if not components_only:
        output = {
            "Average": normalize_np(test.mean(axis=1)),
            "Product": normalize_np(np.exp(np.log(test.clip(EPS, 1.0)).mean(axis=1))),
            "Weighted Average": weighted_baseline(oof, labels, test, structure),
            **output,
        }
    rows = []
    for name, probs in output.items():
        row = {"regime": regime, "method": name, **probability_metrics(test_labels, probs, NC)}
        if name != "Pure RCF":
            row.update(net_correction(probs, pure, test_labels))
        rows.append(row)
    return {"model": model, "fit": fit, "test": test, "output": output, "diag": diag, "rows": rows, "partitions": {"structure": structure, "fusion": fusion, "validation": validation}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate source artifacts")
    parser.add_argument("--artifact", type=Path, default=SOURCE)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--regime", choices=("CE", "ROST", "both"), default="both")
    parser.add_argument("--components-only", action="store_true", help="emit only Pure RCF and component deletion rows")
    args = parser.parse_args()
    if not args.artifact.exists() or not args.manifest.exists():
        raise FileNotFoundError("run train_ravdess_strict_oof_rost.py before Pure RCF evaluation")
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if source_manifest.get("permanent_test_actors") != [5, 10, 15, 20]:
        raise RuntimeError("source artifact does not use the required permanent test actors")
    if args.check:
        with np.load(args.artifact, allow_pickle=False) as archive:
            print(
                f"RAVDESS comparison check passed: development={len(archive['CE_labels'])} "
                f"test={len(archive['test_labels'])}"
            )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.artifact) as archive:
        regimes = ("CE", "ROST") if args.regime == "both" else (args.regime,)
        result = {regime: run_regime(archive, regime, components_only=args.components_only) for regime in regimes}
    rows = [row for regime in result.values() for row in regime["rows"]]
    for regime, value in result.items():
        torch.save({"state_dict": value["model"].state_dict(), "fit": value["fit"], "regime": regime}, args.output / f"{regime.lower()}_pure_rcf.pt")
        np.savez_compressed(
            args.output / f"{regime.lower()}_pure_rcf_predictions.npz",
            **{name.replace(" ", "_").replace("/", "_"): probs for name, probs in value["output"].items()},
            test_labels=np.load(args.artifact)["test_labels"],
            test_expert_posterior=value["test"],
            gate=value["diag"]["gate"],
            disagreement=value["diag"]["disagreement"],
            confidence=value["diag"]["confidence"],
        )
    manifest = {
        "dataset": "RAVDESS",
        "source_artifact": str(args.artifact),
        "source_artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "source_manifest": str(args.manifest),
        "protocol": "Actor-disjoint OOF structure/fusion/validation partitions. Test expert posterior averaged before one frozen Pure RCF inference.",
        "test_aggregation": "mean three fold expert posterior per modality, then one RCF",
        "forbidden": ["test-label fitting", "baseline blend", "candidate router", "pair memory", "class residual MLP"],
        "regime": args.regime,
        "components_only": args.components_only,
        "rows": rows,
    }
    write_json(manifest, args.output / "experiment_manifest.json")
    write_json(rows, args.output / "results.json")
    print(f"RAVDESS Pure RCF results written to {args.output}")
    for row in rows:
        print(f"{row['regime']:<4} {row['method']:<30} acc={row['acc']:.4f} f1={row['f1']:.4f} nll={row['nll']:.4f}")


if __name__ == "__main__":
    main()
