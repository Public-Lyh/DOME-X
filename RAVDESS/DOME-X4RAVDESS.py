from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from dome_x_staged_rcf import (  # type: ignore[import-not-found]
    PoolConfig,
    fit_refiner,
    grouped_role_split,
    normalize_np,
    predict_refiner,
    probability_metrics,
    refit_refiner_fixed_epochs,
    refit_staged_core,
)
from dome_x_staged_runner import (  # type: ignore[import-not-found]
    compact_metadata,
    fit_variant,
    refiner_config,
)


DEFAULT_PROBS = (
    PROJECT_ROOT
    / "data/RAVDESS/checkpoints/domex_x_ravdess_ikun_final_compare/probs.pkl"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data/RAVDESS/checkpoints/domex_v22_ikun_bias_learning/logits.pkl"
)
DEFAULT_OUTPUT = CODE_ROOT / "RAVDESS/checkpoints/staged_rcf_v1/actor_role_core5_s42-46"
DEFAULT_LOG = CODE_ROOT / "RAVDESS/logs/staged_rcf_v1/actor_role_core5_s42-46"
EXPECTED_DEVELOPMENT_ACTORS = set(range(1, 16))
EXPECTED_TEST_ACTORS = set(range(16, 25))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, ensure_ascii=False, indent=2, sort_keys=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain at least three unique integers")
    return seeds


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def actor_ids(keys: list[Any]) -> np.ndarray:
    try:
        actors = np.asarray([int(str(key).rsplit("-", 1)[-1]) for key in keys], dtype=np.int64)
    except ValueError as error:
        raise ValueError("RAVDESS keys do not end in an actor id") from error
    return actors


def load_development(probability_path: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with probability_path.open("rb") as handle:
        pack = pickle.load(handle)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    labels = np.concatenate(
        [np.asarray(pack["y_train"], dtype=np.int64), np.asarray(pack["y_val"], dtype=np.int64)]
    )
    metadata_labels = np.concatenate(
        [
            np.asarray(metadata["labels"]["train"], dtype=np.int64),
            np.asarray(metadata["labels"]["val"], dtype=np.int64),
        ]
    )
    if not np.array_equal(labels, metadata_labels):
        raise RuntimeError("probability and metadata development labels do not align")
    keys = list(metadata["labels"]["keys_train"]) + list(metadata["labels"]["keys_val"])
    groups = actor_ids(keys)
    if set(np.unique(groups).tolist()) != EXPECTED_DEVELOPMENT_ACTORS:
        raise RuntimeError("development actors are not the expected 1--15 partition")
    posterior = np.stack(
        [
            np.concatenate([normalize_np(pack["pa_train"]), normalize_np(pack["pa_val"])]),
            np.concatenate([normalize_np(pack["pv_train"]), normalize_np(pack["pv_val"])]),
        ],
        axis=1,
    )
    if posterior.shape != (len(labels), 2, 8):
        raise RuntimeError(f"unexpected development posterior shape: {posterior.shape}")
    return posterior, labels, groups


def load_test_posterior(probability_path: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with probability_path.open("rb") as handle:
        pack = pickle.load(handle)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    groups = actor_ids(list(metadata["labels"]["keys_test"]))
    if set(np.unique(groups).tolist()) != EXPECTED_TEST_ACTORS:
        raise RuntimeError("test actors are not the expected 16--24 partition")
    posterior = np.stack(
        [normalize_np(pack["pa_test"]), normalize_np(pack["pv_test"])], axis=1
    )
    return posterior, groups


def load_test_labels(probability_path: Path, metadata_path: Path) -> np.ndarray:
    with probability_path.open("rb") as handle:
        pack = pickle.load(handle)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    labels = np.asarray(pack["y_test"], dtype=np.int64)
    metadata_labels = np.asarray(metadata["labels"]["test"], dtype=np.int64)
    if not np.array_equal(labels, metadata_labels):
        raise RuntimeError("probability and metadata test labels do not align")
    return labels


def role_summary(roles: dict[str, np.ndarray], labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    output = {}
    for name, indices in roles.items():
        counts = np.bincount(labels[indices], minlength=8)
        output[name] = {
            "actors": sorted(np.unique(groups[indices]).tolist()),
            "samples": int(len(indices)),
            "class_count_min": int(counts.min()),
            "class_count_max": int(counts.max()),
            "index_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        }
    actor_sets = [set(item["actors"]) for item in output.values()]
    if set.union(*actor_sets) != EXPECTED_DEVELOPMENT_ACTORS:
        raise RuntimeError("role actors do not cover the development partition")
    if sum(len(item) for item in actor_sets) != len(set.union(*actor_sets)):
        raise RuntimeError("an actor appears in more than one RCF role")
    return output


def selection_key(record: dict[str, Any]) -> tuple[float, int, float]:
    complexity = {
        "core5": 0,
        "interaction10": 1,
        "reliability10": 2,
        "prototype13": 3,
        "legacy5": 4,
    }.get(record["design"], 99)
    return record["mean_validation_score"], complexity, record["sampling_power"]


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    seeds = parse_seeds(args.seeds)
    designs = parse_csv(args.designs)
    powers = [float(value) for value in parse_csv(args.sampling_powers)]
    valid_designs = {"core5", "interaction10", "reliability10", "prototype13", "legacy5"}
    if not set(designs) <= valid_designs or any(value < 0.0 for value in powers):
        raise ValueError("invalid refiner design or sampling power")
    if not args.probabilities.exists() or not args.metadata.exists():
        raise FileNotFoundError("the frozen RAVDESS probability/metadata artifacts are required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.log.mkdir(parents=True, exist_ok=True)

    posterior, labels, groups = load_development(args.probabilities, args.metadata)
    prior = np.bincount(labels, minlength=8).astype(np.float32)
    prior /= prior.sum()
    core_by_seed = {}
    core_metadata_by_seed = {}
    roles_by_seed = {}
    refiner_fit_by_key = {}
    tuning_rows = []

    for seed in seeds:
        roles = grouped_role_split(
            labels,
            groups,
            seed,
            calibration_fraction=args.calibration_fraction,
            structure_fraction=args.structure_fraction,
            validation_fraction=args.validation_fraction,
        )
        roles_by_seed[seed] = roles
        core, metadata = fit_variant(
            "Full staged RCF",
            posterior,
            labels,
            roles,
            prior,
            seed,
            args.device,
            args.search,
            {},
            args.selection_tolerance_scale,
            args.include_identity_transport,
            args.include_identity_calibration,
        )
        core_by_seed[seed] = core
        core_metadata_by_seed[seed] = metadata
        for design in designs:
            for power in powers:
                config = refiner_config(args.search, design, power)
                _, fit = fit_refiner(
                    core,
                    posterior,
                    labels,
                    roles["fusion"],
                    roles["validation"],
                    config,
                    seed + 7001,
                    args.device,
                    final_refit_indices=None,
                )
                refiner_fit_by_key[(seed, design, power)] = fit
                metric = fit["best_validation_metric"]
                tuning_rows.append(
                    {
                        "seed": seed,
                        "design": design,
                        "sampling_power": power,
                        "best_epoch": fit["best_epoch"],
                        "validation_score": fit["best_validation_score"],
                        **{f"validation_{key}": metric[key] for key in ("acc", "f1", "nll")},
                    }
                )
                print(
                    f"development seed={seed} design={design} sampling={power:.2f} "
                    f"acc={metric['acc']:.4f} f1={metric['f1']:.4f} "
                    f"nll={metric['nll']:.4f} epoch={fit['best_epoch']}",
                    flush=True,
                )

    design_summary = []
    for design in designs:
        for power in powers:
            selected = [
                row for row in tuning_rows
                if row["design"] == design and row["sampling_power"] == power
            ]
            design_summary.append(
                {
                    "design": design,
                    "sampling_power": power,
                    "mean_validation_score": float(np.mean([row["validation_score"] for row in selected])),
                    "mean_validation_acc": float(np.mean([row["validation_acc"] for row in selected])),
                    "mean_validation_f1": float(np.mean([row["validation_f1"] for row in selected])),
                    "mean_validation_nll": float(np.mean([row["validation_nll"] for row in selected])),
                }
            )
    selected_design = min(design_summary, key=selection_key)
    selected_config = refiner_config(
        args.search,
        selected_design["design"],
        selected_design["sampling_power"],
    )
    development_report = {
        "policy": "Development actors only; no test posterior or test label was loaded during this selection pass.",
        "selection_rule": "Minimum mean validation score across role seeds; deterministic simpler-design tie break.",
        "selected": selected_design,
        "summary": design_summary,
        "rows": tuning_rows,
    }
    save_json(development_report, args.log / "development_selection.json")

    models = {}
    fit_records = {}
    for seed in seeds:
        roles = roles_by_seed[seed]
        core_metadata = core_metadata_by_seed[seed]
        final_indices = np.concatenate([roles["fusion"], roles["validation"]])
        core, pool_refit = refit_staged_core(
            core_by_seed[seed],
            posterior,
            labels,
            final_indices,
            PoolConfig(**core_metadata["selected_candidate"]["pool"]),
            int(core_metadata["selected_pool_fit"]["best_epoch"]),
            seed + 6001,
            args.device,
        )
        selected_fit = refiner_fit_by_key[
            (seed, selected_design["design"], selected_design["sampling_power"])
        ]
        refit_indices = (
            np.arange(len(labels), dtype=np.int64)
            if args.refiner_final_refit_scope == "all_development"
            else final_indices
        )
        model, refiner_refit = refit_refiner_fixed_epochs(
            core,
            posterior,
            labels,
            refit_indices,
            selected_config,
            int(selected_fit["best_epoch"]),
            int(selected_fit["seed"]) + 1702,
            args.device,
        )
        compact = compact_metadata(core_metadata, labels, groups)
        compact["roles"] = role_summary(roles, labels, groups)
        compact["final_pool_refit"] = pool_refit
        compact["refiner"] = {
            "selected_design": selected_design,
            "selection_fit": {
                "seed": selected_fit["seed"],
                "best_epoch": selected_fit["best_epoch"],
                "best_validation_score": selected_fit["best_validation_score"],
                "best_validation_metric": selected_fit["best_validation_metric"],
            },
            "config": asdict(selected_config),
            "final_refit": refiner_refit,
        }
        compact["final_refit_contract"] = {
            "selection_completed_before_refit": True,
            "pool_fit_roles": ["fusion", "validation"],
            "refiner_fit_roles": (
                ["calibration", "structure", "fusion", "validation"]
                if args.refiner_final_refit_scope == "all_development"
                else ["fusion", "validation"]
            ),
            "later_validation_selection": False,
            "frozen_objects": ["calibration", "transport", "candidate", "pool_epochs", "refiner_epochs"],
        }
        models[seed] = model.cpu()
        fit_records[str(seed)] = compact
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_type": type(model).__name__,
                "seed": seed,
                "metadata": compact,
            },
            args.output / f"full_staged_rcf_s{seed}.pt",
        )
    save_json(fit_records, args.log / "fit_records.json")

    test_posterior, test_groups = load_test_posterior(args.probabilities, args.metadata)
    predictions = []
    for seed in seeds:
        prediction, diagnostics = predict_refiner(models[seed], test_posterior, args.device)
        predictions.append(prediction)
        np.savez_compressed(
            args.output / f"full_staged_rcf_s{seed}_prediction.npz",
            prediction=prediction,
            gate=diagnostics["gate"],
            mixture=diagnostics["mixture"],
        )
    stacked = np.stack(predictions)
    ensemble = normalize_np(stacked.mean(0))
    average = normalize_np(test_posterior.mean(1))
    product = normalize_np(np.exp(np.log(np.clip(test_posterior, 1e-10, 1.0)).mean(1)))
    np.savez_compressed(
        args.output / "ensemble_predictions.npz",
        seeds=np.asarray(seeds, dtype=np.int64),
        seed_predictions=stacked,
        ensemble=ensemble,
        average=average,
        product=product,
        test_groups=test_groups,
    )

    test_labels = load_test_labels(args.probabilities, args.metadata)
    seed_metrics = [probability_metrics(test_labels, item, 8) for item in predictions]
    ensemble_metric = probability_metrics(test_labels, ensemble, 8)
    baseline_metrics = {
        "Average": probability_metrics(test_labels, average, 8),
        "Product": probability_metrics(test_labels, product, 8),
    }
    summary = {
        "acc_mean": float(np.mean([item["acc"] for item in seed_metrics])),
        "acc_std_population": float(np.std([item["acc"] for item in seed_metrics], ddof=0)),
        "acc_std_sample": float(np.std([item["acc"] for item in seed_metrics], ddof=1)),
        "f1_mean": float(np.mean([item["f1"] for item in seed_metrics])),
        "f1_std_population": float(np.std([item["f1"] for item in seed_metrics], ddof=0)),
    }
    manifest = {
        "dataset": "RAVDESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "device": args.device,
        "source_artifacts": [
            {"path": args.probabilities, "sha256": sha256(args.probabilities)},
            {"path": args.metadata, "sha256": sha256(args.metadata)},
        ],
        "partition": {
            "development_actors": sorted(EXPECTED_DEVELOPMENT_ACTORS),
            "test_actors": sorted(EXPECTED_TEST_ACTORS),
            "role_unit": "actor",
            "role_fractions": {
                "calibration": args.calibration_fraction,
                "structure": args.structure_fraction,
                "fusion": 1.0 - args.calibration_fraction - args.structure_fraction - args.validation_fraction,
                "validation": args.validation_fraction,
            },
        },
        "contract": [
            "actor_grouped_disjoint_rcf_roles",
            "fit_freeze_calibration",
            "estimate_freeze_calibrated_C_B",
            "fit_freeze_reliability",
            "development_select_refiner",
            "selection_then_fixed_epoch_refit",
            "freeze_all_test_predictions",
            "read_test_labels_once",
        ],
        "test_label_policy": "Test labels are loaded only after every seed prediction and the probability ensemble have been frozen.",
        "upstream_limitation": "The 1440 train posteriors are in-sample predictions rather than expert OOF predictions. Actor isolation is strict across RCF roles and the final test, but this artifact is not strict end-to-end OOF evidence.",
        "selection": selected_design,
        "search": args.search,
        "refiner_final_refit_scope": args.refiner_final_refit_scope,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    results = {
        "manifest": manifest,
        "seed_metrics": {str(seed): metric for seed, metric in zip(seeds, seed_metrics)},
        "role_seed_summary": summary,
        "ensemble": ensemble_metric,
        "baselines": baseline_metrics,
        "development_selection": development_report,
    }
    save_json(manifest, args.log / "manifest.json")
    save_json(results, args.log / "results.json")
    print(
        f"role-seed acc={100.0 * summary['acc_mean']:.2f}+/-"
        f"{100.0 * summary['acc_std_population']:.2f} "
        f"ensemble={100.0 * ensemble_metric['acc']:.2f}",
        flush=True,
    )
    for seed, metric in zip(seeds, seed_metrics):
        print(
            f"seed={seed} acc={100.0 * metric['acc']:.2f} "
            f"f1={100.0 * metric['f1']:.2f} nll={metric['nll']:.4f}",
            flush=True,
        )
    print(f"results={args.log / 'results.json'}", flush=True)
    return args.log / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate source artifacts")
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--search", choices=("quick", "standard", "wide"), default="standard")
    parser.add_argument("--designs", default="core5,interaction10")
    parser.add_argument("--sampling-powers", default="0.0,0.25")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--structure-fraction", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--selection-tolerance-scale", type=float, default=1.0)
    parser.add_argument(
        "--include-identity-transport",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-identity-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--refiner-final-refit-scope",
        choices=("fusion_validation", "all_development"),
        default="all_development",
    )
    args = parser.parse_args()
    if args.check:
        if not args.probabilities.is_file() or not args.metadata.is_file():
            raise FileNotFoundError("RAVDESS probability or metadata artifact is missing")
        posterior, labels, groups = load_development(args.probabilities, args.metadata)
        print(
            f"RAVDESS check passed: development={len(labels)} "
            f"experts={posterior.shape[1]} actors={len(np.unique(groups))}"
        )
        return
    fractions = (args.calibration_fraction, args.structure_fraction, args.validation_fraction)
    if any(value <= 0.0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("invalid role fractions")
    if args.selection_tolerance_scale < 0.0:
        raise ValueError("selection tolerance must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
