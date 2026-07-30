from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--structure-per-class", type=int, default=160)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--concentration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20270728)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "dome_x_channel_mechanism",
    )
    return parser.parse_args()


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    return rows / rows.sum(axis=-1, keepdims=True)


def channel_banks() -> dict[str, np.ndarray]:
    coarse = normalize_rows(
        np.array(
            [
                [0.72, 0.11, 0.11, 0.06],
                [0.72, 0.11, 0.11, 0.06],
                [0.08, 0.06, 0.72, 0.14],
                [0.08, 0.06, 0.72, 0.14],
            ]
        )
    )
    crossed = normalize_rows(
        np.array(
            [
                [0.72, 0.08, 0.12, 0.08],
                [0.08, 0.08, 0.72, 0.12],
                [0.08, 0.08, 0.72, 0.12],
                [0.72, 0.08, 0.12, 0.08],
            ]
        )
    )
    return {
        "redundant": np.stack([coarse, coarse]),
        "complementary_stable": np.stack([coarse, crossed]),
        "complementary_drifted": np.stack([coarse, crossed]),
    }


def sample_posteriors(
    rng: np.random.Generator,
    channels: np.ndarray,
    samples_per_class: int,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray]:
    experts, classes, _ = channels.shape
    labels = np.repeat(np.arange(classes), samples_per_class)
    posterior = np.empty((labels.size, experts, classes), dtype=np.float64)
    for expert in range(experts):
        for label in range(classes):
            index = labels == label
            alpha = concentration * channels[expert, label] + 1e-3
            posterior[index, expert] = rng.dirichlet(alpha, index.sum())
    return posterior, labels


def estimate_reverse_channels(
    posterior: np.ndarray, labels: np.ndarray, smoothing: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    classes = posterior.shape[-1]
    priors = np.bincount(labels, minlength=classes).astype(np.float64)
    priors /= priors.sum()
    confusion = np.stack(
        [
            np.stack([posterior[labels == y, m].mean(axis=0) for y in range(classes)])
            for m in range(posterior.shape[1])
        ]
    )
    joint = priors[None, :, None] * confusion + smoothing
    reverse = joint / joint.sum(axis=1, keepdims=True)
    return confusion, reverse


def reverse_channel_predict(posterior: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    transported = np.einsum("myk,nmk->nmy", reverse, posterior)
    return transported.mean(axis=1).argmax(axis=1)


def prototype_predict(
    posterior: np.ndarray, confusion: np.ndarray
) -> np.ndarray:
    prototypes = confusion.transpose(1, 0, 2).reshape(confusion.shape[1], -1)
    features = posterior.reshape(posterior.shape[0], -1)
    distances = ((features[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def macro_f1(labels: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        true_positive = np.sum((labels == label) & (prediction == label))
        false_positive = np.sum((labels != label) & (prediction == label))
        false_negative = np.sum((labels == label) & (prediction != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def geometry(posterior: np.ndarray, labels: np.ndarray, confusion: np.ndarray) -> dict[str, float]:
    prototypes = confusion.transpose(1, 0, 2).reshape(confusion.shape[1], -1)
    pairwise = np.linalg.norm(prototypes[:, None] - prototypes[None, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    delta_z = float(np.min(pairwise))
    features = posterior.reshape(posterior.shape[0], -1)
    residual = features - prototypes[labels]
    rho = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    gamma = float(delta_z / (2.0 * rho)) if rho > 0 else float("inf")
    return {"delta_z": delta_z, "rho": rho, "gamma": gamma}


def evaluate(
    posterior: np.ndarray,
    labels: np.ndarray,
    confusion: np.ndarray,
    reverse: np.ndarray,
) -> dict[str, float]:
    classes = posterior.shape[-1]
    expert_predictions = posterior.argmax(axis=2)
    expert_accuracy = (expert_predictions == labels[:, None]).mean(axis=0)
    average_prediction = posterior.mean(axis=1).argmax(axis=1)
    rcf_prediction = reverse_channel_predict(posterior, reverse)
    proto_prediction = prototype_predict(posterior, confusion)
    result = {
        "best_expert_acc": float(expert_accuracy.max()),
        "average_acc": float(np.mean(average_prediction == labels)),
        "average_f1": macro_f1(labels, average_prediction, classes),
        "rcf_acc": float(np.mean(rcf_prediction == labels)),
        "rcf_f1": macro_f1(labels, rcf_prediction, classes),
        "prototype_acc": float(np.mean(proto_prediction == labels)),
    }
    result["conditional_utility"] = result["rcf_acc"] - result["best_expert_acc"]
    result.update(geometry(posterior, labels, confusion))
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = channel_banks()
    records: list[dict[str, float | int | str]] = []

    for repetition in range(args.repetitions):
        repetition_seed = args.seed + repetition
        for offset, (condition, structure_bank) in enumerate(banks.items()):
            rng = np.random.default_rng(repetition_seed * 10 + offset)
            structure, structure_labels = sample_posteriors(
                rng,
                structure_bank,
                args.structure_per_class,
                args.concentration,
            )
            confusion, reverse = estimate_reverse_channels(structure, structure_labels)

            test_bank = structure_bank.copy()
            if condition == "complementary_drifted":
                test_bank[1] = test_bank[1, [1, 0, 3, 2]]
            test, test_labels = sample_posteriors(
                rng,
                test_bank,
                args.test_per_class,
                args.concentration,
            )
            record: dict[str, float | int | str] = dict(
                evaluate(test, test_labels, confusion, reverse)
            )
            record.update({"condition": condition, "repetition": repetition})
            records.append(record)

    metric_names = [
        "best_expert_acc",
        "average_acc",
        "average_f1",
        "rcf_acc",
        "rcf_f1",
        "prototype_acc",
        "conditional_utility",
        "delta_z",
        "rho",
        "gamma",
    ]
    summary = {}
    for condition in banks:
        rows = [row for row in records if row["condition"] == condition]
        summary[condition] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in rows])),
                "std": float(np.std([float(row[metric]) for row in rows], ddof=0)),
            }
            for metric in metric_names
        }

    stable = np.array(
        [row["rcf_acc"] for row in records if row["condition"] == "complementary_stable"]
    )
    redundant = np.array(
        [row["rcf_acc"] for row in records if row["condition"] == "redundant"]
    )
    paired_delta = stable - redundant
    summary["paired_stable_minus_redundant_rcf_acc"] = {
        "mean": float(paired_delta.mean()),
        "ci95_repetition_quantiles": [
            float(np.quantile(paired_delta, 0.025)),
            float(np.quantile(paired_delta, 0.975)),
        ],
    }

    csv_path = args.output_dir / "repetition_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "scope": "Controlled channel construction; not neural ROST training or benchmark evidence.",
        "claim_tested": (
            "Stable complementary posterior codes can improve conditional fusion utility "
            "without improving standalone top-1 accuracy."
        ),
        "configuration": {
            "repetitions": args.repetitions,
            "structure_per_class": args.structure_per_class,
            "test_per_class": args.test_per_class,
            "concentration": args.concentration,
            "seed": args.seed,
        },
        "script_sha256": script_hash,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
