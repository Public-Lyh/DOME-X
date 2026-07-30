from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CONDITIONS = (
    "ce_only",
    "rost_rescue_disabled",
    "rost_uniform_rescue",
    "rost_leave_one_rescue",
)
PAIR_INDEX = tuple((a, b) for a in range(4) for b in range(a + 1, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--rescue-weight", type=float, default=4.0)
    parser.add_argument("--rescue-margin", type=float, default=0.027)
    parser.add_argument("--rescue-temperature", type=float, default=0.015)
    parser.add_argument("--initialization-noise", type=float, default=0.02)
    parser.add_argument("--structure-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--concentration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20270728)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "rost_leave_one_training_mechanism",
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.structure_per_class < 1 or args.test_per_class < 1:
        parser.error("sample counts must be positive")
    if args.concentration <= 0 or args.rescue_temperature <= 0:
        parser.error("concentration and rescue temperature must be positive")
    return args


def basis_channels() -> torch.Tensor:
    basis_a = [
        [[0.45, 0.25, 0.15, 0.15],
         [0.45, 0.25, 0.27, 0.03],
         [0.15, 0.15, 0.45, 0.25],
         [0.15, 0.15, 0.45, 0.25]],
        [[0.45, 0.25, 0.15, 0.15],
         [0.45, 0.25, 0.15, 0.15],
         [0.15, 0.15, 0.45, 0.25],
         [0.27, 0.03, 0.45, 0.25]],
    ]
    banks = torch.tensor(basis_a, dtype=torch.float64)
    if not torch.allclose(banks.sum(dim=-1), torch.ones_like(banks[..., 0])):
        raise RuntimeError("basis channel rows must lie on the simplex")
    diagonals = torch.diagonal(banks, dim1=-2, dim2=-1)
    if not torch.allclose(diagonals[0], diagonals[1]):
        raise RuntimeError("basis channels must have identical diagonals")
    predictions = banks.argmax(dim=-1)
    if not torch.equal(predictions[0], predictions[1]):
        raise RuntimeError("basis channels must have identical top-1 decisions")
    return banks


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.clamp_min(1e-12).log() - middle.log())).sum(dim=-1)
        + (right * (right.clamp_min(1e-12).log() - middle.log())).sum(dim=-1)
    )


def pair_distances(codes: torch.Tensor) -> torch.Tensor:
    first = torch.tensor([pair[0] for pair in PAIR_INDEX], dtype=torch.long)
    second = torch.tensor([pair[1] for pair in PAIR_INDEX], dtype=torch.long)
    return js_divergence(codes[..., first, :], codes[..., second, :])


def initial_logits(repetitions: int, seed: int, noise: float) -> torch.Tensor:
    anchor = torch.tensor(
        [[[0.45, 0.00], [0.35, 0.10]]], dtype=torch.float64
    ).expand(repetitions, -1, -1)
    generator = torch.Generator().manual_seed(seed)
    perturbation = noise * torch.randn(
        repetitions, 2, 2, generator=generator, dtype=torch.float64
    )
    return anchor.clone() + perturbation


def train_channels(
    condition: str,
    start: torch.Tensor,
    banks: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    theta = start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=args.learning_rate)
    rescue_mode = {
        "ce_only": "off",
        "rost_rescue_disabled": "off",
        "rost_uniform_rescue": "uniform",
        "rost_leave_one_rescue": "leave_one",
    }[condition]

    last_semantic = 0.0
    last_rescue = 0.0
    for _ in range(args.steps):
        gates = theta.softmax(dim=-1)
        codes = torch.einsum("rmp,pyk->rmyk", gates, banks)
        diagonal = torch.diagonal(codes, dim1=-2, dim2=-1)
        semantic = -diagonal.clamp_min(1e-12).log().mean()
        distances = pair_distances(codes)
        rescue = codes.new_zeros(())
        if rescue_mode != "off":
            terms = []
            for expert in range(2):
                if rescue_mode == "leave_one":
                    weights = torch.softmax(
                        -distances[:, 1 - expert, :].detach()
                        / args.rescue_temperature,
                        dim=-1,
                    )
                else:
                    weights = torch.full_like(
                        distances[:, expert, :], 1.0 / len(PAIR_INDEX)
                    )
                hinge = F.relu(args.rescue_margin - distances[:, expert, :])
                terms.append((weights * hinge).sum(dim=-1))
            rescue = torch.stack(terms, dim=1).mean()
        loss = semantic + args.rescue_weight * rescue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_semantic = float(semantic.detach())
        last_rescue = float(rescue.detach())

    gates = theta.softmax(dim=-1).detach()
    codes = torch.einsum("rmp,pyk->rmyk", gates, banks).detach()
    specialization = (gates.argmax(dim=-1)[:, 0] != gates.argmax(dim=-1)[:, 1])
    detail = {
        "semantic_loss": last_semantic,
        "rescue_loss": last_rescue,
        "specialized_fraction": float(specialization.double().mean()),
        "expert_0_basis_a_weight": float(gates[:, 0, 0].mean()),
        "expert_1_basis_a_weight": float(gates[:, 1, 0].mean()),
    }
    return codes.cpu().numpy(), detail


def sample_posteriors(
    rng: np.random.Generator,
    codes: np.ndarray,
    samples_per_class: int,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(4), samples_per_class)
    posterior = np.empty((labels.size, 2, 4), dtype=np.float64)
    for expert in range(2):
        for label in range(4):
            mask = labels == label
            posterior[mask, expert] = rng.dirichlet(
                concentration * codes[expert, label], int(mask.sum())
            )
    return posterior, labels


def fit_base_rcf(
    posterior: np.ndarray, labels: np.ndarray, smoothing: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.stack(
        [
            np.stack([posterior[labels == y, m].mean(axis=0) for y in range(4)])
            for m in range(2)
        ]
    )
    joint = 0.25 * confusion + smoothing
    reverse = joint / joint.sum(axis=1, keepdims=True)
    return confusion, reverse


def rcf_predict(posterior: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    transported = np.einsum("myk,nmk->nmy", reverse, posterior)
    return transported.mean(axis=1).argmax(axis=1)


def prototype_predict(posterior: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    prototypes = confusion.transpose(1, 0, 2).reshape(4, -1)
    features = posterior.reshape(posterior.shape[0], -1)
    squared_distance = (
        (features[:, None, :] - prototypes[None, :, :]) ** 2
    ).sum(axis=-1)
    return squared_distance.argmin(axis=1)


def effective_rank(prototypes: np.ndarray) -> float:
    centered = prototypes - prototypes.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    normalized = singular / total
    entropy = -(normalized * np.log(np.clip(normalized, 1e-12, None))).sum()
    return float(np.exp(entropy))


def geometry(
    posterior: np.ndarray, labels: np.ndarray, confusion: np.ndarray
) -> dict[str, float]:
    prototypes = confusion.transpose(1, 0, 2).reshape(4, -1)
    pairwise = np.linalg.norm(
        prototypes[:, None, :] - prototypes[None, :, :], axis=-1
    )
    pairwise[np.eye(4, dtype=bool)] = np.inf
    delta_z = float(pairwise.min())
    features = posterior.reshape(posterior.shape[0], -1)
    residual = features - prototypes[labels]
    rho = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return {
        "delta_z": delta_z,
        "rho": rho,
        "gamma": delta_z / (2.0 * rho) if rho > 0 else math.inf,
        "code_effective_rank": effective_rank(prototypes),
    }


def mean_code_delta_z(codes: np.ndarray) -> float:
    prototypes = codes.transpose(1, 0, 2).reshape(4, -1)
    pairwise = np.linalg.norm(
        prototypes[:, None, :] - prototypes[None, :, :], axis=-1
    )
    pairwise[np.eye(4, dtype=bool)] = np.inf
    return float(pairwise.min())


def evaluate_repetition(
    condition: str,
    repetition: int,
    codes: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(args.seed + repetition * 10_007)
    structure, structure_labels = sample_posteriors(
        rng, codes, args.structure_per_class, args.concentration
    )
    confusion, reverse = fit_base_rcf(structure, structure_labels)
    test, test_labels = sample_posteriors(
        rng, codes, args.test_per_class, args.concentration
    )

    expert_predictions = test.argmax(axis=-1)
    expert_accuracy = (expert_predictions == test_labels[:, None]).mean(axis=0)
    average_prediction = test.mean(axis=1).argmax(axis=1)
    product_prediction = np.prod(test, axis=1).argmax(axis=1)
    rcf_prediction = rcf_predict(test, reverse)
    prototype_prediction = prototype_predict(test, confusion)
    full_rcf_accuracy = float(np.mean(rcf_prediction == test_labels))

    single_rcf_accuracy = []
    for expert in range(2):
        prediction = rcf_predict(
            test[:, expert : expert + 1], reverse[expert : expert + 1]
        )
        single_rcf_accuracy.append(float(np.mean(prediction == test_labels)))

    result: dict[str, float | int | str] = {
        "condition": condition,
        "repetition": repetition,
        "expert_0_acc": float(expert_accuracy[0]),
        "expert_1_acc": float(expert_accuracy[1]),
        "mean_expert_acc": float(expert_accuracy.mean()),
        "best_expert_acc": float(expert_accuracy.max()),
        "average_acc": float(np.mean(average_prediction == test_labels)),
        "product_acc": float(np.mean(product_prediction == test_labels)),
        "base_rcf_acc": full_rcf_accuracy,
        "prototype_acc": float(np.mean(prototype_prediction == test_labels)),
        "rcf_single_expert_0_acc": single_rcf_accuracy[0],
        "rcf_single_expert_1_acc": single_rcf_accuracy[1],
        "fusion_gain_over_best_expert": (
            full_rcf_accuracy - float(expert_accuracy.max())
        ),
        "fitted_leave_one_rcf_utility_expert_0": (
            full_rcf_accuracy - single_rcf_accuracy[1]
        ),
        "fitted_leave_one_rcf_utility_expert_1": (
            full_rcf_accuracy - single_rcf_accuracy[0]
        ),
        "fitted_leave_one_rcf_utility_mean": full_rcf_accuracy
        - float(np.mean(single_rcf_accuracy)),
        "mean_code_delta_z": mean_code_delta_z(codes),
    }
    result.update(geometry(test, test_labels, confusion))
    return result


def summarize(
    records: list[dict[str, float | int | str]], metric_names: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        summary[condition] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in rows])),
                "std": float(np.std([float(row[metric]) for row in rows], ddof=0)),
            }
            for metric in metric_names
        }
    return summary


def paired_comparisons(
    records: list[dict[str, float | int | str]], metric_names: list[str]
) -> dict[str, Any]:
    lookup = {
        (str(row["condition"]), int(row["repetition"])): row for row in records
    }
    comparisons = {}
    for baseline in (
        "ce_only",
        "rost_rescue_disabled",
        "rost_uniform_rescue",
    ):
        label = f"rost_leave_one_rescue_minus_{baseline}"
        comparisons[label] = {}
        for metric in metric_names:
            delta = np.array(
                [
                    float(lookup[("rost_leave_one_rescue", repetition)][metric])
                    - float(lookup[(baseline, repetition)][metric])
                    for repetition in range(
                        len(records) // len(CONDITIONS)
                    )
                ]
            )
            comparisons[label][metric] = {
                "mean": float(delta.mean()),
                "std": float(delta.std(ddof=0)),
                "repetition_quantile_interval_95": [
                    float(np.quantile(delta, 0.025)),
                    float(np.quantile(delta, 0.975)),
                ],
            }
    return comparisons


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    banks = basis_channels()
    start = initial_logits(
        args.repetitions, args.seed, args.initialization_noise
    )
    condition_codes: dict[str, np.ndarray] = {}
    training_summary: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        codes, detail = train_channels(condition, start, banks, args)
        condition_codes[condition] = codes
        training_summary[condition] = detail

    records: list[dict[str, float | int | str]] = []
    for repetition in range(args.repetitions):
        for condition in CONDITIONS:
            records.append(
                evaluate_repetition(
                    condition,
                    repetition,
                    condition_codes[condition][repetition],
                    args,
                )
            )

    metric_names = [
        "expert_0_acc",
        "expert_1_acc",
        "mean_expert_acc",
        "best_expert_acc",
        "average_acc",
        "product_acc",
        "base_rcf_acc",
        "prototype_acc",
        "rcf_single_expert_0_acc",
        "rcf_single_expert_1_acc",
        "fusion_gain_over_best_expert",
        "fitted_leave_one_rcf_utility_expert_0",
        "fitted_leave_one_rcf_utility_expert_1",
        "fitted_leave_one_rcf_utility_mean",
        "mean_code_delta_z",
        "delta_z",
        "rho",
        "gamma",
        "code_effective_rank",
    ]
    summary = summarize(records, metric_names)
    comparisons = paired_comparisons(records, metric_names)

    csv_path = args.output_dir / "repetition_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    banks_numpy = banks.numpy()
    mean_code_predictions = banks_numpy.argmax(axis=-1)
    analytic_standalone = float(
        np.mean(mean_code_predictions[0] == np.arange(4))
    )
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "scope": (
            "Controlled population-channel optimization; not a neural training "
            "benchmark and not evidence of an unconditional ROST guarantee."
        ),
        "claim_tested": (
            "Detached leave-one rescue can allocate complementary posterior "
            "codes and improve fitted leave-one RCF utility while the admissible "
            "channels keep standalone mean-channel top-1 accuracy fixed."
        ),
        "design_contract": {
            "class_prior": "uniform over four classes",
            "training_parameterization": (
                "Each expert is a convex gate over two fixed posterior-code bases."
            ),
            "basis_a_role": "encodes pair (0,1); leaves pair (2,3) unresolved",
            "basis_b_role": "encodes pair (2,3); leaves pair (0,1) unresolved",
            "invariant_diagonal": banks_numpy[0].diagonal().tolist(),
            "invariant_top1_predictions": mean_code_predictions[0].tolist(),
            "analytic_mean_channel_standalone_acc": analytic_standalone,
            "semantic_ce_invariant": True,
            "leave_one_weights_detached": True,
            "selection_or_tuning_inside_run": False,
            "evaluation_noise": "Dirichlet around the learned channel rows",
            "reported_utility": (
                "Independent-test accuracy of fitted full base RCF minus fitted "
                "leave-one base RCF; not best-in-family population U_m(G)."
            ),
        },
        "configuration": {
            "repetitions": args.repetitions,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "rescue_weight": args.rescue_weight,
            "rescue_margin": args.rescue_margin,
            "rescue_temperature": args.rescue_temperature,
            "initialization_noise": args.initialization_noise,
            "structure_per_class": args.structure_per_class,
            "test_per_class": args.test_per_class,
            "concentration": args.concentration,
            "seed": args.seed,
        },
        "training_summary": training_summary,
        "summary": summary,
        "paired_comparisons": comparisons,
        "script_sha256": script_hash,
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
