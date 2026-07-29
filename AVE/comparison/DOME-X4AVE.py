"""Compare AVE fusion rules on cached three-fold expert posteriors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("your path")
CODE_ROOT = PROJECT_ROOT / "Code"
ROOT = CODE_ROOT / "AVE"
sys.path.insert(0, str(CODE_ROOT))

from dome_x_pure_rcf import (  # noqa: E402
    EPS,
    PureRCF,
    component_forward,
    estimate_structure,
    normalize_np,
    probability_metrics,
    predict_pure_rcf,
    train_pure_rcf,
)


NC = 28
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ARTIFACT = ROOT / "checkpoints" / "v1" / "DOME_X_AVE_CE_ROST" / "ce_rost_oof_posteriors.npz"
OUT = ROOT / "checkpoints" / "pure_rcf_v1"
OUT.mkdir(parents=True, exist_ok=True)


def write_json(value, path: Path) -> None:
    def cast(item):
        if isinstance(item, dict):
            return {str(key): cast(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [cast(val) for val in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.floating, np.integer)):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item
    path.write_text(json.dumps(cast(value), indent=2, ensure_ascii=False), encoding="utf-8")


def split_oof(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    structure, rest = next(outer.split(indices, labels))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    fusion_local, validation_local = next(inner.split(rest, labels[rest]))
    return structure, rest[fusion_local], rest[validation_local]


def test_posterior(archive, regime: str) -> np.ndarray:
    audio = archive[f"{regime}_test_folds_audio"]
    visual = archive[f"{regime}_test_folds_visual"]
    return np.stack([normalize_np(audio).mean(axis=0), normalize_np(visual).mean(axis=0)], axis=1)


def weighted_average(oof: np.ndarray, labels: np.ndarray, test: np.ndarray, structure: np.ndarray) -> np.ndarray:
    weights = estimate_structure(oof[structure], labels[structure])["reliability"]
    return normalize_np((test * weights[None]).sum(axis=1))


def logistic_stacking(oof: np.ndarray, labels: np.ndarray, test: np.ndarray, fusion: np.ndarray) -> np.ndarray:
    train = np.concatenate([oof[fusion, 0], oof[fusion, 1]], axis=1)
    target = labels[fusion]
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.20, max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED))
    classifier.fit(train, target)
    probs = classifier.predict_proba(np.concatenate([test[:, 0], test[:, 1]], axis=1))
    full = np.zeros((len(test), NC), dtype=np.float32)
    full[:, classifier[-1].classes_] = probs
    return normalize_np(full)


def run_regime(archive, regime: str, labels: np.ndarray, test_labels: np.ndarray) -> dict:
    oof = np.stack([normalize_np(archive[f"{regime}_oof_audio"]), normalize_np(archive[f"{regime}_oof_visual"])], axis=1)
    test = test_posterior(archive, regime)
    structure, fusion, validation = split_oof(labels, SEED)
    model, fit = train_pure_rcf(oof, labels, structure, fusion, validation, seed=SEED, device=DEVICE)
    frozen_test, diag = predict_pure_rcf(model, test, DEVICE)
    output = {
        "Average": normalize_np(test.mean(axis=1)),
        "Product": normalize_np(np.exp(np.log(test.clip(EPS, 1.0)).mean(axis=1))),
        "Weighted Average": weighted_average(oof, labels, test, structure),
        "Logistic Stacking": logistic_stacking(oof, labels, test, fusion),
        "Frozen + Reliability": component_forward(model, test, DEVICE, calibration=False, transport=False, refinement=False),
        "Frozen + Calibration": component_forward(model, test, DEVICE, transport=False, refinement=False),
        "Frozen + Bias Transport": component_forward(model, test, DEVICE, refinement=False),
        "Pure RCF": frozen_test,
        "w/o Reliability": component_forward(model, test, DEVICE, reliability=False),
        "w/o Calibration": component_forward(model, test, DEVICE, calibration=False),
        "w/o Bias Transport": component_forward(model, test, DEVICE, transport=False),
        "w/o Disagreement Refinement": component_forward(model, test, DEVICE, refinement=False),
    }
    rows = [{"regime": regime, "method": name, **probability_metrics(test_labels, probs, NC)} for name, probs in output.items()]
    return {
        "rows": rows,
        "model": model,
        "fit": fit,
        "test_output": output,
        "diagnostic": diag,
        "splits": {"structure": structure, "fusion": fusion, "validation": validation},
    }


def main() -> None:
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"missing verified artifact: {ARTIFACT}")
    with np.load(ARTIFACT) as archive:
        labels = np.asarray(archive["labels"], dtype=np.int64)
        test_labels = np.asarray(archive["test_labels"], dtype=np.int64)
        result = {regime: run_regime(archive, regime, labels, test_labels) for regime in ("CE", "ROST")}
    rows = [row for value in result.values() for row in value["rows"]]
    for regime, value in result.items():
        torch.save(
            {"state_dict": value["model"].state_dict(), "fit": value["fit"], "regime": regime},
            OUT / f"{regime.lower()}_pure_rcf.pt",
        )
        np.savez_compressed(
            OUT / f"{regime.lower()}_pure_rcf_predictions.npz",
            test_labels=test_labels,
            pure_rcf=value["test_output"]["Pure RCF"],
            gate=value["diagnostic"]["gate"],
            disagreement=value["diagnostic"]["disagreement"],
            confidence=value["diagnostic"]["confidence"],
            **{name.replace(" ", "_").replace("/", "_"): probs for name, probs in value["test_output"].items()},
        )
    manifest = {
        "dataset": "AVE",
        "artifact": str(ARTIFACT),
        "artifact_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "protocol": "Cached independent CE/ROST fold experts. OOF only for all RCF statistics and learned fusion. Test labels are read after frozen predictions.",
        "test_aggregation": "mean fold posterior per modality, then one frozen Pure RCF",
        "forbidden": ["baseline blend", "candidate router", "pair memory", "class residual MLP"],
        "rows": rows,
    }
    write_json(manifest, OUT / "experiment_manifest.json")
    write_json(rows, OUT / "results.json")
    print(f"AVE Pure RCF results written to {OUT}")
    for row in rows:
        print(f"{row['regime']:<4} {row['method']:<30} acc={row['acc']:.4f} f1={row['f1']:.4f} nll={row['nll']:.4f}")


if __name__ == "__main__":
    main()
