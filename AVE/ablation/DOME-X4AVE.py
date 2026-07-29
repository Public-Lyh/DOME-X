import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path("your path") / "Code" / "AVE"
REFERENCE_DIR = ROOT / "checkpoints" / "v1" / "DOME_X_AVE_CE_ROST"
CHECKPOINT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "logs"
POSTERIOR_FILE = REFERENCE_DIR / "ce_rost_oof_posteriors.npz"
FULL_CHECKPOINT = REFERENCE_DIR / "rost_rcf.pt"
SEED = int(os.environ.get("DOME_X_SEED", "42"))
FUSION_SEEDS = [SEED + 1000 + i for i in range(int(os.environ.get("DOME_X_FUSION_SEEDS", "5")))]
RCF_EPOCHS = int(os.environ.get("DOME_X_RCF_EPOCHS", "160"))
PATIENCE = int(os.environ.get("DOME_X_FUSION_PATIENCE", "24"))
VERSION = os.environ.get("DOME_X_ABLATION_VERSION", "artifact_v1")
RCF_FILTER = [x.strip() for x in os.environ.get("DOME_X_RCF_VARIANTS", "").split(",") if x.strip()]
EPS = 1e-10
NC = 28
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RCF_VARIANTS = {
    "Average/Base": {"reliability": False, "calibration": False, "transport": False, "refinement": False},
    "+ Reliability": {"reliability": True, "calibration": False, "transport": False, "refinement": False},
    "+ Calibration": {"reliability": True, "calibration": True, "transport": False, "refinement": False},
    "+ Bias Transport": {"reliability": True, "calibration": True, "transport": True, "refinement": False},
    "Full RCF": {"reliability": True, "calibration": True, "transport": True, "refinement": True},
    "w/o Reliability": {"reliability": False, "calibration": True, "transport": True, "refinement": True},
    "w/o Calibration": {"reliability": True, "calibration": False, "transport": True, "refinement": True},
    "w/o Bias Transport": {"reliability": True, "calibration": True, "transport": False, "refinement": True},
    "w/o Disagreement Refinement": {"reliability": True, "calibration": True, "transport": True, "refinement": False},
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def normalize(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, EPS, None)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def normalize_torch(x):
    return x / x.sum(dim=-1, keepdim=True).clamp_min(EPS)


def safe_name(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def metric_dict(labels, probs):
    probs = normalize(probs)
    pred = probs.argmax(1)
    onehot = np.eye(NC, dtype=np.float32)[labels]
    confidence = probs.max(1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs((pred[mask] == labels[mask]).mean() - confidence[mask].mean()))
    return {
        "acc": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "ece": ece,
        "brier": float(np.square(probs - onehot).sum(1).mean()),
        "nll": float(log_loss(labels, probs, labels=np.arange(NC))),
    }


def soft_confusion(labels, probs):
    matrix = np.zeros((NC, NC), dtype=np.float32)
    np.add.at(matrix, labels, normalize(probs))
    count = np.bincount(labels, minlength=NC).astype(np.float32)
    matrix /= np.maximum(count[:, None], 1.0)
    matrix[count == 0] = 1.0 / NC
    return normalize(matrix)


def posterior_context(x):
    entropy = -(x * x.clamp_min(EPS).log()).sum(2) / np.log(NC)
    top2 = x.topk(2, 2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float()
    return torch.cat([
        x.flatten(1),
        x.mean(1),
        x.std(1, unbiased=False),
        x.max(1).values,
        entropy,
        margin,
        disagreement,
    ], 1)


def transport_from_confusion(matrix):
    matrix = normalize(matrix)
    prior = np.full(NC, 1.0 / NC, dtype=np.float32)
    posterior = matrix.T * prior[None]
    posterior /= np.maximum(posterior.sum(1, keepdims=True), EPS)
    return posterior.T.astype(np.float32)


class ReferenceRCF(nn.Module):
    def __init__(self, matrices, config):
        super().__init__()
        self.config = dict(config)
        matrices = np.asarray(matrices, dtype=np.float32)
        diagonal = np.clip(np.diagonal(matrices, axis1=1, axis2=2), 0.02, 1.0)
        diagonal /= diagonal.sum(0, keepdims=True)
        transport = np.stack([transport_from_confusion(matrix) for matrix in matrices])
        self.reliability = nn.Parameter(torch.log(torch.tensor(diagonal)))
        self.transport = nn.Parameter(torch.log(torch.tensor(transport).clamp_min(EPS)))
        self.scale = nn.Parameter(torch.ones(2, NC))
        self.bias = nn.Parameter(torch.zeros(2, NC))
        self.path = nn.Parameter(torch.log(torch.tensor([0.20, 0.25, 0.20, 0.10, 0.25])))
        self.path_gate = nn.Sequential(nn.Linear(5 * NC + 6, 64), nn.GELU(), nn.Linear(64, 5))
        self.gate = nn.Sequential(nn.Linear(5 * NC + 6, 64), nn.GELU(), nn.Linear(64, 1))
        self.residual = nn.Sequential(nn.Linear(5 * NC + 6, 64), nn.GELU(), nn.Linear(64, NC, bias=False))
        nn.init.zeros_(self.path_gate[-1].weight)
        nn.init.zeros_(self.path_gate[-1].bias)
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, x):
        x = x.clamp_min(EPS)
        if self.config["calibration"]:
            calibrated = F.softmax(torch.log(x) * F.softplus(self.scale).unsqueeze(0) + self.bias.unsqueeze(0), 2)
        else:
            calibrated = x
        if self.config["reliability"]:
            weight = F.softmax(self.reliability, 0).unsqueeze(0)
        else:
            weight = torch.full((1, 2, NC), 0.5, dtype=x.dtype, device=x.device)
        if self.config["transport"]:
            transport = F.softmax(self.transport, 1)
            recovered = torch.einsum("nmk,myk->nmy", calibrated, transport)
            recovered = normalize_torch(recovered)
        else:
            recovered = calibrated
        arithmetic = (weight * recovered).sum(1)
        geometric = F.softmax((weight * torch.log(calibrated.clamp_min(EPS))).sum(1), 1)
        recovered_geometric = F.softmax((weight * torch.log(recovered.clamp_min(EPS))).sum(1), 1)
        raw = (weight * x).sum(1)
        product = F.softmax(torch.log(x).sum(1), 1)
        if not self.config["refinement"]:
            return arithmetic
        paths = torch.stack([arithmetic, geometric, recovered_geometric, raw, product], 1)
        context = posterior_context(x)
        mixture = F.softmax(self.path.unsqueeze(0) + 0.5 * self.path_gate(context), 1)
        structured = (mixture.unsqueeze(2) * paths).sum(1)
        disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True)
        uncertainty = 1.0 - x.max(2).values.mean(1, keepdim=True)
        gate = torch.sigmoid(self.gate(context)) * (0.10 + 0.55 * disagreement + 0.35 * uncertainty)
        refined = F.softmax(torch.log(structured.clamp_min(EPS)) + self.residual(context), 1)
        return normalize_torch((1.0 - gate) * structured + gate * refined)


def predict(model, x):
    model = model.to(DEVICE).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            value = torch.from_numpy(x[start:start + 1024]).float().to(DEVICE)
            output.append(model(value).float().cpu().numpy())
    return normalize(np.concatenate(output))


def clone_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_params(value):
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def load_reference(posterior):
    checkpoint = torch.load(FULL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ReferenceRCF(checkpoint["matrices"], RCF_VARIANTS["Full RCF"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def load_artifacts():
    if not POSTERIOR_FILE.exists() or not FULL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing verified AVE artifact under {REFERENCE_DIR}")
    with np.load(POSTERIOR_FILE) as archive:
        required = [
            "ROST_oof_audio", "ROST_oof_visual", "ROST_test_folds_audio",
            "ROST_test_folds_visual", "labels", "test_labels",
        ]
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise RuntimeError(f"Posterior artifact missing keys: {missing}")
        return {
            "oof": np.stack([normalize(archive["ROST_oof_audio"]), normalize(archive["ROST_oof_visual"])], 1),
            "test_folds": [
                np.stack([normalize(archive["ROST_test_folds_audio"][fold]), normalize(archive["ROST_test_folds_visual"][fold])], 1)
                for fold in range(len(archive["ROST_test_folds_audio"]))
            ],
            "labels": np.asarray(archive["labels"], dtype=np.int64),
            "test_labels": np.asarray(archive["test_labels"], dtype=np.int64),
        }


def checkpoint_valid(path, variant, seed, signature):
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        return item.get("variant") == variant and item.get("seed") == seed and item.get("signature") == signature
    except Exception:
        return False


def train_component(config, x, labels, test_folds, seed, root, variant, full_state, matrices, signature):
    if variant == "Average/Base":
        return [normalize(fold.mean(1)) for fold in test_folds], {"params": 0, "epoch": 0, "source": "deterministic"}
    path = root / "fusion" / f"{safe_name(variant)}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = ReferenceRCF(matrices, config)
    if checkpoint_valid(path, variant, seed, signature):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        item = torch.load(path, map_location="cpu", weights_only=False)
        return [predict(model, fold) for fold in test_folds], {"params": item["params"], "epoch": item["epoch"], "source": "cache"}
    seed_all(seed)
    if variant.startswith("w/o "):
        model.load_state_dict(full_state, strict=True)
    enabled = {
        "reliability": config["reliability"],
        "scale": config["calibration"],
        "bias": config["calibration"],
        "transport": config["transport"],
        "path": config["refinement"],
        "path_gate": config["refinement"],
        "gate": config["refinement"],
        "residual": config["refinement"],
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name == key or name.startswith(key + ".") for key, value in enabled.items() if value)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {variant}")
    fit, validation = train_test_split(np.arange(len(labels)), test_size=0.20, random_state=seed, stratify=labels)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RCF_EPOCHS, 2e-5)
    xt = torch.from_numpy(x[fit]).float().to(DEVICE)
    yt = torch.from_numpy(labels[fit]).long().to(DEVICE)
    best = {"score": float("inf"), "state": clone_state(model), "epoch": 0}
    stale = 0
    for epoch in range(1, RCF_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(xt)
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), yt) + 0.05 * F.mse_loss(output, F.one_hot(yt, NC).float())
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 2.0)
        optimizer.step()
        scheduler.step()
        validation_metrics = metric_dict(labels[validation], predict(model, x[validation]))
        score = -validation_metrics["f1"] + 0.003 * validation_metrics["nll"]
        if score < best["score"] - 1e-6:
            best = {"score": score, "state": clone_state(model), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best["state"], strict=True)
    count = parameter_count(model)
    torch.save({"variant": variant, "seed": seed, "signature": signature, "epoch": best["epoch"], "params": count, "state_dict": model.cpu().state_dict()}, path)
    return [predict(model, fold) for fold in test_folds], {"params": count, "epoch": best["epoch"], "source": "trained"}


def plot_confusion(labels, probs, path, title):
    matrix = confusion_matrix(labels, normalize(probs).argmax(1), labels=np.arange(NC)).astype(np.float32)
    matrix /= np.maximum(matrix.sum(1, keepdims=True), 1.0)
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="Blues", vmin=0.0, vmax=1.0, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def bootstrap_delta(labels, first, second, seed):
    rng = np.random.default_rng(seed)
    values = []
    first_pred = normalize(first).argmax(1)
    second_pred = normalize(second).argmax(1)
    for _ in range(2000):
        index = rng.integers(0, len(labels), len(labels))
        values.append(float((first_pred[index] == labels[index]).mean() - (second_pred[index] == labels[index]).mean()))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def print_table(frame):
    print("rank | variant                      | acc    | f1     | precision | recall | ece    | brier  | nll    | params")
    for rank, (_, row) in enumerate(frame.iterrows(), 1):
        print(f"{rank:>4} | {row['variant'][:28]:<28} | {row['acc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} | {format_params(row['params']):>6}")


def main():
    start = time.time()
    seed_all(SEED)
    artifact = load_artifacts()
    root = CHECKPOINT_ROOT / f"ablation_{VERSION}" / "DOME_X_AVE_ABLATION"
    log = LOG_ROOT / f"ablation_{VERSION}" / "DOME_X_AVE_ABLATION"
    root.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    reference, checkpoint = load_reference(artifact["oof"])
    reference_output = normalize(np.mean([predict(reference, fold) for fold in artifact["test_folds"]], 0))
    reference_metrics = metric_dict(artifact["test_labels"], reference_output)
    expected = {"acc": 0.8734793187347932, "f1": 0.8688062519374217, "ece": 0.08941532447208302, "brier": 0.20655918589227185, "nll": 0.45173689798048766}
    delta = {name: reference_metrics[name] - value for name, value in expected.items()}
    if max(abs(value) for value in delta.values()) > 1e-6:
        raise RuntimeError(f"Reference RCF reproduction failed: {delta}")
    signature = hashlib.sha256(json.dumps({"posterior": str(POSTERIOR_FILE), "posterior_mtime": POSTERIOR_FILE.stat().st_mtime_ns, "checkpoint": str(FULL_CHECKPOINT), "checkpoint_mtime": FULL_CHECKPOINT.stat().st_mtime_ns, "epochs": RCF_EPOCHS, "seeds": FUSION_SEEDS}, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "mode": "artifact_first_rcf_ablation",
        "signature": signature,
        "reference_posterior": POSTERIOR_FILE,
        "reference_full_checkpoint": FULL_CHECKPOINT,
        "protocol": "Full ROST experts and Full RCF are fixed verified artifacts. RCF component variants initialize from the matching Full RCF state and train only on ROST OOF posterior labels. Official test labels are used only for final metrics.",
        "explicit_non_goal": "ROST component deletion requires independent expert retraining and is intentionally not approximated from frozen posterior artifacts.",
        "fusion_seeds": FUSION_SEEDS,
        "device": str(DEVICE),
    }
    save_json(manifest, log / "manifest.json")
    active = {name: config for name, config in RCF_VARIANTS.items() if not RCF_FILTER or name in RCF_FILTER}
    if "Full RCF" not in active:
        active = {"Full RCF": RCF_VARIANTS["Full RCF"], **active}
    rows = []
    outputs = {}
    for name, config in active.items():
        if name == "Full RCF":
            output = reference_output
            info = {"params": parameter_count(reference), "epoch": "reference", "source": "verified_reference"}
            seed_outputs = [output]
        else:
            seed_outputs = []
            details = []
            for fusion_seed in FUSION_SEEDS:
                fold_outputs, detail = train_component(config, artifact["oof"], artifact["labels"], artifact["test_folds"], fusion_seed, root, name, checkpoint["state_dict"], checkpoint["matrices"], signature)
                seed_outputs.append(normalize(np.mean(fold_outputs, 0)))
                details.append(detail)
            output = normalize(np.mean(seed_outputs, 0))
            info = {"params": int(round(np.mean([item["params"] for item in details]))), "epoch": [item["epoch"] for item in details], "source": [item["source"] for item in details]}
        outputs[name] = output.astype(np.float32)
        row = {"variant": name, **metric_dict(artifact["test_labels"], output), **info}
        rows.append(row)
        plot_confusion(artifact["test_labels"], output, log / f"cm_norm_rost_{safe_name(name)}.png", f"ROST | {name}")
    frame = pd.DataFrame(rows).sort_values(["acc", "f1", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    frame.to_csv(log / "rcf_component_ablation.csv", index=False)
    np.savez_compressed(root / "rcf_component_predictions.npz", test_labels=artifact["test_labels"], **{safe_name(name): value for name, value in outputs.items()})
    full = outputs["Full RCF"]
    bootstrap = []
    for name, output in outputs.items():
        if name != "Full RCF":
            bootstrap.append({"comparison": f"Full RCF - {name}", "accuracy_delta": metric_dict(artifact["test_labels"], full)["acc"] - metric_dict(artifact["test_labels"], output)["acc"], "accuracy_delta_95ci": bootstrap_delta(artifact["test_labels"], full, output, SEED)})
    result = {"manifest": manifest, "reference_reproduction": {"metrics": reference_metrics, "expected": expected, "delta": delta}, "rows": rows, "paired_bootstrap": bootstrap, "time_minutes": (time.time() - start) / 60.0}
    save_json(result, log / "results.json")
    print(f"AVE ROST/RCF artifact ablation device={DEVICE} OOF={len(artifact['labels'])} test={len(artifact['test_labels'])}")
    print(f"Verified Full ROST + Full RCF acc={reference_metrics['acc']:.4f} f1={reference_metrics['f1']:.4f} nll={reference_metrics['nll']:.4f} params={parameter_count(reference)}")
    print_table(frame)
    print(f"Checkpoints={root}")
    print(f"Logs={log}")
    print(f"TimeMinutes={(time.time() - start) / 60.0:.2f}")


if __name__ == "__main__":
    main()
