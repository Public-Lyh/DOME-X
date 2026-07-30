"""Leakage-free DOME-X ablations for ModelNet40.

This runner intentionally does not reuse the legacy IKUN test-oracle paths in
``DOME-X4ModelNet40-clcp.py``.  Every fusion model is trained from expert OOF
posteriors, while ModelNet40 test labels are consumed only for final reports.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader


PLACEHOLDER_ROOT = Path("your path")
WORKSPACE_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
PROJECT_ROOT = WORKSPACE_ROOT / "Code" / "ModelNet40"
SOURCE_PATH = PROJECT_ROOT / "DOME-X4ModelNet40-clcp.py"
BASE_CKPT_DIR = PROJECT_ROOT / "checkpoints"
BASE_LOG_DIR = PROJECT_ROOT / "logs"
SEED = int(os.environ.get("DOME_X_SEED", "42"))
PIPELINE_SEEDS = [SEED + i for i in range(int(os.environ.get("DOME_X_PIPELINE_SEEDS", "3")))]
FUSION_SEEDS = [SEED + 1000 + i for i in range(int(os.environ.get("DOME_X_FUSION_SEEDS", "5")))]
OUTER_FOLDS = int(os.environ.get("DOME_X_OUTER_FOLDS", "3"))
PRETRAIN_EPOCHS = int(os.environ.get("DOME_X_PRETRAIN_EPOCHS", "110"))
ROST_EPOCHS = int(os.environ.get("DOME_X_ROST_EPOCHS", "70"))
RCF_EPOCHS = int(os.environ.get("DOME_X_RCF_EPOCHS", "180"))
EXPERT_PATIENCE = int(os.environ.get("DOME_X_EXPERT_PATIENCE", "28"))
FUSION_PATIENCE = int(os.environ.get("DOME_X_FUSION_PATIENCE", "28"))
PROFILE_INTERVAL = int(os.environ.get("DOME_X_PROFILE_INTERVAL", "5"))
ROST_BATCH_SIZE = int(os.environ.get("DOME_X_ROST_BATCH_SIZE", "12"))
VARIANT_FILTER = [item.strip() for item in os.environ.get("DOME_X_VARIANTS", "").split(",") if item.strip()]
RCF_FILTER = [item.strip() for item in os.environ.get("DOME_X_RCF_VARIANTS", "").split(",") if item.strip()]
EPS = 1e-8
NC = 40
DEVICE = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

if OUTER_FOLDS < 2:
    raise ValueError("DOME_X_OUTER_FOLDS must be at least 2 for leakage-free OOF ablation")


ROST_VARIANTS = {
    "Full ROST": {"structure": True, "collapse": True, "complement": True, "joint": True},
    "w/o Structure": {"structure": False, "collapse": True, "complement": True, "joint": True},
    "w/o Anti-collapse": {"structure": True, "collapse": False, "complement": True, "joint": True},
    "w/o Complementarity": {"structure": True, "collapse": True, "complement": False, "joint": True},
    "w/o Joint Recovery": {"structure": True, "collapse": True, "complement": True, "joint": False},
}

RCF_VARIANTS = {
    "Average/Base": {"reliability": "none", "calibration": "none", "transport": "none", "refinement": False},
    "+ Class-wise Reliability": {"reliability": "classwise", "calibration": "none", "transport": "none", "refinement": False},
    "+ Posterior Calibration": {"reliability": "classwise", "calibration": "classwise", "transport": "none", "refinement": False},
    "+ Bias Transport": {"reliability": "classwise", "calibration": "classwise", "transport": "learnable", "refinement": False},
    "Full RCF": {"reliability": "classwise", "calibration": "classwise", "transport": "learnable", "refinement": True},
    "w/o Reliability": {"reliability": "none", "calibration": "classwise", "transport": "learnable", "refinement": True},
    "Global Reliability": {"reliability": "global", "calibration": "classwise", "transport": "learnable", "refinement": True},
    "w/o Calibration": {"reliability": "classwise", "calibration": "none", "transport": "learnable", "refinement": True},
    "Shared Calibration": {"reliability": "classwise", "calibration": "shared", "transport": "learnable", "refinement": True},
    "w/o Bias Transport": {"reliability": "classwise", "calibration": "classwise", "transport": "none", "refinement": True},
    "Identity Bias Matrix": {"reliability": "classwise", "calibration": "classwise", "transport": "identity", "refinement": True},
    "Hard Confusion Transport": {"reliability": "classwise", "calibration": "classwise", "transport": "hard", "refinement": True},
    "No Historical Bias": {"reliability": "classwise", "calibration": "classwise", "transport": "no_history", "refinement": True},
    "w/o Disagreement Refinement": {"reliability": "classwise", "calibration": "classwise", "transport": "learnable", "refinement": False},
}


def load_source():
    spec = importlib.util.spec_from_file_location("modelnet40_dome_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the ModelNet40 source: {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SRC = load_source()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def normalize(value):
    value = np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    value = np.clip(value, EPS, None)
    return value / np.maximum(value.sum(axis=1, keepdims=True), EPS)


def rows(value):
    value = np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    value = np.clip(value, 0.0, None)
    return value / np.maximum(value.sum(axis=1, keepdims=True), EPS)


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def config_hash():
    source_hash = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16]
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    payload = {
        "source": source_hash, "runner": runner_hash, "seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS,
        "folds": OUTER_FOLDS, "pretrain": PRETRAIN_EPOCHS, "rost": ROST_EPOCHS,
        "rcf": RCF_EPOCHS, "rost_variants": ROST_VARIANTS, "rcf_variants": RCF_VARIANTS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def output_dirs():
    BASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("DOME_X_ABLATION_VERSION")
    if requested:
        version_text = requested.removeprefix("ablation_v")
        if not version_text.isdigit():
            raise ValueError("DOME_X_ABLATION_VERSION must be a numeric value such as ablation_v12")
        version = int(version_text)
        checkpoint = BASE_CKPT_DIR / f"ablation_v{version}" / "DOME_X_ModelNet40_ABLATION"
        log = BASE_LOG_DIR / f"ablation_v{version}" / "DOME_X_ModelNet40_ABLATION"
        checkpoint.mkdir(parents=True, exist_ok=True)
        log.mkdir(parents=True, exist_ok=True)
        return version, checkpoint, log
    versions = []
    for path in BASE_CKPT_DIR.glob("ablation_v*"):
        match = re.fullmatch(r"ablation_v(\d+)", path.name)
        if path.is_dir() and match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    checkpoint = BASE_CKPT_DIR / f"ablation_v{version}" / "DOME_X_ModelNet40_ABLATION"
    log = BASE_LOG_DIR / f"ablation_v{version}" / "DOME_X_ModelNet40_ABLATION"
    checkpoint.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return version, checkpoint, log


def ece(labels, proba, adaptive=False):
    confidence = proba.max(1)
    correct = proba.argmax(1) == labels
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, 16)) if adaptive else np.linspace(0.0, 1.0, 16)
    edges[0], edges[-1] = 0.0, 1.0
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def metrics(labels, proba):
    proba = normalize(proba)
    prediction = proba.argmax(1)
    one_hot = np.eye(NC)[labels]
    class_ece = []
    for klass in range(NC):
        binary = np.stack([1.0 - proba[:, klass], proba[:, klass]], axis=1)
        class_ece.append(ece((labels == klass).astype(np.int64), binary))
    return {
        "acc": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, average="macro", zero_division="warn")),
        "precision": float(precision_score(labels, prediction, average="macro", zero_division="warn")),
        "recall": float(recall_score(labels, prediction, average="macro", zero_division="warn")),
        "ece": ece(labels, proba), "adaptive_ece": ece(labels, proba, adaptive=True),
        "classwise_ece": float(np.mean(class_ece)),
        "brier": float(np.square(proba - one_hot).sum(1).mean()),
        "nll": float(log_loss(labels, proba, labels=np.arange(NC))),
    }


def mechanism_subset_metrics(labels, experts, reference, candidate, oof_labels=None, oof_experts=None):
    labels = np.asarray(labels)
    experts = np.asarray(experts, dtype=np.float64)
    experts = experts / np.maximum(experts.sum(2, keepdims=True), EPS)
    reference = normalize(reference)
    candidate = normalize(candidate)
    prediction = candidate.argmax(1)
    expert_predictions = experts.argmax(2)
    disagreement = np.any(expert_predictions != expert_predictions[:, :1], axis=1)
    confidence = experts.max(2).mean(1)
    thresholds = {"disagreement": float(disagreement.mean())}
    if oof_labels is not None and oof_experts is not None:
        oof_experts = np.asarray(oof_experts)
        oof_confidence = oof_experts.max(2).mean(1)
        thresholds["low_confidence"] = float(np.quantile(oof_confidence, 0.25))
    else:
        thresholds["low_confidence"] = float(np.quantile(confidence, 0.25))
    low_confidence = confidence <= thresholds["low_confidence"]
    reference_prediction = reference.argmax(1)
    recoverable_error = (reference_prediction != labels) & np.any(expert_predictions == labels[:, None], axis=1)
    rows = {
        "disagreement_rate": float(disagreement.mean()),
        "disagreement_acc": float(accuracy_score(labels[disagreement], prediction[disagreement])) if disagreement.any() else 0.0,
        "low_confidence_rate": float(low_confidence.mean()),
        "low_confidence_acc": float(accuracy_score(labels[low_confidence], prediction[low_confidence])) if low_confidence.any() else 0.0,
        "recoverable_error_rate": float(recoverable_error.mean()),
        "recoverable_error_acc": float(accuracy_score(labels[recoverable_error], prediction[recoverable_error])) if recoverable_error.any() else 0.0,
        "wrong_to_correct": int(((reference_prediction != labels) & (prediction == labels)).sum()),
        "correct_to_wrong": int(((reference_prediction == labels) & (prediction != labels)).sum()),
    }
    hard_classes = []
    if oof_labels is not None and oof_experts is not None:
        oof_prediction = np.asarray(oof_experts).mean(1).argmax(1)
        cm = confusion_matrix(oof_labels, oof_prediction, labels=np.arange(NC)).astype(np.float64)
        recall = np.diag(cm) / np.maximum(cm.sum(1), 1.0)
        hard_classes = np.argsort(recall)[:max(1, NC // 5)].tolist()
    if hard_classes:
        hard_mask = np.isin(labels, hard_classes)
        rows["hard_class_rate"] = float(hard_mask.mean())
        rows["hard_class_acc"] = float(accuracy_score(labels[hard_mask], prediction[hard_mask])) if hard_mask.any() else 0.0
        rows["hard_classes"] = hard_classes
    return rows


def soft_confusion(labels, proba):
    matrix = np.zeros((NC, NC), dtype=np.float64)
    np.add.at(matrix, labels, normalize(proba))
    counts = np.bincount(labels, minlength=NC).astype(np.float64)
    matrix /= np.maximum(counts[:, None], 1.0)
    matrix[counts == 0] = 1.0 / NC
    return rows(matrix)


def js_rows(matrix):
    matrix = rows(matrix)
    left, right = matrix[:, None], matrix[None]
    middle = 0.5 * (left + right)
    return 0.5 * ((left * np.log((left + EPS) / (middle + EPS))).sum(-1) + (right * np.log((right + EPS) / (middle + EPS))).sum(-1)) / np.log(2.0)


def effective_rank(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    probability = singular / max(singular.sum(), EPS)
    return float(np.exp(-(probability * np.log(probability + EPS)).sum()) / min(matrix.shape))


def profile(matrices):
    values = []
    for matrix in matrices:
        matrix = rows(matrix)
        entropy = -(matrix * np.log(matrix + EPS)).sum(1) / np.log(NC)
        usage = matrix.mean(0)
        values.append({
            "row_entropy": float(entropy.mean()),
            "top3_mass": float(np.sort(matrix, axis=1)[:, -3:].sum(1).mean()),
            "separation": float(js_rows(matrix)[~np.eye(NC, dtype=bool)].mean()),
            "column_entropy": float(-(usage * np.log(usage + EPS)).sum() / np.log(NC)),
            "effective_rank": effective_rank(matrix),
            "diagonal": float(np.diag(matrix).mean()),
        })
    joint = np.concatenate([rows(matrix) for matrix in matrices], axis=1)
    distances = 1.0 - (joint @ joint.T) / np.maximum(np.linalg.norm(joint, axis=1)[:, None] * np.linalg.norm(joint, axis=1)[None], EPS)
    relation = [js_rows(matrix) for matrix in matrices]
    direct = np.mean([(matrices[i].ravel() @ matrices[j].ravel()) / max(np.linalg.norm(matrices[i]) * np.linalg.norm(matrices[j]), EPS) for i in range(len(matrices)) for j in range(i + 1, len(matrices))])
    graph = np.mean([(relation[i].ravel() @ relation[j].ravel()) / max(np.linalg.norm(relation[i]) * np.linalg.norm(relation[j]), EPS) for i in range(len(matrices)) for j in range(i + 1, len(matrices))])
    rescue = []
    for current, other in ((i, j) for i in range(len(matrices)) for j in range(len(matrices)) if i != j):
        weights = np.exp(-relation[other] / 0.12)
        np.fill_diagonal(weights, 0.0)
        rescue.append(float((weights * relation[current]).sum() / max(weights.sum(), EPS)))
    jsri = 0.30 * distances[~np.eye(NC, dtype=bool)].mean() + 0.20 * effective_rank(joint) + 0.30 * np.mean(rescue) + 0.20 * (1.0 - 0.5 * (direct + graph))
    return {"experts": values, "direct_redundancy": float(direct), "graph_redundancy": float(graph), "rescue": float(np.mean(rescue)), "jsri": float(np.clip(jsri, 0.0, 1.0))}


def plot_cm(labels, proba, path, title, categories):
    matrix = confusion_matrix(labels, normalize(proba).argmax(1), labels=np.arange(NC)).astype(np.float64)
    matrix = rows(matrix)
    figure, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, ax=axis, xticklabels=categories, yticklabels=categories)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.tick_params(axis="both", labelsize=6)
    plt.setp(axis.get_xticklabels(), rotation=90, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_matrix(matrix, path, title, xlabels, ylabels=None):
    if ylabels is None:
        ylabels = xlabels
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="viridis", ax=axis, xticklabels=xlabels, yticklabels=ylabels)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def model_input(pcs, normals, views, indices, kind, device):
    if kind == "mv":
        return torch.from_numpy(views[indices]).float().to(device, non_blocking=True)
    values = np.concatenate([pcs[indices], normals[indices]], axis=2)
    return torch.from_numpy(values).float().to(device, non_blocking=True)


def predict_expert(model, pcs, normals, views, indices, kind, device=DEVICE):
    model = model.to(device).eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for start in range(0, len(indices), 32):
            index = np.asarray(indices[start:start + 32])
            values.append(F.softmax(model(model_input(pcs, normals, views, index, kind, device)).float(), 1).cpu().numpy())
    return normalize(np.concatenate(values))


def build_loader(pcs, normals, views, labels, indices, kind, augment, batch):
    dataset = SRC.MVDataset(pcs[indices], normals[indices], views[indices], labels[indices], augment=augment) if kind == "mv" else SRC.PCDataset(pcs[indices], normals[indices], labels[indices], augment=augment)
    return DataLoader(dataset, batch_size=batch, shuffle=augment, num_workers=0, pin_memory=DEVICE.type == "cuda", drop_last=augment)


def controller_weights(current_profile, config, epoch):
    progress = epoch / max(ROST_EPOCHS, 1)
    expert = current_profile["experts"]
    entropy = np.mean([item["row_entropy"] for item in expert])
    usage = np.mean([item["column_entropy"] for item in expert])
    rank = np.mean([item["effective_rank"] for item in expert])
    return {
        "ce": float(np.clip(0.88 - 0.22 * progress, 0.64, 0.88)),
        "structure": (0.025 + 0.16 * progress) * (1.0 + max(0.0, entropy - 0.78) + max(0.0, 0.16 - entropy)) if config["structure"] else 0.0,
        "collapse": (0.025 + 0.14 * progress) * (1.0 + max(0.0, 0.92 - usage) + max(0.0, 0.86 - rank)) if config["collapse"] else 0.0,
        "complement": (0.020 + 0.14 * progress) * (1.0 + max(0.0, current_profile["direct_redundancy"] - 0.86) + max(0.0, current_profile["graph_redundancy"] - 0.86)) if config["complement"] else 0.0,
        "joint": (0.030 + 0.14 * progress) * (1.0 + max(0.0, 0.60 - current_profile["jsri"])) if config["joint"] else 0.0,
    }


def torch_confusion(labels, posterior, profile_matrix):
    one_hot = F.one_hot(labels, NC).to(posterior.dtype)
    counts = one_hot.sum(0)
    batch = one_hot.T @ posterior / counts.clamp_min(1).unsqueeze(1)
    profile_matrix = profile_matrix.to(posterior.device, posterior.dtype)
    # Global profile anchors rows absent from a mini-batch and stabilizes rows present.
    matrix = torch.where((counts > 0).unsqueeze(1), 0.55 * batch + 0.45 * profile_matrix, profile_matrix)
    return matrix / matrix.sum(1, keepdim=True).clamp_min(EPS)


def torch_js_rows(matrix):
    matrix = matrix.clamp_min(EPS)
    matrix = matrix / matrix.sum(1, keepdim=True)
    left, right = matrix[:, None], matrix[None]
    middle = 0.5 * (left + right)
    return 0.5 * ((left * (left.log() - middle.log())).sum(-1) + (right * (right.log() - middle.log())).sum(-1)) / np.log(2.0)


class RecoveryProbe(nn.Module):
    """A deliberately small probe; it diagnoses posterior recoverability."""
    def __init__(self, modalities):
        super().__init__()
        self.linear = nn.Linear(modalities * NC, NC)

    def forward(self, posterior):
        return self.linear(posterior.flatten(1))


def roster_loss(logits, labels, profile_matrix, peer_matrices, peer_posterior, config, weights, probe, all_posterior):
    posterior = F.softmax(logits.float(), 1)
    matrix = torch_confusion(labels, posterior, profile_matrix)
    semantic = F.cross_entropy(logits.float(), labels, label_smoothing=0.02)
    js = torch_js_rows(matrix)
    eye = torch.eye(NC, dtype=torch.bool, device=logits.device)
    entropy = -(matrix * matrix.clamp_min(EPS).log()).sum(1) / np.log(NC)
    top3 = matrix.topk(3, 1).values.sum(1)
    structure = F.relu(0.15 - entropy).square().mean() + F.relu(entropy - 0.80).square().mean() + F.relu(0.50 - top3).square().mean() + 0.45 * torch.exp(-js[~eye] / 0.14).mean()
    usage = matrix.mean(0)
    usage_entropy = -(usage * usage.clamp_min(EPS).log()).sum() / np.log(NC)
    singular = torch.linalg.svdvals(matrix.float())
    singular = singular / singular.sum().clamp_min(EPS)
    rank_loss = 1.0 + (singular * singular.clamp_min(EPS).log()).sum() / np.log(NC)
    collapse = F.relu(0.62 - usage_entropy).square() + F.relu(usage.max() - 0.35).square() + 0.22 * rank_loss
    complement = torch.zeros((), device=logits.device)
    if peer_matrices:
        terms = []
        for peer in peer_matrices:
            peer = peer.to(logits.device, logits.dtype)
            peer_js = torch_js_rows(peer)
            rescue_weight = torch.exp(-peer_js / 0.12)
            rescue_weight.fill_diagonal_(0.0)
            direct = F.cosine_similarity(matrix.flatten(), peer.flatten(), dim=0)
            graph = F.cosine_similarity(js.flatten(), peer_js.flatten(), dim=0)
            rescue = -(rescue_weight * js).sum() / rescue_weight.sum().clamp_min(EPS)
            terms.append(0.35 * direct + 0.30 * graph + 0.35 * rescue)
        complement = torch.stack(terms).mean()
    joint = torch.zeros((), device=logits.device)
    probe_loss = torch.zeros((), device=logits.device)
    if config["joint"]:
        smooth = matrix + 1e-3
        smooth = smooth / smooth.sum(1, keepdim=True)
        reverse = smooth.T / smooth.T.sum(1, keepdim=True).clamp_min(EPS)
        decode = -torch.diag(smooth @ reverse).clamp_min(EPS).log().mean()
        probe_loss = F.cross_entropy(probe(all_posterior), labels)
        joint = 0.45 * decode + 0.55 * probe_loss
    loss = weights["ce"] * semantic + weights["structure"] * structure + weights["collapse"] * collapse + weights["complement"] * complement + weights["joint"] * joint
    return loss, {"semantic": float(semantic.detach()), "structure": float(structure.detach()), "collapse": float(collapse.detach()), "complement": float(complement.detach()), "joint": float(joint.detach()), "probe": float(probe_loss.detach())}


def train_semantic_initializer(factory, kind, pcs, normals, views, labels, fit_idx, profile_idx, seed, name):
    seed_all(seed)
    model = factory()
    model, history = SRC.dome_train_stage(model, kind, pcs, normals, views, labels, fit_idx, profile_idx, DEVICE, PRETRAIN_EPOCHS, "CE", seed, tag=f"Shared semantic initialization {name}")
    return model.cpu(), history


def train_rost_models(models, specs, pcs, normals, views, labels, fit_idx, profile_idx, config, seed, fold):
    seed_all(seed)
    names = list(models)
    models = {name: model.to(DEVICE) for name, model in models.items()}
    initial = [soft_confusion(labels[profile_idx], predict_expert(models[name], pcs, normals, views, profile_idx, specs[name][1])) for name in names]
    ema = [matrix.copy() for matrix in initial]
    probe = RecoveryProbe(len(names)).to(DEVICE)
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=2e-4 if specs[name][1] == "pc" else 4e-5, weight_decay=5e-4) for name in names}
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=8e-4, weight_decay=2e-4)
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ROST_EPOCHS, eta_min=1e-6) for name, optimizer in optimizers.items()}
    probe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(probe_optimizer, ROST_EPOCHS, eta_min=1e-6)
    order_batches = [np.asarray(np.random.permutation(fit_idx)[start:start + ROST_BATCH_SIZE]) for start in range(0, len(fit_idx), ROST_BATCH_SIZE)]
    best = {"score": -np.inf, "models": {name: clone_state(model) for name, model in models.items()}, "probe": clone_state(probe), "epoch": 0}
    history = []
    snapshots = {}
    stale = 0
    current_profile = profile(ema)
    weights = controller_weights(current_profile, config, 1)
    for epoch in range(1, ROST_EPOCHS + 1):
        if epoch == 1 or epoch % PROFILE_INTERVAL == 0:
            current = [soft_confusion(labels[profile_idx], predict_expert(models[name], pcs, normals, views, profile_idx, specs[name][1])) for name in names]
            ema = [rows(0.85 * old + 0.15 * new) for old, new in zip(ema, current)]
            current_profile = profile(ema)
            weights = controller_weights(current_profile, config, epoch)
        for model in models.values():
            model.train()
        probe.train()
        total_loss = 0.0
        seen = 0
        for index in order_batches:
            if len(index) < 2:
                continue
            target = torch.from_numpy(labels[index]).long().to(DEVICE)
            inputs = {name: model_input(pcs, normals, views, index, specs[name][1], DEVICE) for name in names}
            logits = {name: models[name](inputs[name]).float() for name in names}
            posterior = {name: F.softmax(logits[name], 1) for name in names}
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            joint_loss = torch.zeros((), device=DEVICE)
            for position, name in enumerate(names):
                all_posterior = torch.stack([posterior[other_name] if other_name == name else posterior[other_name].detach() for other_name in names], 1)
                peers = [torch.from_numpy(ema[peer].astype(np.float32)) for peer in range(len(names)) if peer != position]
                loss, _ = roster_loss(
                    logits[name], target,
                    torch.from_numpy(ema[position].astype(np.float32)), peers, posterior,
                    config, weights, probe, all_posterior,
                )
                joint_loss = joint_loss + loss
                total_loss += float(loss.detach()) * len(index)
                seen += len(index)
            joint_loss.backward()
            for model in models.values():
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            nn.utils.clip_grad_norm_(probe.parameters(), 2.0)
            for optimizer in optimizers.values():
                optimizer.step()
            probe_optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()
        probe_scheduler.step()
        validation = [predict_expert(models[name], pcs, normals, views, profile_idx, specs[name][1]) for name in names]
        matrices = [soft_confusion(labels[profile_idx], value) for value in validation]
        ema = [rows(0.85 * old + 0.15 * new) for old, new in zip(ema, matrices)]
        current_profile = profile(ema)
        if epoch == 1:
            snapshots["rost_early"] = [item.copy() for item in ema]
        if epoch >= max(2, ROST_EPOCHS // 2) and "rost_middle" not in snapshots:
            snapshots["rost_middle"] = [item.copy() for item in ema]
        with torch.no_grad():
            probe_out = F.softmax(probe(torch.from_numpy(np.stack(validation, 1).astype(np.float32)).to(DEVICE)), 1).cpu().numpy()
        probe_metric = metrics(labels[profile_idx], probe_out)
        expert_metrics = [metrics(labels[profile_idx], value) for value in validation]
        score = 0.35 * np.mean([item["acc"] + item["f1"] for item in expert_metrics]) + 0.20 * probe_metric["f1"] + 0.10 * current_profile["jsri"]
        row = {"epoch": epoch, "loss": total_loss / max(seen, 1), "score": float(score), "profile": current_profile, "weights": weights, "probe": probe_metric, "experts": expert_metrics}
        history.append(row)
        if score > best["score"] + 1e-5:
            best = {"score": float(score), "models": {name: clone_state(model) for name, model in models.items()}, "probe": clone_state(probe), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % PROFILE_INTERVAL == 0:
            print(f"ROST fold={fold} epoch={epoch}/{ROST_EPOCHS} score={score:.4f} probe_f1={probe_metric['f1']:.4f} JSRI={current_profile['jsri']:.4f}")
        if epoch >= 20 and stale >= EXPERT_PATIENCE:
            break
        order = np.random.permutation(fit_idx)
        order_batches = [np.asarray(order[start:start + ROST_BATCH_SIZE]) for start in range(0, len(order), ROST_BATCH_SIZE)]
    for name in names:
        models[name].load_state_dict(best["models"][name])
    probe.load_state_dict(best["probe"])
    restored = [predict_expert(models[name], pcs, normals, views, profile_idx, specs[name][1]) for name in names]
    snapshots["rost_final"] = [soft_confusion(labels[profile_idx], item) for item in restored]
    return {name: model.cpu() for name, model in models.items()}, probe.cpu(), history, best["epoch"], snapshots


def valid_expert_artifact(path, tag):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return value.get("tag") == tag and "holdout" in value and "test" in value
    except Exception:
        return False


def valid_semantic_artifact(path, tag):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return value.get("tag") == tag and "states" in value and "fit_idx" in value and "profile_idx" in value and "holdout_idx" in value
    except Exception:
        return False


def semantic_initialization_fold(seed, fold, specs, train, labels, root, tag):
    pcs, normals, views = train
    path = root / "semantic_initialization" / f"shared_s{seed}_f{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid_semantic_artifact(path, tag):
        return torch.load(path, map_location="cpu", weights_only=False)
    splitter = list(StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    outer_fit, holdout = splitter[fold - 1]
    fit_idx, profile_idx = train_test_split(outer_fit, test_size=0.15, stratify=labels[outer_fit], random_state=seed * 100 + fold)
    states, histories, matrices = {}, {}, {}
    for position, (name, (factory, kind)) in enumerate(specs.items()):
        print(f"Shared semantic initialization seed={seed} fold={fold}/{OUTER_FOLDS}: {name}")
        model, history = train_semantic_initializer(factory, kind, pcs, normals, views, labels, fit_idx, profile_idx, seed + fold * 100 + position, name)
        states[name] = clone_state(model)
        histories[name] = history
        matrices[name] = soft_confusion(labels[profile_idx], predict_expert(model, pcs, normals, views, profile_idx, kind))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    artifact = {"tag": tag, "seed": seed, "fold": fold, "fit_idx": np.asarray(fit_idx, dtype=np.int64), "profile_idx": np.asarray(profile_idx, dtype=np.int64), "holdout_idx": np.asarray(holdout, dtype=np.int64), "states": states, "histories": histories, "profile_matrices": matrices}
    torch.save(artifact, path)
    return artifact


def expert_fold(variant, config, seed, fold, specs, train, test, labels, root, tag):
    pcs, normals, views = train
    test_pcs, test_normals, test_views = test
    path = root / "experts" / f"{slug(variant)}_s{seed}_f{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid_expert_artifact(path, tag):
        return torch.load(path, map_location="cpu", weights_only=False)
    initialization = semantic_initialization_fold(seed, fold, specs, train, labels, root, tag)
    holdout, fit_idx, profile_idx = initialization["holdout_idx"], initialization["fit_idx"], initialization["profile_idx"]
    models = {}
    for name, (factory, _) in specs.items():
        model = factory()
        model.load_state_dict(initialization["states"][name])
        models[name] = model
    print(f"{variant} seed={seed} fold={fold}/{OUTER_FOLDS}: controller-driven ROST")
    models, probe, rost_history, rost_epoch, rost_snapshots = train_rost_models(models, specs, pcs, normals, views, labels, fit_idx, profile_idx, config, seed + fold * 1000, fold)
    output = {"tag": tag, "variant": variant, "seed": seed, "fold": fold, "holdout_idx": np.asarray(holdout, dtype=np.int64), "holdout": {}, "test": {}, "profile_idx": np.asarray(profile_idx, dtype=np.int64), "semantic_initialization": {"path": str(root / "semantic_initialization" / f"shared_s{seed}_f{fold}.pt"), "histories": initialization["histories"]}, "rost_history": rost_history, "rost_epoch": rost_epoch, "probe_state": probe.state_dict()}
    profile_posteriors = []
    test_indices = np.arange(len(test_pcs))
    for name, (_, kind) in specs.items():
        output["holdout"][name] = predict_expert(models[name], pcs, normals, views, holdout, kind).astype(np.float32)
        output["test"][name] = predict_expert(models[name], test_pcs, test_normals, test_views, test_indices, kind).astype(np.float32)
        profile_posteriors.append(predict_expert(models[name], pcs, normals, views, profile_idx, kind))
        output[f"{slug(name)}_state"] = models[name].state_dict()
    output["final_profile"] = profile([soft_confusion(labels[profile_idx], item) for item in profile_posteriors])
    output["profile_snapshots"] = {
        "semantic_initialization": [initialization["profile_matrices"][name] for name in specs],
        "final": [soft_confusion(labels[profile_idx], item) for item in profile_posteriors],
        **rost_snapshots,
    }
    torch.save(output, path)
    return output


def transport_from_confusion(matrix):
    """Return P(Y=true | predicted class), shape true x predicted, column stochastic."""
    matrix = rows(matrix) + 1e-3
    reverse = matrix.T / np.maximum(matrix.T.sum(1, keepdims=True), EPS)  # predicted x true
    return reverse.T


def hard_transport_from_confusion(matrix):
    hard = np.zeros_like(matrix, dtype=np.float64)
    hard[np.arange(NC), np.argmax(matrix, axis=1)] = 1.0
    hard += 1e-3
    reverse = hard.T / np.maximum(hard.T.sum(1, keepdims=True), EPS)
    return reverse.T


def rcf_context(x, calibrated, recovered):
    entropy = -(x * x.clamp_min(EPS).log()).sum(2) / np.log(NC)
    top = x.topk(2, dim=2).values
    margin = top[:, :, 0] - top[:, :, 1]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float()
    return torch.cat([x.flatten(1), calibrated.flatten(1), recovered.flatten(1), x.max(2).values, entropy, margin, disagreement], 1)


class ComponentRCF(nn.Module):
    """Full RCF with switchable components for matched checkpoint ablations."""
    def __init__(self, matrices, config):
        super().__init__()
        self.config = config
        self.modalities = len(matrices)
        matrix = np.stack(matrices).astype(np.float32)
        reliability = np.clip(np.diagonal(matrix, axis1=1, axis2=2), 0.02, 1.0)
        reliability /= reliability.sum(0, keepdims=True)
        if config["transport"] == "identity":
            transport = np.stack([np.eye(NC, dtype=np.float32) for _ in matrix])
        elif config["transport"] == "hard":
            transport = np.stack([hard_transport_from_confusion(item) for item in matrix]).astype(np.float32)
        elif config["transport"] == "no_history":
            transport = np.stack([np.eye(NC, dtype=np.float32) for _ in matrix])
        else:
            transport = np.stack([transport_from_confusion(item) for item in matrix]).astype(np.float32)
        self.reliability_logits = nn.Parameter(torch.log(torch.from_numpy(reliability)))
        self.log_temperature = nn.Parameter(torch.zeros(self.modalities, NC))
        self.scale = nn.Parameter(torch.zeros(self.modalities, NC))
        self.bias = nn.Parameter(torch.zeros(self.modalities, NC))
        self.transport_logits = nn.Parameter(torch.log(torch.from_numpy(transport).clamp_min(EPS)))
        context_dim = self.modalities * NC * 3 + self.modalities * 4
        width = max(96, 2 * NC)
        self.path_logits = nn.Parameter(torch.tensor([0.35, 0.30, 0.20, 0.15], dtype=torch.float32))
        gate_output = nn.Linear(width, 1)
        residual_output = nn.Linear(width, NC)
        self.gate = nn.Sequential(nn.Linear(context_dim, width), nn.LayerNorm(width), nn.GELU(), gate_output)
        self.residual = nn.Sequential(nn.Linear(context_dim, width), nn.LayerNorm(width), nn.GELU(), residual_output)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, -3.0)
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

    def enabled_parameters(self):
        enable = {
            "reliability_logits": self.config["reliability"] in ("global", "classwise"),
            "log_temperature": self.config["calibration"] in ("shared", "classwise"), "scale": self.config["calibration"] in ("shared", "classwise"), "bias": self.config["calibration"] in ("shared", "classwise"),
            "transport_logits": self.config["transport"] in ("learnable", "no_history"), "path_logits": self.config["refinement"],
            "gate": self.config["refinement"], "residual": self.config["refinement"],
        }
        parameters = []
        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(name == key or name.startswith(key + ".") for key, active in enable.items() if active)
            if parameter.requires_grad:
                parameters.append(parameter)
        return parameters

    def forward(self, x):
        x = x.clamp_min(EPS)
        if self.config["calibration"] != "none":
            if self.config["calibration"] == "shared":
                log_temperature = self.log_temperature.mean(1, keepdim=True).expand_as(self.log_temperature)
                scale = self.scale.mean(1, keepdim=True).expand_as(self.scale)
                bias = self.bias.mean(1, keepdim=True).expand_as(self.bias)
            else:
                log_temperature, scale, bias = self.log_temperature, self.scale, self.bias
            temperature = F.softplus(log_temperature).unsqueeze(0).clamp_min(0.15)
            calibrated = F.softmax(torch.log(x) / temperature * F.softplus(scale).unsqueeze(0) + bias.unsqueeze(0), 2)
        else:
            calibrated = x
        if self.config["transport"] != "none":
            # Normalizing over true labels makes every predicted-class column sum to one.
            transport = F.softmax(self.transport_logits, dim=1)
            recovered = torch.einsum("nmk,myk->nmy", calibrated, transport)
            recovered = recovered / recovered.sum(2, keepdim=True).clamp_min(EPS)
        else:
            recovered = calibrated
        if self.config["reliability"] == "classwise":
            weights = F.softmax(self.reliability_logits, 0).unsqueeze(0)
        elif self.config["reliability"] == "global":
            global_logits = self.reliability_logits.mean(1, keepdim=True)
            weights = F.softmax(global_logits, 0).expand(-1, NC).unsqueeze(0)
        else:
            weights = torch.full((1, self.modalities, NC), 1.0 / self.modalities, dtype=x.dtype, device=x.device)
        arithmetic = (weights * recovered).sum(1)
        geometric = F.softmax((weights * torch.log(recovered.clamp_min(EPS))).sum(1), 1)
        calibrated_geometric = F.softmax((weights * torch.log(calibrated.clamp_min(EPS))).sum(1), 1)
        raw = (weights * x).sum(1)
        if not self.config["refinement"]:
            return arithmetic / arithmetic.sum(1, keepdim=True).clamp_min(EPS)
        context = rcf_context(x, calibrated, recovered)
        mixture = F.softmax(self.path_logits, 0)
        structured = mixture[0] * arithmetic + mixture[1] * geometric + mixture[2] * calibrated_geometric + mixture[3] * raw
        disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True)
        uncertainty = 1.0 - x.max(2).values.mean(1, keepdim=True)
        active = 0.08 + 0.62 * disagreement + 0.30 * uncertainty
        gate = torch.sigmoid(self.gate(context)) * active
        refined = F.softmax(torch.log(structured.clamp_min(EPS)) + 0.10 * torch.tanh(self.residual(context)), 1)
        output = (1.0 - gate) * structured + gate * refined
        return output / output.sum(1, keepdim=True).clamp_min(EPS)


def rcf_regularization(model):
    value = torch.zeros((), device=DEVICE)
    if model.config["reliability"] in ("global", "classwise"):
        value = value + 0.002 * model.reliability_logits.square().mean()
    if model.config["calibration"] in ("shared", "classwise"):
        value = value + 0.002 * (model.log_temperature.square().mean() + model.scale.square().mean() + model.bias.square().mean())
    if model.config["transport"] in ("learnable", "no_history"):
        value = value + 0.001 * model.transport_logits.square().mean()
    if model.config["refinement"]:
        value = value + 0.001 * model.path_logits.square().mean()
    return value


def rcf_predict(model, x):
    model = model.to(DEVICE).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            output.append(model(torch.from_numpy(x[start:start + 256]).float().to(DEVICE)).cpu().numpy())
    return normalize(np.concatenate(output))


def train_rcf(model, x, labels, fit_idx, monitor_idx, seed):
    seed_all(seed)
    model = model.to(DEVICE)
    params = model.enabled_parameters()
    if not params:
        return model.cpu(), 0, []
    optimizer = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RCF_EPOCHS, eta_min=2e-5)
    x_fit = torch.from_numpy(x[fit_idx]).float().to(DEVICE)
    y_fit = torch.from_numpy(labels[fit_idx]).long().to(DEVICE)
    baseline = metrics(labels[monitor_idx], normalize(x[monitor_idx].mean(1)))
    best_state, best_score, best_epoch, stale = clone_state(model), -baseline["f1"] + 0.003 * baseline["nll"], 0, 0
    history = []
    for epoch in range(1, RCF_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x_fit)
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), y_fit) + 0.05 * F.mse_loss(output, F.one_hot(y_fit, NC).float()) + rcf_regularization(model)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 3.0)
        optimizer.step()
        scheduler.step()
        monitor = rcf_predict(model, x[monitor_idx])
        monitor_metric = metrics(labels[monitor_idx], monitor)
        score = -monitor_metric["f1"] + 0.003 * monitor_metric["nll"]
        history.append({"epoch": epoch, "loss": float(loss.detach()), "monitor_acc": monitor_metric["acc"], "monitor_f1": monitor_metric["f1"], "monitor_nll": monitor_metric["nll"]})
        if score < best_score - 1e-5:
            best_state, best_score, best_epoch, stale = clone_state(model), score, epoch, 0
        else:
            stale += 1
        if stale >= FUSION_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.cpu(), best_epoch, history


def refit_rcf(model, x, labels, epochs, seed):
    seed_all(seed)
    model = model.to(DEVICE)
    params = model.enabled_parameters()
    if not params:
        return model.cpu()
    optimizer = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=2e-5)
    features = torch.from_numpy(x).float().to(DEVICE)
    target = torch.from_numpy(labels).long().to(DEVICE)
    for _ in range(max(epochs, 1)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), target) + 0.05 * F.mse_loss(output, F.one_hot(target, NC).float()) + rcf_regularization(model)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 3.0)
        optimizer.step()
        scheduler.step()
    return model.cpu()


def fit_rcf_variant(name, config, x, labels, test_folds, seed, root, regime, initial_state=None):
    path = root / "fusion" / f"{slug(regime)}_{slug(name)}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("tag") == config_hash() and saved.get("config") == config:
            matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
            model = ComponentRCF(matrices, config)
            model.load_state_dict(saved["state_dict"])
            return [rcf_predict(model, item) for item in test_folds], saved["info"], saved["state_dict"]
    if name == "Average/Base":
        output = [normalize(item.mean(1)) for item in test_folds]
        return output, {"params": 0, "selected_epoch": 0, "history": [], "source": "fixed_average"}, None
    fit_idx, monitor_idx = train_test_split(np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=seed)
    fit_matrices = [soft_confusion(labels[fit_idx], x[fit_idx, modality]) for modality in range(x.shape[1])]
    selected = ComponentRCF(fit_matrices, config)
    if initial_state is not None:
        selected_state = selected.state_dict()
        for key, value in initial_state.items():
            if config["transport"] in ("identity", "hard", "no_history") and key == "transport_logits":
                continue
            selected_state[key] = value.detach().cpu().clone()
        selected.load_state_dict(selected_state, strict=True)
    selected, epoch, history = train_rcf(selected, x, labels, fit_idx, monitor_idx, seed)
    all_matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
    final = ComponentRCF(all_matrices, config)
    if initial_state is not None:
        final_state = final.state_dict()
        for key, value in initial_state.items():
            if config["transport"] in ("identity", "hard", "no_history") and key == "transport_logits":
                continue
            final_state[key] = value.detach().cpu().clone()
        final.load_state_dict(final_state, strict=True)
    else:
        final.load_state_dict(selected.state_dict(), strict=True)
    final = refit_rcf(final, x, labels, epoch, seed + 1)
    params = sum(parameter.numel() for parameter in final.parameters() if parameter.requires_grad)
    info = {"params": params, "selected_epoch": epoch, "history": history, "source": "full_checkpoint_finetune" if initial_state is not None else "independent_full_training"}
    state = clone_state(final)
    torch.save({"tag": config_hash(), "state_dict": state, "config": config, "info": info}, path)
    return [rcf_predict(final, item) for item in test_folds], info, state


def disagreement_metrics(labels, experts, reference, candidate):
    predictions = experts.argmax(2)
    agreement = np.all(predictions == predictions[:, :1], axis=1)
    reference_prediction = reference.argmax(1)
    candidate_prediction = candidate.argmax(1)
    corrected = (reference_prediction != labels) & (candidate_prediction == labels)
    harmed = (reference_prediction == labels) & (candidate_prediction != labels)
    return {
        "disagreement_rate": float((~agreement).mean()),
        "agreement_accuracy": float(accuracy_score(labels[agreement], candidate_prediction[agreement])) if agreement.any() else 0.0,
        "disagreement_accuracy": float(accuracy_score(labels[~agreement], candidate_prediction[~agreement])) if (~agreement).any() else 0.0,
        "wrong_to_correct": int(corrected.sum()), "correct_to_wrong": int(harmed.sum()), "net_correction": int(corrected.sum() - harmed.sum()),
    }


def per_class_recall_delta(labels, reference, candidate):
    reference_cm = confusion_matrix(labels, reference.argmax(1), labels=np.arange(NC)).astype(np.float64)
    candidate_cm = confusion_matrix(labels, candidate.argmax(1), labels=np.arange(NC)).astype(np.float64)
    reference_recall = np.diag(reference_cm) / np.maximum(reference_cm.sum(1), 1.0)
    candidate_recall = np.diag(candidate_cm) / np.maximum(candidate_cm.sum(1), 1.0)
    return (candidate_recall - reference_recall).tolist()


def latency_ms(model_state, matrices, config, x):
    if model_state is None:
        return 0.0
    model = ComponentRCF(matrices, config)
    model.load_state_dict(model_state)
    device = DEVICE
    model.to(device).eval()
    batch = torch.from_numpy(x[:min(256, len(x))]).float().to(device)
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(20): model(batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
    return float((time.perf_counter() - start) * 1000.0 / (20 * len(batch)))


def bootstrap_delta(labels, a, b, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        index = rng.integers(0, len(labels), len(labels))
        values.append(accuracy_score(labels[index], a[index].argmax(1)) - accuracy_score(labels[index], b[index].argmax(1)))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_data(checkpoint_dir):
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"modelnet40_np{SRC.N_POINTS}_views{SRC.N_VIEWS}_size{SRC.VIEW_SIZE}.pkl"
    if cache.exists():
        with open(cache, "rb") as handle:
            data = pickle.load(handle)
    else:
        train_pcs, train_normals, train_views, train_labels = SRC.preprocess_all(SRC.DATA_ROOT, "train", SRC.N_POINTS, SRC.N_VIEWS, SRC.VIEW_SIZE)
        test_pcs, test_normals, test_views, test_labels = SRC.preprocess_all(SRC.DATA_ROOT, "test", SRC.N_POINTS, SRC.N_VIEWS, SRC.VIEW_SIZE)
        data = {"train_pcs": train_pcs, "train_normals": train_normals, "train_views": train_views, "train_labels": train_labels, "test_pcs": test_pcs, "test_normals": test_normals, "test_views": test_views, "test_labels": test_labels}
        with open(cache, "wb") as handle:
            pickle.dump(data, handle)
    return (data["train_pcs"], data["train_normals"], data["train_views"], data["train_labels"]), (data["test_pcs"], data["test_normals"], data["test_views"], data["test_labels"])


def main():
    seed_all(SEED)
    version, root, log = output_dirs()
    tag = config_hash()
    started = time.time()
    if not SRC.DATA_ROOT.exists():
        raise RuntimeError(f"ModelNet40 data root not found: {SRC.DATA_ROOT}")
    train, test = load_data(root)
    pcs, normals, views, labels = train
    test_pcs, test_normals, test_views, test_labels = test
    categories = sorted(path.name for path in SRC.DATA_ROOT.iterdir() if path.is_dir())
    if len(categories) != NC or int(labels.max()) + 1 != NC:
        raise RuntimeError("ModelNet40 category mapping does not match the 40-class label index")
    specs = {
        "PointNet++": (lambda: SRC.PointNetPP(NC, 512), "pc"),
        "DGCNN": (lambda: SRC.DGCNN(NC, 1024, SRC.DGCNN_K), "pc"),
        "MVCNN": (lambda: SRC.MVCNN(NC, SRC.N_VIEWS, 512), "mv"),
    }
    active_rost = {name: config for name, config in ROST_VARIANTS.items() if not VARIANT_FILTER or name in VARIANT_FILTER}
    active_rcf = {name: config for name, config in RCF_VARIANTS.items() if not RCF_FILTER or name in RCF_FILTER}
    manifest = {
        "version": version, "tag": tag, "seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS,
        "outer_folds": OUTER_FOLDS, "experts": list(specs), "rost_variants": active_rost,
        "rcf_variants": active_rcf, "data": {"train": len(labels), "test": len(test_labels)},
        "protocol": "Each ROST variant and seed starts from one shared semantic initialization per outer fold, then independently trains its ROST controller, confusion objectives and recovery probe. Every ROST expert variant is evaluated only by the same Full RCF architecture. Fusion consumes only expert OOF posterior and train-internal calibration/monitor splits. Official ModelNet40 test labels are read only for final metrics, plots, and paired bootstrap confidence intervals.",
        "rcf_protocol": "RCF component ablations use only Full ROST expert posterior. Incremental variants are independently initialized and trained. Removed/replacement variants start from the paired Full RCF state for the same pipeline/fusion seed, disable or replace the named forward component and optimizer parameters, then independently fine-tune on the same OOF source.",
        "excluded_scope": "CE-only, CE incremental objectives, Average, Product, Logistic Stacking, MLP Stacking and other decision-level methods are intentionally excluded. The shared semantic initializer is not an evaluated regime. This runner performs only ROST and RCF ablations.",
        "fixed_observers": "PointNet++, DGCNN and MVCNN are fixed multimodal observers shared by every ROST variant. They are not observer-removal or backbone ablation variants; removing one would change the experimental question and invalidate joint JSRI/complementarity comparisons.",
    }
    save_json(manifest, log / "manifest.json")
    pipeline = {}
    for variant, config in active_rost.items():
        pipeline[variant] = {}
        for seed in PIPELINE_SEEDS:
            oof = {name: np.zeros((len(labels), NC), dtype=np.float32) for name in specs}
            test_by_fold = {name: [] for name in specs}
            profiles = []
            fold_snapshots = []
            trajectories = []
            for fold in range(1, OUTER_FOLDS + 1):
                artifact = expert_fold(variant, config, seed, fold, specs, train[:3], test[:3], labels, root, tag)
                for name in specs:
                    oof[name][artifact["holdout_idx"]] = artifact["holdout"][name]
                    test_by_fold[name].append(artifact["test"][name])
                profiles.append(artifact["final_profile"])
                fold_snapshots.append(artifact["profile_snapshots"])
                for row in artifact["rost_history"]:
                    trajectories.append({"variant": variant, "pipeline_seed": seed, "fold": fold, "epoch": row["epoch"], "loss": row["loss"], "score": row["score"], "jsri": row["profile"]["jsri"], "probe_f1": row["probe"]["f1"], **{f"lambda_{name}": value for name, value in row["weights"].items()}})
            pipeline[variant][seed] = {"oof": oof, "test": {name: np.stack(items, 0) for name, items in test_by_fold.items()}, "profiles": profiles, "snapshots": fold_snapshots}
            trajectory_path = log / "controller_trajectory.csv"
            write_header = not trajectory_path.exists()
            if trajectories:
                pd.DataFrame(trajectories).to_csv(trajectory_path, mode="a", index=False, header=write_header)
            print(f"Completed expert pipeline: {variant} seed={seed}")
    summary_rows, rcf_rows, prediction_cache = [], [], {}
    profile_snapshots = {}
    for variant, regimes in pipeline.items():
        for seed, data in regimes.items():
            x = np.stack([data["oof"][name] for name in specs], 1).astype(np.float32)
            test_folds = [np.stack([data["test"][name][fold] for name in specs], 1).astype(np.float32) for fold in range(OUTER_FOLDS)]
            matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
            full_states = {}
            full_infos = {}
            for fusion_seed in FUSION_SEEDS:
                regime_key = f"{variant}_ps{seed}"
                outputs, info, state = fit_rcf_variant("Full RCF", RCF_VARIANTS["Full RCF"], x, labels, test_folds, fusion_seed, root, regime_key)
                output = normalize(np.mean(outputs, 0))
                full_states[fusion_seed], full_infos[fusion_seed] = state, info
                row = {"variant": variant, "pipeline_seed": seed, "fusion_seed": fusion_seed, "fusion": "Full RCF", **metrics(test_labels, output), **mechanism_subset_metrics(test_labels, test_folds, np.mean(test_folds, 0), output, labels, x), "jsri": profile(matrices)["jsri"], "parameters": info["params"], "profile": profile(matrices), "selected_epoch": info["selected_epoch"]}
                summary_rows.append(row)
                prediction_cache[f"ROST|{variant}|ps{seed}|fs{fusion_seed}"] = output.astype(np.float32)
                plot_cm(test_labels, output, log / f"cm_rost_{slug(variant)}_ps{seed}_fs{fusion_seed}.png", f"ROST {variant} | Full RCF", categories)
            profile_snapshots[f"{slug(variant)}_s{seed}"] = {
                "oof": [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])],
                "fold_profiles": data["profiles"],
                "fold_snapshots": data["snapshots"],
            }
            if variant == "Full ROST":
                for name, config in active_rcf.items():
                    for fusion_seed in FUSION_SEEDS:
                        regime_key = f"{variant}_ps{seed}"
                        full_outputs, full_info, full_state = fit_rcf_variant("Full RCF", RCF_VARIANTS["Full RCF"], x, labels, test_folds, fusion_seed, root, regime_key)
                        reference = normalize(np.mean(full_outputs, 0))
                        incremental = name.startswith("+") or name == "Average/Base"
                        initial_state = None if incremental or name == "Full RCF" else full_state
                        if name == "Full RCF":
                            outputs, info, state = full_outputs, full_info, full_state
                        else:
                            outputs, info, state = fit_rcf_variant(name, config, x, labels, test_folds, fusion_seed, root, regime_key, initial_state)
                        output = normalize(np.mean(outputs, 0))
                        latency = latency_ms(state, matrices, config, x) if state is not None else 0.0
                        initialization = "independent_incremental_training" if incremental else "paired_full_state_finetune"
                        if name == "Full RCF": initialization = "independent_full_training"
                        row = {"expert_regime": variant, "pipeline_seed": seed, "fusion_seed": fusion_seed, "variant": name, **metrics(test_labels, output), **disagreement_metrics(test_labels, np.mean(test_folds, 0), reference, output), **mechanism_subset_metrics(test_labels, test_folds, reference, output, labels, x), "per_class_recall_delta_vs_full": per_class_recall_delta(test_labels, reference, output), "parameters": info["params"], "latency_ms_per_sample": latency, "selected_epoch": info["selected_epoch"], "source": info["source"], "initialization_protocol": initialization}
                        rcf_rows.append(row)
                        prediction_cache[f"RCF|{name}|ps{seed}|fs{fusion_seed}"] = output.astype(np.float32)
                        plot_cm(test_labels, output, log / f"cm_rcf_{slug(variant)}_ps{seed}_fs{fusion_seed}_{slug(name)}.png", f"{variant} | RCF component {name}", categories)
                        if name.startswith("w/o ") and full_state is not None:
                            frozen_model = ComponentRCF(matrices, config)
                            frozen_state = frozen_model.state_dict()
                            for key, value in full_state.items():
                                if config["transport"] in ("identity", "hard", "no_history") and key == "transport_logits":
                                    continue
                                frozen_state[key] = value.detach().cpu().clone()
                            frozen_model.load_state_dict(frozen_state, strict=True)
                            frozen_model.enabled_parameters()
                            frozen_outputs = [rcf_predict(frozen_model, item) for item in test_folds]
                            frozen_output = normalize(np.mean(frozen_outputs, 0))
                            frozen_row = {"expert_regime": variant, "pipeline_seed": seed, "fusion_seed": fusion_seed, "variant": f"Frozen | {name}", **metrics(test_labels, frozen_output), **disagreement_metrics(test_labels, np.mean(test_folds, 0), reference, frozen_output), **mechanism_subset_metrics(test_labels, test_folds, reference, frozen_output, labels, x), "per_class_recall_delta_vs_full": per_class_recall_delta(test_labels, reference, frozen_output), "parameters": int(sum(parameter.numel() for parameter in frozen_model.parameters() if parameter.requires_grad)), "latency_ms_per_sample": latency_ms(frozen_state, matrices, config, x), "selected_epoch": 0, "source": "paired_full_state_frozen_forward", "initialization_protocol": "paired_full_state_frozen_forward"}
                            rcf_rows.append(frozen_row)
                            prediction_cache[f"RCF|Frozen {name}|ps{seed}|fs{fusion_seed}"] = frozen_output.astype(np.float32)
                            plot_cm(test_labels, frozen_output, log / f"cm_rcf_frozen_{slug(variant)}_ps{seed}_fs{fusion_seed}_{slug(name)}.png", f"{variant} | Frozen RCF component {name}", categories)
                        if state is not None and name == "Full RCF":
                            model = ComponentRCF(matrices, config)
                            model.load_state_dict(state)
                            plot_matrix(F.softmax(model.reliability_logits, 0).detach().cpu().numpy(), log / f"reliability_{slug(variant)}_s{seed}_fs{fusion_seed}.png", f"{variant} Full RCF reliability", categories, list(specs))
                            transport = F.softmax(model.transport_logits, 1).detach().cpu().numpy()
                            for modality, expert_name in enumerate(specs):
                                plot_matrix(transport[modality], log / f"transport_{slug(variant)}_{slug(expert_name)}_s{seed}_fs{fusion_seed}.png", f"{variant} transport {expert_name} (true x predicted)", categories)
    pd.DataFrame(summary_rows).to_csv(log / "rost_ablation.csv", index=False)
    pd.DataFrame(rcf_rows).to_csv(log / "rcf_component_ablation.csv", index=False)
    numeric = ["acc", "f1", "precision", "recall", "ece", "adaptive_ece", "classwise_ece", "brier", "nll", "disagreement_acc", "low_confidence_acc", "recoverable_error_acc", "hard_class_acc", "wrong_to_correct", "correct_to_wrong", "net_correction", "parameters", "latency_ms_per_sample"]
    summary_frame = pd.DataFrame(summary_rows)
    rcf_frame = pd.DataFrame(rcf_rows)
    aggregate = summary_frame.groupby(["variant"])[[item for item in numeric if item in summary_frame.columns]].agg(["mean", "std"]).reset_index()
    if not rcf_frame.empty:
        rcf_aggregate = rcf_frame.groupby(["expert_regime", "variant"])[[item for item in numeric if item in rcf_frame.columns]].agg(["mean", "std"]).reset_index()
        rcf_aggregate.columns = [" | ".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in rcf_aggregate.columns]
        rcf_aggregate.insert(0, "table", "RCF component ablation")
        aggregate.columns = [" | ".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in aggregate.columns]
        aggregate.insert(0, "table", "ROST group ablation with Full RCF")
        aggregate = pd.concat([aggregate, rcf_aggregate], ignore_index=True, sort=False)
    aggregate.to_csv(log / "aggregate_mean_std.csv", index=False)
    bootstrap = []
    for seed in PIPELINE_SEEDS:
        for fusion_seed in FUSION_SEEDS:
            rost_key = f"ROST|Full ROST|ps{seed}|fs{fusion_seed}"
            if rost_key in prediction_cache:
                for variant in active_rost:
                    if variant == "Full ROST":
                        continue
                    ablation_key = f"ROST|{variant}|ps{seed}|fs{fusion_seed}"
                    if ablation_key in prediction_cache:
                        bootstrap.append({"pipeline_seed": seed, "fusion_seed": fusion_seed, "comparison": f"Full ROST - {variant} under Full RCF", "accuracy_delta_95ci": bootstrap_delta(test_labels, prediction_cache[rost_key], prediction_cache[ablation_key], seed + fusion_seed)})
    np.savez_compressed(root / "final_seed_predictions.npz", test_labels=test_labels, **prediction_cache)
    save_json(profile_snapshots, log / "profile_snapshots.json")
    save_json({"manifest": manifest, "rost_rows": summary_rows, "rcf_rows": rcf_rows, "paired_bootstrap": bootstrap}, log / "results.json")
    rost_rank = summary_frame.groupby("variant")[["acc", "f1", "ece", "brier", "nll", "jsri", "disagreement_acc", "low_confidence_acc", "recoverable_error_acc", "hard_class_acc"]].mean().sort_values(["acc", "f1"], ascending=False).reset_index()  # pyright: ignore[reportCallIssue]
    rcf_rank = rcf_frame.groupby("variant")[["acc", "f1", "ece", "brier", "nll", "disagreement_acc", "low_confidence_acc", "recoverable_error_acc", "hard_class_acc", "parameters"]].mean().sort_values(["acc", "f1"], ascending=False).reset_index() if not rcf_frame.empty else pd.DataFrame()  # pyright: ignore[reportCallIssue]
    print("ROST ablation under Full RCF")
    print("rank | variant | acc | f1 | nll | JSRI | disagree_acc | low_conf_acc | recoverable_acc | hard_class_acc")
    for rank, (_, row) in enumerate(rost_rank.iterrows(), start=1):
        print(f"{rank:>4} | {row['variant']:<32.32} | {row['acc']:.4f} | {row['f1']:.4f} | {row['nll']:.4f} | {row['jsri']:.4f} | {row['disagreement_acc']:.4f} | {row['low_confidence_acc']:.4f} | {row['recoverable_error_acc']:.4f} | {row['hard_class_acc']:.4f}")
    print("RCF component ablation with Full ROST experts")
    print("rank | variant | acc | f1 | nll | disagree_acc | low_conf_acc | recoverable_acc | hard_class_acc | params")
    for rank, (_, row) in enumerate(rcf_rank.iterrows(), start=1):
        print(f"{rank:>4} | {row['variant']:<32.32} | {row['acc']:.4f} | {row['f1']:.4f} | {row['nll']:.4f} | {row['disagreement_acc']:.4f} | {row['low_confidence_acc']:.4f} | {row['recoverable_error_acc']:.4f} | {row['hard_class_acc']:.4f} | {int(row['parameters'])}")
    print("ModelNet40 DOME-X ROST/RCF ablation complete")
    print(f"Checkpoints={root}")
    print(f"Logs={log}")
    print(f"TimeMinutes={(time.time() - started) / 60.0:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate dependencies and cached inputs")
    return parser.parse_args()


def check_environment():
    required_api = ("MVDataset", "PCDataset", "PointNetPP", "DGCNN", "MVCNN")
    missing = [name for name in required_api if not hasattr(SRC, name)]
    if missing:
        raise RuntimeError(f"ModelNet40 source is missing definitions: {missing}")
    cache_path = PROJECT_ROOT / "cache" / f"modelnet40_np{SRC.N_POINTS}_views{SRC.N_VIEWS}_size{SRC.VIEW_SIZE}.pkl"
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Missing ModelNet40 source: {SOURCE_PATH}")
    if not SRC.DATA_ROOT.is_dir() and not cache_path.is_file():
        raise FileNotFoundError("ModelNet40 data and preprocessing cache are both missing")
    print(
        f"ModelNet40 ablation check passed: source={SOURCE_PATH.name} "
        f"data={SRC.DATA_ROOT.is_dir()} cache={cache_path.is_file()}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        check_environment()
    else:
        main()
