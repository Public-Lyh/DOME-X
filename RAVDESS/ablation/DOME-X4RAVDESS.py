import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PLACEHOLDER_ROOT = Path("your path")
PROJECT_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
ROOT = PROJECT_ROOT / "Code" / "RAVDESS"
SOURCE_SCRIPT = ROOT / "scripts" / "DOME-X4RAVDESS.py"
SOURCE_DIR = PROJECT_ROOT / "data" / "RAVDESS" / "checkpoints" / "domex_x_ravdess_ikun_final_compare"
SOURCE_PROBS = SOURCE_DIR / "probs.pkl"
SOURCE_MODEL = SOURCE_DIR / "domex_x_ikun_final.pth"
CHECKPOINT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "logs"
SEED = int(os.environ.get("DOME_X_SEED", "42"))
EPOCHS = int(os.environ.get("DOME_X_RCF_EPOCHS", "120"))
PATIENCE = int(os.environ.get("DOME_X_FUSION_PATIENCE", "24"))
FULL_RETRAIN = os.environ.get("DOME_X_FULL_RETRAIN", "1") == "1"
FULL_RETRAIN_EPOCHS = int(os.environ.get("DOME_X_FULL_RETRAIN_EPOCHS", "40"))
FULL_RETRAIN_LR = float(os.environ.get("DOME_X_FULL_RETRAIN_LR", "2e-5"))
FUSION_SEEDS = [SEED + 1000 + i for i in range(int(os.environ.get("DOME_X_FUSION_SEEDS", "5")))]
VERSION = os.environ.get("DOME_X_ABLATION_VERSION", "v21_rcf")
FILTER = [item.strip() for item in os.environ.get("DOME_X_RCF_VARIANTS", "").split(",") if item.strip()]
EPS = 1e-8
NC = 8
CLASSES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))

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

ROST_ADAPTER_VARIANTS = {
    "CE-only Residual": {"structure": False, "collapse": False, "complement": False, "joint": False},
    "+ Structure": {"structure": True, "collapse": False, "complement": False, "joint": False},
    "+ Anti-collapse": {"structure": True, "collapse": True, "complement": False, "joint": False},
    "+ Complementarity": {"structure": True, "collapse": True, "complement": True, "joint": False},
    "Full ROST Adapter": {"structure": True, "collapse": True, "complement": True, "joint": True},
    "w/o Structure": {"structure": False, "collapse": True, "complement": True, "joint": True},
    "w/o Anti-collapse": {"structure": True, "collapse": False, "complement": True, "joint": True},
    "w/o Complementarity": {"structure": True, "collapse": True, "complement": False, "joint": True},
    "w/o Joint Recovery": {"structure": True, "collapse": True, "complement": True, "joint": False},
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


def normalize(value):
    value = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    value = np.clip(value, EPS, None)
    return value / np.maximum(value.sum(axis=1, keepdims=True), EPS)


def normalize_torch(value):
    return value / value.sum(dim=-1, keepdim=True).clamp_min(EPS)


def apply_temperature(probs, temperature):
    probs = normalize(probs)
    return normalize(np.exp(np.log(probs) / float(temperature)))


def safe_name(name):
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
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


def metrics(labels, probs):
    probs = normalize(probs)
    prediction = probs.argmax(1)
    matrix = np.zeros((NC, NC), dtype=np.int64)
    np.add.at(matrix, (labels, prediction), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    true_count = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(NC, dtype=np.float64), where=predicted_count > 0)
    recall = np.divide(true_positive, true_count, out=np.zeros(NC, dtype=np.float64), where=true_count > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(NC, dtype=np.float64), where=(precision + recall) > 0)
    onehot = np.eye(NC, dtype=np.float32)[labels]
    confidence = probs.max(1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs((prediction[mask] == labels[mask]).mean() - confidence[mask].mean()))
    return {
        "acc": float((prediction == labels).mean()),
        "f1": float(f1.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "ece": ece,
        "brier": float(np.square(probs - onehot).sum(1).mean()),
        "nll": float(-np.log(probs[np.arange(len(labels)), labels].clip(EPS, 1.0)).mean()),
    }


def js_divergence(first, second):
    first = np.clip(np.asarray(first, dtype=np.float64), EPS, None)
    second = np.clip(np.asarray(second, dtype=np.float64), EPS, None)
    first /= first.sum()
    second /= second.sum()
    middle = 0.5 * (first + second)
    return float(0.5 * np.sum(first * np.log(first / middle)) + 0.5 * np.sum(second * np.log(second / middle)))


def recoverability_profile(labels, probs):
    probs = normalize(probs)
    matrix = np.zeros((NC, NC), dtype=np.float64)
    np.add.at(matrix, labels, probs)
    counts = np.bincount(labels, minlength=NC).astype(np.float64)
    matrix /= np.maximum(counts[:, None], 1.0)
    matrix[counts == 0] = 1.0 / NC
    matrix /= np.maximum(matrix.sum(1, keepdims=True), EPS)
    row_entropy = -(matrix * np.log(matrix + EPS)).sum(1) / np.log(NC)
    usage = matrix.mean(0)
    column_entropy = float(-(usage * np.log(usage + EPS)).sum() / np.log(NC))
    separation = float(np.mean([js_divergence(matrix[i], matrix[j]) / np.log(2) for i in range(NC) for j in range(i + 1, NC)]))
    singular = np.linalg.svd(matrix, compute_uv=False)
    singular /= max(singular.sum(), EPS)
    effective_rank = float(np.exp(-np.sum(singular * np.log(singular + EPS))) / NC)
    prior = counts / max(counts.sum(), 1.0)
    reverse = matrix.T * prior[None]
    reverse /= np.maximum(reverse.sum(1, keepdims=True), EPS)
    decoded = matrix @ reverse
    decodability = float(np.diag(decoded).mean())
    shape = float(np.clip(1.0 - np.maximum(0.20 - row_entropy, 0).mean() - np.maximum(row_entropy - 0.92, 0).mean(), 0.0, 1.0))
    sri = float(0.18 * shape + 0.24 * separation + 0.18 * column_entropy + 0.18 * effective_rank + 0.22 * decodability)
    return {"SRI": sri, "row_entropy": float(row_entropy.mean()), "column_entropy": column_entropy, "row_separation": separation, "effective_rank": effective_rank, "decodability": decodability}


def format_params(value):
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def load_source_module():
    spec = importlib.util.spec_from_file_location("ravdess_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the RAVDESS source: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SRC = load_source_module()


class RAVDESSRCF(SRC.DOME_X_IKUN):
    def __init__(self, mars_init, pair_bias, config):
        super().__init__(mars_init, pair_bias)
        self.config = dict(config)

    def build_mda_features(self, pa, pv, ca, cv, fused, pair, calibrated_logits):
        audio_top = ca.argmax(dim=-1)
        visual_top = cv.argmax(dim=-1)
        disagree = audio_top != visual_top
        uncertainty = 1.0 - torch.maximum(SRC.margin_torch(ca), SRC.margin_torch(cv))
        feature = torch.cat([
            pa, pv, ca, cv, fused, pair,
            torch.abs(pa - pv), torch.abs(ca - cv),
            F.one_hot(audio_top, NC).float(), F.one_hot(visual_top, NC).float(),
            SRC.entropy_torch(pa), SRC.entropy_torch(pv), SRC.entropy_torch(ca), SRC.entropy_torch(cv),
            SRC.margin_torch(pa), SRC.margin_torch(pv), SRC.margin_torch(ca), SRC.margin_torch(cv),
            disagree.float().unsqueeze(1), uncertainty,
            torch.abs(SRC.margin_torch(ca) - SRC.margin_torch(cv)),
            calibrated_logits.flatten(1),
        ], 1)
        return feature, disagree, uncertainty

    def forward(self, pa, pv):
        pa = normalize_torch(pa.clamp_min(EPS))
        pv = normalize_torch(pv.clamp_min(EPS))
        mars_feature = self.build_pre_mars_features(pa, pv)
        if self.config["reliability"]:
            reliability = self.mars(mars_feature)
        else:
            reliability = torch.full((len(pa), 2, NC), 0.5, dtype=pa.dtype, device=pa.device)
        if self.config["calibration"]:
            ca, cv, peace_fused, calibrated_logits = self.peace(pa, pv, reliability)
        else:
            ca, cv = pa, pv
            calibrated_logits = torch.stack([torch.log(pa), torch.log(pv)], 1)
            peace_fused = normalize_torch((reliability * torch.stack([pa, pv], 1)).sum(1))
        if self.config["transport"]:
            pair = self.pair_memory(pa, pv)
        else:
            pair = peace_fused
        feature, disagree, uncertainty = self.build_mda_features(pa, pv, ca, cv, peace_fused, pair, calibrated_logits)
        if self.config["refinement"]:
            final, mda_gate, pair_gate, residual = self.mda(feature, peace_fused, pair, disagree, uncertainty)
        else:
            final = normalize_torch(0.75 * peace_fused + 0.25 * pair) if self.config["transport"] else peace_fused
            mda_gate = torch.zeros((len(pa), 1), dtype=pa.dtype, device=pa.device)
            pair_gate = torch.zeros((len(pa), 1), dtype=pa.dtype, device=pa.device)
            residual = torch.zeros_like(final)
        return {
            "final_probs": normalize_torch(final),
            "peace_fused": normalize_torch(peace_fused),
            "audio_calibrated": ca,
            "visual_calibrated": cv,
            "pair": normalize_torch(pair),
            "mars_weight_sample": reliability,
            "mars_weight_base": self.mars.base_weight(),
            "mda_gate": mda_gate,
            "pair_gate": pair_gate,
            "residual": residual,
            "disagree": disagree.float(),
            "uncertainty": uncertainty,
            "calibrated_logits": calibrated_logits,
        }


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initialize_lazy_layers(model):
    with torch.no_grad():
        model.mars(torch.zeros((1, 46), dtype=torch.float32))
        model.mda(
            torch.zeros((1, 107), dtype=torch.float32),
            torch.full((1, NC), 1.0 / NC, dtype=torch.float32),
            torch.full((1, NC), 1.0 / NC, dtype=torch.float32),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.float32),
        )


def collect(model, audio, visual):
    model = model.to(DEVICE).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(audio), 512):
            out = model(torch.from_numpy(audio[start:start + 512]).float().to(DEVICE), torch.from_numpy(visual[start:start + 512]).float().to(DEVICE))
            outputs.append(out["final_probs"].float().cpu().numpy())
    return normalize(np.concatenate(outputs))


def component_parameters(model, config):
    enabled = {
        "mars": config["reliability"],
        "peace": config["calibration"],
        "pair_memory": config["transport"],
        "mda": config["refinement"],
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name == prefix or name.startswith(prefix + ".") for prefix, active in enabled.items() if active)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def artifact_signature():
    payload = {
        "prob_path": str(SOURCE_PROBS),
        "prob_mtime": SOURCE_PROBS.stat().st_mtime_ns,
        "model_path": str(SOURCE_MODEL),
        "model_mtime": SOURCE_MODEL.stat().st_mtime_ns,
        "script_mtime": Path(__file__).stat().st_mtime_ns,
        "epochs": EPOCHS,
        "seeds": FUSION_SEEDS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cache_valid(path, variant, seed, signature):
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        return item.get("variant") == variant and item.get("seed") == seed and item.get("signature") == signature
    except Exception:
        return False


def train_variant(variant, config, train_audio, train_visual, train_labels, val_audio, val_visual, val_labels, test_audio, test_visual, initial_state, mars_init, pair_bias, seed, root, signature, output_temperature):
    if variant == "Average/Base":
        return apply_temperature(0.5 * test_audio + 0.5 * test_visual, output_temperature), {"params": 0, "epoch": 0, "source": "deterministic"}
    path = root / "fusion" / f"{safe_name(variant)}_seed{seed}.pt"
    model = RAVDESSRCF(mars_init, pair_bias, config)
    if cache_valid(path, variant, seed, signature):
        item = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(item["state_dict"], strict=True)
        return apply_temperature(collect(model, test_audio, test_visual), output_temperature), {"params": item["params"], "epoch": item["epoch"], "source": "cache"}
    seed_all(seed)
    initialize_lazy_layers(model)
    if variant.startswith("w/o "):
        model.load_state_dict(initial_state, strict=True)
    parameters = component_parameters(model, config)
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {variant}")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(parameters, lr=6e-4, weight_decay=6e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 2e-5)
    train_a = torch.from_numpy(train_audio).float().to(DEVICE)
    train_v = torch.from_numpy(train_visual).float().to(DEVICE)
    train_y = torch.from_numpy(train_labels).long().to(DEVICE)
    best = {"score": float("inf"), "state": model_state(model), "epoch": 0}
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train_a, train_v)
        final = output["final_probs"]
        loss = F.nll_loss(torch.log(final.clamp_min(EPS)), train_y)
        if config["calibration"]:
            loss = loss + 0.15 * F.nll_loss(torch.log(output["peace_fused"].clamp_min(EPS)), train_y)
        if config["transport"]:
            loss = loss + 0.04 * F.nll_loss(torch.log(output["pair"].clamp_min(EPS)), train_y)
        loss = loss + model.regularization()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        validation = metrics(val_labels, collect(model, val_audio, val_visual))
        score = -validation["f1"] + 0.003 * validation["nll"]
        if score < best["score"] - 1e-7:
            best = {"score": score, "state": model_state(model), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best["state"], strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = parameter_count(model)
    torch.save({"variant": variant, "seed": seed, "signature": signature, "epoch": best["epoch"], "params": count, "state_dict": model.cpu().state_dict()}, path)
    return apply_temperature(collect(model, test_audio, test_visual), output_temperature), {"params": count, "epoch": best["epoch"], "source": "trained"}


def paired_component_output(config, state, mars_init, pair_bias, test_audio, test_visual, output_temperature):
    model = RAVDESSRCF(mars_init, pair_bias, config)
    initialize_lazy_layers(model)
    model.load_state_dict(state, strict=True)
    return apply_temperature(collect(model, test_audio, test_visual), output_temperature)


def refit_full_state(model, state, train_audio, train_visual, train_labels, val_audio, val_visual, val_labels):
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=FULL_RETRAIN_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FULL_RETRAIN_EPOCHS, FULL_RETRAIN_LR * 0.1)
    audio = torch.from_numpy(train_audio).float().to(DEVICE)
    visual = torch.from_numpy(train_visual).float().to(DEVICE)
    labels = torch.from_numpy(train_labels).long().to(DEVICE)
    best_state = model_state(model)
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, FULL_RETRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(audio, visual)
        loss = F.nll_loss(torch.log(output["final_probs"].clamp_min(EPS)), labels)
        loss = loss + 0.10 * F.nll_loss(torch.log(output["peace_fused"].clamp_min(EPS)), labels)
        loss = loss + 0.03 * F.nll_loss(torch.log(output["pair"].clamp_min(EPS)), labels)
        loss = loss + 0.10 * model.regularization()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        validation = metrics(val_labels, collect(model, val_audio, val_visual))
        score = 0.55 * validation["acc"] + 0.45 * validation["f1"] - 0.01 * validation["nll"]
        if score > best_score:
            best_score = score
            best_state = model_state(model)
            best_epoch = epoch
    model.load_state_dict(best_state, strict=True)
    return best_state, {"epoch": best_epoch, "score": best_score}


def plot_confusion(labels, probs, path, title):
    matrix = np.zeros((NC, NC), dtype=np.float32)
    np.add.at(matrix, (labels, normalize(probs).argmax(1)), 1.0)
    matrix /= np.maximum(matrix.sum(1, keepdims=True), 1.0)
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0, xticklabels=CLASSES, yticklabels=CLASSES, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def bootstrap_delta(labels, first, second, seed):
    rng = np.random.default_rng(seed)
    first = normalize(first).argmax(1)
    second = normalize(second).argmax(1)
    samples = []
    for _ in range(2000):
        index = rng.integers(0, len(labels), len(labels))
        samples.append(float((first[index] == labels[index]).mean() - (second[index] == labels[index]).mean()))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def adapter_features(pa, pv, base):
    pa = normalize(pa)
    pv = normalize(pv)
    base = normalize(base)
    avg = normalize(0.5 * (pa + pv))
    product = normalize(np.sqrt(np.clip(pa, EPS, 1.0) * np.clip(pv, EPS, 1.0)))
    entropy_a = -(pa * np.log(pa + EPS)).sum(1, keepdims=True) / np.log(NC)
    entropy_v = -(pv * np.log(pv + EPS)).sum(1, keepdims=True) / np.log(NC)
    margin_a = np.sort(pa, axis=1)[:, -1:] - np.sort(pa, axis=1)[:, -2:-1]
    margin_v = np.sort(pv, axis=1)[:, -1:] - np.sort(pv, axis=1)[:, -2:-1]
    disagreement = (pa.argmax(1) != pv.argmax(1)).astype(np.float32).reshape(-1, 1)
    return np.concatenate([pa, pv, base, avg, product, entropy_a, entropy_v, margin_a, margin_v, disagreement], 1).astype(np.float32)


class ROSTAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_strength = nn.Parameter(torch.tensor(-1.3041228))
        output_layer = nn.Linear(48, NC)
        self.net = nn.Sequential(
            nn.Linear(45, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.04),
            output_layer,
        )
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, features, base):
        strength = 0.20 * torch.sigmoid(self.log_strength)
        return F.softmax(torch.log(base.clamp_min(EPS)) + strength * torch.tanh(self.net(features)), 1)


def adapter_confusion(probs, labels):
    onehot = F.one_hot(labels, NC).float()
    counts = onehot.sum(0).clamp_min(1.0).unsqueeze(1)
    return onehot.T @ probs / counts


def adapter_structure_loss(probs, labels):
    confusion = adapter_confusion(probs, labels).clamp_min(EPS)
    entropy = -(confusion * confusion.log()).sum(1) / np.log(NC)
    js = []
    for i in range(NC):
        for j in range(i + 1, NC):
            m = 0.5 * (confusion[i] + confusion[j])
            js.append(0.5 * (confusion[i] * (confusion[i].log() - m.log())).sum() + 0.5 * (confusion[j] * (confusion[j].log() - m.log())).sum())
    separation = torch.stack(js).mean()
    return F.relu(0.20 - entropy).square().mean() + F.relu(entropy - 0.92).square().mean() - 0.12 * separation


def adapter_collapse_loss(probs, labels):
    confusion = adapter_confusion(probs, labels).clamp_min(EPS)
    usage = confusion.mean(0)
    entropy = -(usage * usage.log()).sum() / np.log(NC)
    return F.relu(0.72 - entropy).square() + F.relu(usage.max() - 0.30).square()


def adapter_complement_loss(probs, other, labels):
    current = adapter_confusion(probs, labels).clamp_min(EPS)
    peer = adapter_confusion(other.detach(), labels).clamp_min(EPS)
    distances = []
    rescue = []
    for i in range(NC):
        for j in range(i + 1, NC):
            m = 0.5 * (peer[i] + peer[j])
            peer_js = 0.5 * (peer[i] * (peer[i].log() - m.log())).sum() + 0.5 * (peer[j] * (peer[j].log() - m.log())).sum()
            cm = 0.5 * (current[i] + current[j])
            current_js = 0.5 * (current[i] * (current[i].log() - cm.log())).sum() + 0.5 * (current[j] * (current[j].log() - cm.log())).sum()
            distances.append(peer_js.detach())
            rescue.append(current_js)
    weight = torch.softmax(-torch.stack(distances) / 0.12, 0)
    return -(weight * torch.stack(rescue)).sum()


def adapter_joint_loss(probs, base, labels):
    reverse = adapter_confusion(probs, labels).clamp_min(EPS).T
    reverse = reverse / reverse.sum(1, keepdim=True).clamp_min(EPS)
    decoded = adapter_confusion(probs, labels) @ reverse
    return -torch.diag(decoded).clamp_min(EPS).log().mean() + 0.10 * F.kl_div(torch.log(probs.clamp_min(EPS)), base.detach(), reduction="batchmean")


def train_adapter_variant(variant, config, train_features, train_base, train_labels, val_features, val_base, val_labels, test_features, test_base, seed, root, signature):
    path = root / "rost_adapter" / f"{safe_name(variant)}_seed{seed}.pt"
    model = ROSTAdapter()
    if cache_valid(path, variant, seed, signature + "_rost"):
        item = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(item["state_dict"], strict=True)
        return normalize(model(torch.from_numpy(test_features).float(), torch.from_numpy(test_base).float()).detach().numpy()), {"params": item["params"], "epoch": item["epoch"], "source": "cache"}
    seed_all(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 2e-5)
    xa = torch.from_numpy(train_features).float().to(DEVICE)
    xb = torch.from_numpy(train_base).float().to(DEVICE)
    ya = torch.from_numpy(train_labels).long().to(DEVICE)
    best = {"score": float("inf"), "state": model_state(model), "epoch": 0}
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(xa, xb)
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), ya)
        if config["structure"]:
            loss = loss + 0.08 * adapter_structure_loss(output, ya)
        if config["collapse"]:
            loss = loss + 0.06 * adapter_collapse_loss(output, ya)
        if config["complement"]:
            peer = xb
            loss = loss + 0.10 * adapter_complement_loss(output, peer, ya)
        if config["joint"]:
            loss = loss + 0.12 * adapter_joint_loss(output, xb, ya)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_output = model(torch.from_numpy(val_features).float().to(DEVICE), torch.from_numpy(val_base).float().to(DEVICE)).cpu().numpy()
        validation = metrics(val_labels, val_output)
        score = -validation["f1"] + 0.003 * validation["nll"]
        if score < best["score"] - 1e-7:
            best = {"score": score, "state": model_state(model), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best["state"], strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    torch.save({"variant": variant, "seed": seed, "signature": signature + "_rost", "epoch": best["epoch"], "params": count, "state_dict": model.cpu().state_dict()}, path)
    output = model(torch.from_numpy(test_features).float(), torch.from_numpy(test_base).float()).detach().numpy()
    return normalize(output), {"params": count, "epoch": best["epoch"], "source": "trained"}


def print_table(frame):
    print("rank | variant                      | acc    | f1     | precision | recall | ece    | brier  | nll    | params")
    for rank, (_, row) in enumerate(frame.iterrows(), 1):
        print(f"{rank:>4} | {row['variant'][:28]:<28} | {row['acc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} | {format_params(row['params']):>6}")


def main():
    if not SOURCE_PROBS.exists() or not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Missing v21 RAVDESS artifacts under {SOURCE_DIR}")
    start = time.time()
    seed_all(SEED)
    root = CHECKPOINT_ROOT / f"ablation_{VERSION}" / "DOME_X_RAVDESS_ABLATION"
    log = LOG_ROOT / f"ablation_{VERSION}" / "DOME_X_RAVDESS_ABLATION"
    root.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_PROBS, "rb") as handle:
        pack = pickle.load(handle)
    train_audio = normalize(pack["pa_train"])
    train_visual = normalize(pack["pv_train"])
    train_labels = np.asarray(pack["y_train"], dtype=np.int64)
    val_audio = normalize(pack["pa_val"])
    val_visual = normalize(pack["pv_val"])
    val_labels = np.asarray(pack["y_val"], dtype=np.int64)
    test_audio = normalize(pack["pa_test"])
    test_visual = normalize(pack["pv_test"])
    test_labels = np.asarray(pack["y_test"], dtype=np.int64)
    mars_init = SRC.build_mars_init(train_audio, train_visual, train_labels, blend=0.18, floor=0.32)
    pair_bias = SRC.build_pair_bias(train_audio, train_visual, train_labels, smooth=0.40)
    checkpoint = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=False)
    reference = RAVDESSRCF(mars_init, pair_bias, RCF_VARIANTS["Full RCF"])
    reference.load_state_dict(checkpoint["model"], strict=True)
    output_temperature = float(pack.get("domex_calibration", {}).get("temp", 0.60))
    reference_test = apply_temperature(collect(reference, test_audio, test_visual), output_temperature)
    expected = normalize(pack["domex_test"])
    reproduction_l1 = float(np.abs(reference_test - expected).mean())
    if reproduction_l1 > 1e-4:
        raise RuntimeError(f"v21 Full RCF reproduction failed: mean L1={reproduction_l1:.8f}")
    full_state = checkpoint["model"]
    full_selection = {"source": "verified_reference", "epoch": checkpoint.get("epoch", "reference")}
    reference_val = metrics(val_labels, apply_temperature(collect(reference, val_audio, val_visual), output_temperature))
    reference_score = 0.55 * reference_val["acc"] + 0.45 * reference_val["f1"] - 0.01 * reference_val["nll"]
    if FULL_RETRAIN:
        candidate_model = RAVDESSRCF(mars_init, pair_bias, RCF_VARIANTS["Full RCF"])
        initialize_lazy_layers(candidate_model)
        candidate_state, candidate_info = refit_full_state(candidate_model, checkpoint["model"], train_audio, train_visual, train_labels, val_audio, val_visual, val_labels)
        if candidate_info["score"] > reference_score + 1e-7:
            full_state = candidate_state
            full_selection = {"source": "validation_selected_refit", **candidate_info}
            reference.load_state_dict(full_state, strict=True)
            reference_test = apply_temperature(collect(reference, test_audio, test_visual), output_temperature)
    signature = artifact_signature()
    manifest = {
        "mode": "v21_artifact_first_rcf_ablation",
        "signature": signature,
        "source_posterior": SOURCE_PROBS,
        "source_full_checkpoint": SOURCE_MODEL,
        "protocol": "Full RCF is the verified v21 checkpoint. Deletion variants initialize from this Full state, disable the selected forward component, and fine-tune on train posterior only. Progressive variants initialize from train posterior statistics. Validation posterior selects epochs. Official test labels are used only for final metrics and paired bootstrap.",
        "output_temperature": output_temperature,
        "rost_scope": "ROST results are explicitly adapter-level objective ablations matching the v21 limited posterior-adaptation stage. They are not presented as independently retrained audio/visual expert ablations.",
        "fusion_seeds": FUSION_SEEDS,
        "full_selection": full_selection,
        "device": str(DEVICE),
    }
    save_json(manifest, log / "manifest.json")
    active = {name: config for name, config in RCF_VARIANTS.items() if not FILTER or name in FILTER}
    if "Full RCF" not in active:
        active = {"Full RCF": RCF_VARIANTS["Full RCF"], **active}
    paired_rows = []
    paired_outputs = {}
    paired_variants = {
        "Average/Base": RCF_VARIANTS["Average/Base"],
        "Full RCF": RCF_VARIANTS["Full RCF"],
        "w/o Reliability": RCF_VARIANTS["w/o Reliability"],
        "w/o Calibration": RCF_VARIANTS["w/o Calibration"],
        "w/o Bias Transport": RCF_VARIANTS["w/o Bias Transport"],
        "w/o Disagreement Refinement": RCF_VARIANTS["w/o Disagreement Refinement"],
    }
    for name, config in paired_variants.items():
        if name == "Full RCF":
            output = reference_test
            detail = {"params": parameter_count(reference), "epoch": full_selection["epoch"], "source": full_selection["source"]}
        elif name == "Average/Base":
            output = apply_temperature(0.5 * test_audio + 0.5 * test_visual, output_temperature)
            detail = {"params": 0, "epoch": 0, "source": "deterministic"}
        else:
            output = paired_component_output(config, full_state, mars_init, pair_bias, test_audio, test_visual, output_temperature)
            detail = {"params": parameter_count(reference), "epoch": full_selection["epoch"], "source": "paired_full_state"}
        paired_outputs[name] = output.astype(np.float32)
        paired_rows.append({"variant": name, **metrics(test_labels, output), **detail})
        plot_confusion(test_labels, output, log / f"cm_norm_paired_{safe_name(name)}.png", f"RAVDESS v21 paired | {name}")
    paired_frame = pd.DataFrame(paired_rows).sort_values(["acc", "f1", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    paired_frame.to_csv(log / "rcf_paired_component_ablation.csv", index=False)
    paired_full = paired_outputs["Full RCF"]
    paired_bootstrap = [{"comparison": f"Full RCF - {name}", "accuracy_delta": metrics(test_labels, paired_full)["acc"] - metrics(test_labels, output)["acc"], "accuracy_delta_95ci": bootstrap_delta(test_labels, paired_full, output, SEED)} for name, output in paired_outputs.items() if name != "Full RCF"]
    save_json({"protocol": "All rows use the same verified Full RCF state and differ only by a disabled forward component. No retraining or test-based selection is used.", "rows": paired_rows, "paired_bootstrap": paired_bootstrap}, log / "paired_component_results.json")

    reference.load_state_dict(full_state, strict=True)
    base_train = apply_temperature(collect(reference, train_audio, train_visual), output_temperature)
    base_val = apply_temperature(collect(reference, val_audio, val_visual), output_temperature)
    base_test = apply_temperature(collect(reference, test_audio, test_visual), output_temperature)
    feature_train = adapter_features(train_audio, train_visual, base_train)
    feature_val = adapter_features(val_audio, val_visual, base_val)
    feature_test = adapter_features(test_audio, test_visual, base_test)
    rost_rows = []
    rost_outputs = {}
    for name, config in ROST_ADAPTER_VARIANTS.items():
        seed_outputs = []
        details = []
        for pipeline_seed in FUSION_SEEDS:
            output_seed, detail_seed = train_adapter_variant(name, config, feature_train, base_train, train_labels, feature_val, base_val, val_labels, feature_test, base_test, pipeline_seed, root, signature)
            seed_outputs.append(output_seed)
            details.append(detail_seed)
        output = normalize(np.mean(seed_outputs, 0))
        detail = {"params": int(round(np.mean([item["params"] for item in details]))), "epoch": [item["epoch"] for item in details], "source": [item["source"] for item in details], "scope": "v21_posterior_adapter"}
        rost_outputs[name] = output.astype(np.float32)
        rost_rows.append({"variant": name, **metrics(test_labels, output), **recoverability_profile(test_labels, output), "posterior_delta_l1": float(np.abs(output - base_test).mean()), **detail})
        plot_confusion(test_labels, output, log / f"cm_norm_rost_adapter_{safe_name(name)}.png", f"RAVDESS v21 ROST adapter | {name}")
    rost_frame = pd.DataFrame(rost_rows).sort_values(["acc", "f1", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    rost_frame.to_csv(log / "rost_adapter_ablation.csv", index=False)
    np.savez_compressed(root / "rost_adapter_predictions.npz", test_labels=test_labels, **{safe_name(name): output for name, output in rost_outputs.items()})
    rost_full = rost_outputs["Full ROST Adapter"]
    full_profile = recoverability_profile(test_labels, rost_full)
    full_metrics = metrics(test_labels, rost_full)
    for row in rost_rows:
        row["delta_acc_vs_full"] = row["acc"] - full_metrics["acc"]
        row["delta_f1_vs_full"] = row["f1"] - full_metrics["f1"]
        row["delta_nll_vs_full"] = row["nll"] - full_metrics["nll"]
        row["delta_sri_vs_full"] = row["SRI"] - full_profile["SRI"]
    rost_frame = pd.DataFrame(rost_rows).sort_values(["acc", "f1", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    rost_frame.to_csv(log / "rost_adapter_ablation.csv", index=False)
    rost_bootstrap = [{"comparison": f"Full ROST Adapter - {name}", "accuracy_delta": metrics(test_labels, rost_full)["acc"] - metrics(test_labels, output)["acc"], "accuracy_delta_95ci": bootstrap_delta(test_labels, rost_full, output, SEED)} for name, output in rost_outputs.items() if name != "Full ROST Adapter"]
    save_json({"protocol": "All variants are independently trained v21 posterior-adapter objectives with identical architecture, seeds, budget, Full RCF anchor, and validation-only checkpoint selection. This is not expert-level ROST retraining.", "rows": rost_rows, "paired_bootstrap": rost_bootstrap}, log / "rost_adapter_results.json")

    rows = []
    outputs = {}
    for name, config in active.items():
        if name == "Full RCF":
            output = reference_test
            detail = {"params": parameter_count(reference), "epoch": full_selection["epoch"], "source": full_selection["source"]}
        else:
            seed_outputs = []
            details = []
            for fusion_seed in FUSION_SEEDS:
                output_seed, detail_seed = train_variant(name, config, train_audio, train_visual, train_labels, val_audio, val_visual, val_labels, test_audio, test_visual, full_state, mars_init, pair_bias, fusion_seed, root, signature, output_temperature)
                seed_outputs.append(output_seed)
                details.append(detail_seed)
            output = normalize(np.mean(seed_outputs, 0))
            detail = {"params": int(round(np.mean([item["params"] for item in details]))), "epoch": [item["epoch"] for item in details], "source": [item["source"] for item in details]}
        outputs[name] = output.astype(np.float32)
        rows.append({"variant": name, **metrics(test_labels, output), **detail})
        plot_confusion(test_labels, output, log / f"cm_norm_rcf_{safe_name(name)}.png", f"RAVDESS v21 | {name}")
    frame = pd.DataFrame(rows).sort_values(["acc", "f1", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    frame.to_csv(log / "rcf_component_ablation.csv", index=False)
    np.savez_compressed(root / "rcf_component_predictions.npz", test_labels=test_labels, **{safe_name(name): output for name, output in outputs.items()})
    full = outputs["Full RCF"]
    bootstrap = [{"comparison": f"Full RCF - {name}", "accuracy_delta": metrics(test_labels, full)["acc"] - metrics(test_labels, output)["acc"], "accuracy_delta_95ci": bootstrap_delta(test_labels, full, output, SEED)} for name, output in outputs.items() if name != "Full RCF"]
    result = {"manifest": manifest, "reference_reproduction": {"mean_l1": reproduction_l1, "metrics": metrics(test_labels, reference_test)}, "paired_component_rows": paired_rows, "paired_component_bootstrap": paired_bootstrap, "rost_adapter_rows": rost_rows, "rost_adapter_bootstrap": rost_bootstrap, "retrained_rows": rows, "retrained_bootstrap": bootstrap, "time_minutes": (time.time() - start) / 60.0}
    save_json(result, log / "results.json")
    print(f"RAVDESS v21 RCF ablation device={DEVICE} train={len(train_labels)} val={len(val_labels)} test={len(test_labels)}")
    reference_metrics = metrics(test_labels, reference_test)
    print(f"Verified Full RCF acc={reference_metrics['acc']:.4f} f1={reference_metrics['f1']:.4f} nll={reference_metrics['nll']:.4f} params={format_params(parameter_count(reference))}")
    print("paired_full_state_ablation")
    print_table(paired_frame)
    print("rost_adapter_objective_ablation")
    print_table(rost_frame)
    print("retrained_supplementary_ablation")
    print_table(frame)
    print(f"Checkpoints={root}")
    print(f"Logs={log}")
    print(f"TimeMinutes={(time.time() - start) / 60.0:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAVDESS ROST and RCF ablations.")
    parser.add_argument("--check", action="store_true", help="validate source artifacts")
    return parser.parse_args()


def check_environment():
    required = (SOURCE_SCRIPT, SOURCE_PROBS, SOURCE_MODEL)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RAVDESS ablation artifacts: {missing}")
    print(
        f"RAVDESS ablation check passed: source={SOURCE_SCRIPT.name} "
        f"probabilities={SOURCE_PROBS.name} checkpoint={SOURCE_MODEL.name}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        check_environment()
    else:
        main()
