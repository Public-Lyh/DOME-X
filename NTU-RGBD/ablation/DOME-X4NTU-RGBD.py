"""Leakage-free DOME-X ablations for NTU RGB+D 60.

This runner adapts the validated UTD-MHAD ROST/RCF ablation engine to the
feature representation and official protocols in ``ntu_dome_x_train_01.py``.
Official test labels are not consumed until every expert checkpoint and test
posterior has been frozen.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection


PLACEHOLDER_ROOT = Path("your path")
WORKSPACE_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
PROJECT_ROOT = WORKSPACE_ROOT / "Code" / "NTU-RGBD"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SOURCE_PATH = SCRIPT_ROOT / "ntu_dome_x_train_01.py"
CORE_PATH = WORKSPACE_ROOT / "Code" / "UTD-MHAD" / "CongTouYve" / "utd_dome_x_ablation.py"
FEATURE_CACHE = PROJECT_ROOT / "cache" / "ntu60_skeleton_features_xsub_maxNone_v1.npz"
BASE_CKPT_DIR = PROJECT_ROOT / "checkpoints"
BASE_LOG_DIR = PROJECT_ROOT / "logs"
EXP_NAME = "DOME_X_NTU_RGBD60_ABLATION"

NC = 60
EPS = 1e-8
PROTOCOL = os.environ.get("DOME_X_NTU_PROTOCOL", "xsub").lower()
if PROTOCOL not in {"xsub", "xview"}:
    raise ValueError(f"DOME_X_NTU_PROTOCOL must be xsub or xview, got {PROTOCOL!r}")

NTU60_XSUB_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
NTU60_XSUB_TEST_SUBJECTS = set(range(1, 41)) - NTU60_XSUB_TRAIN_SUBJECTS
NTU60_XVIEW_TRAIN_CAMERAS = {2, 3}
NTU60_XVIEW_TEST_CAMERAS = {1}
ALL_VIEWS = ("joint", "bone", "motion", "bone_motion", "part")
DEFAULT_VIEWS = ("joint", "bone", "part")
FUSION_VIEWS = tuple(item.strip() for item in os.environ.get("DOME_X_NTU_VIEWS", ",".join(DEFAULT_VIEWS)).split(",") if item.strip())
if len(FUSION_VIEWS) < 2 or len(set(FUSION_VIEWS)) != len(FUSION_VIEWS) or any(name not in ALL_VIEWS for name in FUSION_VIEWS):
    raise ValueError(f"DOME_X_NTU_VIEWS must select at least two unique views from {ALL_VIEWS}, got {FUSION_VIEWS}")
# Train every reference expert so that the selected joint/bone/part observers
# receive the same seeds and RNG consumption as ntu_dome_x_train_01.py.
VIEW_NAMES = ALL_VIEWS

SEED = int(os.environ.get("DOME_X_SEED", "42"))
PIPELINE_SEEDS = [SEED + index for index in range(int(os.environ.get("DOME_X_PIPELINE_SEEDS", "1")))]
FUSION_SEEDS = [SEED + index for index in range(int(os.environ.get("DOME_X_FUSION_SEEDS", "3")))]
OUTER_FOLDS = int(os.environ.get("DOME_X_OUTER_FOLDS", "5"))
PRETRAIN_EPOCHS = int(os.environ.get("DOME_X_PRETRAIN_EPOCHS", "8"))
ROST_EPOCHS = int(os.environ.get("DOME_X_ROST_EPOCHS", "4"))
RCF_EPOCHS = int(os.environ.get("DOME_X_RCF_EPOCHS", "180"))
EXPERT_PATIENCE = int(os.environ.get("DOME_X_EXPERT_PATIENCE", "6"))
FUSION_PATIENCE = int(os.environ.get("DOME_X_FUSION_PATIENCE", "30"))
PROFILE_INTERVAL = int(os.environ.get("DOME_X_PROFILE_INTERVAL", "1"))
MAX_INPUT_DIM = int(os.environ.get("DOME_X_MAX_INPUT_DIM", "0"))
EXPERT_BATCH = int(os.environ.get("DOME_X_EXPERT_BATCH", "768"))
TRANSFORM_BATCH = int(os.environ.get("DOME_X_TRANSFORM_BATCH", "1024"))
PLOTS = os.environ.get("DOME_X_PLOTS", "0") != "0"
TERMINAL_TOP = int(os.environ.get("DOME_X_TERMINAL_TOP", "35"))
GROUP_OOF = os.environ.get("DOME_X_NTU_GROUP_OOF", "0") == "1"
SYNTHETIC = os.environ.get("DOME_X_NTU_SYNTHETIC", "0") == "1"
VALIDATE_ONLY = os.environ.get("DOME_X_VALIDATE_ONLY", "0") == "1"
SAVE_ALL_PREDICTIONS = os.environ.get("DOME_X_SAVE_ALL_PREDICTIONS", "0") == "1"
KEEP_PREPARED = os.environ.get("DOME_X_KEEP_PREPARED", "0") == "1"

if not PIPELINE_SEEDS or not FUSION_SEEDS:
    raise ValueError("DOME_X_PIPELINE_SEEDS and DOME_X_FUSION_SEEDS must both be positive")
if OUTER_FOLDS < 2:
    raise ValueError("DOME_X_OUTER_FOLDS must be at least 2")
if min(PRETRAIN_EPOCHS, ROST_EPOCHS, RCF_EPOCHS, EXPERT_PATIENCE, FUSION_PATIENCE, PROFILE_INTERVAL, EXPERT_BATCH, TRANSFORM_BATCH) <= 0:
    raise ValueError("Epochs, patience, profile interval, and batch sizes must all be positive")
if MAX_INPUT_DIM < 0:
    raise ValueError("DOME_X_MAX_INPUT_DIM cannot be negative")


def load_core() -> Any:
    if not CORE_PATH.exists():
        raise RuntimeError(f"Validated DOME-X ablation core is missing: {CORE_PATH}")
    spec = importlib.util.spec_from_file_location("ntu_dome_x_ablation_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the ablation core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core: Any = load_core()


class Observer(nn.Module):
    """NTU observer capacity from ntu_dome_x_train_01.py."""

    def __init__(self, input_dim):
        super().__init__()
        hidden = int(np.clip(2 ** round(np.log2(max(np.sqrt(input_dim * NC) * 2, 128))), 256, 768))
        bottleneck = max(hidden // 2, NC * 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(hidden, bottleneck), nn.LayerNorm(bottleneck), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(bottleneck, NC),
        )

    def forward(self, x):
        return self.network(x)


def native_predict(model, features, indices=None, batch=2048):
    model = model.to(core.DEVICE).eval()
    indices = np.arange(len(features)) if indices is None else np.asarray(indices)
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            index = indices[start:start + batch]
            xb = torch.from_numpy(np.asarray(features[index], dtype=np.float32)).to(core.DEVICE, non_blocking=True)
            with torch.autocast(device_type=core.DEVICE.type, dtype=torch.float16, enabled=core.DEVICE.type == "cuda"):
                output.append(F.softmax(model(xb), dim=1).float().cpu().numpy())
    return core.normalize(np.concatenate(output))


def native_component_loss(labels, logits, classes, config, rescue_weights, other_confusions, progress):
    """Reference NTU recoverability loss with independently removable terms."""
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=0.02)
    if not any(config[key] for key in ("structure", "collapse", "complement", "joint")):
        return ce, {"ce": float(ce.detach()), "structure": 0.0, "collapse": 0.0, "complement": 0.0, "joint": 0.0}
    posterior = F.softmax(logits.float(), dim=1)
    matrix, present = native_torch_confusion(labels, posterior, classes)
    matrix = matrix[present]
    if matrix.shape[0] < 4:
        return ce, {"ce": float(ce.detach()), "structure": 0.0, "collapse": 0.0, "complement": 0.0, "joint": 0.0}
    matrix = matrix / matrix.sum(1, keepdim=True).clamp_min(EPS)
    entropy = -(matrix * matrix.clamp_min(EPS).log()).sum(1) / np.log(classes)
    top1 = matrix.max(1).values
    top_r = matrix.topk(min(3, classes), dim=1).values.sum(1)
    distances = native_js_rows(matrix)
    eye = torch.eye(len(matrix), dtype=torch.bool, device=logits.device)
    structure = (
        F.relu(0.18 - entropy).square().mean()
        + F.relu(entropy - 0.78).square().mean()
        + 0.25 * F.relu(top1 - 0.96).square().mean()
        + F.relu(0.45 - top_r).square().mean()
        + F.relu(top_r - 0.99).square().mean()
        + 0.6 * torch.exp(-distances[~eye] / 0.15).mean()
    )
    usage = matrix.mean(0)
    usage_entropy = -(usage * usage.clamp_min(EPS).log()).sum() / np.log(classes)
    uniform = torch.full_like(matrix, 1.0 / classes)
    collapse = (
        0.4 * (F.relu(0.45 - usage_entropy).square() + F.relu(usage_entropy - 0.995).square() + F.relu(usage.max() - 0.45).square())
        - 0.10 * (matrix - uniform).square().sum(1).mean()
    )
    complement = torch.zeros((), device=logits.device)
    if rescue_weights is not None:
        if matrix.shape[0] == classes:
            weight = rescue_weights.to(logits.device, logits.dtype)
        else:
            weight = rescue_weights.to(logits.device, logits.dtype)[present][:, present]
        complement = complement - 0.45 * (weight * distances).sum() / weight.sum().clamp_min(EPS)
    if other_confusions and matrix.shape[0] == classes:
        redundancy = []
        flat = matrix.flatten()
        for other in other_confusions:
            redundancy.append(F.cosine_similarity(flat, other.to(logits.device, logits.dtype).flatten(), dim=0))
        complement = complement + 0.08 * torch.stack(redundancy).mean()
    joint = torch.zeros((), device=logits.device)
    if matrix.shape[0] == classes:
        smooth = matrix + 1e-3
        smooth = smooth / smooth.sum(1, keepdim=True)
        reverse = smooth.T / smooth.T.sum(1, keepdim=True).clamp_min(EPS)
        joint = 0.15 * (-torch.diag(smooth @ reverse).clamp_min(EPS).log().mean())
    scale = 0.05 + 0.20 * progress
    objective = ce
    if config["structure"]:
        objective = objective + scale * structure
    if config["collapse"]:
        objective = objective + scale * collapse
    if config["complement"]:
        objective = objective + scale * complement
    if config["joint"]:
        objective = objective + scale * joint
    return objective, {
        "ce": float(ce.detach()), "structure": float(structure.detach()), "collapse": float(collapse.detach()),
        "complement": float(complement.detach()), "joint": float(joint.detach()),
    }


def native_torch_confusion(labels, probabilities, classes):
    one_hot = F.one_hot(labels, classes).to(probabilities.dtype)
    counts = one_hot.sum(0)
    return one_hot.T @ probabilities / counts.clamp_min(1.0).unsqueeze(1), counts > 0


def native_js_rows(matrix):
    matrix = matrix.clamp_min(EPS)
    matrix = matrix / matrix.sum(1, keepdim=True)
    left, right = matrix[:, None], matrix[None, :]
    middle = 0.5 * (left + right)
    return 0.5 * ((left * (left.log() - middle.log())).sum(-1) + (right * (right.log() - middle.log())).sum(-1)) / np.log(2.0)


def native_rescue_weight_matrix(confusions, excluded, temperature=0.12):
    if len(confusions) < 2:
        raise ValueError("ROST rescue weighting requires at least two expert confusion matrices")
    classes = confusions[0].shape[0]
    other = np.concatenate([matrix for index, matrix in enumerate(confusions) if index != excluded], axis=1)
    weights = np.zeros((classes, classes), dtype=np.float32)
    for first in range(classes):
        for second in range(first + 1, classes):
            left, right = other[first], other[second]
            distance = 1.0 - float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), EPS))
            weights[first, second] = weights[second, first] = np.exp(-distance / temperature)
    weights /= max(weights.sum(), EPS)
    return torch.from_numpy(weights)


def native_js_distance(left, right):
    left = np.clip(np.asarray(left, dtype=np.float64), EPS, None)
    right = np.clip(np.asarray(right, dtype=np.float64), EPS, None)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / middle)) + 0.5 * np.sum(right * np.log(right / middle)))


def native_effective_rank(matrix):
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    probability = singular / max(singular.sum(), EPS)
    return float(np.exp(-np.sum(probability * np.log(probability + EPS))) / matrix.shape[0])


def native_cm_sri(matrix):
    matrix = core.rows(matrix)
    entropy = -(matrix * np.log(matrix + EPS)).sum(axis=1) / np.log(NC)
    top1 = matrix.max(axis=1)
    top3 = np.sort(matrix, axis=1)[:, -min(3, NC):].sum(axis=1)
    multi_peak = np.clip(1.0 - np.mean(np.maximum(0.0, 0.18 - entropy) ** 2 + np.maximum(0.0, entropy - 0.78) ** 2 + 0.25 * np.maximum(0.0, top1 - 0.96) ** 2 + np.maximum(0.0, 0.45 - top3) ** 2 + np.maximum(0.0, top3 - 0.99) ** 2), 0.0, 1.0)
    separation = float(np.mean([native_js_distance(matrix[first], matrix[second]) / np.log(2.0) for first in range(NC) for second in range(first + 1, NC)]))
    usage = matrix.mean(axis=0)
    usage_entropy = float(-(usage * np.log(usage + EPS)).sum() / np.log(NC))
    decode, _, _ = native_bayes_decode(matrix)
    rank = native_effective_rank(matrix)
    score = 0.20 * multi_peak + 0.25 * separation + 0.15 * usage_entropy + 0.15 * rank + 0.25 * decode
    return {"CM_SRI": float(score), "Q_mp": float(multi_peak), "Q_sep": separation, "Q_usage": usage_entropy, "Q_rank": rank, "Q_decode": decode}


def native_cm_jsri(matrices):
    matrices = [core.rows(matrix) for matrix in matrices]
    joint = np.concatenate(matrices, axis=1)
    pair_first, pair_second = np.triu_indices(NC, k=1)
    joint_distance = np.asarray([
        1.0 - float(np.dot(joint[first], joint[second]) / max(np.linalg.norm(joint[first]) * np.linalg.norm(joint[second]), EPS))
        for first, second in zip(pair_first, pair_second)
    ])
    hard_separation = float(np.clip(-0.12 * np.log(np.mean(np.exp(-joint_distance / 0.12)) + EPS), 0.0, 1.0))
    redundancy = []
    rescue = []
    for position, matrix in enumerate(matrices):
        other = np.concatenate([item for index, item in enumerate(matrices) if index != position], axis=1)
        weights, values = [], []
        for first, second in zip(pair_first, pair_second):
            similarity = float(np.dot(other[first], other[second]) / max(np.linalg.norm(other[first]) * np.linalg.norm(other[second]), EPS))
            weights.append(np.exp(-(1.0 - similarity) / 0.12))
            values.append(native_js_distance(matrix[first], matrix[second]) / np.log(2.0))
        weights = np.asarray(weights)
        rescue.append(float(np.dot(weights / max(weights.sum(), EPS), values)))
        for other_matrix in matrices[position + 1:]:
            redundancy.append(float(np.dot(matrix.ravel(), other_matrix.ravel()) / max(np.linalg.norm(matrix) * np.linalg.norm(other_matrix), EPS)))
    decode = float(np.mean([native_bayes_decode(matrix)[0] for matrix in matrices]))
    rank = native_effective_rank(joint)
    red = float(np.mean(redundancy)) if redundancy else 0.0
    score = 0.25 * hard_separation + 0.15 * rank + 0.30 * float(np.mean(rescue)) + 0.15 * decode + 0.15 * (1.0 - red)
    return {"CM_JSRI": float(score), "Q_jsep": hard_separation, "Q_jrank": rank, "Q_rescue": float(np.mean(rescue)), "Q_jdecode": decode, "Q_red": red}


ROST_VARIANTS = {
    "Full ROST": {"structure": True, "collapse": True, "complement": True, "joint": True, "schedule": "progressive"},
    "w/o Structure": {"structure": False, "collapse": True, "complement": True, "joint": True},
    "w/o Anti-collapse": {"structure": True, "collapse": False, "complement": True, "joint": True},
    "w/o Complementarity": {"structure": True, "collapse": True, "complement": False, "joint": True},
    "w/o Joint Recovery": {"structure": True, "collapse": True, "complement": True, "joint": False},
}


RCF_VARIANTS = {
    "Average/Base": {"reliability": False, "calibration": False, "transport": False, "refinement": False},
    "+ Class-wise Reliability": {"reliability": True, "calibration": False, "transport": False, "refinement": False},
    "+ PEACE Calibration": {"reliability": True, "calibration": True, "transport": False, "refinement": False},
    "+ Learnable Bias Transport": {"reliability": True, "calibration": True, "transport": True, "refinement": False},
    "Full RCF": {"reliability": True, "calibration": True, "transport": True, "refinement": True},
    "w/o Reliability": {"reliability": False, "calibration": True, "transport": True, "refinement": True},
    "w/o Calibration": {"reliability": True, "calibration": False, "transport": True, "refinement": True},
    "w/o Bias Transport": {"reliability": True, "calibration": True, "transport": False, "refinement": True},
    "w/o Disagreement Refinement": {"reliability": True, "calibration": True, "transport": True, "refinement": False},
}


def component_active(config, name, cycle):
    if not config[name]:
        return False
    if not config.get("schedule"):
        return True
    if name == "structure":
        return True
    if name == "collapse":
        return cycle >= max(2, int(np.ceil(ROST_EPOCHS / 3.0)))
    if name == "complement":
        return cycle >= max(2, int(np.ceil(2.0 * ROST_EPOCHS / 3.0)))
    return cycle >= max(2, int(np.ceil(3.0 * ROST_EPOCHS / 4.0)))


def native_train_epoch(model, features, labels, sample_indices, config, progress, rescue_weights, other_confusions, optimizer, scaler):
    model.train()
    order = np.random.permutation(sample_indices)
    totals = {"loss": 0.0, "ce": 0.0, "structure": 0.0, "collapse": 0.0, "complement": 0.0, "joint": 0.0}
    for start in range(0, len(order), EXPERT_BATCH):
        index = order[start:start + EXPERT_BATCH]
        xb = torch.from_numpy(np.asarray(features[index], dtype=np.float32)).to(core.DEVICE, non_blocking=True)
        yb = torch.from_numpy(labels[index]).long().to(core.DEVICE, non_blocking=True)
        xb = xb + 0.015 * torch.randn_like(xb)
        xb = xb * (torch.rand_like(xb) > 0.015)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=core.DEVICE.type, dtype=torch.float16, enabled=core.DEVICE.type == "cuda"):
            logits = model(xb)
            loss, parts = native_component_loss(yb, logits, NC, config, rescue_weights, other_confusions, progress)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach()) * len(index)
        for name, value in parts.items():
            totals[name] += value * len(index)
    return {name: value / len(order) for name, value in totals.items()}


def native_posterior_context(x):
    x = x.clamp_min(EPS)
    mean = x.mean(1)
    std = x.std(1, unbiased=False)
    maximum = x.max(1).values
    entropy = -(x * x.log()).sum(2) / np.log(x.shape[2])
    top2 = x.topk(min(2, x.shape[2]), dim=2).values
    margin = top2[:, :, 0] - top2[:, :, -1]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float()
    return torch.cat([x.flatten(1), mean, std, maximum, entropy, margin, disagreement], dim=1)


class NativeRCF(nn.Module):
    """The v22 RCFNet with only declared paths switchable for ablation."""

    def __init__(self, initial_confusions, config):
        super().__init__()
        self.config = dict(config)
        self.modalities = len(initial_confusions)
        self.classes = initial_confusions[0].shape[0]
        context_dim = self.modalities * self.classes + 3 * self.classes + 3 * self.modalities
        reverse, reliability = [], []
        for confusion in initial_confusions:
            _, backward, decoded = native_bayes_decode(confusion)
            reverse.append(backward.T)
            reliability.append(np.clip(np.diag(decoded), 0.02, 1.0))
        reliability = np.stack(reliability)
        reliability /= reliability.sum(0, keepdims=True)
        self.reliability_logits = nn.Parameter(torch.log(torch.tensor(reliability, dtype=torch.float32)))
        self.calibration_scale = nn.Parameter(torch.ones(self.modalities, self.classes))
        self.calibration_bias = nn.Parameter(torch.zeros(self.modalities, self.classes))
        self.transport_logits = nn.Parameter(torch.log(torch.tensor(np.stack(reverse), dtype=torch.float32).clamp_min(EPS)))
        self.mix_logits = nn.Parameter(torch.log(torch.tensor([0.15, 0.20, 0.15, 0.05, 0.45], dtype=torch.float32)))
        width = max(64, self.classes)
        path_output = nn.Linear(width, 5)
        residual_output = nn.Linear(width, self.classes, bias=False)
        self.path_gate = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), path_output)
        self.disagreement_gate = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), nn.Linear(width, 1))
        self.residual = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), residual_output)
        nn.init.zeros_(path_output.weight)
        nn.init.zeros_(path_output.bias)
        nn.init.zeros_(residual_output.weight)

    def enabled_parameters(self):
        enabled = {
            "reliability_logits": self.config["reliability"],
            "calibration_scale": self.config["calibration"], "calibration_bias": self.config["calibration"],
            "transport_logits": self.config["transport"],
            "mix_logits": self.config["refinement"], "path_gate": self.config["refinement"],
            "disagreement_gate": self.config["refinement"], "residual": self.config["refinement"],
        }
        selected = []
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(any(name == key or name.startswith(key + ".") for key, active in enabled.items() if active))
            if parameter.requires_grad:
                selected.append(parameter)
        return selected

    def forward(self, x):
        x = x.clamp_min(EPS)
        calibrated = (
            F.softmax(torch.log(x) * F.softplus(self.calibration_scale).unsqueeze(0) + self.calibration_bias.unsqueeze(0), dim=2)
            if self.config["calibration"] else x
        )
        weights = F.softmax(self.reliability_logits, dim=0).unsqueeze(0) if self.config["reliability"] else torch.full((1, self.modalities, self.classes), 1.0 / self.modalities, dtype=x.dtype, device=x.device)
        if self.config["transport"]:
            transport = F.softmax(self.transport_logits, dim=1)
            recovered = torch.einsum("nmk,myk->nmy", calibrated, transport)
            recovered = recovered / recovered.sum(2, keepdim=True).clamp_min(EPS)
        else:
            recovered = calibrated
        arithmetic = (weights * recovered).sum(1)
        geometric = F.softmax((weights * torch.log(calibrated.clamp_min(EPS))).sum(1), dim=1)
        bias_geometric = F.softmax((weights * torch.log(recovered.clamp_min(EPS))).sum(1), dim=1)
        raw = (weights * x).sum(1)
        product = F.softmax(torch.log(x.clamp_min(EPS)).sum(1), dim=1)
        if not self.config["refinement"]:
            return arithmetic / arithmetic.sum(1, keepdim=True).clamp_min(EPS)
        context = native_posterior_context(x)
        mix = F.softmax(self.mix_logits.unsqueeze(0) + 0.5 * self.path_gate(context), dim=1)
        structured = mix[:, 0:1] * arithmetic + mix[:, 1:2] * geometric + mix[:, 2:3] * bias_geometric + mix[:, 3:4] * raw + mix[:, 4:5] * product
        disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True)
        uncertainty = 1.0 - x.max(2).values.mean(1, keepdim=True)
        gate = torch.sigmoid(self.disagreement_gate(context)) * (0.10 + 0.55 * disagreement + 0.35 * uncertainty)
        residual = F.softmax(torch.log(structured.clamp_min(EPS)) + self.residual(context), dim=1)
        output = (1.0 - gate) * structured + gate * residual
        return output / output.sum(1, keepdim=True).clamp_min(EPS)


def native_bayes_decode(confusion, smoothing=1e-3):
    matrix = core.rows(np.asarray(confusion, dtype=np.float64) + smoothing)
    prior = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    backward = (prior[:, None] * matrix).T
    backward /= np.maximum(backward.sum(axis=1, keepdims=True), EPS)
    decoded = matrix @ backward
    return float(np.diag(decoded).mean()), backward, decoded


def native_augment_posteriors(x):
    logits = torch.log(x.clamp_min(EPS))
    temperature = torch.empty((len(x), x.shape[1], 1), device=x.device).uniform_(0.85, 1.15)
    augmented = F.softmax(logits / temperature + 0.025 * torch.randn_like(logits), dim=2)
    keep = (torch.rand((len(x), x.shape[1], 1), device=x.device) > 0.08).to(x.dtype)
    fallback = augmented.mean(1, keepdim=True)
    augmented = augmented * keep + fallback * (1.0 - keep)
    return augmented / augmented.sum(2, keepdim=True).clamp_min(EPS)


def native_rcf_predict(model, x, batch=4096):
    model = model.to(core.DEVICE).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(np.asarray(x[start:start + batch], dtype=np.float32)).to(core.DEVICE)
            output.append(model(xb).float().cpu().numpy())
    return core.normalize(np.concatenate(output))


def fit_native_rcf_variant(name, config, x, labels, test_folds, seed, root, context, tag, initial_state=None, source="independent"):
    path = root / "fusion" / f"native_{core.slug(context)}_{core.slug(name)}_fs{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    posterior_hash = array_hash(x, labels, *test_folds)
    upstream_hash = core.state_hash(initial_state)
    matrices = [core.soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
    if path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("tag") == tag and saved.get("config") == config and saved.get("source") == source and saved.get("posterior_hash") == posterior_hash and saved.get("upstream_hash") == upstream_hash:
                model = NativeRCF(matrices, config)
                model.load_state_dict(saved["state_dict"], strict=True)
                return [native_rcf_predict(model, fold_x) for fold_x in test_folds], saved["info"], saved["state_dict"]
        except (OSError, KeyError, RuntimeError):
            pass
    if name == "Average/Base":
        return [core.normalize(fold_x.mean(1)) for fold_x in test_folds], {"params": 0, "selected_epoch": 0, "source": "fixed_average", "selection_state": None}, None
    selection_idx, validation_idx = train_test_split(np.arange(len(labels)), test_size=0.22, random_state=SEED, stratify=labels)
    seed_all(seed)
    model = NativeRCF(matrices, config)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    parameters = model.enabled_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RCF_EPOCHS, eta_min=1e-5)
    best_state, best_score, best_epoch, stale, history = core.clone_state(model), float("inf"), 0, 0, []
    for epoch in range(1, RCF_EPOCHS + 1):
        model.train()
        order = np.random.permutation(selection_idx)
        total = 0.0
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            xb = torch.from_numpy(x[index]).float().to(core.DEVICE)
            yb = torch.from_numpy(labels[index]).long().to(core.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(native_augment_posteriors(xb))
            loss = F.nll_loss(torch.log(output.clamp_min(EPS)), yb) + 0.08 * F.mse_loss(output, F.one_hot(yb, NC).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        validation = native_rcf_predict(model, x[validation_idx])
        f1 = float(f1_score(labels[validation_idx], validation.argmax(1), average="macro", zero_division="warn"))
        nll = core.metrics(labels[validation_idx], validation)["nll"]
        score = -f1 + 0.002 * nll
        history.append({"epoch": epoch, "train_loss": total / len(order), "selection_f1": f1, "selection_nll": nll, "selection_score": score})
        if score < best_score - 1e-5:
            best_state, best_score, best_epoch, stale = core.clone_state(model), score, epoch, 0
        else:
            stale += 1
        if stale >= FUSION_PATIENCE:
            break
    model.load_state_dict(best_state)
    state = core.clone_state(model)
    info = {"params": sum(parameter.numel() for parameter in model.enabled_parameters()), "selected_epoch": best_epoch, "source": source, "selection_state": state, "history": history}
    torch.save({"tag": tag, "config": config, "source": source, "posterior_hash": posterior_hash, "upstream_hash": upstream_hash, "state_dict": state, "info": info}, path)
    return [native_rcf_predict(model, fold_x) for fold_x in test_folds], info, state


class NativeStackingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, NC),
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


def native_fusion_features(x):
    x = np.clip(np.asarray(x, dtype=np.float32), EPS, 1.0)
    mean, std, maximum = x.mean(1), x.std(1), x.max(1)
    entropy = -(x * np.log(x)).sum(2) / np.log(x.shape[2])
    ordered = np.sort(x, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).astype(np.float32)
    return np.concatenate([x.reshape(len(x), -1), mean, std, maximum, entropy, margin, disagreement], 1).astype(np.float32)


def fit_native_mlp(x, labels, test_x, seed):
    seed_all(seed)
    features, test_features = native_fusion_features(x), native_fusion_features(test_x)
    fit_idx, validation_idx = train_test_split(np.arange(len(labels)), test_size=0.22, random_state=SEED, stratify=labels)
    model = NativeStackingMLP(features.shape[1]).to(core.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best_state, best_loss, stale = core.clone_state(model), float("inf"), 0
    for _ in range(RCF_EPOCHS):
        model.train()
        order = np.random.permutation(fit_idx)
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            xb = torch.from_numpy(features[index]).float().to(core.DEVICE)
            yb = torch.from_numpy(labels[index]).long().to(core.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = F.nll_loss(torch.log(output.clamp_min(EPS)), yb) + 0.08 * F.mse_loss(output, F.one_hot(yb, NC).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = model(torch.from_numpy(features[validation_idx]).float().to(core.DEVICE))
            value = float(F.nll_loss(torch.log(validation.clamp_min(EPS)), torch.from_numpy(labels[validation_idx]).long().to(core.DEVICE)))
        if value < best_loss - 1e-5:
            best_state, best_loss, stale = core.clone_state(model), value, 0
        else:
            stale += 1
        if stale >= FUSION_PATIENCE:
            break
    model.load_state_dict(best_state)
    return native_rcf_predict(model, test_features), sum(parameter.numel() for parameter in model.parameters())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def file_fingerprint(path):
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def array_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def config_hash():
    payload = {
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "core": hashlib.sha256(CORE_PATH.read_bytes()).hexdigest()[:16],
        "source": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16],
        "feature_cache": file_fingerprint(FEATURE_CACHE),
        "protocol": PROTOCOL, "views": VIEW_NAMES, "classes": NC,
        "pipeline_seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS,
        "folds": OUTER_FOLDS, "group_oof": GROUP_OOF,
        "pretrain_epochs": PRETRAIN_EPOCHS, "rost_epochs": ROST_EPOCHS,
        "rcf_epochs": RCF_EPOCHS, "expert_patience": EXPERT_PATIENCE,
        "fusion_patience": FUSION_PATIENCE, "profile_interval": PROFILE_INTERVAL,
        "expert_batch": EXPERT_BATCH, "max_input_dim": MAX_INPUT_DIM,
        "rost_variants": core.ROST_VARIANTS, "rcf_variants": core.RCF_VARIANTS,
        "synthetic": SYNTHETIC,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def output_dirs(tag):
    if SYNTHETIC:
        checkpoint = Path("/tmp/kilo/ntu_dome_x_ablation/checkpoints") / tag
        log = Path("/tmp/kilo/ntu_dome_x_ablation/logs") / tag
        checkpoint.mkdir(parents=True, exist_ok=True)
        log.mkdir(parents=True, exist_ok=True)
        return 0, checkpoint, log
    BASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("DOME_X_NTU_ABLATION_VERSION", os.environ.get("DOME_X_ABLATION_VERSION"))
    if requested:
        version = int(re.sub(r"\D", "", requested))
    else:
        versions = []
        for path in BASE_CKPT_DIR.glob("ablation_v*"):
            match = re.fullmatch(r"ablation_v(\d+)", path.name)
            if path.is_dir() and match:
                versions.append(int(match.group(1)))
        version = None
        if os.environ.get("DOME_X_NEW_ABLATION", "0") != "1":
            for candidate in sorted(versions, reverse=True):
                manifest_path = BASE_LOG_DIR / f"ablation_v{candidate}" / EXP_NAME / "manifest.json"
                try:
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        if json.load(handle).get("tag") == tag:
                            version = candidate
                            break
                except (OSError, ValueError):
                    continue
        if version is None:
            version = max(versions, default=0) + 1
    checkpoint = BASE_CKPT_DIR / f"ablation_v{version}" / EXP_NAME
    log = BASE_LOG_DIR / f"ablation_v{version}" / EXP_NAME
    checkpoint.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return version, checkpoint, log


def synthetic_features():
    rng = np.random.default_rng(SEED)
    train_ids = sorted(NTU60_XSUB_TRAIN_SUBJECTS)[:4] if PROTOCOL == "xsub" else [2, 3]
    test_ids = sorted(NTU60_XSUB_TEST_SUBJECTS)[:4] if PROTOCOL == "xsub" else [1]
    repeats = 2 if PROTOCOL == "xsub" else 4
    labels, split_ids, groups, cameras = [], [], [], []
    rows_by_view = {name: [] for name in VIEW_NAMES}
    for split_id in [*train_ids, *test_ids]:
        for repeat in range(repeats):
            for label in range(NC):
                labels.append(label)
                split_ids.append(split_id)
                subject = split_id if PROTOCOL == "xsub" else repeat + 1
                groups.append(subject)
                cameras.append(split_id if PROTOCOL == "xview" else 1 + repeat % 3)
                latent = rng.normal(0.0, 0.25, 48).astype(np.float32)
                latent[label % 48] += 2.5
                for position, name in enumerate(VIEW_NAMES):
                    rows_by_view[name].append(latent + rng.normal(0.0, 0.18 + 0.02 * position, 48))
    views = {name: np.asarray(rows, dtype=np.float32) for name, rows in rows_by_view.items()}
    return views, np.asarray(labels, dtype=np.int64), np.asarray(split_ids), np.asarray(groups), np.asarray(cameras)


def ensure_memmap_cache():
    if not FEATURE_CACHE.exists():
        raise RuntimeError(
            f"NTU feature cache is missing: {FEATURE_CACHE}. Run ntu_dome_x_train_01.py once to build it."
        )
    source_tag = hashlib.sha256(file_fingerprint(FEATURE_CACHE).encode()).hexdigest()[:12]
    target = FEATURE_CACHE.with_suffix("")
    target = target.parent / f"{target.name}_memmap_{source_tag}"
    target.mkdir(parents=True, exist_ok=True)
    required = {**{name: f"X_{name}" for name in VIEW_NAMES}, "labels": "y", "subjects": "subjects", "cameras": "cameras"}
    with np.load(FEATURE_CACHE, allow_pickle=False) as source:
        for name, key in required.items():
            path = target / f"{name}.npy"
            if path.exists():
                try:
                    cached = np.load(path, mmap_mode="r")
                    if cached.shape == source[key].shape and cached.dtype == source[key].dtype:
                        continue
                except (OSError, ValueError):
                    pass
            np.save(path, np.asarray(source[key]))
    return target


class IndexedRows:
    """A zero-copy local-index view over a source feature memmap."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), source.shape[1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.source[self.indices[item]]


_development_groups = None
_test_groups = None


def load_features():
    global _development_groups, _test_groups
    if SYNTHETIC:
        views, labels, split_ids, groups, _ = synthetic_features()
    else:
        cache = ensure_memmap_cache()
        views = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in VIEW_NAMES}
        labels = np.load(cache / "labels.npy", mmap_mode="r")
        subjects = np.load(cache / "subjects.npy", mmap_mode="r")
        cameras = np.load(cache / "cameras.npy", mmap_mode="r")
        split_ids = subjects if PROTOCOL == "xsub" else cameras
        groups = subjects
    train_values = NTU60_XSUB_TRAIN_SUBJECTS if PROTOCOL == "xsub" else NTU60_XVIEW_TRAIN_CAMERAS
    test_values = NTU60_XSUB_TEST_SUBJECTS if PROTOCOL == "xsub" else NTU60_XVIEW_TEST_CAMERAS
    train_mask = np.isin(split_ids, sorted(train_values))
    test_mask = np.isin(split_ids, sorted(test_values))
    _development_groups = np.asarray(groups[train_mask], dtype=np.int64)
    _test_groups = np.asarray(groups[test_mask], dtype=np.int64)
    return views, np.asarray(labels, dtype=np.int64), np.asarray(split_ids, dtype=np.int64)


def grouped_splits(labels, groups, seed):
    if not GROUP_OOF:
        from sklearn.model_selection import StratifiedKFold
        return list(StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    if len(np.unique(groups)) < OUTER_FOLDS:
        raise RuntimeError(f"Need at least {OUTER_FOLDS} development groups, found {len(np.unique(groups))}")
    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def profile_split(outer_fit, labels, groups, seed):
    if not GROUP_OOF:
        outer_labels = labels[outer_fit]
        class_count = len(np.unique(outer_labels))
        requested = max(int(np.ceil(0.18 * len(outer_fit))), class_count)
        if requested >= len(outer_fit):
            raise RuntimeError("Outer fold is too small to create a class-complete ROST profile split")
        return train_test_split(outer_fit, test_size=requested, stratify=outer_labels, random_state=seed)
    unique_groups = np.unique(groups[outer_fit])
    folds = min(5, len(unique_groups))
    if folds < 2:
        raise RuntimeError("A group-isolated profile split needs at least two outer-fit groups")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fit_local, profile_local = next(splitter.split(np.zeros(len(outer_fit)), labels[outer_fit], groups[outer_fit]))
    fit_idx, profile_idx = outer_fit[fit_local], outer_fit[profile_local]
    expected = set(range(NC))
    if set(np.unique(labels[fit_idx])) != expected or set(np.unique(labels[profile_idx])) != expected:
        raise RuntimeError("Group-isolated fit/profile split does not contain all 60 classes")
    return fit_idx, profile_idx


def fusion_group_split(values, test_size=0.20, stratify=None, random_state=None, **kwargs):
    values = np.asarray(values)
    if not GROUP_OOF or _development_groups is None or len(values) != len(_development_groups):
        return train_test_split(values, test_size=test_size, stratify=stratify, random_state=random_state, **kwargs)
    folds = max(2, min(int(round(1.0 / float(test_size))), len(np.unique(_development_groups))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fit_local, monitor_local = next(splitter.split(np.zeros(len(values)), stratify, _development_groups))
    return values[fit_local], values[monitor_local]


def cluster_bootstrap_delta(labels, candidate, reference, groups, seed, rounds=1000):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values = []
    candidate_prediction = candidate.argmax(1)
    reference_prediction = reference.argmax(1)
    for _ in range(rounds):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        values.append(
            accuracy_score(labels[indices], candidate_prediction[indices])
            - accuracy_score(labels[indices], reference_prediction[indices])
        )
    values = np.asarray(values)
    p_value = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    return {
        "delta": float(values.mean()), "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)), "bootstrap_p": min(1.0, p_value),
        "unit": "subject", "groups": int(len(unique_groups)),
    }


def prepare_view_pair(raw_train, raw_test, fit_idx, seed, path_prefix, tag):
    train_path = path_prefix.with_name(path_prefix.name + "_train.npy")
    test_path = path_prefix.with_name(path_prefix.name + "_test.npy")
    meta_path = path_prefix.with_name(path_prefix.name + "_meta.json")
    fit_hash = array_hash(fit_idx)
    if train_path.exists() and test_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("tag") == tag and meta.get("fit_hash") == fit_hash:
                return np.load(train_path, mmap_mode="r"), np.load(test_path, mmap_mode="r"), meta
        except (OSError, ValueError):
            pass

    scaler = StandardScaler()
    for start in range(0, len(fit_idx), TRANSFORM_BATCH):
        scaler.partial_fit(np.asarray(raw_train[fit_idx[start:start + TRANSFORM_BATCH]], dtype=np.float32))
    projector = None
    projected_scaler = None
    output_dim = raw_train.shape[1]
    if MAX_INPUT_DIM > 0 and raw_train.shape[1] > MAX_INPUT_DIM:
        projector = SparseRandomProjection(
            n_components=MAX_INPUT_DIM,  # pyright: ignore[reportArgumentType]
            density="auto",
            random_state=seed,
        )
        projector.fit(np.zeros((1, raw_train.shape[1]), dtype=np.float32))
        projected_scaler = StandardScaler()
        for start in range(0, len(fit_idx), TRANSFORM_BATCH):
            chunk = scaler.transform(np.asarray(raw_train[fit_idx[start:start + TRANSFORM_BATCH]], dtype=np.float32))
            projected_scaler.partial_fit(projector.transform(chunk))
        output_dim = MAX_INPUT_DIM

    train_output = np.lib.format.open_memmap(train_path, mode="w+", dtype=np.float32, shape=(len(raw_train), output_dim))
    test_output = np.lib.format.open_memmap(test_path, mode="w+", dtype=np.float32, shape=(len(raw_test), output_dim))
    for source, output in ((raw_train, train_output), (raw_test, test_output)):
        for start in range(0, len(source), TRANSFORM_BATCH):
            stop = min(start + TRANSFORM_BATCH, len(source))
            chunk = scaler.transform(np.asarray(source[start:stop], dtype=np.float32))
            if projector is not None:
                if projected_scaler is None:
                    raise RuntimeError("Projected feature scaler was not initialized")
                chunk = projected_scaler.transform(projector.transform(chunk))
            output[start:stop] = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        output.flush()
    meta = {
        "tag": tag, "fit_hash": fit_hash, "seed": seed,
        "input_dim": int(raw_train.shape[1]), "output_dim": int(output_dim),
        "train_rows": len(raw_train), "test_rows": len(raw_test),
        "projection": "SparseRandomProjection" if projector is not None else None,
        "fit_scope": "outer-fit gradient subset only",
    }
    core.save_json(meta, meta_path)
    return np.load(train_path, mmap_mode="r"), np.load(test_path, mmap_mode="r"), meta


def valid_expert_artifact(path, tag, variant, seed, fold, holdout, profile_idx, test_size):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("tag") != tag or value.get("variant") != variant or value.get("seed") != seed or value.get("fold") != fold:
            return False
        if set(value.get("holdout", {})) != set(VIEW_NAMES) or set(value.get("test", {})) != set(VIEW_NAMES):
            return False
        if not np.array_equal(value.get("holdout_idx"), holdout) or not np.array_equal(value.get("profile_idx"), profile_idx):
            return False
        probabilities = [*value["holdout"].values(), *value["test"].values()]
        if any(item.ndim != 2 or item.shape[1] != NC for item in probabilities):
            return False
        if any(not np.isfinite(item).all() or not np.allclose(item.sum(1), 1.0, atol=1e-4) for item in probabilities):
            return False
        return all(item.shape[0] == len(holdout) for item in value["holdout"].values()) and all(item.shape[0] == test_size for item in value["test"].values())
    except Exception:
        return False


def expert_fold(variant, config, seed, fold, raw_train, raw_test, labels, root, tag, prepared):
    training_variant = variant
    path = root / "experts" / f"{core.slug(training_variant)}_s{seed}_f{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    outer_fit, holdout = grouped_splits(labels, _development_groups, seed)[fold - 1]
    fit_idx, profile_idx = profile_split(outer_fit, labels, _development_groups, seed + fold)
    if valid_expert_artifact(path, tag, training_variant, seed, fold, holdout, profile_idx, len(next(iter(raw_test.values())))):
        return torch.load(path, map_location="cpu", weights_only=False)

    features, test_features, transforms = {}, {}, {}
    prepared_root = root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    for position, name in enumerate(VIEW_NAMES):
        prefix = prepared_root / f"{tag}_s{seed}_f{fold}_{name}"
        train_view, test_view, transform = prepare_view_pair(
            raw_train[name], raw_test[name], fit_idx, seed + fold * 10 + position, prefix, tag
        )
        features[name], test_features[name], transforms[name] = train_view, test_view, transform

    base_seed = seed + fold * 1000
    training_config = dict(config)
    # Full ROST is the exact strict counterpart of the NTU reference ROST
    # loss. Cumulative additions retain progressive activation; removals use
    # all retained reference terms throughout their matched continuation.
    if variant == "Full ROST" or variant.startswith("w/o"):
        training_config.pop("schedule", None)
    warmup_path = root / "experts" / f"shared_ce_warmup_s{seed}_f{fold}.pt"
    warmup = None
    if warmup_path.exists():
        try:
            candidate = torch.load(warmup_path, map_location="cpu", weights_only=False)
            if candidate.get("tag") == tag and candidate.get("fit_hash") == array_hash(fit_idx) and set(candidate.get("models", {})) == set(VIEW_NAMES) and set(candidate.get("optimizers", {})) == set(VIEW_NAMES):
                warmup = candidate
        except Exception:
            warmup = None
    if warmup is None:
        models, optimizers, ce_history = {}, {}, {}
        for position, name in enumerate(VIEW_NAMES):
            seed_all(base_seed + position)
            model = Observer(features[name].shape[1])
            models[name] = model.to(core.DEVICE)
            optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=1.5e-3, weight_decay=2e-4)
            ce_history[name] = []
        amp = torch.GradScaler("cuda", enabled=core.DEVICE.type == "cuda")
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            for name in VIEW_NAMES:
                ce_config = {"structure": False, "collapse": False, "complement": False, "joint": False}
                ce_history[name].append(native_train_epoch(models[name], features[name], labels, fit_idx, ce_config, 0.0, None, None, optimizers[name], amp))
            print(f"Shared CE warmup seed={seed} fold={fold}/{OUTER_FOLDS} epoch={epoch}/{PRETRAIN_EPOCHS}")
        warmup = {
            "tag": tag, "seed": seed, "fold": fold, "fit_hash": array_hash(fit_idx),
            "models": {name: core.clone_state(model) for name, model in models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "history": ce_history,
        }
        torch.save(warmup, warmup_path)
    models, optimizers = {}, {}
    for name in VIEW_NAMES:
        model = Observer(features[name].shape[1])
        model.load_state_dict(warmup["models"][name])
        models[name] = model.to(core.DEVICE)
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=1.5e-3, weight_decay=2e-4)
        optimizers[name].load_state_dict(warmup["optimizers"][name])
    ce_history = warmup["history"]
    ce_snapshots = [core.soft_confusion(labels[profile_idx], native_predict(models[name], features[name], profile_idx)) for name in VIEW_NAMES]
    amp = torch.GradScaler("cuda", enabled=core.DEVICE.type == "cuda")

    if training_variant == "CE-only":
        continuation = {}
        for cycle in range(1, ROST_EPOCHS + 1):
            for name in VIEW_NAMES:
                ce_config = {"structure": False, "collapse": False, "complement": False, "joint": False}
                continuation.setdefault(name, []).append(native_train_epoch(models[name], features[name], labels, fit_idx, ce_config, 0.0, None, None, optimizers[name], amp))
            print(f"CE-only seed={seed} fold={fold}/{OUTER_FOLDS} continuation={cycle}/{ROST_EPOCHS}")
        ce_history = {**ce_history, "matched_control": continuation}
        rost_history, selected_epoch = [], ROST_EPOCHS
        final_snapshots = [core.soft_confusion(labels[profile_idx], native_predict(models[name], features[name], profile_idx)) for name in VIEW_NAMES]
        snapshots = {"ce_warmup": ce_snapshots, "final": final_snapshots}
    else:
        rost_history, snapshots = [], {"ce_warmup": ce_snapshots}
        matrices = ce_snapshots
        for cycle in range(1, ROST_EPOCHS + 1):
            controller_prob = {name: native_predict(models[name], features[name], profile_idx) for name in VIEW_NAMES}
            matrices = [core.soft_confusion(labels[profile_idx], controller_prob[name]) for name in VIEW_NAMES]
            active = {name: component_active(training_config, name, cycle) for name in ("structure", "collapse", "complement", "joint")}
            cycle_config = {**active, "schedule": training_config.get("schedule")}
            losses = {}
            for position, name in enumerate(VIEW_NAMES):
                rescue = native_rescue_weight_matrix(matrices, position)
                peers = [torch.from_numpy(matrix.astype(np.float32)) for peer, matrix in enumerate(matrices) if peer != position]
                losses[name] = native_train_epoch(models[name], features[name], labels, fit_idx, cycle_config, cycle / ROST_EPOCHS, rescue, peers, optimizers[name], amp)
            controller_prob = {name: native_predict(models[name], features[name], profile_idx) for name in VIEW_NAMES}
            matrices = [core.soft_confusion(labels[profile_idx], controller_prob[name]) for name in VIEW_NAMES]
            joint_profile = native_cm_jsri(matrices)
            rost_history.append({
                "epoch": cycle, "train_loss": float(np.mean([item["loss"] for item in losses.values()])),
                "score": float(np.mean([accuracy_score(labels[profile_idx], value.argmax(1)) for value in controller_prob.values()])),
                "profile": {"jsri": joint_profile["CM_JSRI"], "experts": [{"sri": native_cm_sri(matrix)["CM_SRI"], "row_entropy": 0.0, "column_entropy": 0.0, "effective_rank": 0.0} for matrix in matrices], "direct_redundancy": joint_profile["Q_red"], "graph_redundancy": joint_profile["Q_red"], "rescue": joint_profile["Q_rescue"]},
                "probe": {"f1": 0.0, "nll": 0.0}, "weights": {f"active_{name}": float(value) for name, value in active.items()},
            })
            print(f"{training_variant} seed={seed} fold={fold}/{OUTER_FOLDS} cycle={cycle}/{ROST_EPOCHS} CM-JSRI={joint_profile['CM_JSRI']:.4f}")
        selected_epoch = ROST_EPOCHS
        snapshots["final"] = matrices

    output = {
        "tag": tag, "variant": training_variant, "seed": seed, "fold": fold,
        "holdout_idx": np.asarray(holdout, dtype=np.int64),
        "profile_idx": np.asarray(profile_idx, dtype=np.int64),
        "holdout": {}, "test": {}, "ce_history": ce_history, "rost_history": rost_history,
        "rost_epoch": selected_epoch, "probe_state": None, "profile_snapshots": snapshots,
        "models": {}, "preprocessing": transforms,
        "parameter_counts": {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()},
    }
    profile_posteriors = []
    for name, model in models.items():
        output["holdout"][name] = native_predict(model, features[name], holdout).astype(np.float32)
        output["test"][name] = native_predict(model, test_features[name]).astype(np.float32)
        profile_posteriors.append(native_predict(model, features[name], profile_idx))
        output["models"][name] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output["final_profile"] = core.profile([core.soft_confusion(labels[profile_idx], item) for item in profile_posteriors])
    output["final_probe"] = {"f1": 0.0, "nll": 0.0}
    torch.save(output, path)
    return output


def configure_core():
    core.PROJECT_ROOT = PROJECT_ROOT
    core.SOURCE_PATH = SOURCE_PATH
    core.FEATURE_CACHE = FEATURE_CACHE
    core.BASE_CKPT_DIR = BASE_CKPT_DIR
    core.BASE_LOG_DIR = BASE_LOG_DIR
    core.EXP_NAME = EXP_NAME
    core.NC = NC
    core.ACTION_NAMES = [f"A{index + 1:03d}" for index in range(NC)]
    core.EPS = EPS
    core.SEED = SEED
    core.PIPELINE_SEEDS = PIPELINE_SEEDS
    core.FUSION_SEEDS = FUSION_SEEDS
    core.OUTER_FOLDS = OUTER_FOLDS
    core.PRETRAIN_EPOCHS = PRETRAIN_EPOCHS
    core.ROST_EPOCHS = ROST_EPOCHS
    core.RCF_EPOCHS = RCF_EPOCHS
    core.EXPERT_PATIENCE = EXPERT_PATIENCE
    core.FUSION_PATIENCE = FUSION_PATIENCE
    core.PROFILE_INTERVAL = PROFILE_INTERVAL
    core.MAX_INPUT_DIM = MAX_INPUT_DIM
    core.EXPERT_BATCH = EXPERT_BATCH
    core.PLOTS = PLOTS
    core.TERMINAL_TOP = TERMINAL_TOP
    core.VARIANT_FILTER = [item.strip() for item in os.environ.get("DOME_X_VARIANTS", "").split(",") if item.strip()]
    core.RCF_FILTER = [item.strip() for item in os.environ.get("DOME_X_RCF_VARIANTS", "").split(",") if item.strip()]
    core.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    core.Observer = Observer
    core.ROST_VARIANTS = ROST_VARIANTS
    core.RCF_VARIANTS = RCF_VARIANTS
    core.rescue_weight_matrix = native_rescue_weight_matrix
    core.seed_all = seed_all
    core.valid_expert_artifact = valid_expert_artifact
    core.expert_fold = expert_fold
    core.train_test_split = fusion_group_split


def validate_numerics():
    seed_all(SEED)
    labels = np.repeat(np.arange(NC), 2)
    rng = np.random.default_rng(SEED)
    experts = rng.dirichlet(np.ones(NC), size=(len(labels), len(VIEW_NAMES))).astype(np.float32)
    matrices = [core.soft_confusion(labels, experts[:, index]) for index in range(experts.shape[1])]
    transport = core.transport_from_confusion(matrices[0], np.bincount(labels, minlength=NC) + 1.0)
    if not np.allclose(transport.sum(0), 1.0, atol=1e-6):
        raise AssertionError("Bias transport is not column stochastic")
    for name, config in core.RCF_VARIANTS.items():
        model = core.ComponentRCF(matrices, config)
        output = model(torch.from_numpy(experts[:8]))
        if output.shape != (8, NC) or not torch.isfinite(output).all() or not torch.allclose(output.sum(1), torch.ones(8), atol=1e-5):
            raise AssertionError(f"Invalid RCF output for {name}")
    observer = Observer(48)
    logits = observer(torch.randn(16, 48))
    torch.nn.functional.cross_entropy(logits, torch.arange(16) % NC).backward()
    print("NTU DOME-X validation passed: observer backward, all RCF variants, probability normalization, transport columns")


def rcf_parameter_count(matrices, config):
    model = core.ComponentRCF(matrices, config)
    return sum(parameter.numel() for parameter in model.enabled_parameters())


def posterior_is_valid(path, tag, labels):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as value:
            if str(value["tag"].item()) != tag or not np.array_equal(value["labels"], labels):
                return False
            for name in VIEW_NAMES:
                oof, test = value[f"oof_{name}"], value[f"test_{name}"]
                if oof.shape != (len(labels), NC) or test.ndim != 3 or test.shape[0] != OUTER_FOLDS or test.shape[2] != NC:
                    return False
                if not np.isfinite(oof).all() or not np.isfinite(test).all():
                    return False
                if not np.allclose(oof.sum(1), 1.0, atol=1e-4) or not np.allclose(test.sum(2), 1.0, atol=1e-4):
                    return False
        return True
    except Exception:
        return False


def save_posterior(path, tag, labels, oof, test_by_fold):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"tag": np.asarray(tag), "labels": labels}
    payload.update({f"oof_{name}": value for name, value in oof.items()})
    payload.update({f"test_{name}": np.stack(test_by_fold[name]) for name in VIEW_NAMES})
    np.savez_compressed(path, **payload)


def main():
    configure_core()
    seed_all(SEED)
    if VALIDATE_ONLY:
        validate_numerics()
        return
    raw_features, all_labels, split_ids = load_features()
    tag = config_hash()
    version, root, log = output_dirs(tag)
    started = time.time()
    train_values = NTU60_XSUB_TRAIN_SUBJECTS if PROTOCOL == "xsub" else NTU60_XVIEW_TRAIN_CAMERAS
    test_values = NTU60_XSUB_TEST_SUBJECTS if PROTOCOL == "xsub" else NTU60_XVIEW_TEST_CAMERAS
    train_index = np.flatnonzero(np.isin(split_ids, sorted(train_values)))
    test_index = np.flatnonzero(np.isin(split_ids, sorted(test_values)))
    raw_train = {name: IndexedRows(raw_features[name], train_index) for name in VIEW_NAMES}
    raw_test = {name: IndexedRows(raw_features[name], test_index) for name in VIEW_NAMES}
    labels = all_labels[train_index]
    if set(np.unique(labels)) != set(range(NC)):
        raise RuntimeError("NTU RGB+D development partition does not contain all 60 classes")

    active_rost = {name: config for name, config in core.ROST_VARIANTS.items() if core.selected(name, core.VARIANT_FILTER)}
    active_rcf = {name: config for name, config in core.RCF_VARIANTS.items() if core.selected(name, core.RCF_FILTER)}
    if not active_rost:
        raise ValueError(f"DOME_X_VARIANTS selected no known variant: {core.VARIANT_FILTER}")
    if core.RCF_FILTER and not active_rcf:
        raise ValueError(f"DOME_X_RCF_VARIANTS selected no known variant: {core.RCF_FILTER}")
    add_chain = ["Average/Base", "+ Class-wise Reliability", "+ PEACE Calibration", "+ Learnable Bias Transport", "+ Disagreement Refinement"]
    component_execution = set(active_rcf)
    for name in active_rcf:
        if name in add_chain:
            component_execution.update(add_chain[:add_chain.index(name) + 1])
    executable_rcf = {name: config for name, config in core.RCF_VARIANTS.items() if name in component_execution}
    manifest = {
        "version": version, "tag": tag, "source": str(SOURCE_PATH), "core": str(CORE_PATH),
        "feature_cache": str(FEATURE_CACHE), "protocol": PROTOCOL, "classes": NC,
        "pipeline_seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS,
        "outer_folds": OUTER_FOLDS, "group_oof": GROUP_OOF,
        "pretrain_epochs": PRETRAIN_EPOCHS, "rost_epochs": ROST_EPOCHS,
        "rcf_epochs": RCF_EPOCHS, "max_input_dim": MAX_INPUT_DIM,
        "experts": list(VIEW_NAMES), "view_dimensions": {name: int(raw_train[name].shape[1]) for name in VIEW_NAMES},
        "view_selection": "Frozen CE-OOF selection from the reference NTU run by default; override only through DOME_X_NTU_VIEWS before an experiment.",
        "rost_variants": active_rost, "rcf_variants": active_rcf,
        "data": {"development": len(labels), "test": len(test_index)},
        "train_partition": sorted(train_values), "test_partition": sorted(test_values),
        "label_isolation": "Outer holdout labels never affect their expert. Inner profile labels control ROST/checkpoints. Official test labels are indexed only after every requested expert, fusion, component checkpoint and prediction is frozen.",
        "fairness": "All variants share official split, OOF groups, profile groups, preprocessing, observer architecture, CE warmup checkpoint, continuation budget, seeds, selected views and fusion protocol.",
        "fusion_protocol": "Every fusion is trained from development OOF posteriors. Test inference is performed per expert fold and then averaged for learned RCF component models; baseline inputs use the same fold-ensemble posterior mean.",
        "resume": "Prepared fold features, shared CE warmups, expert variants and RCF selection/refit checkpoints are configuration-tagged. Posterior artifacts are integrity-checked before fusion.",
    }
    core.save_json(manifest, log / "manifest.json")
    print(f"NTU RGB+D 60 DOME-X ablation v{version} tag={tag} protocol={PROTOCOL} device={core.DEVICE} development={len(labels)} test={len(test_index)} views={VIEW_NAMES}")

    pipeline_index = {}
    trajectory_path = log / "controller_trajectory.csv"
    if trajectory_path.exists():
        trajectory_path.unlink()
    prepared = {}
    for variant, config in active_rost.items():
        pipeline_index[variant] = {}
        for pipeline_seed in PIPELINE_SEEDS:
            posterior_path = root / "posteriors" / f"posterior_{core.slug(variant)}_s{pipeline_seed}.npz"
            oof = {name: np.zeros((len(labels), NC), dtype=np.float32) for name in VIEW_NAMES}
            test_by_fold = {name: [] for name in VIEW_NAMES}
            profiles, snapshots, trajectories, fold_probes, selected_epochs, expert_parameters = [], [], [], [], [], []
            for fold in range(1, OUTER_FOLDS + 1):
                artifact = expert_fold(variant, config, pipeline_seed, fold, raw_train, raw_test, labels, root, tag, prepared)
                for name in VIEW_NAMES:
                    oof[name][artifact["holdout_idx"]] = artifact["holdout"][name]
                    test_by_fold[name].append(artifact["test"][name])
                profiles.append(artifact["final_profile"])
                snapshots.append(artifact["profile_snapshots"])
                fold_probes.append(artifact["final_probe"])
                selected_epochs.append(artifact["rost_epoch"])
                expert_parameters.append(artifact["parameter_counts"])
                for row in artifact["rost_history"]:
                    trajectories.append({
                        "variant": variant, "pipeline_seed": pipeline_seed, "fold": fold, "epoch": row["epoch"],
                        "loss": row["train_loss"], "score": row["score"], "jsri": row["profile"]["jsri"],
                        "probe_f1": row["probe"]["f1"], "probe_nll": row["probe"]["nll"],
                        "mean_sri": float(np.mean([item["sri"] for item in row["profile"]["experts"]])),
                        "row_entropy": float(np.mean([item["row_entropy"] for item in row["profile"]["experts"]])),
                        "column_entropy": float(np.mean([item["column_entropy"] for item in row["profile"]["experts"]])),
                        "effective_rank": float(np.mean([item["effective_rank"] for item in row["profile"]["experts"]])),
                        "direct_redundancy": row["profile"]["direct_redundancy"], "graph_redundancy": row["profile"]["graph_redundancy"],
                        "rescue": row["profile"]["rescue"], **{f"lambda_{name}": value for name, value in row["weights"].items()},
                    })
            core.append_trajectory(trajectory_path, trajectories)
            if any(np.any(np.isclose(value.sum(1), 0.0)) for value in oof.values()):
                raise RuntimeError(f"Incomplete OOF posterior for {variant}, seed={pipeline_seed}")
            save_posterior(posterior_path, tag, labels, oof, test_by_fold)
            pipeline_index[variant][pipeline_seed] = {
                "path": str(posterior_path), "profiles": profiles, "snapshots": snapshots,
                "fold_probes": fold_probes, "selected_epochs": selected_epochs,
                "expert_parameters": expert_parameters,
            }
            print(f"Completed expert pipeline: {variant} seed={pipeline_seed}")

    # Freeze every expert posterior and every fusion/component prediction before
    # official test labels enter the evaluation phase.
    del raw_features, raw_train, raw_test
    if not KEEP_PREPARED:
        shutil.rmtree(root / "prepared", ignore_errors=True)
    gc.collect()
    evaluation_root = root / "evaluation_predictions"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    evaluation_index, component_index, profile_snapshots = [], [], {}
    for variant, regimes in pipeline_index.items():
        for pipeline_seed, metadata in regimes.items():
            posterior_path = Path(metadata["path"])
            if not posterior_is_valid(posterior_path, tag, labels):
                raise RuntimeError(f"Invalid posterior artifact: {posterior_path}")
            with np.load(posterior_path, allow_pickle=False) as data:
                oof = {name: np.asarray(data[f"oof_{name}"], dtype=np.float32) for name in VIEW_NAMES}
                test = {name: np.asarray(data[f"test_{name}"], dtype=np.float32) for name in VIEW_NAMES}
            x = np.stack([oof[name] for name in VIEW_NAMES], 1).astype(np.float32)
            test_folds = [np.stack([test[name][fold] for name in VIEW_NAMES], 1).astype(np.float32) for fold in range(OUTER_FOLDS)]
            mean_experts = np.mean(test_folds, 0)
            matrices = [core.soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
            diagnostics = core.profile(matrices)
            average = core.normalize(np.mean([core.normalize(item.mean(1)) for item in test_folds], 0))
            product = core.normalize(np.mean([core.normalize(np.exp(np.log(np.clip(item, EPS, 1.0)).mean(1))) for item in test_folds], 0))
            reliability = np.stack([core.confusion_reliability(matrix) for matrix in matrices])
            reliability /= reliability.sum(0, keepdims=True)
            weighted = core.normalize(np.mean([core.normalize((item * reliability[None]).sum(1)) for item in test_folds], 0))
            logistic_outputs, logistic_params = core.logistic_stacking(x, labels, test_folds, pipeline_seed)
            logistic = core.normalize(np.mean(logistic_outputs, 0))
            mlp_outputs, mlp_params = core.train_mlp_stacking(x, labels, test_folds, pipeline_seed)
            mlp = core.normalize(np.mean(mlp_outputs, 0))
            context = f"{PROTOCOL}_{variant}_pipeline_s{pipeline_seed}"
            component_regime = variant in ("CE-only", "Full ROST") and pipeline_seed == PIPELINE_SEEDS[0]
            needs_full_checkpoints = any(name == "Full RCF" or name.startswith("w/o") for name in active_rcf)
            required_fusion_seeds = FUSION_SEEDS if component_regime and needs_full_checkpoints else FUSION_SEEDS[:1]
            full_outputs, full_infos, full_states = {}, {}, {}
            for fusion_seed in required_fusion_seeds:
                fold_outputs, info, state = core.fit_rcf_variant(
                    "Full RCF", core.RCF_VARIANTS["Full RCF"], x, labels, test_folds,
                    fusion_seed, root, context, tag,
                )
                full_outputs[fusion_seed] = core.normalize(np.mean(fold_outputs, 0))
                full_infos[fusion_seed], full_states[fusion_seed] = info, state
            full = full_outputs[FUSION_SEEDS[0]]
            profile_snapshots[f"{core.slug(variant)}_s{pipeline_seed}"] = {
                "oof": matrices, "fold_profiles": metadata["profiles"], "fold_snapshots": metadata["snapshots"],
            }
            expert_outputs = {f"Submodel {name}": core.normalize(test[name].mean(0)) for name in VIEW_NAMES}
            result_set = {
                **expert_outputs, "Average": average, "Product": product, "Weighted Average": weighted,
                "Logistic Stacking": logistic, "MLP Stacking": mlp, "Full RCF": full,
            }
            parameters = {
                **{f"Submodel {name}": int(np.mean([item[name] for item in metadata["expert_parameters"]])) for name in VIEW_NAMES},
                "Average": 0, "Product": 0, "Weighted Average": reliability.size,
                "Logistic Stacking": logistic_params, "MLP Stacking": mlp_params,
                "Full RCF": int(np.mean([item["params"] for item in full_infos.values()])),
            }
            mean_sri = float(np.mean([item["sri"] for item in diagnostics["experts"]]))
            probe_f1 = float(np.mean([item["f1"] for item in metadata["fold_probes"]]))
            probe_nll = float(np.mean([item["nll"] for item in metadata["fold_probes"]]))
            selected_epoch = float(np.mean(metadata["selected_epochs"]))
            result_paths = {}
            for fusion_name, output in result_set.items():
                key = f"{core.slug(variant)}_s{pipeline_seed}_{core.slug(fusion_name)}"
                path = evaluation_root / f"{key}.npy"
                np.save(path, output.astype(np.float32))
                result_paths[fusion_name] = str(path)
            mean_experts_path = evaluation_root / f"{core.slug(variant)}_s{pipeline_seed}_expert_tensor.npy"
            np.save(mean_experts_path, mean_experts.astype(np.float32))
            evaluation_index.append({
                "variant": variant, "pipeline_seed": pipeline_seed, "result_paths": result_paths,
                "parameters": parameters, "sri": mean_sri, "jsri": diagnostics["jsri"],
                "probe_f1": probe_f1, "probe_nll": probe_nll, "selected_epoch": selected_epoch,
            })

            if component_regime:
                component_context = f"{PROTOCOL}_component_{variant}_pipeline_s{pipeline_seed}"
                for fusion_seed in FUSION_SEEDS:
                    outputs_by_name, states_by_name = {}, {}
                    seed_records = []
                    previous_selection_state, previous_refit_state = None, None
                    for component_name, component_config in executable_rcf.items():
                        if component_name == "Full RCF":
                            output, state = full_outputs[fusion_seed], full_states[fusion_seed]
                            info = {**full_infos[fusion_seed], "source": "actual_pipeline_full_checkpoint"}
                        else:
                            if component_name.startswith("+"):
                                selection_initial_state, refit_initial_state, source = previous_selection_state, previous_refit_state, "incremental_add_chain"
                            elif component_name.startswith("w/o"):
                                selection_initial_state = full_infos[fusion_seed]["selection_state"]
                                refit_initial_state, source = full_states[fusion_seed], "full_checkpoint_removal"
                            else:
                                selection_initial_state, refit_initial_state, source = None, None, "fixed_average"
                            fold_outputs, info, state = core.fit_rcf_variant(
                                component_name, component_config, x, labels, test_folds, fusion_seed,
                                root, component_context, tag, selection_initial_state, refit_initial_state, source,
                            )
                            output = core.normalize(np.mean(fold_outputs, 0))
                        outputs_by_name[component_name], states_by_name[component_name] = output, state
                        if component_name.startswith("+"):
                            previous_selection_state, previous_refit_state = info["selection_state"], state
                        if component_name not in active_rcf:
                            continue
                        component_path = evaluation_root / f"rcf_{core.slug(variant)}_fs{fusion_seed}_{core.slug(component_name)}.npy"
                        np.save(component_path, output.astype(np.float32))
                        seed_records.append({
                            "expert_regime": variant, "pipeline_seed": pipeline_seed, "fusion_seed": fusion_seed,
                            "variant": component_name, "prediction_path": str(component_path),
                            "average_path": result_paths["Average"], "mean_experts_path": str(mean_experts_path),
                            "parameters": rcf_parameter_count(matrices, component_config),
                            "latency_ms_per_sample": core.latency_ms(state, matrices, component_config, x),
                            "selected_epoch": info["selected_epoch"], "source": info["source"],
                        })
                    full_component = outputs_by_name.get("Full RCF", full_outputs.get(fusion_seed, average))
                    full_component_path = evaluation_root / f"rcf_{core.slug(variant)}_fs{fusion_seed}_full_reference.npy"
                    np.save(full_component_path, full_component.astype(np.float32))
                    for record in seed_records:
                        record["full_reference_path"] = str(full_component_path)
                    component_index.extend(seed_records)

            del oof, test, x, test_folds, mean_experts
            gc.collect()

    test_labels = all_labels[test_index]
    if set(np.unique(test_labels)) != set(range(NC)):
        raise RuntimeError("NTU RGB+D official test partition does not contain all 60 classes")
    summary_rows, rcf_rows, saved_predictions, bootstrap = [], [], {}, []
    for record in evaluation_index:
        outputs = {name: np.load(path, mmap_mode="r") for name, path in record["result_paths"].items()}
        best_expert = max(core.metrics(test_labels, output)["acc"] for name, output in outputs.items() if name.startswith("Submodel "))
        for fusion_name, output in outputs.items():
            metric = core.metrics(test_labels, output)
            summary_rows.append({
                "variant": record["variant"], "pipeline_seed": record["pipeline_seed"], "fusion": fusion_name, **metric,
                "fusion_gain": metric["acc"] - best_expert, "sri": record["sri"], "jsri": record["jsri"],
                "probe_f1": record["probe_f1"], "probe_nll": record["probe_nll"],
                "selected_epoch": record["selected_epoch"], "parameters": record["parameters"].get(fusion_name, 0),
            })
            key = f"{core.slug(record['variant'])}_s{record['pipeline_seed']}_{core.slug(fusion_name)}"
            if SAVE_ALL_PREDICTIONS or (record["variant"] in ("CE-only", "Full ROST") and fusion_name == "Full RCF"):
                saved_predictions[key] = np.asarray(output, dtype=np.float32)
            core.plot_cm(test_labels, output, log / f"cm_{key}.png", f"{record['variant']} | seed {record['pipeline_seed']} | {fusion_name}")

    for record in component_index:
        output = np.load(record["prediction_path"], mmap_mode="r")
        average = np.load(record["average_path"], mmap_mode="r")
        mean_experts = np.load(record["mean_experts_path"], mmap_mode="r")
        full_reference = np.load(record["full_reference_path"], mmap_mode="r")
        diagnostic = core.disagreement_metrics(test_labels, mean_experts, average, output)
        rcf_rows.append({
            **{key: value for key, value in record.items() if not key.endswith("_path")},
            **core.metrics(test_labels, output), **diagnostic,
            "per_class_recall_delta_vs_average": core.per_class_recall_delta(test_labels, average, output),
            "per_class_recall_delta_vs_full": core.per_class_recall_delta(test_labels, full_reference, output),
        })

    summary_frame = core.pd.DataFrame(summary_rows)
    rcf_frame = core.pd.DataFrame(rcf_rows)
    summary_frame.to_csv(log / "rost_pipeline_summary.csv", index=False)
    rcf_frame.to_csv(log / "rcf_component_ablation.csv", index=False)
    numeric = ["acc", "f1", "precision", "recall", "ece", "adaptive_ece", "classwise_ece", "brier", "nll", "fusion_gain", "sri", "jsri", "probe_f1", "probe_nll", "selected_epoch", "parameters", "latency_ms_per_sample"]
    aggregate = summary_frame.groupby(["variant", "fusion"])[[name for name in numeric if name in summary_frame]].agg(["mean", "std"]).reset_index()
    aggregate.columns = [" | ".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in aggregate.columns]
    aggregate.insert(0, "table", "ROST pipeline")
    if not rcf_frame.empty:
        rcf_aggregate = rcf_frame.groupby(["expert_regime", "variant"])[[name for name in numeric if name in rcf_frame]].agg(["mean", "std"]).reset_index()
        rcf_aggregate.columns = [" | ".join(str(part) for part in column if part) if isinstance(column, tuple) else column for column in rcf_aggregate.columns]
        rcf_aggregate.insert(0, "table", "RCF component ablation")
        aggregate = core.pd.concat([aggregate, rcf_aggregate], ignore_index=True, sort=False)
    aggregate.to_csv(log / "aggregate_mean_std.csv", index=False)

    for pipeline_seed in PIPELINE_SEEDS:
        ce_key = f"ce_only_s{pipeline_seed}_full_rcf"
        rost_key = f"full_rost_s{pipeline_seed}_full_rcf"
        if ce_key in saved_predictions and rost_key in saved_predictions:
            bootstrap.append({
                "pipeline_seed": pipeline_seed, "comparison": "Full ROST Full RCF - CE-only Full RCF",
                **cluster_bootstrap_delta(test_labels, saved_predictions[rost_key], saved_predictions[ce_key], _test_groups, pipeline_seed),
            })
    correlation_rows = []
    for fusion_name, group in summary_frame[~summary_frame["fusion"].str.startswith("Submodel")].groupby("fusion"):
        if len(group) >= 3 and group["jsri"].nunique() > 1 and group["fusion_gain"].nunique() > 1:
            correlation_rows.append({
                "fusion": fusion_name, "pearson_jsri_gain": group["jsri"].corr(group["fusion_gain"], method="pearson"),
                "spearman_jsri_gain": group["jsri"].corr(group["fusion_gain"], method="spearman"), "n": len(group),
            })
    core.pd.DataFrame(correlation_rows).to_csv(log / "recoverability_gain_correlation.csv", index=False)
    np.savez_compressed(root / "final_seed_predictions.npz", test_labels=test_labels, **saved_predictions)
    core.save_json(profile_snapshots, log / "profile_snapshots.json")
    core.save_json({"manifest": manifest, "rost_rows": summary_rows, "rcf_rows": rcf_rows, "paired_bootstrap": bootstrap, "correlations": correlation_rows}, log / "results.json")
    core.save_json(bootstrap, log / "paired_bootstrap.json")
    core.print_terminal_summary(summary_frame, rcf_frame, log)
    print("NTU RGB+D 60 DOME-X ablation complete")
    print(f"Checkpoints={root}")
    print(f"Logs={log}")
    print(f"TimeMinutes={(time.time() - started) / 60.0:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate dependencies and cached inputs")
    return parser.parse_args()


def check_environment():
    required_api = ("normalize", "metrics", "fit_rcf_variant", "soft_confusion")
    missing = [name for name in required_api if not hasattr(core, name)]
    if missing:
        raise RuntimeError(f"Ablation core is missing functions: {missing}")
    if not SOURCE_PATH.is_file() or not CORE_PATH.is_file():
        raise FileNotFoundError("NTU source or shared ablation core is missing")
    print(
        f"NTU RGB+D ablation check passed: source={SOURCE_PATH.name} "
        f"cache={FEATURE_CACHE.exists()} protocol={PROTOCOL}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        check_environment()
    else:
        main()
