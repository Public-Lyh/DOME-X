"""Run UTD-MHAD component ablations under the odd/even subject split."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection


PLACEHOLDER_ROOT = Path("your path")
WORKSPACE_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
PROJECT_ROOT = WORKSPACE_ROOT / "Code" / "UTD-MHAD"
SCRIPT_ROOT = PROJECT_ROOT / "CongTouYve"
SOURCE_PATH = SCRIPT_ROOT / "utd_dome_x_train_02.py"
FEATURE_CACHE = PROJECT_ROOT / "cache" / "utd_mhad_ce_rost_features_v1.npz"
BASE_CKPT_DIR = PROJECT_ROOT / "checkpoints"
BASE_LOG_DIR = PROJECT_ROOT / "logs"
EXP_NAME = "DOME_X_UTD_MHAD_ABLATION"

NC = 27
ACTION_NAMES = [f"a{index + 1}" for index in range(NC)]
TRAIN_SUBJECTS = {1, 3, 5, 7}
TEST_SUBJECTS = {2, 4, 6, 8}
EPS = 1e-8
SEED = int(os.environ.get("DOME_X_SEED", "42"))
PIPELINE_SEEDS = [SEED + index for index in range(int(os.environ.get("DOME_X_PIPELINE_SEEDS", "3")))]
FUSION_SEEDS = [SEED + 1000 + index for index in range(int(os.environ.get("DOME_X_FUSION_SEEDS", "5")))]
OUTER_FOLDS = int(os.environ.get("DOME_X_OUTER_FOLDS", "3"))
PRETRAIN_EPOCHS = int(os.environ.get("DOME_X_PRETRAIN_EPOCHS", "60"))
ROST_EPOCHS = int(os.environ.get("DOME_X_ROST_EPOCHS", "40"))
RCF_EPOCHS = int(os.environ.get("DOME_X_RCF_EPOCHS", "180"))
EXPERT_PATIENCE = int(os.environ.get("DOME_X_EXPERT_PATIENCE", "18"))
FUSION_PATIENCE = int(os.environ.get("DOME_X_FUSION_PATIENCE", "30"))
PROFILE_INTERVAL = int(os.environ.get("DOME_X_PROFILE_INTERVAL", "3"))
MAX_INPUT_DIM = int(os.environ.get("DOME_X_MAX_INPUT_DIM", "0"))
EXPERT_BATCH = int(os.environ.get("DOME_X_EXPERT_BATCH", "192"))
PLOTS = os.environ.get("DOME_X_PLOTS", "1") != "0"
TERMINAL_TOP = int(os.environ.get("DOME_X_TERMINAL_TOP", "35"))
VARIANT_FILTER = [item.strip() for item in os.environ.get("DOME_X_VARIANTS", "").split(",") if item.strip()]
RCF_FILTER = [item.strip() for item in os.environ.get("DOME_X_RCF_VARIANTS", "").split(",") if item.strip()]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ROST_VARIANTS = {
    "CE-only": {"structure": False, "collapse": False, "complement": False, "joint": False},
    "CE+Structure": {"structure": True, "collapse": False, "complement": False, "joint": False, "schedule": "progressive"},
    "CE+Structure+Anti-collapse": {"structure": True, "collapse": True, "complement": False, "joint": False, "schedule": "progressive"},
    "CE+Structure+Anti-collapse+Complementarity": {"structure": True, "collapse": True, "complement": True, "joint": False, "schedule": "progressive"},
    "Full ROST": {"structure": True, "collapse": True, "complement": True, "joint": True, "schedule": "progressive"},
    "w/o Structure": {"structure": False, "collapse": True, "complement": True, "joint": True, "schedule": "progressive"},
    "w/o Anti-collapse": {"structure": True, "collapse": False, "complement": True, "joint": True, "schedule": "progressive"},
    "w/o Complementarity": {"structure": True, "collapse": True, "complement": False, "joint": True, "schedule": "progressive"},
    "w/o Joint Recovery": {"structure": True, "collapse": True, "complement": True, "joint": False, "schedule": "progressive"},
}

RCF_VARIANTS = {
    "Average/Base": {"reliability": False, "calibration": False, "transport": False, "refinement": False},
    "+ Class-wise Reliability": {"reliability": True, "calibration": False, "transport": False, "refinement": False},
    "+ PEACE Calibration": {"reliability": True, "calibration": True, "transport": False, "refinement": False},
    "+ Learnable Bias Transport": {"reliability": True, "calibration": True, "transport": True, "refinement": False},
    "+ Disagreement Refinement": {"reliability": True, "calibration": True, "transport": True, "refinement": True},
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def selected(name, filters):
    return not filters or name in filters or slug(name) in {slug(item) for item in filters}


def normalize(value):
    value = np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    value = np.clip(value, EPS, None)
    return value / np.maximum(value.sum(1, keepdims=True), EPS)


def rows(value):
    value = np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    value = np.clip(value, 0.0, None)
    return value / np.maximum(value.sum(1, keepdims=True), EPS)


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_hash(state):
    if state is None:
        return "none"
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()[:16]


def array_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def config_hash():
    cache_hash = hashlib.sha256(FEATURE_CACHE.read_bytes()).hexdigest()[:16] if FEATURE_CACHE.exists() else "missing"
    payload = {
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "source": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16],
        "feature_cache": cache_hash,
        "pipeline_seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS,
        "folds": OUTER_FOLDS, "pretrain_epochs": PRETRAIN_EPOCHS,
        "rost_epochs": ROST_EPOCHS, "rcf_epochs": RCF_EPOCHS,
        "expert_patience": EXPERT_PATIENCE, "fusion_patience": FUSION_PATIENCE,
        "profile_interval": PROFILE_INTERVAL, "expert_batch": EXPERT_BATCH,
        "max_input_dim": MAX_INPUT_DIM, "rost_variants": ROST_VARIANTS,
        "rcf_variants": RCF_VARIANTS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def output_dirs(tag):
    BASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    requested = os.environ.get("DOME_X_UTD_ABLATION_VERSION", os.environ.get("DOME_X_ABLATION_VERSION"))
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


def load_features():
    if not FEATURE_CACHE.exists():
        # Cache generation is delegated to the reference implementation only on first use.
        spec = importlib.util.spec_from_file_location("utd_dome_source", SOURCE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load the UTD-MHAD source: {SOURCE_PATH}")
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        source.build_or_load_features()
    if not FEATURE_CACHE.exists():
        raise RuntimeError(f"UTD-MHAD feature cache was not created: {FEATURE_CACHE}")
    with np.load(FEATURE_CACHE, allow_pickle=False) as data:
        return ({name: data[name].astype(np.float32) for name in ("skeleton", "inertial", "rgb")}, data["labels"].astype(np.int64), data["subjects"].astype(np.int64))


def prepare_view(features, fit_idx, seed):
    features = np.asarray(features, dtype=np.float32)
    raw_scaler = StandardScaler().fit(features[fit_idx])
    scaled = np.asarray(raw_scaler.transform(features), dtype=np.float32)
    projector = None
    final_scaler = None
    if MAX_INPUT_DIM > 0 and scaled.shape[1] > MAX_INPUT_DIM:
        projector = SparseRandomProjection(
            n_components=MAX_INPUT_DIM,  # pyright: ignore[reportArgumentType]
            density="auto",
            random_state=seed,
        )
        projector.fit(scaled[fit_idx])
        projected = projector.transform(scaled).astype(np.float32)
        final_scaler = StandardScaler().fit(projected[fit_idx])
        transformed = final_scaler.transform(projected)
    else:
        transformed = scaled
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    transform = {"raw_scaler": raw_scaler, "projector": projector, "final_scaler": final_scaler, "input_dim": features.shape[1], "output_dim": transformed.shape[1]}
    return transformed, transform


def ece(labels, proba, adaptive=False):
    confidence = proba.max(1)
    correct = proba.argmax(1) == labels
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, 16)) if adaptive else np.linspace(0.0, 1.0, 16)
    edges[0], edges[-1] = 0.0, 1.0
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & (confidence <= high if index == len(edges) - 2 else confidence < high)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def metrics(labels, proba):
    proba = normalize(proba)
    prediction = proba.argmax(1)
    one_hot = np.eye(NC)[labels]
    class_ece = []
    for klass in range(NC):
        binary = np.stack([1.0 - proba[:, klass], proba[:, klass]], 1)
        class_ece.append(ece((labels == klass).astype(np.int64), binary))
    return {
        "acc": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, average="macro", zero_division="warn")),
        "precision": float(precision_score(labels, prediction, average="macro", zero_division="warn")),
        "recall": float(recall_score(labels, prediction, average="macro", zero_division="warn")),
        "ece": ece(labels, proba), "adaptive_ece": ece(labels, proba, True),
        "classwise_ece": float(np.mean(class_ece)),
        "brier": float(np.square(proba - one_hot).sum(1).mean()),
        "nll": float(log_loss(labels, proba, labels=np.arange(NC))),
    }


def soft_confusion(labels, proba):
    matrix = np.zeros((NC, NC), dtype=np.float64)
    np.add.at(matrix, labels, normalize(proba))
    counts = np.bincount(labels, minlength=NC).astype(np.float64)
    matrix /= np.maximum(counts[:, None], 1.0)
    matrix[counts == 0] = 1.0 / NC
    return rows(matrix)


def js_rows_numpy(matrix):
    matrix = rows(matrix)
    left, right = matrix[:, None], matrix[None]
    middle = 0.5 * (left + right)
    return 0.5 * ((left * np.log((left + EPS) / (middle + EPS))).sum(-1) + (right * np.log((right + EPS) / (middle + EPS))).sum(-1)) / np.log(2.0)


def effective_rank(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    probability = singular / max(singular.sum(), EPS)
    return float(np.exp(-(probability * np.log(probability + EPS)).sum()) / min(matrix.shape))


def profile(matrices):
    matrices = [rows(matrix) for matrix in matrices]
    experts = []
    relations = []
    for matrix in matrices:
        entropy = -(matrix * np.log(matrix + EPS)).sum(1) / np.log(NC)
        usage = matrix.mean(0)
        relation = js_rows_numpy(matrix)
        relations.append(relation)
        top3 = np.sort(matrix, axis=1)[:, -3:].sum(1)
        shape = float(np.clip(1.0 - np.mean(np.maximum(0.0, 0.14 - entropy) ** 2 + np.maximum(0.0, entropy - 0.82) ** 2 + np.maximum(0.0, 0.48 - top3) ** 2), 0.0, 1.0))
        smooth = rows(matrix + 1e-3)
        reverse = smooth.T / np.maximum(smooth.T.sum(1, keepdims=True), EPS)
        decode = float(np.diag(smooth @ reverse).mean())
        separation = float(relation[~np.eye(NC, dtype=bool)].mean())
        column_entropy = float(-(usage * np.log(usage + EPS)).sum() / np.log(NC))
        rank = effective_rank(matrix)
        experts.append({
            "row_entropy": float(entropy.mean()),
            "top3_mass": float(top3.mean()), "separation": separation,
            "column_entropy": column_entropy, "effective_rank": rank,
            "diagonal": float(np.diag(matrix).mean()), "decode": decode,
            "sri": float(0.22 * shape + 0.28 * separation + 0.15 * column_entropy + 0.15 * rank + 0.20 * decode),
        })
    joint = np.concatenate(matrices, 1)
    denominator = np.maximum(np.linalg.norm(joint, axis=1)[:, None] * np.linalg.norm(joint, axis=1)[None], EPS)
    joint_distance = 1.0 - (joint @ joint.T) / denominator
    pairs = [(left, right) for left in range(len(matrices)) for right in range(left + 1, len(matrices))]
    direct = np.mean([(matrices[a].ravel() @ matrices[b].ravel()) / max(np.linalg.norm(matrices[a]) * np.linalg.norm(matrices[b]), EPS) for a, b in pairs])
    graph = np.mean([(relations[a].ravel() @ relations[b].ravel()) / max(np.linalg.norm(relations[a]) * np.linalg.norm(relations[b]), EPS) for a, b in pairs])
    rescue = []
    for current in range(len(matrices)):
        other = np.mean([relations[index] for index in range(len(matrices)) if index != current], 0)
        weight = np.exp(-other / 0.12)
        np.fill_diagonal(weight, 0.0)
        rescue.append(float((weight * relations[current]).sum() / max(weight.sum(), EPS)))
    jsri = 0.30 * joint_distance[~np.eye(NC, dtype=bool)].mean() + 0.20 * effective_rank(joint) + 0.30 * np.mean(rescue) + 0.20 * (1.0 - 0.5 * (direct + graph))
    return {"experts": experts, "direct_redundancy": float(direct), "graph_redundancy": float(graph), "rescue": float(np.mean(rescue)), "jsri": float(np.clip(jsri, 0.0, 1.0))}


class Observer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        # Match the reference observer capacity for every original feature view.
        hidden = int(np.clip(2 ** round(np.log2(max(np.sqrt(dimension * NC) * 2, 96))), 128, 512))
        bottleneck = max(NC * 2, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.18),
            nn.Linear(hidden, bottleneck), nn.LayerNorm(bottleneck), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(bottleneck, NC),
        )

    def forward(self, x):
        return self.net(x)


class RecoveryProbe(nn.Module):
    def __init__(self, modalities):
        super().__init__()
        self.linear = nn.Linear(modalities * NC, NC)

    def forward(self, posterior):
        return self.linear(posterior.flatten(1))


def predict_observer(model, features, indices):
    model = model.to(DEVICE).eval()
    output = []
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
        for start in range(0, len(indices), 512):
            index = np.asarray(indices[start:start + 512])
            logits = model(torch.from_numpy(features[index]).float().to(DEVICE)).float()
            output.append(F.softmax(logits, 1).cpu().numpy())
    return normalize(np.concatenate(output))


def class_weights(labels):
    count = np.bincount(labels, minlength=NC).astype(np.float32) + 1.0
    weight = 1.0 / np.sqrt(count)
    return torch.from_numpy(weight / weight.mean()).float().to(DEVICE)


def train_semantic(model, features, labels, fit_idx, profile_idx, epochs, seed, tag, lr=1.8e-3):
    seed_all(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=2e-5)
    amp = torch.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    weight = class_weights(labels[fit_idx])
    initial = metrics(labels[profile_idx], predict_observer(model, features, profile_idx))
    best = clone_state(model)
    best_score = 0.55 * initial["f1"] + 0.45 * initial["acc"]
    stale = 0
    history = [{"epoch": 0, "loss": None, "val_acc": initial["acc"], "val_f1": initial["f1"], "val_nll": initial["nll"]}]
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(fit_idx)
        total = 0.0
        for start in range(0, len(order), EXPERT_BATCH):
            index = order[start:start + EXPERT_BATCH]
            x = torch.from_numpy(features[index]).float().to(DEVICE)
            y = torch.from_numpy(labels[index]).long().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(x + 0.01 * torch.randn_like(x))
                loss = F.cross_entropy(logits.float(), y, weight=weight, label_smoothing=0.03)
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            amp.step(optimizer)
            amp.update()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        validation = predict_observer(model, features, profile_idx)
        metric = metrics(labels[profile_idx], validation)
        score = 0.55 * metric["f1"] + 0.45 * metric["acc"]
        history.append({"epoch": epoch, "loss": total / len(fit_idx), "val_acc": metric["acc"], "val_f1": metric["f1"], "val_nll": metric["nll"]})
        if score > best_score + 1e-5:
            best, best_score, stale = clone_state(model), score, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 12 == 0:
            print(f"{tag} epoch={epoch}/{epochs} val_acc={metric['acc']:.4f} val_f1={metric['f1']:.4f}")
        if epoch >= min(14, epochs) and stale >= EXPERT_PATIENCE:
            break
    model.load_state_dict(best)
    return model.cpu(), history


def torch_confusion(labels, posterior, anchor):
    one_hot = F.one_hot(labels, NC).to(posterior.dtype)
    counts = one_hot.sum(0)
    batch = one_hot.T @ posterior / counts.clamp_min(1.0).unsqueeze(1)
    anchor = anchor.to(posterior.device, posterior.dtype)
    matrix = torch.where((counts > 0).unsqueeze(1), 0.60 * batch + 0.40 * anchor, anchor)
    return matrix / matrix.sum(1, keepdim=True).clamp_min(EPS)


def torch_js_rows(matrix):
    matrix = matrix.clamp_min(EPS)
    matrix = matrix / matrix.sum(1, keepdim=True)
    left, right = matrix[:, None], matrix[None]
    middle = 0.5 * (left + right)
    return 0.5 * ((left * (left.log() - middle.log())).sum(-1) + (right * (right.log() - middle.log())).sum(-1)) / np.log(2.0)


def controller_weights(current, config, epoch, semantic_drop=0.0):
    progress = epoch / max(ROST_EPOCHS, 1)
    progressive = config.get("schedule") == "progressive"
    structure_active = config["structure"]
    collapse_active = config["collapse"] and (not progressive or progress >= 1.0 / 3.0)
    complement_active = config["complement"] and (not progressive or progress >= 2.0 / 3.0)
    joint_active = config["joint"] and (not progressive or progress >= 3.0 / 4.0)
    entropy = np.mean([item["row_entropy"] for item in current["experts"]])
    usage = np.mean([item["column_entropy"] for item in current["experts"]])
    rank = np.mean([item["effective_rank"] for item in current["experts"]])
    guard = float(np.clip(semantic_drop / 0.08, 0.0, 1.0))
    structure_weight = (0.006 + 0.032 * progress) * (1.0 + max(0.0, entropy - 0.80) + max(0.0, 0.15 - entropy)) if structure_active else 0.0
    collapse_weight = (0.006 + 0.026 * progress) * (1.0 + max(0.0, 0.90 - usage) + max(0.0, 0.82 - rank)) if collapse_active else 0.0
    complement_weight = (0.004 + 0.022 * progress) * (1.0 + max(0.0, current["direct_redundancy"] - 0.82) + max(0.0, current["graph_redundancy"] - 0.82)) if complement_active else 0.0
    joint_weight = (0.008 + 0.030 * progress) * (1.0 + max(0.0, 0.58 - current["jsri"])) if joint_active else 0.0
    attenuation = 1.0 - 0.70 * guard
    return {
        "ce": 1.0,
        "anchor": 0.10 + 0.04 * progress + 0.22 * guard,
        "structure": structure_weight * attenuation,
        "collapse": collapse_weight * attenuation,
        "complement": complement_weight * attenuation,
        "joint": joint_weight * attenuation,
        "semantic_guard": guard,
    }


def rost_loss(logits, target, semantic_weight, teacher_posterior, profile_matrix, peer_matrices, config, weights, probe, all_posterior, position):
    posterior = F.softmax(logits.float(), 1)
    matrix = torch_confusion(target, posterior, profile_matrix)
    semantic = F.cross_entropy(logits.float(), target, weight=semantic_weight, label_smoothing=0.03)
    teacher = teacher_posterior.to(posterior.device, posterior.dtype).clamp_min(EPS)
    middle = 0.5 * (posterior + teacher)
    anchor = 0.5 * (
        F.kl_div(middle.log(), posterior, reduction="batchmean")
        + F.kl_div(middle.log(), teacher, reduction="batchmean")
    )
    distance = torch_js_rows(matrix)
    eye = torch.eye(NC, dtype=torch.bool, device=logits.device)
    entropy = -(matrix * matrix.clamp_min(EPS).log()).sum(1) / np.log(NC)
    top3 = matrix.topk(3, 1).values.sum(1)
    structure = F.relu(0.14 - entropy).square().mean() + F.relu(entropy - 0.82).square().mean() + F.relu(0.48 - top3).square().mean() + 0.40 * torch.exp(-distance[~eye] / 0.14).mean()
    usage = matrix.mean(0)
    usage_entropy = -(usage * usage.clamp_min(EPS).log()).sum() / np.log(NC)
    singular = torch.linalg.svdvals(matrix.float())
    singular = singular / singular.sum().clamp_min(EPS)
    rank_loss = 1.0 + (singular * singular.clamp_min(EPS).log()).sum() / np.log(NC)
    collapse = F.relu(0.60 - usage_entropy).square() + F.relu(usage.max() - 0.36).square() + 0.22 * rank_loss
    complement = torch.zeros((), device=logits.device)
    if peer_matrices:
        terms = []
        for peer in peer_matrices:
            peer = peer.to(logits.device, logits.dtype)
            peer_distance = torch_js_rows(peer)
            rescue_weight = torch.exp(-peer_distance / 0.12)
            rescue_weight.fill_diagonal_(0.0)
            direct = F.cosine_similarity(matrix.flatten(), peer.flatten(), dim=0)
            graph = F.cosine_similarity(distance.flatten(), peer_distance.flatten(), dim=0)
            rescue = -(rescue_weight * distance).sum() / rescue_weight.sum().clamp_min(EPS)
            terms.append(0.32 * direct + 0.30 * graph + 0.38 * rescue)
        complement = torch.stack(terms).mean()
    joint = torch.zeros((), device=logits.device)
    probe_loss = torch.zeros((), device=logits.device)
    marginal = torch.zeros((), device=logits.device)
    fusion_recovery = torch.zeros((), device=logits.device)
    if config["joint"]:
        smooth = (matrix + 1e-3) / (matrix + 1e-3).sum(1, keepdim=True)
        reverse = smooth.T / smooth.T.sum(1, keepdim=True).clamp_min(EPS)
        decode = -torch.diag(smooth @ reverse).clamp_min(EPS).log().mean()
        probe_loss = F.cross_entropy(probe(all_posterior), target)
        fusion_logits = torch.log(all_posterior.clamp_min(EPS)).mean(1)
        fusion_recovery = F.cross_entropy(fusion_logits, target)
        masked = all_posterior.detach().clone()
        masked[:, position] = 1.0 / NC
        masked_loss = F.cross_entropy(probe(masked), target).detach()
        marginal = F.relu(probe_loss - masked_loss + 0.04)
        joint = 0.25 * (decode / np.log(NC)) + 0.30 * (probe_loss / np.log(NC)) + 0.30 * (fusion_recovery / np.log(NC)) + 0.15 * marginal
    loss = weights["ce"] * semantic + weights["anchor"] * anchor + weights["structure"] * structure + weights["collapse"] * collapse + weights["complement"] * complement + weights["joint"] * joint
    parts = {"semantic": float(semantic.detach()), "anchor": float(anchor.detach()), "structure": float(structure.detach()), "collapse": float(collapse.detach()), "complement": float(complement.detach()), "joint": float(joint.detach()), "probe": float(probe_loss.detach()), "fusion_recovery": float(fusion_recovery.detach()), "marginal": float(marginal.detach())}
    return loss, parts


def probe_step(probe, models, features, labels, indices, optimizer, steps=1):
    names = list(models)
    probe.train()
    for _ in range(steps):
        order = np.random.permutation(indices)
        for start in range(0, len(order), EXPERT_BATCH):
            index = order[start:start + EXPERT_BATCH]
            target = torch.from_numpy(labels[index]).long().to(DEVICE)
            with torch.no_grad():
                posterior = torch.stack([F.softmax(models[name](torch.from_numpy(features[name][index]).float().to(DEVICE)).float(), 1) for name in names], 1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(posterior), target)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 2.0)
            optimizer.step()


def evaluate_probe(probe, models, features, labels, indices):
    names = list(models)
    posterior = [predict_observer(models[name], features[name], indices) for name in names]
    probe = probe.to(DEVICE).eval()
    with torch.no_grad():
        output = F.softmax(probe(torch.from_numpy(np.stack(posterior, 1).astype(np.float32)).to(DEVICE)), 1).cpu().numpy()
    return metrics(labels[indices], output), posterior


def train_rost_models(models, features, labels, fit_idx, profile_idx, config, seed, fold):
    seed_all(seed)
    names = list(models)
    models = {name: model.to(DEVICE) for name, model in models.items()}
    all_indices = np.arange(len(labels))
    teacher_posteriors = {name: predict_observer(models[name], features[name], all_indices).astype(np.float32) for name in names}
    initial_posteriors = [predict_observer(models[name], features[name], profile_idx) for name in names]
    warmup_f1 = float(np.mean([metrics(labels[profile_idx], item)["f1"] for item in initial_posteriors]))
    warmup_metrics = [metrics(labels[profile_idx], item) for item in initial_posteriors]
    warmup_score = 0.50 * np.mean([item["acc"] + item["f1"] for item in warmup_metrics])
    ema = [soft_confusion(labels[profile_idx], item) for item in initial_posteriors]
    probe = RecoveryProbe(len(names)).to(DEVICE)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=9e-4, weight_decay=2e-4)
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=3e-4, weight_decay=3e-4) for name in names}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ROST_EPOCHS, eta_min=1e-5) for name, optimizer in optimizers.items()}
    best = {"score": float(warmup_score), "models": {name: clone_state(model) for name, model in models.items()}, "probe": clone_state(probe), "epoch": 0}
    history, stale = [], 0
    snapshots = {"ce_warmup": [item.copy() for item in ema]}
    current = profile(ema)
    semantic_drop = 0.0
    weights = controller_weights(current, config, 1, semantic_drop)
    semantic_weight = class_weights(labels[fit_idx])
    eligible_start = 1
    if config.get("schedule") == "progressive":
        if config["joint"]:
            eligible_start = max(1, int(np.ceil(3.0 * ROST_EPOCHS / 4.0)))
        elif config["complement"]:
            eligible_start = max(1, int(np.ceil(2.0 * ROST_EPOCHS / 3.0)))
        elif config["collapse"]:
            eligible_start = max(1, int(np.ceil(ROST_EPOCHS / 3.0)))
    minimum_final_phase = min(4, max(1, ROST_EPOCHS - eligible_start + 1))
    for epoch in range(1, ROST_EPOCHS + 1):
        if epoch == 1 or epoch % PROFILE_INTERVAL == 0:
            values = [predict_observer(models[name], features[name], profile_idx) for name in names]
            ema = [rows(0.85 * old + 0.15 * soft_confusion(labels[profile_idx], new)) for old, new in zip(ema, values)]
            current = profile(ema)
            weights = controller_weights(current, config, epoch, semantic_drop)
        for model in models.values():
            model.eval()
        probe_step(probe, models, features, labels, fit_idx, probe_optimizer, steps=1)
        order = np.random.permutation(fit_idx)
        totals = {key: 0.0 for key in ("loss", "semantic", "anchor", "structure", "collapse", "complement", "joint", "probe", "fusion_recovery", "marginal")}
        seen = 0
        for start in range(0, len(order), EXPERT_BATCH):
            index = order[start:start + EXPERT_BATCH]
            target = torch.from_numpy(labels[index]).long().to(DEVICE)
            for position, name in enumerate(names):
                for other_name, model in models.items():
                    model.train(other_name == name)
                for parameter in probe.parameters():
                    parameter.requires_grad_(False)
                optimizers[name].zero_grad(set_to_none=True)
                posterior = []
                current_logits = None
                for other_position, other_name in enumerate(names):
                    x = torch.from_numpy(features[other_name][index]).float().to(DEVICE)
                    if other_position == position:
                        current_logits = models[other_name](x).float()
                        posterior.append(F.softmax(current_logits, 1))
                    else:
                        with torch.no_grad():
                            posterior.append(F.softmax(models[other_name](x).float(), 1))
                all_posterior = torch.stack(posterior, 1)
                peers = [torch.from_numpy(ema[peer].astype(np.float32)) for peer in range(len(names)) if peer != position]
                teacher = torch.from_numpy(teacher_posteriors[name][index])
                loss, parts = rost_loss(current_logits, target, semantic_weight, teacher, torch.from_numpy(ema[position].astype(np.float32)), peers, config, weights, probe, all_posterior, position)
                loss.backward()
                nn.utils.clip_grad_norm_(models[name].parameters(), 3.0)
                optimizers[name].step()
                for parameter in probe.parameters():
                    parameter.requires_grad_(True)
                totals["loss"] += float(loss.detach()) * len(index)
                for key, value in parts.items():
                    totals[key] += value * len(index)
                seen += len(index)
        for scheduler in schedulers.values():
            scheduler.step()
        probe_metric, validation = evaluate_probe(probe, models, features, labels, profile_idx)
        matrices = [soft_confusion(labels[profile_idx], item) for item in validation]
        ema = [rows(0.85 * old + 0.15 * new) for old, new in zip(ema, matrices)]
        current = profile(ema)
        expert_metrics = [metrics(labels[profile_idx], item) for item in validation]
        semantic_drop = max(0.0, warmup_f1 - float(np.mean([item["f1"] for item in expert_metrics])) - 0.01)
        # A common semantic selector prevents recovery diagnostics from leaking
        # joint-recovery pressure into variants where that component is absent.
        score = 0.50 * np.mean([item["acc"] + item["f1"] for item in expert_metrics])
        recoverability_score = score + 0.025 * probe_metric["f1"] + 0.015 * current["jsri"]
        selection_allowed = not config["joint"] or score >= warmup_score - 0.015
        selection_score = recoverability_score if config["joint"] else score
        row = {
            "epoch": epoch, "score": float(score), "selection_score": float(selection_score), "profile": current,
            "weights": weights, "probe": probe_metric, "experts": expert_metrics,
            **{f"train_{key}": value / max(seen, 1) for key, value in totals.items()},
        }
        history.append(row)
        if epoch == 1:
            snapshots["rost_early"] = [item.copy() for item in ema]
        if epoch >= max(2, ROST_EPOCHS // 2) and "rost_middle" not in snapshots:
            snapshots["rost_middle"] = [item.copy() for item in ema]
        if epoch == eligible_start and eligible_start > 1 and not config["joint"]:
            best["score"], stale = -np.inf, 0
        if epoch >= eligible_start and selection_allowed and selection_score > best["score"] + 1e-5:
            best = {"score": float(selection_score), "models": {name: clone_state(model) for name, model in models.items()}, "probe": clone_state(probe), "epoch": epoch}
            stale = 0
        elif epoch >= eligible_start:
            stale += 1
        if epoch == 1 or epoch % PROFILE_INTERVAL == 0:
            print(f"ROST fold={fold} epoch={epoch}/{ROST_EPOCHS} score={score:.4f} probe_f1={probe_metric['f1']:.4f} JSRI={current['jsri']:.4f}")
        if epoch >= max(min(12, ROST_EPOCHS), eligible_start + minimum_final_phase - 1) and stale >= EXPERT_PATIENCE:
            break
    for name in names:
        models[name].load_state_dict(best["models"][name])
    probe.load_state_dict(best["probe"])
    _, final_posteriors = evaluate_probe(probe, models, features, labels, profile_idx)
    snapshots["final"] = [soft_confusion(labels[profile_idx], item) for item in final_posteriors]
    return {name: model.cpu() for name, model in models.items()}, probe.cpu(), history, best["epoch"], snapshots


def train_diagnostic_probe(models, features, labels, fit_idx, profile_idx, seed):
    seed_all(seed)
    models = {name: model.to(DEVICE).eval() for name, model in models.items()}
    probe = RecoveryProbe(len(models)).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=2e-4)
    best, best_score, stale = clone_state(probe), np.inf, 0
    for _ in range(50):
        probe_step(probe, models, features, labels, fit_idx, optimizer)
        metric, _ = evaluate_probe(probe, models, features, labels, profile_idx)
        score = -metric["f1"] + 0.002 * metric["nll"]
        if score < best_score - 1e-5:
            best, best_score, stale = clone_state(probe), score, 0
        else:
            stale += 1
        if stale >= 8:
            break
    probe.load_state_dict(best)
    return probe.cpu()


def valid_expert_artifact(path, tag, variant, seed, fold, holdout, profile_idx, test_size):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("tag") != tag or value.get("variant") != variant or value.get("seed") != seed or value.get("fold") != fold:
            return False
        if set(value.get("holdout", {})) != {"skeleton", "inertial", "rgb"} or set(value.get("test", {})) != {"skeleton", "inertial", "rgb"}:
            return False
        if not np.array_equal(value.get("holdout_idx"), holdout) or not np.array_equal(value.get("profile_idx"), profile_idx):
            return False
        if any(item.shape[0] != len(holdout) for item in value["holdout"].values()) or any(item.shape[0] != test_size for item in value["test"].values()):
            return False
        if not all(np.isfinite(item).all() and item.ndim == 2 and item.shape[1] == NC and np.allclose(item.sum(1), 1.0, atol=1e-4) for item in [*value["holdout"].values(), *value["test"].values()]):
            return False
        return "parameter_counts" in value and "final_probe" in value and "preprocessing" in value
    except Exception:
        return False


def expert_fold(variant, config, seed, fold, raw_train, raw_test, labels, root, tag, prepared):
    path = root / "experts" / f"{slug(variant)}_s{seed}_f{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    outer = list(StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    outer_fit, holdout = outer[fold - 1]
    fit_idx, profile_idx = train_test_split(outer_fit, test_size=0.20, stratify=labels[outer_fit], random_state=seed * 100 + fold)
    test_size = len(next(iter(raw_test.values())))
    if valid_expert_artifact(path, tag, variant, seed, fold, holdout, profile_idx, test_size):
        return torch.load(path, map_location="cpu", weights_only=False)
    features, transforms = {}, {}
    for position, name in enumerate(raw_train):
        cache_key = (seed, fold, name)
        if cache_key not in prepared:
            joined = np.concatenate([raw_train[name], raw_test[name]], 0)
            transformed, transform = prepare_view(joined, fit_idx, seed + fold * 10 + position)
            prepared[cache_key] = (transformed[:len(labels)], transformed[len(labels):], transform)
        features[name] = prepared[cache_key][0]
        transforms[name] = prepared[cache_key][2]
    models, ce_history = {}, {}
    # Independent retraining uses matched random seeds across variants.
    base_seed = seed + fold * 1000
    for position, name in enumerate(features):
        seed_all(base_seed + position)
        model = Observer(features[name].shape[1])
        print(f"{variant} seed={seed} fold={fold}/{OUTER_FOLDS}: CE warmup {name}")
        model, history = train_semantic(model, features[name], labels, fit_idx, profile_idx, PRETRAIN_EPOCHS, base_seed + position, f"CE {name}")
        models[name], ce_history[name] = model, history
    ce_snapshots = [soft_confusion(labels[profile_idx], predict_observer(models[name], features[name], profile_idx)) for name in models]
    if variant == "CE-only":
        continuation = {}
        for position, name in enumerate(features):
            models[name], continuation[name] = train_semantic(models[name], features[name], labels, fit_idx, profile_idx, ROST_EPOCHS, base_seed + 100 + position, f"CE control {name}", lr=3e-4)
        ce_history["matched_control"] = continuation
        probe = train_diagnostic_probe(models, features, labels, fit_idx, profile_idx, base_seed + 500)
        rost_history, selected_epoch = [], ROST_EPOCHS
        final_snapshots = [soft_confusion(labels[profile_idx], predict_observer(models[name], features[name], profile_idx)) for name in models]
        snapshots = {"ce_warmup": ce_snapshots, "final": final_snapshots}
    else:
        print(f"{variant} seed={seed} fold={fold}/{OUTER_FOLDS}: controller-driven ROST")
        models, probe, rost_history, selected_epoch, snapshots = train_rost_models(models, features, labels, fit_idx, profile_idx, config, base_seed + 500, fold)
    output = {
        "tag": tag, "variant": variant, "seed": seed, "fold": fold,
        "holdout_idx": np.asarray(holdout, dtype=np.int64),
        "profile_idx": np.asarray(profile_idx, dtype=np.int64),
        "holdout": {}, "test": {}, "ce_history": ce_history, "rost_history": rost_history,
        "rost_epoch": selected_epoch, "probe_state": probe.state_dict(), "profile_snapshots": snapshots,
        "models": {}, "preprocessing": transforms,
        "parameter_counts": {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()},
    }
    profile_posteriors = []
    for name, model in models.items():
        train_view, test_view, _ = prepared[(seed, fold, name)]
        output["holdout"][name] = predict_observer(model, train_view, holdout).astype(np.float32)
        output["test"][name] = predict_observer(model, test_view, np.arange(len(test_view))).astype(np.float32)
        profile_posteriors.append(predict_observer(model, train_view, profile_idx))
        output["models"][name] = model.state_dict()
    output["final_profile"] = profile([soft_confusion(labels[profile_idx], item) for item in profile_posteriors])
    output["final_probe"], _ = evaluate_probe(probe, models, features, labels, profile_idx)
    torch.save(output, path)
    return output


def transport_from_confusion(matrix, class_prior=None):
    """P(Y=true | predicted class), represented as true x predicted columns."""
    matrix = rows(matrix)
    if class_prior is None:
        class_prior = np.full(NC, 1.0 / NC, dtype=np.float64)
    class_prior = np.asarray(class_prior, dtype=np.float64)
    class_prior = class_prior / max(class_prior.sum(), EPS)
    mass = matrix * class_prior[:, None] + 1e-3 / NC
    return mass / np.maximum(mass.sum(0, keepdims=True), EPS)


def confusion_reliability(matrix):
    matrix = rows(matrix + 1e-3)
    reverse = matrix.T / np.maximum(matrix.T.sum(1, keepdims=True), EPS)
    return np.clip(np.diag(matrix @ reverse), 0.02, 1.0)


def posterior_context(x):
    mean = x.mean(1)
    std = x.std(1, unbiased=False)
    maximum = x.max(1).values
    entropy = -(x * x.clamp_min(EPS).log()).sum(2) / np.log(NC)
    top = x.topk(2, dim=2).values
    margin = top[:, :, 0] - top[:, :, 1]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float()
    return torch.cat([x.flatten(1), mean, std, maximum, entropy, margin, disagreement], 1)


class ComponentRCF(nn.Module):
    def __init__(self, matrices, config, class_prior=None):
        super().__init__()
        self.config = dict(config)
        self.modalities = len(matrices)
        matrix = np.stack(matrices).astype(np.float32)
        reliability = np.stack([confusion_reliability(item) for item in matrix]).astype(np.float32)
        reliability /= reliability.sum(0, keepdims=True)
        transport = np.stack([transport_from_confusion(item, class_prior) for item in matrix]).astype(np.float32)
        identity_softplus = float(np.log(np.expm1(1.0)))
        self.reliability_logits = nn.Parameter(torch.log(torch.from_numpy(reliability)))
        self.log_temperature = nn.Parameter(torch.full((self.modalities, NC), identity_softplus))
        self.scale = nn.Parameter(torch.full((self.modalities, NC), identity_softplus))
        self.bias = nn.Parameter(torch.zeros(self.modalities, NC))
        self.transport_logits = nn.Parameter(torch.log(torch.from_numpy(transport).clamp_min(EPS)))
        self.transport_gate = nn.Parameter(torch.full((self.modalities, NC), -3.2))
        context_dim = self.modalities * NC + 3 * NC + 3 * self.modalities
        width = 96
        self.path_logits = nn.Parameter(torch.log(torch.tensor([0.18, 0.22, 0.18, 0.07, 0.35], dtype=torch.float32)))
        self.refinement_strength = nn.Parameter(torch.tensor(-1.10, dtype=torch.float32))
        path_output = nn.Linear(width, 5)
        gate_output = nn.Linear(width, 1)
        residual_output = nn.Linear(width, NC, bias=False)
        self.path_gate = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), path_output)
        self.gate = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), gate_output)
        self.residual = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), residual_output)
        nn.init.zeros_(path_output.weight)
        nn.init.zeros_(path_output.bias)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, -3.0)
        nn.init.zeros_(residual_output.weight)

    def enabled_parameters(self):
        enabled = {
            "reliability_logits": self.config["reliability"],
            "log_temperature": self.config["calibration"], "scale": self.config["calibration"], "bias": self.config["calibration"],
            "transport_logits": self.config["transport"], "transport_gate": self.config["transport"],
            "path_logits": self.config["refinement"], "refinement_strength": self.config["refinement"], "path_gate": self.config["refinement"],
            "gate": self.config["refinement"], "residual": self.config["refinement"],
        }
        parameters = []
        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(name == key or name.startswith(key + ".") for key, active in enabled.items() if active)
            if parameter.requires_grad:
                parameters.append(parameter)
        return parameters

    def forward(self, x):
        x = x.clamp_min(EPS)
        if self.config["calibration"]:
            temperature = F.softplus(self.log_temperature).unsqueeze(0).clamp_min(0.15)
            calibrated = F.softmax(torch.log(x) / temperature * F.softplus(self.scale).unsqueeze(0) + self.bias.unsqueeze(0), 2)
        else:
            calibrated = x
        if self.config["transport"]:
            transport = F.softmax(self.transport_logits, dim=1)
            transported = torch.einsum("nmk,myk->nmy", calibrated, transport)
            transported = transported / transported.sum(2, keepdim=True).clamp_min(EPS)
            disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True).unsqueeze(2)
            uncertainty = (1.0 - x.max(2).values.mean(1, keepdim=True)).unsqueeze(2)
            active = 0.10 + 0.55 * disagreement + 0.35 * uncertainty
            strength = torch.sigmoid(self.transport_gate).unsqueeze(0) * active
            recovered = (1.0 - strength) * calibrated + strength * transported
            recovered = recovered / recovered.sum(2, keepdim=True).clamp_min(EPS)
        else:
            recovered = calibrated
        if self.config["reliability"]:
            weight = F.softmax(self.reliability_logits, 0).unsqueeze(0)
        else:
            weight = torch.full((1, self.modalities, NC), 1.0 / self.modalities, dtype=x.dtype, device=x.device)
        arithmetic = (weight * recovered).sum(1)
        geometric = F.softmax((weight * torch.log(recovered.clamp_min(EPS))).sum(1), 1)
        calibrated_geometric = F.softmax((weight * torch.log(calibrated.clamp_min(EPS))).sum(1), 1)
        raw = (weight * x).sum(1)
        raw_product = F.softmax(torch.log(x).mean(1), 1)
        if not self.config["refinement"]:
            return arithmetic / arithmetic.sum(1, keepdim=True).clamp_min(EPS)
        context = posterior_context(x)
        paths = torch.stack([arithmetic, geometric, calibrated_geometric, raw, raw_product], 1)
        mixture = F.softmax(self.path_logits.unsqueeze(0) + 0.35 * self.path_gate(context), 1)
        structured = (mixture.unsqueeze(2) * paths).sum(1)
        disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True)
        uncertainty = 1.0 - x.max(2).values.mean(1, keepdim=True)
        active = 0.08 + 0.62 * disagreement + 0.30 * uncertainty
        gate = torch.sigmoid(self.gate(context)) * active
        refined = F.softmax(torch.log(structured.clamp_min(EPS)) + 0.08 * torch.tanh(self.residual(context)), 1)
        learned = (1.0 - gate) * structured + gate * refined
        correction = torch.sigmoid(self.refinement_strength)
        output = (1.0 - correction) * raw_product + correction * learned
        return output / output.sum(1, keepdim=True).clamp_min(EPS)


def rcf_regularization(model):
    value = torch.zeros((), device=DEVICE)
    if model.config["reliability"]:
        value = value + 0.002 * model.reliability_logits.square().mean()
    if model.config["calibration"]:
        value = value + 0.002 * (model.log_temperature.square().mean() + model.scale.square().mean() + model.bias.square().mean())
    if model.config["transport"]:
        value = value + 0.001 * model.transport_logits.square().mean() + 0.002 * (torch.sigmoid(model.transport_gate) - 0.04).square().mean()
    if model.config["refinement"]:
        value = value + 0.001 * model.path_logits.square().mean() + 0.001 * model.refinement_strength.square()
    return value


def rcf_predict(model, x):
    model = model.to(DEVICE).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            output.append(model(torch.from_numpy(x[start:start + 512]).float().to(DEVICE)).cpu().numpy())
    return normalize(np.concatenate(output))


def train_rcf(model, x, labels, fit_idx, monitor_idx, seed):
    seed_all(seed)
    model = model.to(DEVICE)
    parameters = model.enabled_parameters()
    if not parameters:
        return model.cpu(), 0, []
    optimizer = torch.optim.AdamW(parameters, lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RCF_EPOCHS, eta_min=2e-5)
    features = torch.from_numpy(x).float().to(DEVICE)
    target = torch.from_numpy(labels).long().to(DEVICE)
    baseline = metrics(labels[monitor_idx], rcf_predict(model, x[monitor_idx]))
    best, best_score, best_epoch, stale = clone_state(model), -baseline["f1"] + 0.003 * baseline["nll"], 0, 0
    history = []
    for epoch in range(1, RCF_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_x = features[fit_idx]
        temperature = torch.empty((len(train_x), train_x.shape[1], 1), device=DEVICE).uniform_(0.90, 1.10)
        augmented = F.softmax(torch.log(train_x.clamp_min(EPS)) / temperature + 0.015 * torch.randn_like(train_x), 2)
        output = model(augmented)
        raw_product = F.softmax(torch.log(train_x.clamp_min(EPS)).mean(1), 1).detach()
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), target[fit_idx]) + 0.05 * F.mse_loss(output, F.one_hot(target[fit_idx], NC).float()) + 0.02 * F.kl_div(torch.log(output.clamp_min(EPS)), raw_product, reduction="batchmean") + rcf_regularization(model)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 3.0)
        optimizer.step()
        scheduler.step()
        monitor = rcf_predict(model, x[monitor_idx])
        metric = metrics(labels[monitor_idx], monitor)
        score = -metric["f1"] + 0.003 * metric["nll"]
        history.append({"epoch": epoch, "loss": float(loss.detach()), "monitor_acc": metric["acc"], "monitor_f1": metric["f1"], "monitor_nll": metric["nll"]})
        if score < best_score - 1e-5:
            best, best_score, best_epoch, stale = clone_state(model), score, epoch, 0
        else:
            stale += 1
        if stale >= FUSION_PATIENCE:
            break
    model.load_state_dict(best)
    return model.cpu(), best_epoch, history


def refit_rcf(model, x, labels, epochs, seed):
    if epochs <= 0:
        return model.cpu()
    seed_all(seed)
    model = model.to(DEVICE)
    parameters = model.enabled_parameters()
    if not parameters:
        return model.cpu()
    optimizer = torch.optim.AdamW(parameters, lr=1.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=2e-5)
    features = torch.from_numpy(x).float().to(DEVICE)
    target = torch.from_numpy(labels).long().to(DEVICE)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.empty((len(features), features.shape[1], 1), device=DEVICE).uniform_(0.90, 1.10)
        augmented = F.softmax(torch.log(features.clamp_min(EPS)) / temperature + 0.015 * torch.randn_like(features), 2)
        output = model(augmented)
        raw_product = F.softmax(torch.log(features.clamp_min(EPS)).mean(1), 1).detach()
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), target) + 0.05 * F.mse_loss(output, F.one_hot(target, NC).float()) + 0.02 * F.kl_div(torch.log(output.clamp_min(EPS)), raw_product, reduction="batchmean") + rcf_regularization(model)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 3.0)
        optimizer.step()
        scheduler.step()
    return model.cpu()


def fit_rcf_variant(name, config, x, labels, test_folds, seed, root, context, tag, selection_initial_state=None, refit_initial_state=None, source="independent"):
    path = root / "fusion" / f"{slug(context)}_{slug(name)}_fs{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    upstream = {"selection": state_hash(selection_initial_state), "refit": state_hash(refit_initial_state)}
    posterior_hash = array_hash(x, labels, *test_folds)
    if path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            states = [saved["state_dict"], saved["info"]["selection_state"]]
            valid_states = all(state and all(torch.isfinite(value).all() for value in state.values()) for state in states)
        except Exception:
            saved, valid_states = {}, False
        if valid_states and saved.get("tag") == tag and saved.get("config") == config and saved.get("source") == source and saved.get("upstream") == upstream and saved.get("posterior_hash") == posterior_hash:
            matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
            prior = np.bincount(labels, minlength=NC).astype(np.float64) + 1.0
            model = ComponentRCF(matrices, config, prior)
            try:
                model.load_state_dict(saved["state_dict"], strict=True)
                return [rcf_predict(model, item) for item in test_folds], saved["info"], saved["state_dict"]
            except (KeyError, RuntimeError):
                pass
    if name == "Average/Base":
        output = [normalize(item.mean(1)) for item in test_folds]
        return output, {"params": 0, "selected_epoch": 0, "history": [], "source": "fixed_average"}, None
    fit_idx, monitor_idx = train_test_split(np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=seed)
    fit_matrices = [soft_confusion(labels[fit_idx], x[fit_idx, modality]) for modality in range(x.shape[1])]
    fit_prior = np.bincount(labels[fit_idx], minlength=NC).astype(np.float64) + 1.0
    selected_model = ComponentRCF(fit_matrices, config, fit_prior)
    if selection_initial_state is not None:
        selected_model.load_state_dict(selection_initial_state, strict=True)
    selected_model, epoch, history = train_rcf(selected_model, x, labels, fit_idx, monitor_idx, seed)
    selection_state = clone_state(selected_model)
    all_matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
    all_prior = np.bincount(labels, minlength=NC).astype(np.float64) + 1.0
    final = ComponentRCF(all_matrices, config, all_prior)
    if refit_initial_state is not None:
        final.load_state_dict(refit_initial_state, strict=True)
    final = refit_rcf(final, x, labels, epoch, seed + 1)
    state = clone_state(final)
    params = sum(parameter.numel() for parameter in final.parameters() if parameter.requires_grad)
    info = {"params": params, "selected_epoch": epoch, "history": history, "source": source, "selection_state": selection_state}
    torch.save({"tag": tag, "state_dict": state, "config": config, "source": source, "upstream": upstream, "posterior_hash": posterior_hash, "info": info}, path)
    return [rcf_predict(final, item) for item in test_folds], info, state


def fusion_features(x):
    x = np.clip(np.asarray(x, dtype=np.float32), EPS, 1.0)
    mean = x.mean(1)
    std = x.std(1)
    maximum = x.max(1)
    entropy = -(x * np.log(x)).sum(2) / np.log(NC)
    sorted_x = np.sort(x, axis=2)
    margin = sorted_x[:, :, -1] - sorted_x[:, :, -2]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).astype(np.float32)
    return np.concatenate([x.reshape(len(x), -1), mean, std, maximum, entropy, margin, disagreement], 1).astype(np.float32)


def logistic_stacking(x, labels, test_folds, seed):
    features = fusion_features(x)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.20, max_iter=1200, solver="lbfgs", class_weight="balanced", random_state=seed))
    model.fit(features, labels)
    outputs = [normalize(model.predict_proba(fusion_features(item))) for item in test_folds]
    params = sum(np.asarray(getattr(model[-1], field)).size for field in ("coef_", "intercept_"))
    return outputs, params


class StackingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, NC))

    def forward(self, x):
        return F.softmax(self.net(x), 1)


def train_mlp_stacking(x, labels, test_folds, seed):
    fit_idx, monitor_idx = train_test_split(np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=seed)
    seed_all(seed)
    feature_array = fusion_features(x)
    test_feature_arrays = [fusion_features(item) for item in test_folds]
    model = StackingMLP(feature_array.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RCF_EPOCHS, eta_min=2e-5)
    features = torch.from_numpy(feature_array).float().to(DEVICE)
    target = torch.from_numpy(labels).long().to(DEVICE)
    best, best_score, stale = clone_state(model), np.inf, 0
    for _ in range(RCF_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features[fit_idx])
        loss = F.nll_loss(torch.log(output.clamp_min(EPS)), target[fit_idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            monitor = model(features[monitor_idx]).cpu().numpy()
        metric = metrics(labels[monitor_idx], monitor)
        score = -metric["f1"] + 0.003 * metric["nll"]
        if score < best_score - 1e-5:
            best, best_score, stale = clone_state(model), score, 0
        else:
            stale += 1
        if stale >= FUSION_PATIENCE:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        outputs = [normalize(model(torch.from_numpy(item).float().to(DEVICE)).cpu().numpy()) for item in test_feature_arrays]
    return outputs, sum(parameter.numel() for parameter in model.parameters())


def disagreement_metrics(labels, experts, reference, candidate):
    predictions = experts.argmax(2)
    agreement = np.all(predictions == predictions[:, :1], 1)
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


def latency_ms(state, matrices, config, x):
    if state is None or len(x) == 0:
        return 0.0
    model = ComponentRCF(matrices, config).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    batch = torch.from_numpy(x[:min(256, len(x))]).float().to(DEVICE)
    with torch.no_grad():
        for _ in range(4):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(15):
            model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    return float((time.perf_counter() - started) * 1000.0 / (15 * len(batch)))


def bootstrap_delta(labels, candidate, reference, seed, rounds=1000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        index = rng.integers(0, len(labels), len(labels))
        values.append(accuracy_score(labels[index], candidate[index].argmax(1)) - accuracy_score(labels[index], reference[index].argmax(1)))
    values = np.asarray(values)
    p_value = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    return {"delta": float(values.mean()), "ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5)), "bootstrap_p": min(1.0, p_value)}


def plot_cm(labels, proba, path, title):
    if not PLOTS:
        return
    matrix = rows(confusion_matrix(labels, normalize(proba).argmax(1), labels=np.arange(NC)))
    figure, axis = plt.subplots(figsize=(13, 11))
    sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, ax=axis, xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.tick_params(axis="both", labelsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_matrix(matrix, path, title, xlabels, ylabels=None):
    if not PLOTS:
        return
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="viridis", ax=axis, xticklabels=xlabels, yticklabels=ylabels if ylabels is not None else xlabels)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_reliability(labels, before, after, path, title):
    if not PLOTS:
        return
    figure, axis = plt.subplots(figsize=(6, 6))
    for proba, name in ((before, "Average"), (after, "Full RCF")):
        confidence = proba.max(1)
        correct = proba.argmax(1) == labels
        centers, accuracy = [], []
        for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
            if mask.any():
                centers.append(float(confidence[mask].mean()))
                accuracy.append(float(correct[mask].mean()))
        axis.plot(centers, accuracy, marker="o", label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set(xlabel="Confidence", ylabel="Accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def append_trajectory(path, rows_to_write):
    if not rows_to_write:
        return
    frame = pd.DataFrame(rows_to_write)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def parameter_text(value):
    value = int(round(float(value)))
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def print_terminal_summary(summary_frame, rcf_frame, log):
    ranked = summary_frame.groupby(["variant", "fusion"], as_index=False).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"), f1=("f1", "mean"), f1_std=("f1", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"), ece=("ece", "mean"),
        brier=("brier", "mean"), nll=("nll", "mean"), fusion_gain=("fusion_gain", "mean"),
        sri=("sri", "mean"), jsri=("jsri", "mean"), parameters=("parameters", "mean"),
    ).sort_values(["acc", "f1"], ascending=False).reset_index(drop=True)
    ranked[["acc_std", "f1_std"]] = ranked[["acc_std", "f1_std"]].fillna(0.0)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(log / "ranked_rost_ablation.csv", index=False)
    print("\nrank | variant                         | model                | acc mean±std  | f1 mean±std   | precision | recall | ece    | brier  | nll    | gain    | JSRI   | params")
    displayed = ranked.iloc[:TERMINAL_TOP] if TERMINAL_TOP > 0 else ranked
    for row in displayed.itertuples(index=False):
        print(f"{row.rank:>4} | {row.variant:<31.31} | {row.fusion:<20.20} | {row.acc:.4f}±{row.acc_std:.4f} | {row.f1:.4f}±{row.f1_std:.4f} | {row.precision:.4f}    | {row.recall:.4f} | {row.ece:.4f} | {row.brier:.4f} | {row.nll:.4f} | {row.fusion_gain:+.4f} | {row.jsri:.4f} | {parameter_text(row.parameters):>7}")
    if len(displayed) < len(ranked):
        print(f"... showing top {len(displayed)} of {len(ranked)} rows; full ranking: {log / 'ranked_rost_ablation.csv'}")

    comparison_rows = []
    if {"CE-only", "Full ROST"}.issubset(set(summary_frame["variant"])):
        means = summary_frame.groupby(["variant", "fusion"], as_index=False)[["acc", "f1", "nll", "ece", "fusion_gain"]].mean()
        for model in sorted(set(means.loc[means["variant"] == "CE-only", "fusion"]) & set(means.loc[means["variant"] == "Full ROST", "fusion"])):
            ce = means[(means["variant"] == "CE-only") & (means["fusion"] == model)].iloc[0]
            rost = means[(means["variant"] == "Full ROST") & (means["fusion"] == model)].iloc[0]
            comparison_rows.append({
                "model": model, "ce_acc": ce["acc"], "rost_acc": rost["acc"], "delta_acc": rost["acc"] - ce["acc"],
                "ce_f1": ce["f1"], "rost_f1": rost["f1"], "delta_f1": rost["f1"] - ce["f1"],
                "ce_nll": ce["nll"], "rost_nll": rost["nll"], "delta_nll": rost["nll"] - ce["nll"],
                "ce_ece": ce["ece"], "rost_ece": rost["ece"], "delta_ece": rost["ece"] - ce["ece"],
            })
        comparison = pd.DataFrame(comparison_rows).sort_values("rost_acc", ascending=False)
        comparison.to_csv(log / "ce_only_vs_full_rost.csv", index=False)
        print("\nmodel                | CE acc | ROST acc | delta acc | CE f1  | ROST f1 | delta f1 | CE nll | ROST nll | delta nll")
        for row in comparison.to_dict("records"):
            print(f"{row['model']:<20.20} | {row['ce_acc']:.4f} | {row['rost_acc']:.4f}   | {row['delta_acc']:+.4f}   | {row['ce_f1']:.4f} | {row['rost_f1']:.4f}  | {row['delta_f1']:+.4f}  | {row['ce_nll']:.4f} | {row['rost_nll']:.4f}   | {row['delta_nll']:+.4f}")

    if not rcf_frame.empty:
        components = rcf_frame.groupby(["expert_regime", "variant"], as_index=False).agg(
            acc=("acc", "mean"), acc_std=("acc", "std"), f1=("f1", "mean"), f1_std=("f1", "std"),
            ece=("ece", "mean"), brier=("brier", "mean"), nll=("nll", "mean"),
            net_correction=("net_correction", "mean"), parameters=("parameters", "mean"),
        ).sort_values(["expert_regime", "acc"], ascending=[True, False])
        components[["acc_std", "f1_std"]] = components[["acc_std", "f1_std"]].fillna(0.0)
        components.to_csv(log / "ranked_rcf_components.csv", index=False)
        print("\nexpert regime | RCF component                  | acc mean±std  | f1 mean±std   | ece    | brier  | nll    | net corr | params")
        for row in components.itertuples(index=False):
            print(f"{row.expert_regime:<13.13} | {row.variant:<30.30} | {row.acc:.4f}±{row.acc_std:.4f} | {row.f1:.4f}±{row.f1_std:.4f} | {row.ece:.4f} | {row.brier:.4f} | {row.nll:.4f} | {row.net_correction:+8.2f} | {parameter_text(row.parameters):>7}")


def main():
    seed_all(SEED)
    raw_features, all_labels, subjects = load_features()
    tag = config_hash()
    version, root, log = output_dirs(tag)
    started = time.time()
    train_index = np.flatnonzero(np.isin(subjects, sorted(TRAIN_SUBJECTS)))
    test_index = np.flatnonzero(np.isin(subjects, sorted(TEST_SUBJECTS)))
    raw_train = {name: value[train_index] for name, value in raw_features.items()}
    raw_test = {name: value[test_index] for name, value in raw_features.items()}
    labels = all_labels[train_index]
    if set(np.unique(labels)) != set(range(NC)):
        raise RuntimeError("UTD-MHAD development partition does not contain all 27 classes")
    active_rost = {name: config for name, config in ROST_VARIANTS.items() if selected(name, VARIANT_FILTER)}
    active_rcf = {name: config for name, config in RCF_VARIANTS.items() if selected(name, RCF_FILTER)}
    if not active_rost:
        raise ValueError(f"DOME_X_VARIANTS selected no known variant: {VARIANT_FILTER}")
    if RCF_FILTER and not active_rcf:
        raise ValueError(f"DOME_X_RCF_VARIANTS selected no known variant: {RCF_FILTER}")
    add_chain = ["Average/Base", "+ Class-wise Reliability", "+ PEACE Calibration", "+ Learnable Bias Transport", "+ Disagreement Refinement"]
    component_execution = set(active_rcf)
    for name in active_rcf:
        if name in add_chain:
            component_execution.update(add_chain[:add_chain.index(name) + 1])
    executable_rcf = {name: config for name, config in RCF_VARIANTS.items() if name in component_execution}
    manifest = {
        "version": version, "tag": tag, "source": str(SOURCE_PATH), "feature_cache": str(FEATURE_CACHE),
        "pipeline_seeds": PIPELINE_SEEDS, "fusion_seeds": FUSION_SEEDS, "outer_folds": OUTER_FOLDS,
        "pretrain_epochs": PRETRAIN_EPOCHS, "rost_epochs": ROST_EPOCHS, "rcf_epochs": RCF_EPOCHS,
        "max_input_dim": MAX_INPUT_DIM, "experts": list(raw_train), "rost_variants": active_rost,
        "rcf_variants": active_rcf, "data": {"development": len(labels), "test": len(test_index)},
        "train_subjects": sorted(TRAIN_SUBJECTS), "test_subjects": sorted(TEST_SUBJECTS),
        "protocol": "Odd subjects form the development set. Each ROST variant and pipeline seed independently trains three outer-fold expert ensembles. Inner profile labels control checkpoints and ROST. Even-subject labels are consumed only by final metrics, plots, and paired bootstrap tests.",
        "runtime": "Cached statistics, large GPU batches, early stopping, stage checkpoints, and one fusion fit per OOF pipeline bound runtime. Original feature dimensions are preserved by default; train-only sparse random projection remains an opt-in debug control through DOME_X_MAX_INPUT_DIM.",
        "rcf_protocol": "Incremental add variants follow the add chain. Every removal variant starts from the corresponding independently trained Full RCF state, disables removed forward paths and optimizer parameters, then fine-tunes on the same OOF split.",
        "rost_note": "The cumulative CE+Structure(+Anti-collapse/+Complementarity) groups progressively activate newly added blocks. Removal groups activate all retained blocks for the full ROST stage. This distinguishes the preregistered cumulative +Complementarity row from w/o Joint Recovery while preserving their component sets.",
    }
    save_json(manifest, log / "manifest.json")
    print(f"UTD-MHAD DOME-X ablation v{version} tag={tag} device={DEVICE} development={len(labels)} test={len(test_index)}")

    prepared = {}
    pipeline = {}
    trajectory_path = log / "controller_trajectory.csv"
    if trajectory_path.exists():
        trajectory_path.unlink()
    for variant, config in active_rost.items():
        pipeline[variant] = {}
        for seed in PIPELINE_SEEDS:
            oof = {name: np.zeros((len(labels), NC), dtype=np.float32) for name in raw_train}
            test_by_fold = {name: [] for name in raw_train}
            profiles, snapshots, trajectories, fold_probes, selected_epochs, expert_parameters = [], [], [], [], [], []
            for fold in range(1, OUTER_FOLDS + 1):
                artifact = expert_fold(variant, config, seed, fold, raw_train, raw_test, labels, root, tag, prepared)
                for name in raw_train:
                    oof[name][artifact["holdout_idx"]] = artifact["holdout"][name]
                    test_by_fold[name].append(artifact["test"][name])
                profiles.append(artifact["final_profile"])
                snapshots.append(artifact["profile_snapshots"])
                fold_probes.append(artifact["final_probe"])
                selected_epochs.append(artifact["rost_epoch"])
                expert_parameters.append(artifact["parameter_counts"])
                for row in artifact["rost_history"]:
                    trajectories.append({
                        "variant": variant, "pipeline_seed": seed, "fold": fold, "epoch": row["epoch"],
                        "loss": row["train_loss"], "score": row["score"], "jsri": row["profile"]["jsri"],
                        "probe_f1": row["probe"]["f1"], "probe_nll": row["probe"]["nll"],
                        "mean_sri": float(np.mean([item["sri"] for item in row["profile"]["experts"]])),
                        "row_entropy": float(np.mean([item["row_entropy"] for item in row["profile"]["experts"]])),
                        "column_entropy": float(np.mean([item["column_entropy"] for item in row["profile"]["experts"]])),
                        "effective_rank": float(np.mean([item["effective_rank"] for item in row["profile"]["experts"]])),
                        "direct_redundancy": row["profile"]["direct_redundancy"],
                        "graph_redundancy": row["profile"]["graph_redundancy"],
                        "rescue": row["profile"]["rescue"],
                        **{f"lambda_{name}": value for name, value in row["weights"].items()},
                    })
            append_trajectory(trajectory_path, trajectories)
            if any(np.any(np.isclose(value.sum(1), 0.0)) for value in oof.values()):
                raise RuntimeError(f"Incomplete OOF posterior for {variant}, seed={seed}")
            pipeline[variant][seed] = {
                "oof": oof, "test": {name: np.stack(items) for name, items in test_by_fold.items()},
                "profiles": profiles, "snapshots": snapshots, "fold_probes": fold_probes,
                "selected_epochs": selected_epochs, "expert_parameters": expert_parameters,
            }
            np.savez_compressed(root / f"posterior_{slug(variant)}_s{seed}.npz", labels=labels, **{f"oof_{name}": value for name, value in oof.items()}, **{f"test_{name}": value for name, value in pipeline[variant][seed]["test"].items()})
            print(f"Completed expert pipeline: {variant} seed={seed}")

    # All expert objectives, checkpoints, OOF posteriors, and test posteriors are now frozen.
    test_labels = all_labels[test_index]
    if set(np.unique(test_labels)) != set(range(NC)):
        raise RuntimeError("UTD-MHAD final test partition does not contain all 27 classes")
    summary_rows, rcf_rows, predictions, profile_snapshots = [], [], {}, {}
    full_state_cache = {}
    for variant, regimes in pipeline.items():
        for pipeline_seed, data in regimes.items():
            names = list(raw_train)
            x = np.stack([data["oof"][name] for name in names], 1).astype(np.float32)
            test_folds = [np.stack([data["test"][name][fold] for name in names], 1).astype(np.float32) for fold in range(OUTER_FOLDS)]
            mean_experts = np.mean(test_folds, 0)
            matrices = [soft_confusion(labels, x[:, modality]) for modality in range(x.shape[1])]
            diagnostics = profile(matrices)
            average = normalize(mean_experts.mean(1))
            product = normalize(np.exp(np.log(mean_experts).mean(1)))
            reliability = np.stack([confusion_reliability(matrix) for matrix in matrices])
            reliability /= reliability.sum(0, keepdims=True)
            weighted = normalize((mean_experts * reliability[None]).sum(1))
            logistic_outputs, logistic_params = logistic_stacking(x, labels, [mean_experts], pipeline_seed)
            logistic = logistic_outputs[0]
            mlp_outputs, mlp_params = train_mlp_stacking(x, labels, [mean_experts], pipeline_seed)
            mlp = mlp_outputs[0]
            context = f"{variant}_pipeline_s{pipeline_seed}"
            full_outputs, full_infos, full_states = {}, {}, {}
            component_regime = variant in ("CE-only", "Full ROST") and pipeline_seed == PIPELINE_SEEDS[0]
            needs_full_checkpoints = any(name == "Full RCF" or name.startswith("w/o") for name in active_rcf)
            required_fusion_seeds = FUSION_SEEDS if component_regime and needs_full_checkpoints else FUSION_SEEDS[:1]
            for fusion_seed in required_fusion_seeds:
                fold_outputs, info, state = fit_rcf_variant("Full RCF", RCF_VARIANTS["Full RCF"], x, labels, test_folds, fusion_seed, root, context, tag)
                full_outputs[fusion_seed] = normalize(np.mean(fold_outputs, 0))
                full_infos[fusion_seed], full_states[fusion_seed] = info, state
            full = full_outputs[FUSION_SEEDS[0]]
            full_state_cache[(variant, pipeline_seed)] = (full_states, full_infos)
            profile_snapshots[f"{slug(variant)}_s{pipeline_seed}"] = {"oof": matrices, "fold_profiles": data["profiles"], "fold_snapshots": data["snapshots"]}
            for name, matrix in zip(names, matrices):
                plot_matrix(js_rows_numpy(matrix), log / f"row_relation_{slug(variant)}_s{pipeline_seed}_{name}.png", f"{variant} | seed {pipeline_seed} | {name} row relation", ACTION_NAMES)
            expert_outputs = {f"Submodel {name}": normalize(data["test"][name].mean(0)) for name in names}
            best_expert = max(metrics(test_labels, output)["acc"] for output in expert_outputs.values())
            result_set = {**expert_outputs, "Average": average, "Product": product, "Weighted Average": weighted, "Logistic Stacking": logistic, "MLP Stacking": mlp, "Full RCF": full}
            parameters = {
                **{f"Submodel {name}": int(np.mean([item[name] for item in data["expert_parameters"]])) for name in names},
                "Average": 0, "Product": 0, "Weighted Average": reliability.size, "Logistic Stacking": logistic_params,
                "MLP Stacking": mlp_params, "Full RCF": int(np.mean([item["params"] for item in full_infos.values()])),
            }
            mean_sri = float(np.mean([item["sri"] for item in diagnostics["experts"]]))
            probe_f1 = float(np.mean([item["f1"] for item in data["fold_probes"]]))
            probe_nll = float(np.mean([item["nll"] for item in data["fold_probes"]]))
            selected_epoch = float(np.mean(data["selected_epochs"]))
            for fusion_name, output in result_set.items():
                metric = metrics(test_labels, output)
                summary_rows.append({"variant": variant, "pipeline_seed": pipeline_seed, "fusion": fusion_name, **metric, "fusion_gain": metric["acc"] - best_expert, "sri": mean_sri, "jsri": diagnostics["jsri"], "probe_f1": probe_f1, "probe_nll": probe_nll, "selected_epoch": selected_epoch, "parameters": parameters.get(fusion_name, 0)})
                predictions[f"{slug(variant)}_s{pipeline_seed}_{slug(fusion_name)}"] = output.astype(np.float32)
                plot_cm(test_labels, output, log / f"cm_{slug(variant)}_s{pipeline_seed}_{slug(fusion_name)}.png", f"{variant} | seed {pipeline_seed} | {fusion_name}")

            if component_regime:
                full_reference_states, full_reference_infos = full_state_cache[(variant, pipeline_seed)]
                component_context = f"component_{variant}_pipeline_s{pipeline_seed}"
                for fusion_seed in FUSION_SEEDS:
                    outputs_by_name, states_by_name = {}, {}
                    previous_selection_state, previous_refit_state = None, None
                    for component_name, component_config in executable_rcf.items():
                        output = average
                        state = None
                        info = {"params": 0, "selected_epoch": 0, "source": "uninitialized"}
                        selection_initial_state, refit_initial_state = None, None
                        if component_name == "Full RCF":
                            output = full_outputs[fusion_seed]
                            state = full_reference_states[fusion_seed]
                            reference_info = full_reference_infos[fusion_seed]
                            info = {**reference_info, "source": "actual_pipeline_full_checkpoint"}
                            fold_outputs = None
                            source = "actual_pipeline_full_checkpoint"
                        elif component_name.startswith("+"):
                            selection_initial_state, refit_initial_state = previous_selection_state, previous_refit_state
                            source = "incremental_add_chain"
                        elif component_name.startswith("w/o"):
                            selection_initial_state = full_reference_infos[fusion_seed]["selection_state"]
                            refit_initial_state = full_reference_states[fusion_seed]
                            source = "full_checkpoint_removal"
                        else:
                            selection_initial_state, refit_initial_state, source = None, None, "fixed_average"
                        if component_name != "Full RCF":
                            fold_outputs, info, state = fit_rcf_variant(
                                component_name, component_config, x, labels, test_folds,
                                fusion_seed, root, component_context, tag,
                                selection_initial_state, refit_initial_state, source,
                            )
                            output = normalize(np.mean(fold_outputs, 0))
                        outputs_by_name[component_name], states_by_name[component_name] = output, state
                        if component_name.startswith("+"):
                            previous_selection_state = info["selection_state"]
                            previous_refit_state = state
                        if component_name not in active_rcf:
                            continue
                        average_diagnostic = disagreement_metrics(test_labels, mean_experts, average, output)
                        row = {
                            "expert_regime": variant, "pipeline_seed": pipeline_seed, "fusion_seed": fusion_seed,
                            "variant": component_name, **metrics(test_labels, output), **average_diagnostic,
                            "per_class_recall_delta_vs_average": per_class_recall_delta(test_labels, average, output),
                            "parameters": info["params"], "latency_ms_per_sample": latency_ms(state, matrices, component_config, x),
                            "selected_epoch": info["selected_epoch"], "source": info["source"],
                        }
                        rcf_rows.append(row)
                        plot_cm(test_labels, output, log / f"cm_rcf_{slug(variant)}_fs{fusion_seed}_{slug(component_name)}.png", f"{variant} | fusion seed {fusion_seed} | {component_name}")
                    full_component = outputs_by_name.get("Full RCF", full_outputs.get(fusion_seed, average))
                    for row in rcf_rows:
                        if row["expert_regime"] == variant and row["fusion_seed"] == fusion_seed:
                            candidate = outputs_by_name[row["variant"]]
                            row["per_class_recall_delta_vs_full"] = per_class_recall_delta(test_labels, full_component, candidate)
                    full_state = states_by_name.get("Full RCF", full_reference_states.get(fusion_seed))
                    if full_state is not None:
                        model = ComponentRCF(matrices, RCF_VARIANTS["Full RCF"])
                        model.load_state_dict(full_state)
                        plot_matrix(F.softmax(model.reliability_logits, 0).detach().numpy(), log / f"reliability_{slug(variant)}_fs{fusion_seed}.png", f"{variant} Full RCF class-wise reliability", ACTION_NAMES, names)
                        transport = F.softmax(model.transport_logits, 1).detach().numpy()
                        for modality, expert_name in enumerate(names):
                            plot_matrix(transport[modality], log / f"transport_{slug(variant)}_{expert_name}_fs{fusion_seed}.png", f"{variant} {expert_name} transport (true x predicted)", ACTION_NAMES)
                    plot_reliability(test_labels, average, full_component, log / f"calibration_{slug(variant)}_fs{fusion_seed}.png", f"{variant} calibration")

    summary_frame = pd.DataFrame(summary_rows)
    rcf_frame = pd.DataFrame(rcf_rows)
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
        aggregate = pd.concat([aggregate, rcf_aggregate], ignore_index=True, sort=False)
    aggregate.to_csv(log / "aggregate_mean_std.csv", index=False)

    bootstrap = []
    for pipeline_seed in PIPELINE_SEEDS:
        ce_key = f"ce_only_s{pipeline_seed}_full_rcf"
        rost_key = f"full_rost_s{pipeline_seed}_full_rcf"
        if ce_key in predictions and rost_key in predictions:
            bootstrap.append({"pipeline_seed": pipeline_seed, "comparison": "Full ROST Full RCF - CE-only Full RCF", **bootstrap_delta(test_labels, predictions[rost_key], predictions[ce_key], pipeline_seed)})
    correlation_rows = []
    for fusion_name, group in summary_frame[~summary_frame["fusion"].str.startswith("Submodel")].groupby("fusion"):
        if len(group) >= 3 and group["jsri"].nunique() > 1 and group["fusion_gain"].nunique() > 1:  # pyright: ignore[reportAttributeAccessIssue]
            correlation_rows.append({"fusion": fusion_name, "pearson_jsri_gain": group["jsri"].corr(group["fusion_gain"], method="pearson"), "spearman_jsri_gain": group["jsri"].corr(group["fusion_gain"], method="spearman"), "n": len(group)})  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType, reportCallIssue]
    pd.DataFrame(correlation_rows).to_csv(log / "recoverability_gain_correlation.csv", index=False)
    if PLOTS:
        figure, axis = plt.subplots(figsize=(8, 6))
        scatter = summary_frame[summary_frame["fusion"].isin(["Average", "Logistic Stacking", "MLP Stacking", "Full RCF"])]
        sns.scatterplot(data=scatter, x="jsri", y="fusion_gain", hue="fusion", style="fusion", ax=axis)  # pyright: ignore[reportArgumentType]
        axis.set_title("Recoverability versus fusion gain")
        figure.tight_layout()
        figure.savefig(log / "fusion_gain_vs_jsri.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    np.savez_compressed(root / "final_seed_predictions.npz", test_labels=test_labels, **predictions)
    save_json(profile_snapshots, log / "profile_snapshots.json")
    save_json({"manifest": manifest, "rost_rows": summary_rows, "rcf_rows": rcf_rows, "paired_bootstrap": bootstrap, "correlations": correlation_rows}, log / "results.json")
    save_json(bootstrap, log / "paired_bootstrap.json")
    print_terminal_summary(summary_frame, rcf_frame, log)
    print("UTD-MHAD DOME-X ablation complete")
    print(f"Checkpoints={root}")
    print(f"Logs={log}")
    print(f"TimeMinutes={(time.time() - started) / 60.0:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate source and feature cache")
    return parser.parse_args()


def check_environment():
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Missing UTD-MHAD source: {SOURCE_PATH}")
    if not FEATURE_CACHE.is_file():
        raise FileNotFoundError(f"Missing UTD-MHAD feature cache: {FEATURE_CACHE}")
    print(
        f"UTD-MHAD ablation check passed: source={SOURCE_PATH.name} "
        f"cache={FEATURE_CACHE.name}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        check_environment()
    else:
        main()
