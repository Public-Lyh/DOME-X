import argparse
import os
import re
import json
import pickle
import random
import warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, log_loss

try:
    import joblib
except Exception:
    joblib = None

PLACEHOLDER_ROOT = Path("your path")
WORKSPACE_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
DATA_ROOT = WORKSPACE_ROOT / "data" / "nturgb+d_skeletons"
PROJECT_ROOT = WORKSPACE_ROOT / "Code" / "NTU-RGBD"
BASE_CKPT_DIR = PROJECT_ROOT / "checkpoints"
BASE_LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
EXP_NAME = "NTU_RGBD60_DOME_X"
PROTOCOL = "xsub"
MAX_FILES = None
FORCE_REBUILD_CACHE = False
TARGET_LEN = 64
NUM_JOINTS = 25
MAX_BODIES = 2
SEED = 42
EPS = 1e-8
FEATURE_JOBS = 16
WARMUP_EPOCHS = 20
ROST_CYCLES = 10
OOF_FOLDS = 5
OOF_WARMUP_EPOCHS = 8
OOF_ROST_CYCLES = 4
OBSERVER_BATCH = 768
FUSION_EPOCHS = 180
LEAK_FUSION_EPOCHS = 260
RCF_TRACE_INTERVAL = 5
RCF_SELECTION_NLL_TIEBREAK = 0.002
COMPARISON_FUSIONS = ("Average", "Product", "Weighted Average", "Logistic Stacking", "MLP Stacking", "DOME-X RCF")

NTU60_XSUB_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
NTU60_XVIEW_TRAIN_CAMERAS = {2, 3}
NTU60_XVIEW_TEST_CAMERAS = {1}
NTU_BONES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]
NTU_PARTS = [[0, 1, 2, 3, 20], [4, 5, 6, 7, 21, 22], [8, 9, 10, 11, 23, 24], [12, 13, 14, 15], [16, 17, 18, 19]]
NTU_NAME_PATTERN = re.compile(r"S(?P<S>\d{3})C(?P<C>\d{3})P(?P<P>\d{3})R(?P<R>\d{3})A(?P<A>\d{3})\.skeleton$")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


seed_everything(SEED)
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device("cpu")]


def next_output_dirs():
    BASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in BASE_CKPT_DIR.glob("v*"):
        m = re.fullmatch(r"v(\d+)", p.name)
        if p.is_dir() and m:
            versions.append(int(m.group(1)))
    version = max(versions, default=0) + 1
    ckpt = BASE_CKPT_DIR / f"v{version}" / EXP_NAME
    log = BASE_LOG_DIR / f"v{version}" / EXP_NAME
    ckpt.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return version, ckpt, log


RUN_VERSION = 0
CKPT_DIR = BASE_CKPT_DIR
LOG_DIR = BASE_LOG_DIR


def initialize_output_dirs():
    global RUN_VERSION, CKPT_DIR, LOG_DIR
    RUN_VERSION, CKPT_DIR, LOG_DIR = next_output_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def json_value(x):
    if isinstance(x, dict):
        return {str(k): json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_value(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)
    return x


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_value(obj), f, indent=2, ensure_ascii=False, allow_nan=False)


def normalize_proba(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, EPS, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def row_normalize(c):
    c = np.asarray(c, dtype=np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(c, 0.0, None)
    return c / np.maximum(c.sum(axis=1, keepdims=True), EPS)


def parse_name(path):
    m = NTU_NAME_PATTERN.match(path.name)
    if not m:
        return None
    action = int(m.group("A"))
    return {"setup": int(m.group("S")), "camera": int(m.group("C")), "subject": int(m.group("P")), "label": action - 1, "path": path}


def read_skeleton(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        index = 0
        frames = int(lines[index].strip())
        index += 1
        output = np.zeros((frames, MAX_BODIES, NUM_JOINTS, 3), dtype=np.float32)
        tracked = {}
        for t in range(frames):
            bodies = int(lines[index].strip())
            index += 1
            current = []
            for _ in range(bodies):
                body_info = lines[index].split()
                body_id = body_info[0] if body_info else str(len(current))
                index += 1
                joints = int(lines[index].strip())
                index += 1
                xyz = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
                for j in range(joints):
                    values = lines[index].split()
                    index += 1
                    if j < NUM_JOINTS and len(values) >= 3:
                        xyz[j] = [float(values[0]), float(values[1]), float(values[2])]
                energy = float(np.square(xyz).sum())
                tracked[body_id] = tracked.get(body_id, 0.0) + energy
                current.append((body_id, xyz))
            ranked = sorted(current, key=lambda item: tracked[item[0]], reverse=True)[:MAX_BODIES]
            for slot, (_, xyz) in enumerate(ranked):
                output[t, slot] = xyz
        if not np.isfinite(output).all() or np.abs(output).sum() < EPS:
            return None
        return output
    except Exception:
        return None


def normalize_skeleton(seq):
    seq = np.asarray(seq, dtype=np.float32).copy()
    valid = np.linalg.norm(seq, axis=-1, keepdims=True) > 0
    for body in range(seq.shape[1]):
        root = seq[:, body:body + 1, 0:1, :]
        centered = seq[:, body:body + 1] - root
        seq[:, body:body + 1] = np.where(valid[:, body:body + 1], centered, 0.0)
    lengths = []
    for a, b in NTU_BONES:
        d = np.linalg.norm(seq[:, 0, a] - seq[:, 0, b], axis=-1)
        lengths.append(d[d > 1e-6])
    lengths = np.concatenate([x for x in lengths if len(x)]) if any(len(x) for x in lengths) else np.array([1.0])
    scale = float(np.median(lengths))
    seq /= max(scale, 1e-4)
    return np.clip(np.nan_to_num(seq), -10.0, 10.0).astype(np.float32)


def resample(seq):
    if len(seq) == TARGET_LEN:
        return seq
    indices = np.linspace(0, max(len(seq) - 1, 0), TARGET_LEN).astype(np.int64)
    return seq[indices]


def stats(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32).reshape(len(x), -1))
    groups = []
    for z in (x, np.diff(x, axis=0), np.diff(x, n=2, axis=0)):
        if len(z) == 0:
            z = np.zeros((1, x.shape[1]), dtype=np.float32)
        groups.extend([z.mean(0), z.std(0), z.min(0), z.max(0), np.median(z, axis=0), np.percentile(z, 25, axis=0), np.percentile(z, 75, axis=0), np.percentile(z, 90, axis=0) - np.percentile(z, 10, axis=0), np.abs(z).mean(0), np.square(z).mean(0)])
    for segments in (4, 8):
        for s in range(segments):
            a = int(round(s * len(x) / segments))
            b = max(int(round((s + 1) * len(x) / segments)), a + 1)
            groups.extend([x[a:b].mean(0), x[a:b].std(0)])
    spectrum = np.abs(np.fft.rfft(x, axis=0))
    total = spectrum.sum(0) + EPS
    n = len(spectrum)
    for a, b in ((0, max(1, n // 4)), (max(1, n // 4), max(2, n // 2)), (max(2, n // 2), n)):
        groups.append(spectrum[a:b].sum(0) / total)
    return np.clip(np.nan_to_num(np.concatenate(groups)), -1e6, 1e6).astype(np.float32)


def extract_views(seq):
    seq = resample(normalize_skeleton(seq))
    joint = seq.reshape(len(seq), -1)
    bones = np.stack([seq[:, :, b] - seq[:, :, a] for a, b in NTU_BONES], axis=2).reshape(len(seq), -1)
    motion = np.zeros_like(seq)
    motion[1:] = seq[1:] - seq[:-1]
    motion = motion.reshape(len(seq), -1)
    bone_motion = np.zeros_like(bones)
    bone_motion[1:] = bones[1:] - bones[:-1]
    part = np.concatenate([stats(seq[:, :, joints].reshape(len(seq), -1)) for joints in NTU_PARTS])
    return {"joint": stats(joint), "bone": stats(bones), "motion": stats(motion), "bone_motion": stats(bone_motion), "part": part.astype(np.float32)}


def process_file(path):
    meta = parse_name(path)
    if meta is None:
        return None
    seq = read_skeleton(path)
    if seq is None:
        return None
    try:
        return meta, extract_views(seq)
    except Exception:
        return None


def build_or_load_features():
    cache = CACHE_DIR / f"ntu60_skeleton_features_{PROTOCOL}_max{MAX_FILES}_v1.npz"
    meta_file = CACHE_DIR / f"ntu60_skeleton_features_{PROTOCOL}_max{MAX_FILES}_v1_meta.pkl"
    if cache.exists() and meta_file.exists() and not FORCE_REBUILD_CACHE:
        data = np.load(cache, allow_pickle=False, mmap_mode="r")
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        views = {key[2:]: np.asarray(data[key], dtype=np.float32) for key in data.files if key.startswith("X_")}
        return views, np.asarray(data["y"]), np.asarray(data["subjects"]), np.asarray(data["cameras"]), meta
    files = sorted(DATA_ROOT.glob("*.skeleton"))
    if MAX_FILES is not None:
        files = files[:MAX_FILES]
    if joblib is not None:
        records = joblib.Parallel(n_jobs=FEATURE_JOBS, backend="loky", verbose=5)(
            joblib.delayed(process_file)(path) for path in files
        )
    else:
        records = [process_file(p) for p in files]
    records = [r for r in records if r is not None]
    if not records:
        raise RuntimeError("No valid NTU RGB+D skeleton samples were found")
    names = list(records[0][1])
    views = {name: np.stack([r[1][name] for r in records]).astype(np.float32) for name in names}
    y = np.array([r[0]["label"] for r in records], dtype=np.int64)
    subjects = np.array([r[0]["subject"] for r in records], dtype=np.int64)
    cameras = np.array([r[0]["camera"] for r in records], dtype=np.int64)
    np.savez_compressed(cache, **{f"X_{k}": v for k, v in views.items()}, y=y, subjects=subjects, cameras=cameras, setups=np.array([r[0]["setup"] for r in records], dtype=np.int64))
    meta = {"paths": [str(r[0]["path"]) for r in records], "view_names": names, "created_at": datetime.now().isoformat()}
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)
    return views, y, subjects, cameras, meta


def official_split(subjects, cameras):
    if PROTOCOL == "xsub":
        train = np.array([s in NTU60_XSUB_TRAIN_SUBJECTS for s in subjects])
        return train, ~train
    train = np.array([c in NTU60_XVIEW_TRAIN_CAMERAS for c in cameras])
    test = np.array([c in NTU60_XVIEW_TEST_CAMERAS for c in cameras])
    return train, test


def split_train_indices(indices, labels):
    base, remaining = train_test_split(indices, test_size=0.28, random_state=SEED, stratify=labels[indices])
    controller, fusion = train_test_split(remaining, test_size=0.5, random_state=SEED + 1, stratify=labels[remaining])
    return np.sort(base), np.sort(controller), np.sort(fusion)


def soft_confusion(y, p, classes):
    c = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(c, y, p)
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    c /= np.maximum(counts[:, None], 1.0)
    missing = counts == 0
    c[missing] = 1.0 / classes
    return row_normalize(c)


def entropy_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def js_np(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def effective_rank(c):
    singular = np.linalg.svd(c, compute_uv=False)
    p = singular / max(singular.sum(), EPS)
    return float(np.exp(-(p * np.log(p + EPS)).sum()) / c.shape[0])


def bayes_decode(c, prior=None, smoothing=1e-3):
    classes = c.shape[0]
    c = row_normalize(c + smoothing)
    prior = np.ones(classes) / classes if prior is None else np.asarray(prior, dtype=np.float64)
    prior /= prior.sum()
    reverse = (prior[:, None] * c).T
    reverse /= np.maximum(reverse.sum(axis=1, keepdims=True), EPS)
    decoded = c @ reverse
    return float(np.diag(decoded).mean()), reverse, decoded


def row_graph(c):
    classes = c.shape[0]
    graph = np.zeros((classes, classes), dtype=np.float64)
    for a in range(classes):
        for b in range(a + 1, classes):
            graph[a, b] = graph[b, a] = js_np(c[a], c[b]) / np.log(2)
    return graph


def cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def cm_sri(c):
    c = row_normalize(c)
    classes = c.shape[0]
    entropies = np.array([entropy_np(row) / np.log(classes) for row in c])
    top1 = c.max(axis=1)
    top_r = np.sort(c, axis=1)[:, -min(3, classes):].sum(axis=1)
    mp_penalty = np.maximum(0, 0.18 - entropies) ** 2 + np.maximum(0, entropies - 0.78) ** 2 + 0.25 * np.maximum(0, top1 - 0.96) ** 2 + np.maximum(0, 0.45 - top_r) ** 2 + np.maximum(0, top_r - 0.99) ** 2
    q_mp = float(np.clip(1.0 - mp_penalty.mean(), 0.0, 1.0))
    distances = [js_np(c[a], c[b]) / np.log(2) for a in range(classes) for b in range(a + 1, classes)]
    q_sep = float(np.mean(distances))
    usage = c.mean(axis=0)
    usage_entropy = entropy_np(usage) / np.log(classes)
    q_usage = float(np.clip(1.0 - max(0, 0.45 - usage_entropy) ** 2 - max(0, usage_entropy - 0.995) ** 2 - max(0, usage.max() - 0.45) ** 2, 0.0, 1.0))
    q_rank = effective_rank(c)
    q_decode, _, _ = bayes_decode(c)
    uniform = np.ones(classes) / classes
    q_nonrand = float(np.mean([js_np(row, uniform) / np.log(2) for row in c]))
    score = 0.20 * q_mp + 0.25 * q_sep + 0.15 * q_usage + 0.15 * q_rank + 0.20 * q_decode + 0.05 * q_nonrand
    return {"CM_SRI": float(score), "Q_mp": q_mp, "Q_sep": q_sep, "Q_usage": q_usage, "Q_rank": q_rank, "Q_decode": q_decode, "Q_nonrand": q_nonrand}


def pair_redundancy(a, b):
    return float(np.clip(0.5 * cosine(a, b) + 0.5 * cosine(row_graph(a), row_graph(b)), 0.0, 1.0))


def joint_separation(c_list, temperature=0.12):
    z = np.concatenate(c_list, axis=1)
    distances = np.array([1.0 - cosine(z[a], z[b]) for a in range(z.shape[0]) for b in range(a + 1, z.shape[0])])
    hard = -temperature * np.log(np.mean(np.exp(-distances / temperature)) + EPS)
    return float(np.mean(distances)), float(np.clip(hard, 0.0, 1.0)), effective_rank(z)


def rescue_score(c_list, temperature=0.12):
    if len(c_list) < 2:
        return 0.0
    values = []
    classes = c_list[0].shape[0]
    for m, current in enumerate(c_list):
        others = np.concatenate([c for i, c in enumerate(c_list) if i != m], axis=1)
        pairs = [(a, b) for a in range(classes) for b in range(a + 1, classes)]
        other_distances = np.array([1.0 - cosine(others[a], others[b]) for a, b in pairs])
        weights = np.exp(-other_distances / temperature)
        weights /= weights.sum() + EPS
        values.append(float(np.sum(weights * np.array([js_np(current[a], current[b]) / np.log(2) for a, b in pairs]))))
    return float(np.mean(values))


def cm_jsri(c_list):
    c_list = [row_normalize(c) for c in c_list]
    javg, jhard, jrank = joint_separation(c_list)
    redundancies = [pair_redundancy(c_list[a], c_list[b]) for a in range(len(c_list)) for b in range(a + 1, len(c_list))]
    red = float(np.mean(redundancies)) if redundancies else 0.0
    rescue = rescue_score(c_list)
    decode = float(np.mean([bayes_decode(c)[0] for c in c_list]))
    score = 0.25 * jhard + 0.15 * jrank + 0.30 * rescue + 0.15 * decode + 0.15 * (1.0 - red)
    return {"CM_JSRI": float(np.clip(score, 0.0, 1.0)), "Q_jsep_avg": javg, "Q_jsep_hard": jhard, "Q_jrank": jrank, "Q_rescue": rescue, "Q_jdecode": decode, "Q_red": red}


class Observer(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        hidden = int(np.clip(2 ** round(np.log2(max(np.sqrt(input_dim * classes) * 2, 128))), 256, 768))
        bottleneck = max(hidden // 2, classes * 2)
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, bottleneck), nn.LayerNorm(bottleneck), nn.GELU(), nn.Dropout(0.15), nn.Linear(bottleneck, classes))

    def forward(self, x):
        return self.network(x)


def torch_confusion(labels, probabilities, classes):
    one_hot = F.one_hot(labels, classes).to(probabilities.dtype)
    counts = one_hot.sum(0)
    c = one_hot.T @ probabilities
    c = c / counts.clamp_min(1.0).unsqueeze(1)
    return c, counts > 0


def js_torch_rows(c):
    c = c.clamp_min(EPS)
    c = c / c.sum(1, keepdim=True)
    p = c[:, None, :]
    q = c[None, :, :]
    middle = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - middle.log())).sum(-1) + (q * (q.log() - middle.log())).sum(-1)) / np.log(2)


def recoverability_loss(labels, logits, classes, rescue_weights=None, other_confusions=None, progress=1.0):
    ce = F.cross_entropy(logits, labels, label_smoothing=0.02)
    p = F.softmax(logits.float(), dim=1)
    c, present = torch_confusion(labels, p, classes)
    c = c[present]
    if c.shape[0] < 4:
        return ce
    c = c / c.sum(1, keepdim=True).clamp_min(EPS)
    entropy = -(c * c.clamp_min(EPS).log()).sum(1) / np.log(classes)
    top1 = c.max(1).values
    top_r = c.topk(min(3, classes), dim=1).values.sum(1)
    multi_peak = F.relu(0.18 - entropy).square().mean() + F.relu(entropy - 0.78).square().mean() + 0.25 * F.relu(top1 - 0.96).square().mean() + F.relu(0.45 - top_r).square().mean() + F.relu(top_r - 0.99).square().mean()
    distances = js_torch_rows(c)
    eye = torch.eye(len(c), dtype=torch.bool, device=c.device)
    separation = torch.exp(-distances[~eye] / 0.15).mean()
    usage = c.mean(0)
    usage_entropy = -(usage * usage.clamp_min(EPS).log()).sum() / np.log(classes)
    usage_loss = F.relu(0.45 - usage_entropy).square() + F.relu(usage_entropy - 0.995).square() + F.relu(usage.max() - 0.45).square()
    uniform = torch.full_like(c, 1.0 / classes)
    nonrandom = -(c - uniform).square().sum(1).mean()
    decode_loss = torch.zeros((), device=logits.device)
    if c.shape[0] == classes:
        smooth = c + 1e-3
        smooth = smooth / smooth.sum(1, keepdim=True)
        reverse = smooth.T / smooth.T.sum(1, keepdim=True).clamp_min(EPS)
        decoded = smooth @ reverse
        decode_loss = -torch.diag(decoded).clamp_min(EPS).log().mean()
    rescue_loss = torch.zeros((), device=logits.device)
    redundancy_loss = torch.zeros((), device=logits.device)
    if rescue_weights is not None and c.shape[0] == classes:
        rescue_loss = -(rescue_weights.to(c.device) * distances).sum() / rescue_weights.sum().clamp_min(EPS)
    elif rescue_weights is not None:
        local_weights = rescue_weights.to(c.device)[present][:, present]
        rescue_loss = -(local_weights * distances).sum() / local_weights.sum().clamp_min(EPS)
    if other_confusions and c.shape[0] == classes:
        flat = c.flatten()
        values = []
        for other in other_confusions:
            other = other.to(c.device).flatten()
            values.append(F.cosine_similarity(flat, other, dim=0))
        redundancy_loss = torch.stack(values).mean()
    recoverability = multi_peak + 0.6 * separation + 0.4 * usage_loss + 0.10 * nonrandom + 0.15 * decode_loss + 0.45 * rescue_loss + 0.08 * redundancy_loss
    return ce + (0.05 + 0.20 * progress) * recoverability


class ObserverBundle:
    def __init__(self, name, input_dim, classes, device):
        self.name = name
        self.classes = classes
        self.device = device
        self.scaler = StandardScaler()
        self.model = Observer(input_dim, classes).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.5e-3, weight_decay=2e-4)
        self.grad_scaler = torch.GradScaler("cuda", enabled=device.type == "cuda")
        self.best_state = None
        self.best_controller_loss = float("inf")

    def transform(self, x, fit=False):
        output = self.scaler.fit_transform(x) if fit else self.scaler.transform(x)
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def scale_inplace(self, x, fit_indices, batch=2048):
        for start in range(0, len(fit_indices), batch):
            self.scaler.partial_fit(x[fit_indices[start:start + batch]])
        for start in range(0, len(x), batch):
            stop = min(start + batch, len(x))
            x[start:stop] = np.nan_to_num(self.scaler.transform(x[start:stop]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def train_epoch(self, x, y, sample_indices, classes, regime, progress=0.0, rescue_weights=None, other_confusions=None, augment=False):
        self.model.train()
        order = np.random.permutation(sample_indices)
        total = 0.0
        for start in range(0, len(order), OBSERVER_BATCH):
            idx = order[start:start + OBSERVER_BATCH]
            xb = torch.from_numpy(x[idx]).to(self.device, non_blocking=True)
            yb = torch.from_numpy(y[idx]).long().to(self.device, non_blocking=True)
            if augment:
                xb = xb + 0.015 * torch.randn_like(xb)
                keep = torch.rand_like(xb) > 0.015
                xb = xb * keep
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.02) if regime == "CE" else recoverability_loss(yb, logits, classes, rescue_weights, other_confusions, progress)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            total += float(loss.detach()) * len(idx)
        return total / len(order)

    def predict(self, x, indices=None, batch=2048):
        self.model.eval()
        output = []
        indices = np.arange(len(x)) if indices is None else np.asarray(indices)
        with torch.no_grad():
            for start in range(0, len(indices), batch):
                idx = indices[start:start + batch]
                xb = torch.from_numpy(x[idx]).to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                    p = F.softmax(self.model(xb), dim=1)
                output.append(p.float().cpu().numpy())
        return normalize_proba(np.concatenate(output))

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())


def rescue_weight_matrix(c_list, excluded, temperature=0.12):
    classes = c_list[0].shape[0]
    others = np.concatenate([c for i, c in enumerate(c_list) if i != excluded], axis=1)
    weights = np.zeros((classes, classes), dtype=np.float32)
    for a in range(classes):
        for b in range(a + 1, classes):
            distance = 1.0 - cosine(others[a], others[b])
            weights[a, b] = weights[b, a] = np.exp(-distance / temperature)
    weights /= max(weights.sum(), EPS)
    return torch.from_numpy(weights)


def train_observers(views, y, base_idx, controller_idx, classes, regime, fold_seed, warmup_epochs=OOF_WARMUP_EPOCHS, rost_cycles=OOF_ROST_CYCLES, report_prefix="Observers", reference_confusions=None):
    bundles = {}
    arrays = {}
    for position, (name, x) in enumerate(views.items()):
        seed_everything(fold_seed + position)
        bundle = ObserverBundle(name, x.shape[1], classes, DEVICES[position % len(DEVICES)])
        bundle.scale_inplace(x, base_idx)
        arrays[name] = x
        bundles[name] = bundle
    for epoch in range(warmup_epochs):
        for name, bundle in bundles.items():
            bundle.train_epoch(arrays[name], y, base_idx, classes, "CE", augment=True)
        print(f"{report_prefix} warm-up {epoch + 1}/{warmup_epochs}")
    if regime == "CE":
        for epoch in range(rost_cycles):
            for name, bundle in bundles.items():
                bundle.train_epoch(arrays[name], y, base_idx, classes, "CE", augment=True)
            print(f"{report_prefix} CE epoch {epoch + 1}/{rost_cycles}")
        return bundles, arrays, []
    history = []
    for cycle in range(rost_cycles):
        controller_prob = {name: bundle.predict(arrays[name], controller_idx) for name, bundle in bundles.items()}
        c_list = [soft_confusion(y[controller_idx], controller_prob[name], classes) for name in bundles]
        if reference_confusions is not None and cycle == 0:
            c_list = [0.5 * c + 0.5 * reference_confusions[name] for c, name in zip(c_list, bundles)]
        jsri_before = cm_jsri(c_list)
        for index, (name, bundle) in enumerate(bundles.items()):
            rescue = rescue_weight_matrix(c_list, index)
            others = [torch.from_numpy(c.astype(np.float32)) for j, c in enumerate(c_list) if j != index]
            bundle.train_epoch(arrays[name], y, base_idx, classes, "ROST", (cycle + 1) / rost_cycles, rescue, others, augment=True)
        controller_prob = {name: bundle.predict(arrays[name], controller_idx) for name, bundle in bundles.items()}
        c_list = [soft_confusion(y[controller_idx], controller_prob[name], classes) for name in bundles]
        jsri_after = cm_jsri(c_list)
        history.append({"cycle": cycle + 1, "before": jsri_before, "after": jsri_after, "CM_SRI": {name: cm_sri(c) for name, c in zip(bundles, c_list)}})
        print(f"{report_prefix} cycle {cycle + 1}/{rost_cycles} CM-JSRI={jsri_after['CM_JSRI']:.4f} rescue={jsri_after['Q_rescue']:.4f}")
    return bundles, arrays, history


def select_observers(names, controller_prob, y_controller, classes):
    confusions = {name: soft_confusion(y_controller, controller_prob[name], classes) for name in names}
    rankings = []
    for size in range(2, len(names) + 1):
        for members in combinations(names, size):
            c_list = [confusions[name] for name in members]
            joint = cm_jsri(c_list)
            individual = float(np.mean([cm_sri(c)["CM_SRI"] for c in c_list]))
            stacked = np.stack([controller_prob[name] for name in members], axis=1)
            probe_scores = []
            inner_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            for fit_idx, val_idx in inner_splitter.split(np.zeros(len(y_controller)), y_controller):
                weight = weighted_fusion_fit(y_controller[fit_idx], stacked[fit_idx], classes)
                weighted = weighted_fusion(stacked[val_idx], weight)
                product = product_fusion(stacked[val_idx])
                features_fit = top_level_features(stacked[fit_idx])
                features_val = top_level_features(stacked[val_idx])
                probe = LogisticRegression(C=0.15, max_iter=800, solver="lbfgs", class_weight="balanced", random_state=SEED)
                probe.fit(features_fit, y_controller[fit_idx])
                linear = sklearn_proba(probe, features_val, classes)
                probe_scores.append({"weighted": float(accuracy_score(y_controller[val_idx], weighted.argmax(1))), "product": float(accuracy_score(y_controller[val_idx], product.argmax(1))), "linear": float(accuracy_score(y_controller[val_idx], linear.argmax(1)))} )
            recovery_probe = {key: float(np.mean([row[key] for row in probe_scores])) for key in probe_scores[0]}
            score = 0.35 * recovery_probe["weighted"] + 0.30 * recovery_probe["product"] + 0.20 * recovery_probe["linear"] + 0.10 * joint["CM_JSRI"] + 0.05 * individual
            rankings.append({"members": list(members), "score": score, "CM_JSRI": joint, "mean_CM_SRI": individual, "recovery_probe": recovery_probe})
    rankings.sort(key=lambda row: row["score"], reverse=True)
    return rankings[0]["members"], confusions, rankings


def cross_fitted_posteriors(views, y, train_idx, test_idx, classes, regime, ce_fold_confusions=None):
    names = list(views)
    oof = {name: np.zeros((len(train_idx), classes), dtype=np.float32) for name in names}
    test_sum = {name: np.zeros((len(test_idx), classes), dtype=np.float64) for name in names}
    test_folds = {name: [] for name in names}
    parameter_counts = {name: [] for name in names}
    checkpoint_index = {name: [] for name in names}
    fold_reports = []
    fold_confusions = {}
    splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)
    local_labels = y[train_idx]
    for fold, (fit_local, holdout_local) in enumerate(splitter.split(np.zeros(len(train_idx)), local_labels), 1):
        fold_views = {name: views[name][train_idx].copy() for name in names}
        inner_fit, inner_controller = train_test_split(fit_local, test_size=0.18, random_state=SEED + fold, stratify=local_labels[fit_local])
        reference = ce_fold_confusions.get(fold) if ce_fold_confusions is not None else None
        bundles, arrays, history = train_observers(fold_views, local_labels, inner_fit, inner_controller, classes, regime, SEED + fold * 1000, OOF_WARMUP_EPOCHS, OOF_ROST_CYCLES, f"{regime} OOF fold {fold}", reference)
        fold_metrics = {}
        for name, bundle in bundles.items():
            holdout_probability = bundle.predict(arrays[name], holdout_local)
            oof[name][holdout_local] = holdout_probability.astype(np.float32)
            test_scaled = bundle.transform(views[name][test_idx])
            test_probability = bundle.predict(test_scaled)
            test_sum[name] += test_probability
            test_folds[name].append(test_probability.astype(np.float32))
            fold_metrics[name] = {"holdout_acc": float(accuracy_score(local_labels[holdout_local], holdout_probability.argmax(1))), "CM_SRI": cm_sri(soft_confusion(local_labels[holdout_local], holdout_probability, classes))}
            path = CKPT_DIR / f"{regime.lower()}_oof_observer_{safe_name(name)}_fold{fold}.pt"
            torch.save({"state_dict": bundle.model.cpu().state_dict(), "scaler_mean": bundle.scaler.mean_, "scaler_scale": bundle.scaler.scale_, "classes": bundle.classes, "input_dim": bundle.scaler.n_features_in_, "fold": fold, "regime": regime}, path)
            parameter_counts[name].append(bundle.count_params())
            checkpoint_index[name].append(str(path))
            del bundle
        fold_confusions[fold] = {name: soft_confusion(local_labels[holdout_local], oof[name][holdout_local], classes) for name in names}
        fold_reports.append({"fold": fold, "history": history, "metrics": fold_metrics})
        del fold_views, arrays, bundles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{regime} OOF fold {fold}/{OOF_FOLDS} completed")
    test_mean = {name: normalize_proba(test_sum[name] / OOF_FOLDS) for name in names}
    test_by_fold = {name: np.stack(test_folds[name], axis=0) for name in names}
    return oof, test_mean, test_by_fold, fold_reports, {name: int(np.mean(counts)) for name, counts in parameter_counts.items()}, checkpoint_index, fold_confusions


def posterior_context(x):
    x = x.clamp_min(EPS)
    mean = x.mean(1)
    std = x.std(1, unbiased=False)
    maximum = x.max(1).values
    entropy = -(x * x.log()).sum(2) / np.log(x.shape[2])
    top2 = x.topk(min(2, x.shape[2]), dim=2).values
    margin = top2[:, :, 0] - top2[:, :, -1]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float()
    return torch.cat([x.flatten(1), mean, std, maximum, entropy, margin, disagreement], dim=1)


def top_level_features(x):
    x = np.clip(np.asarray(x, dtype=np.float32), EPS, 1.0)
    mean = x.mean(1)
    std = x.std(1)
    maximum = x.max(1)
    entropy = -(x * np.log(x)).sum(2) / np.log(x.shape[2])
    sorted_p = np.sort(x, axis=2)
    margin = sorted_p[:, :, -1] - sorted_p[:, :, -2]
    disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).astype(np.float32)
    return np.concatenate([x.reshape(len(x), -1), mean, std, maximum, entropy, margin, disagreement], axis=1).astype(np.float32)


def augment_posteriors(x):
    logits = torch.log(x.clamp_min(EPS))
    temperature = torch.empty((len(x), x.shape[1], 1), device=x.device).uniform_(0.85, 1.15)
    jitter = 0.025 * torch.randn_like(logits)
    augmented = F.softmax(logits / temperature + jitter, dim=2)
    keep = (torch.rand((len(x), x.shape[1], 1), device=x.device) > 0.08).to(x.dtype)
    fallback = augmented.mean(1, keepdim=True)
    augmented = augmented * keep + fallback * (1.0 - keep)
    return augmented / augmented.sum(2, keepdim=True).clamp_min(EPS)


class PosteriorNet(nn.Module):
    def __init__(self, input_dim, classes, hidden_sizes):
        super().__init__()
        layers = []
        current = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15)])
            current = hidden
        layers.append(nn.Linear(current, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


class MoENet(nn.Module):
    def __init__(self, modalities, classes):
        super().__init__()
        input_dim = modalities * classes
        hidden = max(64, input_dim // 2)
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, modalities))

    def forward(self, x):
        gate = F.softmax(self.gate(x.flatten(1)), dim=1)
        return (gate.unsqueeze(2) * x).sum(1).clamp_min(EPS)


class RCFNet(nn.Module):
    def __init__(self, initial_confusions):
        super().__init__()
        modalities = len(initial_confusions)
        classes = initial_confusions[0].shape[0]
        context_dim = modalities * classes + 3 * classes + 3 * modalities
        reverse = []
        reliability = []
        for c in initial_confusions:
            decode, b, decoded = bayes_decode(c)
            reverse.append(b.T)
            reliability.append(np.clip(np.diag(decoded), 0.02, 1.0))
        reliability = np.stack(reliability)
        reliability /= reliability.sum(0, keepdims=True)
        self.reliability_logits = nn.Parameter(torch.log(torch.tensor(reliability, dtype=torch.float32)))
        self.calibration_scale = nn.Parameter(torch.ones(modalities, classes))
        self.calibration_bias = nn.Parameter(torch.zeros(modalities, classes))
        self.transport_logits = nn.Parameter(torch.log(torch.tensor(np.stack(reverse), dtype=torch.float32).clamp_min(EPS)))
        self.mix_logits = nn.Parameter(torch.log(torch.tensor([0.15, 0.20, 0.15, 0.05, 0.45], dtype=torch.float32)))
        path_output = nn.Linear(max(64, classes), 5)
        residual_output = nn.Linear(max(64, classes), classes, bias=False)
        self.path_gate = nn.Sequential(nn.Linear(context_dim, max(64, classes)), nn.GELU(), path_output)
        self.disagreement_gate = nn.Sequential(nn.Linear(context_dim, max(64, classes)), nn.GELU(), nn.Linear(max(64, classes), 1))
        self.residual = nn.Sequential(nn.Linear(context_dim, max(64, classes)), nn.GELU(), residual_output)
        nn.init.zeros_(path_output.weight)
        nn.init.zeros_(path_output.bias)
        nn.init.zeros_(residual_output.weight)

    def forward(self, x):
        x = x.clamp_min(EPS)
        calibrated = F.softmax(torch.log(x) * F.softplus(self.calibration_scale).unsqueeze(0) + self.calibration_bias.unsqueeze(0), dim=2)
        weights = F.softmax(self.reliability_logits, dim=0).unsqueeze(0)
        transport = F.softmax(self.transport_logits, dim=1)
        recovered = torch.einsum("nmk,myk->nmy", calibrated, transport)
        recovered = recovered / recovered.sum(2, keepdim=True).clamp_min(EPS)
        arithmetic = (weights * recovered).sum(1)
        geometric = F.softmax((weights * torch.log(calibrated.clamp_min(EPS))).sum(1), dim=1)
        bias_geometric = F.softmax((weights * torch.log(recovered.clamp_min(EPS))).sum(1), dim=1)
        raw = (weights * x).sum(1)
        product = F.softmax(torch.log(x.clamp_min(EPS)).sum(1), dim=1)
        context = posterior_context(x)
        mix = F.softmax(self.mix_logits.unsqueeze(0) + 0.5 * self.path_gate(context), dim=1)
        structured = mix[:, 0:1] * arithmetic + mix[:, 1:2] * geometric + mix[:, 2:3] * bias_geometric + mix[:, 3:4] * raw + mix[:, 4:5] * product
        disagreement = (x.argmax(2) != x.argmax(2)[:, :1]).float().mean(1, keepdim=True)
        uncertainty = 1.0 - x.max(2).values.mean(1, keepdim=True)
        gate = torch.sigmoid(self.disagreement_gate(context)) * (0.10 + 0.55 * disagreement + 0.35 * uncertainty)
        residual = F.softmax(torch.log(structured.clamp_min(EPS)) + self.residual(context), dim=1)
        output = (1.0 - gate) * structured + gate * residual
        return output / output.sum(1, keepdim=True).clamp_min(EPS)


class TemperatureCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.log(x.mean(1).clamp_min(EPS)) / (F.softplus(self.log_temperature) + 0.05)
        return F.softmax(logits, dim=1)


class MatrixCalibration(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear = nn.Linear(classes, classes)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.softmax(self.linear(torch.log(x.mean(1).clamp_min(EPS))), dim=1)


class DirichletCalibration(nn.Module):
    def __init__(self, modalities, classes):
        super().__init__()
        self.linear = nn.Linear(modalities * classes, classes)

    def forward(self, x):
        return F.softmax(self.linear(torch.log(x.clamp_min(EPS)).flatten(1)), dim=1)


def fit_torch_fusion(model, x, y, device, epochs=FUSION_EPOCHS, batch=1024, posterior_augmentation=False):
    model = model.to(device)
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.22, random_state=SEED, stratify=y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best = None
    best_loss = float("inf")
    patience = 25
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            xb = torch.from_numpy(x[idx]).float().to(device)
            yb = torch.from_numpy(y[idx]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            if posterior_augmentation:
                xb = augment_posteriors(xb)
            output = model(xb)
            loss = F.nll_loss(torch.log(output.clamp_min(EPS)), yb) + 0.08 * F.mse_loss(output, F.one_hot(yb, output.shape[1]).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(x[val_idx]).float().to(device))
            val_loss = float(F.nll_loss(torch.log(output.clamp_min(EPS)), torch.from_numpy(y[val_idx]).long().to(device)))
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best)
    return model


def torch_predict(model, x, device, batch=4096):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start:start + batch]).float().to(device)
            output.append(model(xb).float().cpu().numpy())
    return normalize_proba(np.concatenate(output))


def rcf_state_summary(model, initial_state):
    with torch.no_grad():
        reliability = F.softmax(model.reliability_logits, dim=0)
        transport = F.softmax(model.transport_logits, dim=1)
        mixture = F.softmax(model.mix_logits, dim=0)
        reliability_entropy = -(reliability * reliability.clamp_min(EPS).log()).sum(0).mean() / np.log(reliability.shape[0])
        transport_diagonal = torch.diagonal(transport, dim1=1, dim2=2).mean()
        transport_entropy = -(transport * transport.clamp_min(EPS).log()).sum(1).mean() / np.log(transport.shape[1])
        total_delta = 0.0
        total_initial = 0.0
        groups = {"reliability": 0.0, "calibration": 0.0, "transport": 0.0, "mixture": 0.0, "path_gate": 0.0, "gate": 0.0, "residual": 0.0}
        for name, parameter in model.state_dict().items():
            delta = float((parameter.detach().cpu() - initial_state[name]).float().norm())
            initial_norm = float(initial_state[name].float().norm())
            total_delta += delta ** 2
            total_initial += initial_norm ** 2
            key = "path_gate" if name.startswith("path_gate") else "gate" if name.startswith("disagreement_gate") else "residual" if name.startswith("residual") else "calibration" if name.startswith("calibration") else "transport" if name.startswith("transport") else "reliability" if name.startswith("reliability") else "mixture"
            groups[key] += delta ** 2
        return {"reliability_entropy": float(reliability_entropy), "reliability_max": float(reliability.max()), "transport_diagonal": float(transport_diagonal), "transport_entropy": float(transport_entropy), "calibration_scale_mean": float(F.softplus(model.calibration_scale).mean()), "calibration_bias_abs_mean": float(model.calibration_bias.abs().mean()), "mixture": mixture.detach().cpu().numpy(), "parameter_delta_ratio": float(np.sqrt(total_delta) / max(np.sqrt(total_initial), EPS)), "parameter_delta_groups": {name: float(np.sqrt(value)) for name, value in groups.items()}}


def fit_rcf_with_trace(model, x, y, device, monitor_x, monitor_y, mode, epochs, initial_state, monitor_fold_x=None):
    model = model.to(device)
    if mode == "standard":
        train_idx, validation_idx = train_test_split(np.arange(len(y)), test_size=0.22, random_state=SEED, stratify=y)
        selection_x, selection_y = x[validation_idx], y[validation_idx]
    else:
        train_idx = np.arange(len(y))
        selection_x, selection_y = x, y
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_state = None
    best_score = float("inf")
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        train_loss_sum = 0.0
        for start in range(0, len(order), 1024):
            idx = order[start:start + 1024]
            xb = torch.from_numpy(x[idx]).float().to(device)
            yb = torch.from_numpy(y[idx]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            xb = augment_posteriors(xb)
            output = model(xb)
            loss = F.nll_loss(torch.log(output.clamp_min(EPS)), yb) + 0.08 * F.mse_loss(output, F.one_hot(yb, output.shape[1]).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            selected_output = model(torch.from_numpy(selection_x).float().to(device))
            selected_loss = float(F.nll_loss(torch.log(selected_output.clamp_min(EPS)), torch.from_numpy(selection_y).long().to(device)))
            selected_f1 = float(f1_score(selection_y, selected_output.argmax(1).cpu().numpy(), average="macro", zero_division="warn"))
        selection_score = -selected_f1 + RCF_SELECTION_NLL_TIEBREAK * selected_loss if mode == "standard" else selected_loss
        if selection_score < best_score - 1e-5:
            best_score = selection_score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % RCF_TRACE_INTERVAL == 0 or epoch == epochs:
            if monitor_fold_x is None:
                monitor_probability = torch_predict(model, monitor_x, device)
            else:
                monitor_probability = normalize_proba(np.mean([torch_predict(model, fold_x, device) for fold_x in monitor_fold_x], axis=0))
            monitor_metric = compute_metrics(monitor_y, monitor_probability, monitor_probability.shape[1])
            row = {"epoch": epoch, "train_loss": train_loss_sum / len(order), "selection_loss": selected_loss, "selection_f1": selected_f1, "selection_score": selection_score, "monitor_acc": monitor_metric["acc"], "monitor_f1": monitor_metric["f1"], "monitor_ece": monitor_metric["ece"], "monitor_brier": monitor_metric["brier"], "monitor_nll": monitor_metric["nll"]}
            row.update(rcf_state_summary(model, initial_state))
            trace.append(row)
            print(f"DOME-X {mode} epoch={epoch}/{epochs} selection_nll={selected_loss:.4f} diagnostic_acc={monitor_metric['acc']:.4f} diagnostic_nll={monitor_metric['nll']:.4f}")
        if mode == "standard" and stale >= 30:
            break
    if best_state is None:
        raise RuntimeError("RCF training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    return model, trace


def compare_rcf_traces(standard_trace, leak_trace):
    leak_by_epoch = {row["epoch"]: row for row in leak_trace}
    comparison = []
    for standard in standard_trace:
        leak = leak_by_epoch.get(standard["epoch"])
        if leak is None:
            continue
        comparison.append({"epoch": standard["epoch"], "acc_gap_leak_minus_standard": leak["monitor_acc"] - standard["monitor_acc"], "nll_gap_leak_minus_standard": leak["monitor_nll"] - standard["monitor_nll"], "reliability_entropy_gap": leak["reliability_entropy"] - standard["reliability_entropy"], "transport_diagonal_gap": leak["transport_diagonal"] - standard["transport_diagonal"], "transport_entropy_gap": leak["transport_entropy"] - standard["transport_entropy"], "parameter_delta_ratio_gap": leak["parameter_delta_ratio"] - standard["parameter_delta_ratio"], "mixture_gap": (np.asarray(leak["mixture"]) - np.asarray(standard["mixture"])).tolist(), "parameter_delta_group_gap": {name: leak["parameter_delta_groups"][name] - standard["parameter_delta_groups"][name] for name in standard["parameter_delta_groups"]}})
    return comparison


def plot_rcf_traces(standard_trace, leak_trace, path):
    if not standard_trace or not leak_trace:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for trace, label, color in ((standard_trace, "DOME-X", "#1f77b4"), (leak_trace, "DOME-X_leak", "#d62728")):
        epoch = [row["epoch"] for row in trace]
        axes[0, 0].plot(epoch, [row["monitor_acc"] for row in trace], label=label, color=color)
        axes[0, 1].plot(epoch, [row["monitor_nll"] for row in trace], label=label, color=color)
        axes[1, 0].plot(epoch, [row["transport_diagonal"] for row in trace], label=label, color=color)
        axes[1, 1].plot(epoch, [row["parameter_delta_ratio"] for row in trace], label=label, color=color)
    axes[0, 0].set_title("Diagnostic accuracy")
    axes[0, 1].set_title("Diagnostic NLL")
    axes[1, 0].set_title("Bias transport diagonal")
    axes[1, 1].set_title("Parameter delta ratio")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sklearn_proba(model, x, classes):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
    else:
        scores = model.decision_function(x)
        scores -= scores.max(axis=1, keepdims=True)
        raw = np.exp(scores)
    output = np.full((len(x), classes), EPS, dtype=np.float64)
    for column, label in enumerate(model.classes_):
        output[:, int(label)] = raw[:, column]
    return normalize_proba(output)


def average_fusion(x):
    return normalize_proba(x.mean(1))


def product_fusion(x):
    return normalize_proba(np.exp(np.log(np.clip(x, EPS, 1.0)).mean(1)))


def weighted_fusion_fit(y, x, classes):
    reliability = []
    for m in range(x.shape[1]):
        c = soft_confusion(y, x[:, m], classes)
        reliability.append(np.clip(np.diag(bayes_decode(c)[2]), 0.02, 1.0))
    weights = np.stack(reliability)
    return weights / weights.sum(0, keepdims=True)


def weighted_fusion(x, weights):
    return normalize_proba((x * weights[None]).sum(1))


def add_torch_method(name, model, fusion_train_x, fusion_y, test_x, results, models, device):
    fitted = fit_torch_fusion(model, fusion_train_x, fusion_y, device, posterior_augmentation=fusion_train_x.ndim == 3)
    results[name] = torch_predict(fitted, test_x, device)
    models[name] = fitted.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_fusions(fusion_train_x, fusion_y, test_x, test_fold_x, confusions, classes, regime):
    results = {"Average": average_fusion(test_x), "Product": product_fusion(test_x)}
    models = {}
    weights = weighted_fusion_fit(fusion_y, fusion_train_x, classes)
    results["Weighted Average"] = weighted_fusion(test_x, weights)
    models["Weighted Average"] = weights
    flat_train = top_level_features(fusion_train_x)
    flat_test = top_level_features(test_x)
    input_dim = flat_train.shape[1]
    device = DEVICES[0]
    seed_everything(SEED)
    rcf = RCFNet(confusions)
    rcf_initial_state = {name: value.detach().cpu().clone() for name, value in rcf.state_dict().items()}
    standard_rcf, standard_trace = fit_rcf_with_trace(rcf, fusion_train_x, fusion_y, device, fusion_train_x, fusion_y, "standard", FUSION_EPOCHS, rcf_initial_state)
    fold_predictions = [torch_predict(standard_rcf, fold_x, device) for fold_x in test_fold_x]
    results["DOME-X RCF"] = normalize_proba(np.mean(fold_predictions, axis=0))
    models["DOME-X RCF"] = standard_rcf.cpu()
    save_json({"regime": regime, "training": "OOF posterior only", "standard_trace": standard_trace}, LOG_DIR / f"{regime.lower()}_rcf_training_trace.json")
    add_torch_method("MLP Stacking", PosteriorNet(input_dim, classes, [128, 64]), flat_train, fusion_y, flat_test, results, models, device)
    lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=SEED)
    lr.fit(flat_train, fusion_y)
    results["Logistic Regression Stacking"] = sklearn_proba(lr, flat_test, classes)
    models["Logistic Regression Stacking"] = lr
    return results, models, {"standard_trace": standard_trace}


def compute_metrics(y, p, classes):
    p = normalize_proba(p)
    pred = p.argmax(1)
    one_hot = np.eye(classes)[y]
    confidence = p.max(1)
    correctness = pred == y
    ece = 0.0
    bins = np.linspace(0, 1, 16)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, average="macro", zero_division="warn")), "precision": float(precision_score(y, pred, average="macro", zero_division="warn")), "recall": float(recall_score(y, pred, average="macro", zero_division="warn")), "ece": float(ece), "brier": float(np.square(p - one_hot).sum(1).mean()), "nll": float(log_loss(y, p, labels=np.arange(classes))), "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(classes)).tolist()}


def plot_confusion(y, p, title, path, classes):
    cm = confusion_matrix(y, np.argmax(p, axis=1), labels=np.arange(classes)).astype(np.float64)
    cm = row_normalize(cm)
    size = max(10, min(20, classes * 0.27))
    plt.figure(figsize=(size, size * 0.9))
    sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, square=True, cbar=True)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_soft_confusion(c, title, path):
    size = max(10, min(20, c.shape[0] * 0.27))
    plt.figure(figsize=(size, size * 0.9))
    sns.heatmap(row_normalize(c), cmap="Blues", vmin=0, vmax=1, square=True, cbar=True)
    plt.xlabel("Posterior class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parameter_count(model):
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    if isinstance(model, np.ndarray):
        return int(model.size)
    total = 0
    for attribute in ("coef_", "intercept_"):
        if hasattr(model, attribute):
            total += int(np.asarray(getattr(model, attribute)).size)
    return total


def format_parameters(value):
    value = int(value or 0)
    if value >= 1000000:
        return f"{value / 1000000:.3f}M"
    if value >= 1000:
        return f"{value / 1000:.2f}K"
    return str(value)


def print_rank_table(summary):
    header = "rank | model | acc | f1 | precision | recall | ece | brier | nll | params"
    print(header)
    for rank, row in enumerate(summary, 1):
        print(f"{rank:>4} | {row['model']:<30.30} | {row['acc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} | {format_parameters(row['parameters'])}")


def save_checkpoints(bundles, selected, fusion_models, metadata, observer_index=None):
    observer_index = {} if observer_index is None else observer_index
    for name, bundle in bundles.items():
        path = CKPT_DIR / f"observer_{safe_name(name)}.pt"
        torch.save({"state_dict": bundle.model.cpu().state_dict(), "scaler_mean": bundle.scaler.mean_, "scaler_scale": bundle.scaler.scale_, "classes": bundle.classes, "input_dim": bundle.scaler.n_features_in_}, path)
        observer_index[name] = str(path)
    fusion_index = {}
    sklearn_models = {}
    for name, model in fusion_models.items():
        if isinstance(model, nn.Module):
            path = CKPT_DIR / f"fusion_{safe_name(name)}.pt"
            torch.save({"state_dict": model.state_dict(), "class": model.__class__.__name__}, path)
            fusion_index[name] = str(path)
        else:
            sklearn_models[name] = model
    if sklearn_models:
        path = CKPT_DIR / "fusion_sklearn_models.pkl"
        with open(path, "wb") as f:
            pickle.dump(sklearn_models, f)
        fusion_index["sklearn"] = str(path)
    save_json({"selected": selected, "observers": observer_index, "fusion_models": fusion_index, "metadata": metadata}, CKPT_DIR / "checkpoint_index.json")


def main():
    initialize_output_dirs()
    print(f"NTU RGB+D 60 DOME-X CE/ROST v{RUN_VERSION} protocol={PROTOCOL} devices={[str(d) for d in DEVICES]}")
    views, y, subjects, cameras, feature_meta = build_or_load_features()
    classes = int(np.max(y)) + 1
    train_mask, test_mask = official_split(subjects, cameras)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    dataset_info = {"total": len(y), "official_train": len(train_idx), "oof_folds": OOF_FOLDS, "test": len(test_idx), "classes": classes, "views": {name: int(x.shape[1]) for name, x in views.items()}, "protocol": PROTOCOL}
    save_json(dataset_info, LOG_DIR / "dataset_info.json")
    print(f"Samples total={len(y)} official_train={len(train_idx)} oof_folds={OOF_FOLDS} test={len(test_idx)}")
    ce_data = cross_fitted_posteriors(views, y, train_idx, test_idx, classes, "CE")
    ce_oof, ce_test, ce_test_folds, ce_history, ce_parameters, ce_checkpoints, ce_fold_confusions = ce_data
    save_json(ce_history, LOG_DIR / "ce_oof_history.json")
    rost_data = cross_fitted_posteriors(views, y, train_idx, test_idx, classes, "ROST", ce_fold_confusions)
    rost_oof, rost_test, rost_test_folds, rost_history, rost_parameters, rost_checkpoints, _ = rost_data
    save_json(rost_history, LOG_DIR / "rost_oof_history.json")
    selected, _, combo_ranking = select_observers(list(ce_oof), ce_oof, y[train_idx], classes)
    save_json(combo_ranking, LOG_DIR / "ce_selected_observer_ranking.json")
    print(f"Fixed observers from CE OOF selection={selected}")
    all_regimes = {"CE": (ce_oof, ce_test, ce_test_folds, ce_parameters, ce_checkpoints), "ROST": (rost_oof, rost_test, rost_test_folds, rost_parameters, rost_checkpoints)}
    all_metrics = {}; summary = []; recoverability = {}; model_index = {}
    for regime, (oof_prob, test_prob, test_fold_prob, observer_parameters, observer_checkpoints) in all_regimes.items():
        observer_report = {}
        for name in oof_prob:
            confusion = soft_confusion(y[train_idx], oof_prob[name], classes)
            observer_report[name] = {"CM_SRI": cm_sri(confusion), "parameters": observer_parameters[name], "selected": name in selected, "oof_acc": float(accuracy_score(y[train_idx], oof_prob[name].argmax(1)))}
            plot_soft_confusion(confusion, f"{regime} {name} OOF soft confusion", LOG_DIR / f"cm_soft_{regime.lower()}_{safe_name(name)}.png")
            plot_confusion(y[test_idx], test_prob[name], f"{regime} {name} normalized confusion matrix", LOG_DIR / f"cm_norm_{regime.lower()}_submodel_{safe_name(name)}.png", classes)
        selected_confusions = [soft_confusion(y[train_idx], oof_prob[name], classes) for name in selected]
        recoverability[regime] = {"observers": observer_report, "CM_JSRI": cm_jsri(selected_confusions)}
        fusion_train_x = np.stack([oof_prob[name] for name in selected], axis=1).astype(np.float32)
        test_x = np.stack([test_prob[name] for name in selected], axis=1).astype(np.float32)
        test_fold_x = [np.stack([test_fold_prob[name][fold] for name in selected], axis=1).astype(np.float32) for fold in range(OOF_FOLDS)]
        fusion_results, fusion_models, fusion_trace = run_fusions(fusion_train_x, y[train_idx], test_x, test_fold_x, selected_confusions, classes, regime)
        recoverability[regime]["rcf_trace"] = fusion_trace
        all_results = {f"Submodel {name}": test_prob[name] for name in oof_prob}
        all_results.update(fusion_results)
        best_expert_acc = max(accuracy_score(y[test_idx], test_prob[name].argmax(1)) for name in oof_prob)
        for name, probability in all_results.items():
            if name not in COMPARISON_FUSIONS and not name.startswith("Submodel "):
                continue
            metric = compute_metrics(y[test_idx], probability, classes); key = f"{regime} | {name}"; all_metrics[key] = metric
            parameters = observer_parameters[name.replace("Submodel ", "")] if name.startswith("Submodel ") else parameter_count(fusion_models.get(name))
            summary.append({"regime": regime, "model": name, "acc": metric["acc"], "f1": metric["f1"], "precision": metric["precision"], "recall": metric["recall"], "ece": metric["ece"], "brier": metric["brier"], "nll": metric["nll"], "fusion_gain": metric["acc"] - best_expert_acc, "parameters": parameters})
            plot_confusion(y[test_idx], probability, f"{regime} {name} normalized confusion matrix", LOG_DIR / f"cm_norm_{regime.lower()}_{safe_name(name)}.png", classes)
        model_index[regime] = {"observers": observer_checkpoints, "fusion": fusion_models}
    comparison = []
    for method in sorted(set(row["model"] for row in summary)):
        ce_row = next(row for row in summary if row["regime"] == "CE" and row["model"] == method)
        rost_row = next(row for row in summary if row["regime"] == "ROST" and row["model"] == method)
        comparison.append({"model": method, "ce_acc": ce_row["acc"], "rost_acc": rost_row["acc"], "delta_acc": rost_row["acc"] - ce_row["acc"], "ce_f1": ce_row["f1"], "rost_f1": rost_row["f1"], "delta_f1": rost_row["f1"] - ce_row["f1"], "ce_ece": ce_row["ece"], "rost_ece": rost_row["ece"], "delta_ece": rost_row["ece"] - ce_row["ece"], "ce_nll": ce_row["nll"], "rost_nll": rost_row["nll"], "delta_nll": rost_row["nll"] - ce_row["nll"]})
    summary.sort(key=lambda row: row["acc"], reverse=True)
    metadata = {"version": RUN_VERSION, "timestamp": datetime.now().isoformat(), "selected_observers": selected, "dataset": dataset_info, "feature_meta": feature_meta, "fusion_training": "out_of_fold_posteriors", "comparison_scope": "CE and ROST share views, folds, observer architecture, initialization seeds, augmentation, optimizer, early stopping and fusion budgets. Only the observer objective differs.", "test_protocol": "Official test labels are used only for final evaluation."}
    save_json(metadata, LOG_DIR / "run_meta.json")
    save_json(all_metrics, LOG_DIR / "all_metrics.json")
    save_json(summary, LOG_DIR / "ranked_summary.json")
    save_json(comparison, LOG_DIR / "ce_vs_rost_comparison.json")
    save_json(recoverability, LOG_DIR / "recoverability_report.json")
    with open(LOG_DIR / "ranked_summary.csv", "w", encoding="utf-8") as f:
        f.write("regime,model,acc,f1,precision,recall,ece,brier,nll,fusion_gain,parameters\n")
        for row in summary: f.write(",".join(str(row[key]) for key in ("regime", "model", "acc", "f1", "precision", "recall", "ece", "brier", "nll", "fusion_gain", "parameters")) + "\n")
    with open(LOG_DIR / "ce_vs_rost_comparison.csv", "w", encoding="utf-8") as f:
        f.write("model,ce_acc,rost_acc,delta_acc,ce_f1,rost_f1,delta_f1,ce_ece,rost_ece,delta_ece,ce_nll,rost_nll,delta_nll\n")
        for row in comparison: f.write(",".join(str(row[key]) for key in ("model", "ce_acc", "rost_acc", "delta_acc", "ce_f1", "rost_f1", "delta_f1", "ce_ece", "rost_ece", "delta_ece", "ce_nll", "rost_nll", "delta_nll")) + "\n")
    for regime, items in model_index.items():
        with open(CKPT_DIR / f"{regime.lower()}_fusion_models.pkl", "wb") as f: pickle.dump({name: model for name, model in items["fusion"].items() if not isinstance(model, nn.Module)}, f)
        for name, model in items["fusion"].items():
            if isinstance(model, nn.Module): torch.save({"state_dict": model.state_dict(), "regime": regime, "class": model.__class__.__name__}, CKPT_DIR / f"{regime.lower()}_fusion_{safe_name(name)}.pt")
    print("rank | regime | model | acc | f1 | precision | recall | ece | brier | nll | fusion_gain | params")
    for rank, row in enumerate(summary, 1): print(f"{rank:>4} | {row['regime']:<6} | {row['model']:<20.20} | {row['acc']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} | {row['fusion_gain']:+.4f} | {format_parameters(row['parameters'])}")
    print("model | CE acc | ROST acc | delta acc | CE f1 | ROST f1 | delta f1 | CE nll | ROST nll")
    for row in comparison: print(f"{row['model']:<20.20} | {row['ce_acc']:.4f} | {row['rost_acc']:.4f} | {row['delta_acc']:+.4f} | {row['ce_f1']:.4f} | {row['rost_f1']:.4f} | {row['delta_f1']:+.4f} | {row['ce_nll']:.4f} | {row['rost_nll']:.4f}")
    print(f"Checkpoints={CKPT_DIR}")
    print(f"Logs={LOG_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DOME-X on NTU RGB+D 60.")
    parser.add_argument("--check", action="store_true", help="validate dependencies and data paths")
    return parser.parse_args()


def check_environment():
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Missing project code directory: {PROJECT_ROOT}")
    if not DATA_ROOT.is_dir() and not CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing NTU data and feature cache: {DATA_ROOT}")
    print(
        f"NTU RGB+D check passed: data={DATA_ROOT.is_dir()} cache={CACHE_DIR.exists()} "
        f"device_count={len(DEVICES)}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        check_environment()
    else:
        main()
