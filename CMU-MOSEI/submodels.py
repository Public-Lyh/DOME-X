# -*- coding: utf-8 -*-
import os
import json
import math
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42

ROOT = Path("/home/luoyh/deep_learning_project/Code/CMU-MOSEI")
PROCESSED_DIR = ROOT / "processed"
EXPERT_ROOT = PROCESSED_DIR / "CMU_MOSEI_RAW_EXPERTS_V1"
OUTPUT_DIR = EXPERT_ROOT / "outputs"
MODEL_DIR = EXPERT_ROOT / "models"
REPORT_DIR = EXPERT_ROOT / "reports"

EXPERTS = ["text", "audio", "text_audio"]

K = 7
CLASS_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 90
PATIENCE = 18
LR = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 512
EMBED = 256
DROPOUT = 0.25
GRAD_CLIP = 1.0
NUM_WORKERS = 2

FOCAL_GAMMA = 1.5
PRIOR_TAU = 0.55
SOFT_LABEL_SIGMA = 0.75
EMA_MOMENTUM = 0.94
MIN_CLASS_BATCH = 2
EPS = 1e-8

CE_W = 1.00
FOCAL_W = 0.45
ORD_W = 0.28
REG_W = 0.18
STAB_W = 0.10
COV_W = 0.18
SEP_W = 0.12
PEAK_W = 0.04
PRIOR_W = 0.06
ENT_W = 0.02


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def entropy_np(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, EPS, 1.0)
    return -np.sum(prob * np.log(prob), axis=-1)


def normalize_prob_np(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.maximum(prob, EPS)
    prob = prob / np.maximum(prob.sum(axis=-1, keepdims=True), EPS)
    return prob.astype(np.float32)


def prob_to_score_np(prob):
    prob = normalize_prob_np(prob)
    return np.sum(prob * CLASS_VALUES.reshape(1, -1), axis=1).astype(np.float32)


def score_to_label7(reg):
    reg = np.asarray(reg, dtype=np.float32)
    y = np.rint(reg).astype(np.int64) + 3
    return np.clip(y, 0, K - 1)


def label7_to_reg(y):
    return CLASS_VALUES[np.asarray(y, dtype=np.int64)]


def corrcoef(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def acc2(score, reg):
    score = np.asarray(score).reshape(-1)
    reg = np.asarray(reg).reshape(-1)
    mask = reg != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((score[mask] > 0) == (reg[mask] > 0)))


def metrics_from_prob(prob, y, reg):
    prob = normalize_prob_np(prob)
    pred = prob.argmax(axis=1)
    score = prob_to_score_np(prob)
    return {
        "acc7": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(reg, score)),
        "corr": corrcoef(reg, score),
        "acc2": acc2(score, reg),
        "pred_count": np.bincount(pred, minlength=K).tolist(),
    }


def print_metric(name, metric):
    print(
        f"{name:<26} "
        f"Acc7={metric['acc7'] * 100:7.2f}% "
        f"F1M={metric['f1_macro'] * 100:7.2f}% "
        f"F1W={metric['f1_weighted'] * 100:7.2f}% "
        f"MAE={metric['mae']:.4f} "
        f"Corr={metric['corr']:.4f} "
        f"Acc2={metric['acc2'] * 100:7.2f}% "
        f"pred={metric['pred_count']}"
    )


def candidate_npz_files():
    files = []
    roots = [
        EXPERT_ROOT / "features",
        EXPERT_ROOT / "outputs",
        PROCESSED_DIR,
    ]
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.npz"))
    return sorted(set(files))


def has_basic_keys(data):
    keys = set(data.files)
    has_id = "ids" in keys or "id" in keys
    has_y = "y" in keys or "label" in keys or "labels" in keys or "reg" in keys
    return has_id and has_y


def modality_score(path, keys, expert, split):
    text = f"{path.name.lower()} {str(path.parent).lower()} {' '.join([k.lower() for k in keys])}"
    score = 0

    if split in text:
        score += 80

    if expert == "text":
        positives = ["text", "bert", "language", "transcript"]
        negatives = ["audio", "covarep", "wav", "acoustic", "visual", "video", "image", "facet"]
    elif expert == "audio":
        positives = ["audio", "covarep", "wav", "acoustic"]
        negatives = ["text", "bert", "language", "transcript", "visual", "video", "image", "facet"]
    else:
        positives = []
        negatives = []

    for word in positives:
        if word in text:
            score += 20
    for word in negatives:
        if word in text:
            score -= 40

    if "outputs" in text and expert in text and split in text:
        score += 30

    return score


def pick_feature_key(data, expert):
    keys = list(data.files)
    bad = {
        "ids", "id", "y", "label", "labels", "reg", "score", "prob", "pred",
        "split", "text", "raw_text", "start", "end", "video_id", "segment_id",
    }

    preferred = []
    for key in keys:
        low = key.lower()
        if expert == "text" and any(w in low for w in ["text", "bert", "language", "transcript"]):
            preferred.append(key)
        if expert == "audio" and any(w in low for w in ["audio", "covarep", "wav", "acoustic"]):
            preferred.append(key)

    candidates = preferred + [k for k in keys if k.lower() not in bad]
    best_key = None
    best_dim = -1

    for key in candidates:
        try:
            arr = data[key]
        except Exception:
            continue
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] >= 10:
            if arr.shape[1] > best_dim:
                best_key = key
                best_dim = arr.shape[1]

    return best_key


def find_single_modality_split(expert, split):
    best = None
    best_score = -10**9

    for path in candidate_npz_files():
        low_path = str(path).lower()
        if split not in low_path:
            continue

        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue

        if not has_basic_keys(data):
            continue

        key = pick_feature_key(data, expert)
        if key is None:
            continue

        score = modality_score(path, data.files, expert, split)

        if score > best_score:
            best_score = score
            best = (path, key, list(data.files))

    if best is None or best_score < 20:
        raise RuntimeError(
            f"找不到 {expert}/{split} 的可信特征文件。为避免误读模态，本脚本不会用其他模态替代。"
        )

    return best


def read_npz_split(expert, split):
    path, feature_key, keys = find_single_modality_split(expert, split)
    data = np.load(path, allow_pickle=True)

    ids_key = "ids" if "ids" in data.files else "id"
    ids = data[ids_key].astype(str)

    if "y" in data.files:
        y = data["y"].astype(np.int64)
        if y.min() < 0:
            y = score_to_label7(y)
    elif "label" in data.files:
        y = data["label"].astype(np.int64)
        if y.min() < 0:
            y = score_to_label7(y)
    elif "labels" in data.files:
        y = data["labels"].astype(np.int64)
        if y.min() < 0:
            y = score_to_label7(y)
    else:
        y = score_to_label7(data["reg"])

    if "reg" in data.files:
        reg = data["reg"].astype(np.float32)
    elif "score" in data.files:
        reg = data["score"].astype(np.float32)
    else:
        reg = label7_to_reg(y).astype(np.float32)

    x = data[feature_key].astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if len(ids) != len(y) or len(y) != x.shape[0]:
        raise RuntimeError(
            f"{expert}/{split} 长度不一致: ids={len(ids)}, y={len(y)}, x={x.shape}, path={path}"
        )

    return {
        "expert": expert,
        "split": split,
        "ids": ids,
        "y": y,
        "reg": reg,
        "x": x,
        "source": str(path),
        "feature_key": feature_key,
        "keys": keys,
    }


def align_by_ids(objs):
    common = set(objs[0]["ids"].tolist())
    for obj in objs[1:]:
        common &= set(obj["ids"].tolist())
    common = sorted(common)

    if len(common) == 0:
        raise RuntimeError("不同模态之间没有共同 ids，无法构造 text_audio。")

    aligned = []
    for obj in objs:
        idx_map = {sid: i for i, sid in enumerate(obj["ids"].tolist())}
        idx = np.asarray([idx_map[sid] for sid in common], dtype=np.int64)
        aligned.append({
            **obj,
            "ids": obj["ids"][idx],
            "y": obj["y"][idx],
            "reg": obj["reg"][idx],
            "x": obj["x"][idx],
        })

    y0 = aligned[0]["y"]
    r0 = aligned[0]["reg"]
    for obj in aligned[1:]:
        if not np.array_equal(y0, obj["y"]):
            raise RuntimeError("对齐后不同模态 y 不一致。")

    return aligned, np.asarray(common), y0, r0


def load_expert_data(expert):
    if expert in ["text", "audio"]:
        return {split: read_npz_split(expert, split) for split in ["train", "val", "test"]}

    if expert == "text_audio":
        out = {}
        for split in ["train", "val", "test"]:
            text_obj = read_npz_split("text", split)
            audio_obj = read_npz_split("audio", split)
            aligned, ids, y, reg = align_by_ids([text_obj, audio_obj])
            x = np.concatenate([aligned[0]["x"], aligned[1]["x"]], axis=1).astype(np.float32)
            out[split] = {
                "expert": expert,
                "split": split,
                "ids": ids,
                "y": y,
                "reg": reg,
                "x": x,
                "source": f"{aligned[0]['source']} + {aligned[1]['source']}",
                "feature_key": f"{aligned[0]['feature_key']} + {aligned[1]['feature_key']}",
                "keys": [],
            }
        return out

    raise ValueError(expert)


class ExpertDataset(Dataset):
    def __init__(self, obj, scaler=None, fit=False):
        self.ids = obj["ids"].astype(str)
        self.y = obj["y"].astype(np.int64)
        self.reg = obj["reg"].astype(np.float32)

        x = obj["x"].astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self.scaler = StandardScaler()
            self.x = self.scaler.fit_transform(x).astype(np.float32)
        else:
            self.scaler = scaler
            self.x = self.scaler.transform(x).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "reg": torch.tensor(self.reg[idx], dtype=torch.float32),
        }


def make_sampler(y):
    count = np.bincount(y, minlength=K).astype(np.float64)
    count = np.maximum(count, 1.0)
    w = 1.0 / np.power(count[y], 0.55)
    w = w / w.mean()
    return WeightedRandomSampler(
        torch.tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
    )


def make_loader(ds, train=False):
    if train:
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            sampler=make_sampler(ds.y),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


class OrdinalDomeExpert(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, EMBED),
            nn.LayerNorm(EMBED),
            nn.GELU(),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED, K),
        )
        self.reg_head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED, 1),
        )
        self.unc_head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.class_head(feat)
        reg = self.reg_head(feat).squeeze(1)
        unc = torch.sigmoid(self.unc_head(feat).squeeze(1))
        return logits, reg, unc, feat


class EMAConfusion:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.cm = torch.ones(num_classes, num_classes, device=device) / num_classes
        self.ready = False

    @torch.no_grad()
    def update(self, prob, y):
        onehot = F.one_hot(y, self.num_classes).float()
        mat = onehot.t() @ prob
        row_sum = mat.sum(dim=1, keepdim=True)
        valid = row_sum.squeeze(1) >= MIN_CLASS_BATCH
        mat = mat / row_sum.clamp_min(EPS)

        if valid.any():
            if not self.ready:
                self.cm[valid] = mat[valid]
                self.ready = True
            else:
                self.cm[valid] = EMA_MOMENTUM * self.cm[valid] + (1.0 - EMA_MOMENTUM) * mat[valid]

        self.cm = self.cm / self.cm.sum(dim=1, keepdim=True).clamp_min(EPS)

    def get(self):
        return self.cm.detach()


def entropy_torch(prob, dim=-1):
    prob = torch.clamp(prob, EPS, 1.0)
    return -(prob * torch.log(prob)).sum(dim=dim)


def soft_ordinal_target(y, sigma=SOFT_LABEL_SIGMA):
    values = torch.arange(K, device=y.device, dtype=torch.float32).view(1, -1)
    center = y.float().view(-1, 1)
    q = torch.exp(-((values - center) ** 2) / (2.0 * sigma * sigma))
    q = q / q.sum(dim=1, keepdim=True).clamp_min(EPS)
    return q


def class_weights_np(y):
    count = np.bincount(y, minlength=K).astype(np.float64)
    w = count.sum() / np.maximum(count, 1.0)
    w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def balanced_softmax_loss(logits, y, class_count):
    prior = class_count / class_count.sum().clamp_min(EPS)
    adjusted = logits + torch.log(prior.clamp_min(EPS)).view(1, -1)
    return F.cross_entropy(adjusted, y)


def focal_loss(logits, y, class_weight):
    ce = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** FOCAL_GAMMA) * ce).mean()


def soft_label_ordinal_loss(logits, y):
    logp = F.log_softmax(logits, dim=1)
    target = soft_ordinal_target(y)
    return F.kl_div(logp, target, reduction="batchmean")


def expected_score(prob):
    values = torch.tensor(CLASS_VALUES, dtype=prob.dtype, device=prob.device)
    return (prob * values.view(1, -1)).sum(dim=1)


def ordinal_distance_loss(prob, y):
    values = torch.arange(K, device=prob.device, dtype=prob.dtype).view(1, -1)
    center = y.float().view(-1, 1)
    dist = torch.abs(values - center) / (K - 1)
    return (prob * dist).sum(dim=1).mean()


def sebi_regularizers(prob, y, ema_cm, target_prior):
    ema_cm.update(prob.detach(), y)

    onehot = F.one_hot(y, K).float()
    batch_mat = onehot.t() @ prob
    row_count = onehot.sum(dim=0).view(K, 1)
    valid = row_count.squeeze(1) >= MIN_CLASS_BATCH
    batch_cm = batch_mat / row_count.clamp_min(EPS)

    cm = ema_cm.get().clone()
    if valid.any():
        cm[valid] = 0.45 * cm[valid] + 0.55 * batch_cm[valid]
    cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(EPS)

    row_entropy = entropy_torch(cm, dim=1).mean() / math.log(K)

    q = cm.mean(dim=0)
    q = q / q.sum().clamp_min(EPS)
    coverage_loss = F.kl_div(torch.log(q.clamp_min(EPS)), target_prior, reduction="sum")

    sims = []
    cm_norm = F.normalize(cm, p=2, dim=1)
    for i in range(K):
        for j in range(i + 1, K):
            sims.append((cm_norm[i] * cm_norm[j]).sum())
    sep_loss = torch.stack(sims).mean() if sims else prob.new_tensor(0.0)

    peak_loss = 1.0 - cm.max(dim=1).values.mean()
    pred_ent = entropy_torch(prob, dim=1).mean() / math.log(K)

    return row_entropy, coverage_loss, sep_loss, peak_loss, pred_ent, cm


def prediction_balance_penalty(prob, target_prior):
    q = prob.mean(dim=0)
    q = q / q.sum().clamp_min(EPS)
    return F.kl_div(torch.log(q.clamp_min(EPS)), target_prior, reduction="sum")


@torch.no_grad()
def infer(model, loader):
    model.eval()
    ids_all = []
    y_all = []
    reg_all = []
    prob_all = []
    feat_all = []
    unc_all = []

    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)
        logits, reg_pred, unc, feat = model(x)
        prob = F.softmax(logits, dim=1)

        ids_all.extend(batch["id"])
        y_all.append(batch["y"].numpy())
        reg_all.append(batch["reg"].numpy())
        prob_all.append(prob.cpu().numpy())
        feat_all.append(feat.cpu().numpy())
        unc_all.append(unc.cpu().numpy())

    ids = np.asarray(ids_all).astype(str)
    y = np.concatenate(y_all).astype(np.int64)
    reg = np.concatenate(reg_all).astype(np.float32)
    prob = normalize_prob_np(np.concatenate(prob_all, axis=0))
    feat = np.concatenate(feat_all, axis=0).astype(np.float32)
    unc = np.concatenate(unc_all, axis=0).astype(np.float32)

    score = prob_to_score_np(prob)
    ent = entropy_np(prob).reshape(-1, 1) / math.log(K)

    feature = np.concatenate(
        [
            feat,
            prob,
            np.log(np.maximum(prob, EPS)),
            score.reshape(-1, 1),
            prob.max(axis=1, keepdims=True),
            ent,
            unc.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    return ids, y, reg, prob, feature


def save_outputs(expert, split, ids, y, reg, prob, feature):
    mkdir(OUTPUT_DIR)
    prob = normalize_prob_np(prob)
    score = prob_to_score_np(prob)
    pred = prob.argmax(axis=1).astype(np.int64)

    np.savez_compressed(
        OUTPUT_DIR / f"{expert}_{split}_outputs.npz",
        ids=ids.astype(str),
        y=y.astype(np.int64),
        reg=reg.astype(np.float32),
        prob=prob.astype(np.float32),
        score=score.astype(np.float32),
        pred=pred,
        feature=feature.astype(np.float32),
    )


def save_confusion(name, y, prob):
    mkdir(REPORT_DIR)
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(K))).astype(np.float64)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    np.save(REPORT_DIR / f"{name}_cm_norm.npy", cmn.astype(np.float32))


def train_one_expert(expert):
    print("\n" + "=" * 130)
    print(f"TRAIN DOME-X ORDINAL CEPL EXPERT: {expert}")
    print("=" * 130)

    raw = load_expert_data(expert)

    for split in ["train", "val", "test"]:
        obj = raw[split]
        print(f"{expert}/{split}: n={len(obj['y'])} x={obj['x'].shape} source={obj['source']} key={obj['feature_key']}")
        print(f"{expert}/{split}: label7={dict(sorted(Counter(obj['y'].tolist()).items()))}")

    train_set = ExpertDataset(raw["train"], fit=True)
    val_set = ExpertDataset(raw["val"], scaler=train_set.scaler)
    test_set = ExpertDataset(raw["test"], scaler=train_set.scaler)

    train_loader = make_loader(train_set, train=True)
    eval_train_loader = make_loader(train_set, train=False)
    val_loader = make_loader(val_set, train=False)
    test_loader = make_loader(test_set, train=False)

    model = OrdinalDomeExpert(train_set.x.shape[1]).to(DEVICE)

    count_np = np.bincount(train_set.y, minlength=K).astype(np.float32)
    count_t = torch.tensor(count_np, dtype=torch.float32, device=DEVICE).clamp_min(1.0)
    class_weight = class_weights_np(train_set.y)

    prior = count_t / count_t.sum()
    target_prior = torch.pow(prior, PRIOR_TAU)
    target_prior = target_prior / target_prior.sum().clamp_min(EPS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ema_cm = EMAConfusion(K, DEVICE)

    best_score = -1e18
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = defaultdict(float)
        total_n = 0

        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            reg = batch["reg"].to(DEVICE, non_blocking=True)

            logits, reg_pred, unc, feat = model(x)
            prob = F.softmax(logits, dim=1)

            loss_ce = balanced_softmax_loss(logits, y, count_t)
            loss_focal = focal_loss(logits, y, class_weight)
            loss_ord_soft = soft_label_ordinal_loss(logits, y)
            loss_ord_dist = ordinal_distance_loss(prob, y)
            loss_reg = F.smooth_l1_loss(reg_pred, reg)
            loss_score = F.smooth_l1_loss(expected_score(prob), reg)

            loss_stab, loss_cov, loss_sep, loss_peak, loss_ent, cm = sebi_regularizers(prob, y, ema_cm, target_prior)
            loss_prior = prediction_balance_penalty(prob, target_prior)

            r = min(1.0, max(0.0, (epoch - 6) / max(1, EPOCHS - 6)))

            loss = (
                CE_W * loss_ce
                + FOCAL_W * loss_focal
                + ORD_W * (0.60 * loss_ord_soft + 0.40 * loss_ord_dist)
                + REG_W * (0.50 * loss_reg + 0.50 * loss_score)
                + r * STAB_W * loss_stab
                + r * COV_W * loss_cov
                + r * SEP_W * loss_sep
                + r * PEAK_W * loss_peak
                + r * PRIOR_W * loss_prior
                + r * ENT_W * loss_ent
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bs = len(y)
            total["loss"] += float(loss.detach().cpu()) * bs
            total["ce"] += float(loss_ce.detach().cpu()) * bs
            total["focal"] += float(loss_focal.detach().cpu()) * bs
            total["ord"] += float((loss_ord_soft + loss_ord_dist).detach().cpu()) * bs
            total["reg"] += float((loss_reg + loss_score).detach().cpu()) * bs
            total["stab"] += float(loss_stab.detach().cpu()) * bs
            total["cov"] += float(loss_cov.detach().cpu()) * bs
            total["sep"] += float(loss_sep.detach().cpu()) * bs
            total["peak"] += float(loss_peak.detach().cpu()) * bs
            total_n += bs

        scheduler.step()

        ids_v, y_v, reg_v, prob_v, feat_v = infer(model, val_loader)
        mv = metrics_from_prob(prob_v, y_v, reg_v)

        pred_count = np.asarray(mv["pred_count"], dtype=np.float32)
        pred_ratio = pred_count / max(pred_count.sum(), 1.0)
        used = int((pred_count > 0).sum())
        max_ratio = float(pred_ratio.max())
        pred_entropy = float(-(pred_ratio[pred_ratio > 0] * np.log(pred_ratio[pred_ratio > 0])).sum() / math.log(K))

        score = (
            0.34 * mv["acc7"]
            + 0.22 * mv["f1_macro"]
            + 0.24 * mv["corr"]
            + 0.10 * pred_entropy
            + 0.10 * used / K
            - 0.10 * max_ratio
        )

        row = {
            "epoch": epoch,
            "score": float(score),
            "val_acc7": mv["acc7"],
            "val_f1_macro": mv["f1_macro"],
            "val_corr": mv["corr"],
            "used_classes": used,
            "max_pred_ratio": max_ratio,
            "pred_entropy": pred_entropy,
        }
        for key, value in total.items():
            row[key] = value / max(total_n, 1)
        history.append(row)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{expert}_e{epoch:03d} "
                f"loss={row['loss']:.4f} "
                f"val_acc7={mv['acc7'] * 100:.2f}% "
                f"f1m={mv['f1_macro'] * 100:.2f}% "
                f"corr={mv['corr']:.4f} "
                f"used={used}/{K} "
                f"max_ratio={max_ratio:.3f} "
                f"entropy={pred_entropy:.3f} "
                f"pred={mv['pred_count']}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "scaler": train_set.scaler,
                "input_dim": train_set.x.shape[1],
                "class_count": count_np.tolist(),
                "target_prior": target_prior.detach().cpu().numpy().tolist(),
                "history": history,
                "expert": expert,
            }
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"{expert}: early_stop at epoch={epoch}, best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state["model"])

    ids_tr, y_tr, reg_tr, prob_tr, feat_tr = infer(model, eval_train_loader)
    ids_v, y_v, reg_v, prob_v, feat_v = infer(model, val_loader)
    ids_te, y_te, reg_te, prob_te, feat_te = infer(model, test_loader)

    mtr = metrics_from_prob(prob_tr, y_tr, reg_tr)
    mv = metrics_from_prob(prob_v, y_v, reg_v)
    mt = metrics_from_prob(prob_te, y_te, reg_te)

    print("\n[BEST]")
    print(f"{expert}: best_epoch={best_epoch}")
    print_metric(f"{expert}_train", mtr)
    print_metric(f"{expert}_val", mv)
    print_metric(f"{expert}_test", mt)

    save_outputs(expert, "train", ids_tr, y_tr, reg_tr, prob_tr, feat_tr)
    save_outputs(expert, "val", ids_v, y_v, reg_v, prob_v, feat_v)
    save_outputs(expert, "test", ids_te, y_te, reg_te, prob_te, feat_te)

    save_confusion(f"{expert}_train", y_tr, prob_tr)
    save_confusion(f"{expert}_val", y_v, prob_v)
    save_confusion(f"{expert}_test", y_te, prob_te)

    mkdir(MODEL_DIR)
    torch.save(best_state, MODEL_DIR / f"{expert}_ordinal_cepl.pt")

    return {
        "expert": expert,
        "best_epoch": best_epoch,
        "source": {
            "train": raw["train"]["source"],
            "val": raw["val"]["source"],
            "test": raw["test"]["source"],
            "feature_key_train": raw["train"]["feature_key"],
            "feature_key_val": raw["val"]["feature_key"],
            "feature_key_test": raw["test"]["feature_key"],
        },
        "train": mtr,
        "val": mv,
        "test": mt,
        "history": history,
    }


def main():
    seed_all(SEED)
    mkdir(OUTPUT_DIR)
    mkdir(MODEL_DIR)
    mkdir(REPORT_DIR)

    print("=" * 130)
    print("DOME-X CMU-MOSEI SUBMODEL TRAINING: ORDINAL CEPL WITHOUT XIEXIU")
    print("=" * 130)
    print("device:", DEVICE)
    print("experts:", EXPERTS)
    print("output_protocol: ids / y / reg / prob / score / pred / feature")
    print("design: Ordinal DOME-X + CEPL + SEBI regularization + strict modality source check")
    print("note: CMU-MOSEI is treated as a limitation dataset, not a DOME-X-friendly benchmark.")

    reports = {}
    for expert in EXPERTS:
        reports[expert] = train_one_expert(expert)

    with open(REPORT_DIR / "submodel_ordinal_cepl_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 130)
    print("DONE")
    print("=" * 130)
    print("outputs:", OUTPUT_DIR)
    print("models :", MODEL_DIR)
    print("report :", REPORT_DIR / "submodel_ordinal_cepl_report.json")
    print("下一步运行正式 fusion.py；融合代码中不应再加入 xiexiu/test-label training。")


if __name__ == "__main__":
    main()
