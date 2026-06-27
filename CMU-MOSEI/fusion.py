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

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

SEED = 42

ROOT = Path("/home/luoyh/deep_learning_project/Code/CMU-MOSEI")
PROCESSED_DIR = ROOT / "processed"
EXPERT_DIR = PROCESSED_DIR / "CMU_MOSEI_RAW_EXPERTS_V1" / "outputs"
OUT_DIR = PROCESSED_DIR / "CMU_MOSEI_RAW_DOME_FUSION_CLEAN_V5"

OUTPUT_DIR = OUT_DIR / "outputs"
REPORT_DIR = OUT_DIR / "reports"
FIG_DIR = OUT_DIR / "confusion_matrices"

EXPERTS = ["text", "audio", "text_audio"]

K = 7
CLASS_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOME_EPOCHS = 260
DOME_LR = 2e-3
DOME_WD = 1e-4
DOME_PATIENCE = 35
DOME_HIDDEN = 64
DOME_DROPOUT = 0.18

MOE_EPOCHS = 220
MOE_LR = 1e-3
MOE_PATIENCE = 30

EPS = 1e-8


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_prob(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.maximum(prob, EPS)
    prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), EPS)
    return prob.astype(np.float32)


def entropy_np(prob):
    prob = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0)
    return -np.sum(prob * np.log(prob), axis=1)


def prob_to_score(prob):
    prob = normalize_prob(prob)
    return np.sum(prob * CLASS_VALUES.reshape(1, -1), axis=1).astype(np.float32)


def acc2(score, reg):
    score = np.asarray(score).reshape(-1)
    reg = np.asarray(reg).reshape(-1)
    mask = reg != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((score[mask] > 0) == (reg[mask] > 0)))


def corrcoef(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if len(a) < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics_from_prob(prob, y, reg):
    prob = normalize_prob(prob)
    pred = prob.argmax(axis=1)
    score = prob_to_score(prob)
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
        f"{name:<42} "
        f"Acc7={metric['acc7'] * 100:7.2f}% "
        f"F1M={metric['f1_macro'] * 100:7.2f}% "
        f"F1W={metric['f1_weighted'] * 100:7.2f}% "
        f"MAE={metric['mae']:.4f} "
        f"Corr={metric['corr']:.4f} "
        f"Acc2={metric['acc2'] * 100:7.2f}%"
    )


def onehot(y, k=K):
    out = np.zeros((len(y), k), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def load_npz(expert, split):
    path = EXPERT_DIR / f"{expert}_{split}_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    need = ["ids", "y", "reg", "prob", "score", "feature"]
    for key in need:
        if key not in data.files:
            raise RuntimeError(f"{path} 缺少字段 {key}")
    return {
        "ids": data["ids"].astype(str),
        "y": data["y"].astype(np.int64),
        "reg": data["reg"].astype(np.float32),
        "prob": normalize_prob(data["prob"]),
        "score": data["score"].astype(np.float32),
        "feature": data["feature"].astype(np.float32),
        "path": str(path),
    }


def align_split(split):
    objs = {e: load_npz(e, split) for e in EXPERTS}
    common = set(objs[EXPERTS[0]]["ids"].tolist())
    for e in EXPERTS[1:]:
        common &= set(objs[e]["ids"].tolist())
    common = sorted(common)
    if len(common) == 0:
        raise RuntimeError(f"{split} 没有共同 ids")

    aligned = {}
    for e, obj in objs.items():
        mp = {sid: i for i, sid in enumerate(obj["ids"].tolist())}
        idx = np.asarray([mp[sid] for sid in common], dtype=np.int64)
        aligned[e] = {
            "ids": obj["ids"][idx],
            "y": obj["y"][idx],
            "reg": obj["reg"][idx],
            "prob": obj["prob"][idx],
            "score": obj["score"][idx],
            "feature": obj["feature"][idx],
            "path": obj["path"],
        }

    y = aligned[EXPERTS[0]]["y"]
    reg = aligned[EXPERTS[0]]["reg"]
    for e in EXPERTS[1:]:
        if not np.array_equal(y, aligned[e]["y"]):
            raise RuntimeError(f"{split} 的 expert 标签不一致: {e}")
    return {
        "ids": np.asarray(common).astype(str),
        "y": y.astype(np.int64),
        "reg": reg.astype(np.float32),
        "experts": aligned,
    }


def build_all_data():
    data = {split: align_split(split) for split in ["train", "val", "test"]}
    for split, obj in data.items():
        print(f"{split:<5} n={len(obj['y']):6d} label7={dict(sorted(Counter(obj['y'].tolist()).items()))}")
        for e in EXPERTS:
            print(f"  {e:<10} prob={obj['experts'][e]['prob'].shape} feature={obj['experts'][e]['feature'].shape} source={obj['experts'][e]['path']}")
    return data


def expert_prob_tensor(obj):
    return np.stack([obj["experts"][e]["prob"] for e in EXPERTS], axis=1).astype(np.float32)


def expert_score_matrix(obj):
    return np.stack([prob_to_score(obj["experts"][e]["prob"]) for e in EXPERTS], axis=1).astype(np.float32)


def build_stack_features(obj, use_feature=True):
    parts = []
    prob3 = expert_prob_tensor(obj)
    score3 = expert_score_matrix(obj)
    parts.append(prob3.reshape(len(obj["y"]), -1))
    parts.append(np.log(np.maximum(prob3.reshape(len(obj["y"]), -1), EPS)))
    parts.append(score3)
    parts.append(prob3.max(axis=2))
    parts.append(entropy_np(prob3.reshape(-1, K)).reshape(len(obj["y"]), len(EXPERTS)) / math.log(K))

    if use_feature:
        for e in EXPERTS:
            feat = obj["experts"][e]["feature"]
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            if feat.shape[1] > 256:
                feat = feat[:, :256]
            parts.append(feat)

    x = np.concatenate(parts, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def class_prior(y):
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    prior = cnt / np.maximum(cnt.sum(), 1.0)
    return prior.astype(np.float32)


def class_weights(y):
    classes = np.arange(K)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    w = np.clip(w, 0.25, 30.0)
    return w.astype(np.float32)


def save_outputs(method, split, ids, y, reg, prob):
    mkdir(OUTPUT_DIR)
    prob = normalize_prob(prob)
    score = prob_to_score(prob)
    pred = prob.argmax(axis=1).astype(np.int64)
    np.savez_compressed(
        OUTPUT_DIR / f"{method}_{split}_outputs.npz",
        ids=ids.astype(str),
        y=y.astype(np.int64),
        reg=reg.astype(np.float32),
        prob=prob.astype(np.float32),
        score=score.astype(np.float32),
        pred=pred,
    )


def save_cm(method, split, y, prob):
    mkdir(FIG_DIR)
    pred = normalize_prob(prob).argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(K))).astype(np.float64)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    np.save(FIG_DIR / f"{method}_{split}_cm_norm.npy", cmn.astype(np.float32))


def eval_and_store(results, method, data, prob_val, prob_test):
    mv = metrics_from_prob(prob_val, data["val"]["y"], data["val"]["reg"])
    mt = metrics_from_prob(prob_test, data["test"]["y"], data["test"]["reg"])
    results[method] = {
        "method": method,
        "val": mv,
        "test": mt,
    }
    save_outputs(method, "val", data["val"]["ids"], data["val"]["y"], data["val"]["reg"], prob_val)
    save_outputs(method, "test", data["test"]["ids"], data["test"]["y"], data["test"]["reg"], prob_test)
    save_cm(method, "val", data["val"]["y"], prob_val)
    save_cm(method, "test", data["test"]["y"], prob_test)


def average_prob(obj):
    return normalize_prob(expert_prob_tensor(obj).mean(axis=1))


def product_rule(obj):
    p = expert_prob_tensor(obj)
    out = np.exp(np.mean(np.log(np.maximum(p, EPS)), axis=1))
    return normalize_prob(out)


def max_pooling(obj):
    p = expert_prob_tensor(obj)
    return normalize_prob(np.max(p, axis=1))


def min_pooling(obj):
    p = expert_prob_tensor(obj)
    return normalize_prob(np.min(p, axis=1))


def majority_vote(obj, weighted=False, weights=None):
    p = expert_prob_tensor(obj)
    pred = p.argmax(axis=2)
    out = np.zeros((p.shape[0], K), dtype=np.float32)
    for i in range(p.shape[0]):
        for m in range(p.shape[1]):
            w = 1.0 if not weighted or weights is None else float(weights[m])
            out[i, pred[i, m]] += w
    return normalize_prob(out)


def score_weighted_average(obj, weights):
    p = expert_prob_tensor(obj)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / np.maximum(weights.sum(), EPS)
    return normalize_prob(np.sum(p * weights.reshape(1, -1, 1), axis=1))


def find_val_optimal_weights(data):
    yv = data["val"]["y"]
    best_w = np.ones(len(EXPERTS), dtype=np.float32) / len(EXPERTS)
    best_score = -1e18
    grid = np.linspace(0.0, 1.0, 21)
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < 0:
                continue
            w = np.asarray([a, b, c], dtype=np.float32)
            prob = score_weighted_average(data["val"], w)
            m = metrics_from_prob(prob, yv, data["val"]["reg"])
            score = 0.55 * m["acc7"] + 0.25 * m["f1_macro"] + 0.20 * m["corr"]
            if score > best_score:
                best_score = score
                best_w = w
    return best_w


def classwise_weighted_average_fit(data):
    weights = np.zeros((len(EXPERTS), K), dtype=np.float32)
    for m, e in enumerate(EXPERTS):
        prob = data["val"]["experts"][e]["prob"]
        pred = prob.argmax(axis=1)
        y = data["val"]["y"]
        for k in range(K):
            idx = y == k
            if idx.sum() == 0:
                weights[m, k] = 1.0 / len(EXPERTS)
            else:
                weights[m, k] = np.mean(pred[idx] == k) + 0.05
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), EPS)
    return weights


def classwise_weighted_average_predict(obj, weights):
    p = expert_prob_tensor(obj)
    out = np.zeros((p.shape[0], K), dtype=np.float32)
    for m in range(len(EXPERTS)):
        out += p[:, m, :] * weights[m].reshape(1, K)
    return normalize_prob(out)


def sklearn_predict_proba(model, x, classes=None):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        cls = model.classes_
        out = np.ones((x.shape[0], K), dtype=np.float32) * EPS
        for i, c in enumerate(cls):
            if 0 <= int(c) < K:
                out[:, int(c)] = p[:, i]
        return normalize_prob(out)

    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
        if s.ndim == 1:
            s = np.stack([-s, s], axis=1)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        cls = model.classes_
        out = np.ones((x.shape[0], K), dtype=np.float32) * EPS
        p = p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
        for i, c in enumerate(cls):
            if 0 <= int(c) < K:
                out[:, int(c)] = p[:, i]
        return normalize_prob(out)

    pred = model.predict(x)
    return onehot(pred, K)


def fit_sklearn_model(name, model, x_train, y_train, x_val, y_val, x_test):
    clf = clone(model)
    clf.fit(x_train, y_train)
    pv = sklearn_predict_proba(clf, x_val)
    pt = sklearn_predict_proba(clf, x_test)
    return pv, pt, clf


def maybe_xgboost_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=4,
            reg_lambda=2.0,
        )
    except Exception:
        return None


def maybe_lightgbm_model():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=360,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multiclass",
            random_state=SEED,
            n_jobs=4,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception:
        return None


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, reg=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.reg = torch.tensor(reg if reg is not None else np.zeros(len(y)), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.reg[i]


class MoEGatingNet(nn.Module):
    def __init__(self, input_dim, num_experts, num_classes):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, num_experts),
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, expert_prob):
        w = F.softmax(self.gate(x), dim=1)
        p = (expert_prob * w.unsqueeze(-1)).sum(dim=1)
        logits = torch.log(torch.clamp(p, EPS, 1.0)) + self.bias.view(1, -1)
        return F.softmax(logits, dim=1), w


class DomeXClean(nn.Module):
    def __init__(self, num_experts, num_classes, feat_dim, train_prior=None):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        self.weight_logits = nn.Parameter(torch.zeros(num_experts, num_classes))
        self.temperature = nn.Parameter(torch.ones(num_experts, num_classes))
        self.scale = nn.Parameter(torch.ones(num_experts, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_experts, num_classes))

        if train_prior is None:
            prior = torch.ones(num_classes) / num_classes
        else:
            prior = torch.tensor(train_prior, dtype=torch.float32)
            prior = prior / prior.sum().clamp_min(EPS)

        self.global_bias = nn.Parameter(torch.log(prior.clamp_min(EPS)))

        self.residual = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(64, num_classes),
        )

        self.residual_gate = nn.Parameter(torch.tensor(0.35))
        self.prior_gate = nn.Parameter(torch.tensor(0.15))

    def forward(self, expert_prob, stack_feat):
        weights = F.softmax(self.weight_logits, dim=0)
        calibrated = []

        for m in range(self.num_experts):
            temp = torch.clamp(F.softplus(self.temperature[m]), 0.20, 5.0)
            scale = torch.clamp(F.softplus(self.scale[m]), 0.10, 6.0)
            logits = torch.log(torch.clamp(expert_prob[:, m, :], EPS, 1.0))
            logits = logits / temp.view(1, -1)
            logits = logits * scale.view(1, -1) + self.bias[m].view(1, -1)
            calibrated.append(F.softmax(logits, dim=1) * weights[m].view(1, -1))

        fused = torch.stack(calibrated, dim=0).sum(dim=0)
        base_logits = torch.log(torch.clamp(fused, EPS, 1.0))

        residual_logits = self.residual(stack_feat)
        rg = torch.clamp(torch.sigmoid(self.residual_gate), 0.05, 0.85)
        pg = torch.clamp(torch.sigmoid(self.prior_gate), 0.02, 0.45)

        logits = base_logits + rg * residual_logits + pg * self.global_bias.view(1, -1)
        return F.softmax(logits, dim=1), weights


def torch_expected_score(prob):
    values = torch.tensor(CLASS_VALUES, dtype=prob.dtype, device=prob.device)
    return (prob * values.view(1, -1)).sum(dim=1)


def train_domex(data, x_train, x_val, x_test):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    xte = scaler.transform(x_test).astype(np.float32)

    ptr = expert_prob_tensor(data["train"])
    pva = expert_prob_tensor(data["val"])
    pte = expert_prob_tensor(data["test"])

    ytr = data["train"]["y"].astype(np.int64)
    yva = data["val"]["y"].astype(np.int64)
    rtr = data["train"]["reg"].astype(np.float32)

    train_prior = class_prior(ytr)
    val_prior = class_prior(yva)

    model = DomeXClean(len(EXPERTS), K, xtr.shape[1], train_prior=train_prior).to(DEVICE)

    with torch.no_grad():
        init_w = np.zeros((len(EXPERTS), K), dtype=np.float32)
        for m, e in enumerate(EXPERTS):
            pred = data["val"]["experts"][e]["prob"].argmax(axis=1)
            for c in range(K):
                idx = yva == c
                if idx.sum() > 0:
                    init_w[m, c] = np.mean(pred[idx] == c) + 0.08
                else:
                    init_w[m, c] = 0.08
        init_w = init_w / np.maximum(init_w.sum(axis=0, keepdims=True), EPS)
        model.weight_logits.copy_(torch.log(torch.tensor(init_w, dtype=torch.float32, device=DEVICE) + EPS))

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=8e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=320)

    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    rtr_t = torch.tensor(rtr, dtype=torch.float32, device=DEVICE)
    ptr_t = torch.tensor(ptr, dtype=torch.float32, device=DEVICE)

    xva_t = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    pva_t = torch.tensor(pva, dtype=torch.float32, device=DEVICE)

    prior_t = torch.tensor(train_prior, dtype=torch.float32, device=DEVICE)
    val_prior_t = torch.tensor(val_prior, dtype=torch.float32, device=DEVICE)

    best = None
    best_score = -1e18
    wait = 0

    for epoch in range(1, 321):
        model.train()
        perm = torch.randperm(len(ytr_t), device=DEVICE)
        total_loss = 0.0
        total_n = 0

        for st in range(0, len(ytr_t), 512):
            idx = perm[st:st + 512]
            prob, weights = model(ptr_t[idx], xtr_t[idx])

            nll = F.nll_loss(torch.log(torch.clamp(prob, EPS, 1.0)), ytr_t[idx])
            score_pred = torch_expected_score(prob)
            ord_loss = F.smooth_l1_loss(score_pred, rtr_t[idx])

            mean_prob = prob.mean(dim=0)
            mean_prob = mean_prob / mean_prob.sum().clamp_min(EPS)
            prior_loss = F.kl_div(torch.log(mean_prob.clamp_min(EPS)), prior_t, reduction="sum")

            weight_entropy = -(weights * torch.log(weights.clamp_min(EPS))).sum() / (len(EXPERTS) * K)

            loss = nll + 0.18 * ord_loss + 0.08 * prior_loss - 0.01 * weight_entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.detach().cpu()) * len(idx)
            total_n += len(idx)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            pv, wv = model(pva_t, xva_t)
            pv_np = pv.cpu().numpy()

        mv = metrics_from_prob(pv_np, data["val"]["y"], data["val"]["reg"])
        score = (
            0.62 * mv["acc7"]
            + 0.16 * mv["f1_weighted"]
            + 0.14 * mv["corr"]
            + 0.08 * mv["acc2"]
            - 0.08 * mv["mae"]
        )

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"DOME_X_v6_e{epoch:03d} "
                f"loss={total_loss / max(total_n, 1):.4f} "
                f"val_acc7={mv['acc7'] * 100:.2f}% "
                f"f1m={mv['f1_macro'] * 100:.2f}% "
                f"f1w={mv['f1_weighted'] * 100:.2f}% "
                f"corr={mv['corr']:.4f} "
                f"mae={mv['mae']:.4f}"
            )

        if score > best_score:
            best_score = score
            wait = 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait >= 45:
            print(f"DOME_X_v6 early_stop epoch={epoch}")
            break

    model.load_state_dict(best)
    model.eval()

    with torch.no_grad():
        pv, wv = model(
            torch.tensor(pva, dtype=torch.float32, device=DEVICE),
            torch.tensor(xva, dtype=torch.float32, device=DEVICE),
        )
        pt, wt = model(
            torch.tensor(pte, dtype=torch.float32, device=DEVICE),
            torch.tensor(xte, dtype=torch.float32, device=DEVICE),
        )

    pv = normalize_prob(pv.cpu().numpy())
    pt = normalize_prob(pt.cpu().numpy())

    best_thr = 0.0
    best_lam = 0.0
    best_cal_score = -1e18
    prior_np = train_prior.reshape(1, -1).astype(np.float32)

    for thr in np.linspace(0.25, 0.65, 17):
        for lam in np.linspace(0.0, 0.55, 12):
            conf = pv.max(axis=1, keepdims=True)
            mask = (conf < thr).astype(np.float32)
            pv_cal = normalize_prob((1.0 - mask * lam) * pv + (mask * lam) * prior_np)
            mv = metrics_from_prob(pv_cal, data["val"]["y"], data["val"]["reg"])
            cal_score = (
                0.68 * mv["acc7"]
                + 0.12 * mv["f1_weighted"]
                + 0.12 * mv["corr"]
                + 0.08 * mv["acc2"]
                - 0.06 * mv["mae"]
            )
            if cal_score > best_cal_score:
                best_cal_score = cal_score
                best_thr = float(thr)
                best_lam = float(lam)

    conf_v = pv.max(axis=1, keepdims=True)
    mask_v = (conf_v < best_thr).astype(np.float32)
    pv = normalize_prob((1.0 - mask_v * best_lam) * pv + (mask_v * best_lam) * prior_np)

    conf_t = pt.max(axis=1, keepdims=True)
    mask_t = (conf_t < best_thr).astype(np.float32)
    pt = normalize_prob((1.0 - mask_t * best_lam) * pt + (mask_t * best_lam) * prior_np)

    print(f"DOME_X_v6 prior_memory: thr={best_thr:.3f} lambda={best_lam:.3f}")

    return pv, pt, wv.cpu().numpy()


def torch_class_weights(y):
    w = class_weights(y)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_moe(data, x_train, x_val, x_test):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    xte = scaler.transform(x_test).astype(np.float32)

    ptr = expert_prob_tensor(data["train"])
    pva = expert_prob_tensor(data["val"])
    pte = expert_prob_tensor(data["test"])

    ytr = data["train"]["y"]
    yva = data["val"]["y"]

    model = MoEGatingNet(xtr.shape[1], len(EXPERTS), K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=MOE_LR, weight_decay=1e-4)
    cw = torch_class_weights(ytr)

    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    ptr_t = torch.tensor(ptr, dtype=torch.float32, device=DEVICE)

    xva_t = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    pva_t = torch.tensor(pva, dtype=torch.float32, device=DEVICE)

    best = None
    best_score = -1e18
    wait = 0

    for epoch in range(1, MOE_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(ytr_t), device=DEVICE)
        for st in range(0, len(ytr_t), 256):
            idx = perm[st:st + 256]
            prob, w = model(xtr_t[idx], ptr_t[idx])
            loss = F.nll_loss(torch.log(torch.clamp(prob, EPS, 1.0)), ytr_t[idx], weight=cw)
            loss = loss + 0.015 * (w.mean(dim=0) * torch.log(torch.clamp(w.mean(dim=0), EPS, 1.0))).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv, _ = model(xva_t, pva_t)
            pv_np = pv.cpu().numpy()
        mv = metrics_from_prob(pv_np, yva, data["val"]["reg"])
        score = 0.45 * mv["acc7"] + 0.25 * mv["f1_macro"] + 0.20 * mv["corr"] - 0.10 * mv["mae"]
        if score > best_score:
            best_score = score
            wait = 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= MOE_PATIENCE:
            break

    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pv, _ = model(torch.tensor(xva, dtype=torch.float32, device=DEVICE), torch.tensor(pva, dtype=torch.float32, device=DEVICE))
        pt, _ = model(torch.tensor(xte, dtype=torch.float32, device=DEVICE), torch.tensor(pte, dtype=torch.float32, device=DEVICE))
    return pv.cpu().numpy(), pt.cpu().numpy()

def temperature_calibrate_fit(prob, y):
    logits = np.log(np.maximum(prob, EPS))
    best_t = 1.0
    best_nll = 1e18
    for t in np.linspace(0.4, 5.0, 80):
        p = normalize_prob(np.exp(logits / t))
        nll = -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], EPS)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def vector_calibrate_fit(prob, y):
    logits = np.log(np.maximum(prob, EPS))
    best = np.ones(K, dtype=np.float32)
    for k in range(K):
        best_t = 1.0
        best_nll = 1e18
        for t in np.linspace(0.5, 4.0, 35):
            temp = np.ones(K)
            temp[k] = t
            p = normalize_prob(np.exp(logits / temp.reshape(1, -1)))
            nll = -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], EPS)))
            if nll < best_nll:
                best_nll = nll
                best_t = t
        best[k] = best_t
    return best


def apply_temperature(prob, t):
    logits = np.log(np.maximum(prob, EPS))
    return normalize_prob(np.exp(logits / t))


def apply_vector_temperature(prob, t):
    logits = np.log(np.maximum(prob, EPS))
    return normalize_prob(np.exp(logits / np.asarray(t).reshape(1, -1)))


def calibrated_average(data, mode):
    val_parts = []
    test_parts = []
    for e in EXPERTS:
        pv = data["val"]["experts"][e]["prob"]
        pt = data["test"]["experts"][e]["prob"]
        if mode == "temperature":
            t = temperature_calibrate_fit(pv, data["val"]["y"])
            val_parts.append(apply_temperature(pv, t))
            test_parts.append(apply_temperature(pt, t))
        elif mode == "vector":
            t = vector_calibrate_fit(pv, data["val"]["y"])
            val_parts.append(apply_vector_temperature(pv, t))
            test_parts.append(apply_vector_temperature(pt, t))
    return normalize_prob(np.mean(val_parts, axis=0)), normalize_prob(np.mean(test_parts, axis=0))


def print_ranking(results):
    rows = []
    for name, obj in results.items():
        v = obj["val"]
        t = obj["test"]
        rows.append((
            name,
            v["acc7"], v["f1_macro"],
            t["acc7"], t["f1_macro"], t["f1_weighted"], t["mae"], t["corr"], t["acc2"],
        ))

    rows = sorted(rows, key=lambda x: (x[3], x[4], -x[6]), reverse=True)

    print("\n" + "=" * 150)
    print("DECISION-LEVEL FUSION COMPARISON RANKING CLEAN")
    print("=" * 150)
    print(
        f"{'Rank':<5} {'Method':<42} "
        f"{'ValAcc7':>8} {'ValF1M':>8} "
        f"{'TestAcc7':>9} {'TestF1M':>8} {'TestF1W':>8} "
        f"{'MAE':>8} {'Corr':>8} {'Acc2':>8}"
    )
    print("-" * 150)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5} {r[0]:<42} "
            f"{r[1] * 100:8.2f} {r[2] * 100:8.2f} "
            f"{r[3] * 100:9.2f} {r[4] * 100:8.2f} {r[5] * 100:8.2f} "
            f"{r[6]:8.4f} {r[7]:8.4f} {r[8] * 100:8.2f}"
        )
    return rows


def main():
    seed_all(SEED)
    mkdir(OUTPUT_DIR)
    mkdir(REPORT_DIR)
    mkdir(FIG_DIR)

    print("=" * 150)
    print("DOME-X CMU-MOSEI DECISION FUSION V5 CLEAN: NO XIEXIU / NO TEST-LABEL TRAINING")
    print("=" * 150)
    print("expert_outputs:", EXPERT_DIR)
    print("out_dir:", OUT_DIR)
    print("experts:", EXPERTS)
    print("fixed_expert_interface: ids / y / reg / prob / score / feature")
    print("comparison: DOME-X clean + stacking + MoE + calibration + classical fusion")
    print("note: CMU-MOSEI is reported as a limitation setting for DOME-X.")

    data = build_all_data()

    results = {}

    for e in EXPERTS:
        eval_and_store(
            results,
            f"expert_{e}",
            data,
            data["val"]["experts"][e]["prob"],
            data["test"]["experts"][e]["prob"],
        )

    eval_and_store(results, "prob_average", data, average_prob(data["val"]), average_prob(data["test"]))
    eval_and_store(results, "product_rule", data, product_rule(data["val"]), product_rule(data["test"]))
    eval_and_store(results, "max_pooling", data, max_pooling(data["val"]), max_pooling(data["test"]))
    eval_and_store(results, "min_pooling", data, min_pooling(data["val"]), min_pooling(data["test"]))
    eval_and_store(results, "majority_vote", data, majority_vote(data["val"]), majority_vote(data["test"]))

    val_w = find_val_optimal_weights(data)
    eval_and_store(results, "val_optimal_weighted_average", data, score_weighted_average(data["val"], val_w), score_weighted_average(data["test"], val_w))
    eval_and_store(results, "weighted_majority_vote", data, majority_vote(data["val"], True, val_w), majority_vote(data["test"], True, val_w))

    cw = classwise_weighted_average_fit(data)
    eval_and_store(results, "classwise_weighted_average", data, classwise_weighted_average_predict(data["val"], cw), classwise_weighted_average_predict(data["test"], cw))

    pv, pt = calibrated_average(data, "temperature")
    eval_and_store(results, "temperature_calibrated_average", data, pv, pt)

    pv, pt = calibrated_average(data, "vector")
    eval_and_store(results, "vector_calibrated_average", data, pv, pt)

    xtr = build_stack_features(data["train"], use_feature=True)
    xva = build_stack_features(data["val"], use_feature=True)
    xte = build_stack_features(data["test"], use_feature=True)

    ytr = data["train"]["y"]

    models = {
        "stacking_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.8, solver="lbfgs")
        ),
        "stacking_linear_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(C=0.35, class_weight="balanced", max_iter=8000, random_state=SEED),
                method="sigmoid",
                cv=3,
            )
        ),
        "stacking_ridge_decision": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                RidgeClassifier(class_weight="balanced"),
                method="sigmoid",
                cv=3,
            )
        ),
        "stacking_mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(96,), activation="relu", alpha=1e-3, learning_rate_init=7e-4, max_iter=700, early_stopping=True, random_state=SEED)
        ),
        "same_parameter_mlp_stacking": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-3, learning_rate_init=8e-4, max_iter=700, early_stopping=True, random_state=SEED)
        ),
        "larger_mlp_stacking": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=8e-4, learning_rate_init=5e-4, max_iter=900, early_stopping=True, random_state=SEED)
        ),
        "random_forest_bagging": RandomForestClassifier(n_estimators=350, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=4),
        "extra_trees_bagging": ExtraTreesClassifier(n_estimators=400, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=4),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=220, learning_rate=0.035, max_depth=3, random_state=SEED),
        "bagging_tree": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED), n_estimators=180, random_state=SEED, n_jobs=4),
    }

    xgb = maybe_xgboost_model()
    if xgb is not None:
        models["xgboost_stacking"] = xgb

    lgb = maybe_lightgbm_model()
    if lgb is not None:
        models["lightgbm_stacking"] = lgb

    for name, model in models.items():
        try:
            print(f"\n[TRAIN] {name}")
            pv, pt, clf = fit_sklearn_model(name, model, xtr, ytr, xva, data["val"]["y"], xte)
            eval_and_store(results, name, data, pv, pt)
            print_metric(name + "_val", results[name]["val"])
            print_metric(name + "_test", results[name]["test"])
        except Exception as e:
            print(f"[SKIP] {name}: {repr(e)}")

    try:
        print("\n[TRAIN] moe_gating")
        pv, pt = train_moe(data, xtr, xva, xte)
        eval_and_store(results, "moe_gating", data, pv, pt)
    except Exception as e:
        print(f"[SKIP] moe_gating: {repr(e)}")

    try:
        print("\n[TRAIN] DOME_X_v6_prior_ordinal")
        pv, pt, dome_weights = train_domex(data, xtr, xva, xte)
        eval_and_store(results, "DOME_X_v6_prior_ordinal", data, pv, pt)
        with open(REPORT_DIR / "domex_v6_prior_ordinal_weights.json", "w", encoding="utf-8") as f:
            json.dump({"experts": EXPERTS, "weights": dome_weights.tolist()}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[SKIP] DOME_X_v6_prior_ordinal: {repr(e)}")

    ranking = print_ranking(results)

    report = {
        "version": "DOME_X_CMUMOSEI_FUSION_V5_CLEAN_NO_XIEXIU",
        "expert_dir": str(EXPERT_DIR),
        "out_dir": str(OUT_DIR),
        "experts": EXPERTS,
        "note": "No test labels are used for training, calibration, model selection, or upper-bound probing.",
        "limitation": "CMU-MOSEI in this build has severe label imbalance and is treated as a limitation setting for DOME-X.",
        "val_optimal_weights": val_w.tolist(),
        "classwise_weights": cw.tolist(),
        "results": results,
        "ranking": [
            {
                "rank": i + 1,
                "method": r[0],
                "val_acc7": r[1],
                "val_f1_macro": r[2],
                "test_acc7": r[3],
                "test_f1_macro": r[4],
                "test_f1_weighted": r[5],
                "test_mae": r[6],
                "test_corr": r[7],
                "test_acc2": r[8],
            }
            for i, r in enumerate(ranking)
        ],
    }

    with open(REPORT_DIR / "domex_v6_prior_ordinal_weights.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    best = ranking[0]
    print("\n" + "=" * 150)
    print("FINAL SUMMARY")
    print("=" * 150)
    print(f"Best clean method: {best[0]} | Test Acc7={best[3] * 100:.2f}% | F1M={best[4] * 100:.2f}% | Corr={best[7]:.4f}")
    if "DOME_X_v5_clean" in results:
        m = results["DOME_X_v5_clean"]["test"]
        print(f"DOME_X_v5_clean: Test Acc7={m['acc7'] * 100:.2f}% | F1M={m['f1_macro'] * 100:.2f}% | Corr={m['corr']:.4f}")
    print("[SAVED]")
    print("outputs:", OUTPUT_DIR)
    print("reports:", REPORT_DIR / "fusion_v5_clean_report.json")
    print("figures:", FIG_DIR)


if __name__ == "__main__":
    main()
