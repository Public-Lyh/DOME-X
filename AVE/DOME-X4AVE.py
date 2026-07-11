import math
import time
import copy
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

warnings.filterwarnings("ignore")

SEED = 42
EPS = 1e-10
NC = 28

SAVE_DIR = Path("/home/luoyh/DATA/fusion/data/AVE/checkpoints/v9")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_KEYS = ["AudioNet", "VisualNet"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + EPS)


def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, EPS, None)
    return x / (x.sum(axis=-1, keepdims=True) + EPS)


def normalize_torch(x):
    return x / x.sum(dim=-1, keepdim=True).clamp_min(EPS)


def entropy_np(p):
    p = normalize_np(p)
    return -(p * np.log(p + EPS)).sum(axis=1, keepdims=True) / math.log(p.shape[1])


def margin_np(p):
    p = normalize_np(p)
    s = np.sort(p, axis=1)
    return (s[:, -1:] - s[:, -2:-1]).clip(0.0, 1.0)


def eval_probs(probs, labels):
    probs = normalize_np(probs)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs.argmax(axis=1)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "f1w": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "f1m": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }


def clcp_np(probs, labels):
    probs = normalize_np(probs)
    labels = np.asarray(labels, dtype=np.int64)
    c = np.zeros((NC, NC), dtype=np.float64)
    present = np.zeros(NC, dtype=np.float64)

    for k in range(NC):
        idx = labels == k
        if idx.any():
            c[k] = probs[idx].mean(axis=0)
            present[k] = 1.0

    denom = math.log(NC)
    valid = max(present.sum(), 1.0)

    row_entropy = -(c * np.log(c + EPS)).sum(axis=1) / denom
    rcs = 1.0 - float((row_entropy * present).sum() / valid)

    col_mass = c.sum(axis=0)
    col_dist = c / (col_mass[None, :] + EPS)
    col_entropy = -(col_dist * np.log(col_dist + EPS)).sum(axis=0) / denom
    cps = 1.0 - float((col_mass * col_entropy).sum() / (col_mass.sum() + EPS))

    usage = col_mass / (col_mass.sum() + EPS)
    cus = float(-(usage * np.log(usage + EPS)).sum() / denom)

    peak = c.max(axis=1)
    ps = float((peak * present).sum() / valid)

    return float(np.clip(rcs, EPS, 1.0) * np.clip(cps, EPS, 1.0) * np.clip(cus, EPS, 1.0) * np.clip(ps, EPS, 1.0))


def val_score(probs, labels):
    ev = eval_probs(probs, labels)
    return float(0.45 * ev["acc"] + 0.35 * ev["f1m"] + 0.15 * ev["f1w"] + 0.05 * clcp_np(probs, labels))


def format_params(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def load_logits():
    p1 = SAVE_DIR / "all_logits_ave_v8.pkl"
    p2 = SAVE_DIR / "all_logits_ave_v9.pkl"

    if p1.exists():
        path = p1
    elif p2.exists():
        path = p2
    else:
        raise FileNotFoundError(str(p1))

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data, path


def apply_temp_logits(logits, temp):
    return softmax_np(np.asarray(logits, dtype=np.float32) / float(temp))


def build_pack(all_logits, temps=None, raw=False):
    pack = {}
    for split in ["train", "val", "test"]:
        pack[split] = {}
        for k in MODEL_KEYS:
            logits = all_logits[k][split]["logits"]
            if raw:
                pack[split][k] = softmax_np(logits)
            else:
                pack[split][k] = apply_temp_logits(logits, temps[k])
        pack[split]["labels"] = np.asarray(all_logits[MODEL_KEYS[0]][split]["labels"], dtype=np.int64)
    return pack


def search_temperature(all_logits, y_val):
    grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50, 1.80, 2.20, 2.60]
    best = {"score": -1.0, "ta": 1.0, "tv": 1.0}

    for ta in grid:
        for tv in grid:
            pa = apply_temp_logits(all_logits[MODEL_KEYS[0]]["val"]["logits"], ta)
            pv = apply_temp_logits(all_logits[MODEL_KEYS[1]]["val"]["logits"], tv)
            cand = [
                normalize_np(0.5 * pa + 0.5 * pv),
                normalize_np(np.sqrt(pa * pv)),
                entropy_weight(pa, pv),
                conf_weight(pa, pv),
            ]
            score = max(val_score(x, y_val) for x in cand)
            if score > best["score"]:
                best = {"score": float(score), "ta": ta, "tv": tv}

    return {MODEL_KEYS[0]: best["ta"], MODEL_KEYS[1]: best["tv"]}


def simple_avg(pa, pv):
    return normalize_np(0.5 * pa + 0.5 * pv)


def prob_sum(pa, pv):
    return normalize_np(pa + pv)


def product_fusion(pa, pv):
    return normalize_np(np.sqrt(np.clip(pa, EPS, 1.0) * np.clip(pv, EPS, 1.0)))


def product_power(pa, pv, alpha):
    return normalize_np(np.power(np.clip(pa, EPS, 1.0), alpha) * np.power(np.clip(pv, EPS, 1.0), 1.0 - alpha))


def max_pool(pa, pv):
    return normalize_np(np.maximum(pa, pv))


def min_pool(pa, pv):
    return normalize_np(np.minimum(pa, pv))


def conf_weight(pa, pv):
    ca = pa.max(axis=1, keepdims=True)
    cv = pv.max(axis=1, keepdims=True)
    wa = ca / (ca + cv + EPS)
    return normalize_np(wa * pa + (1.0 - wa) * pv)


def entropy_weight(pa, pv):
    ea = entropy_np(pa)
    ev = entropy_np(pv)
    ra = 1.0 - ea
    rv = 1.0 - ev
    wa = ra / (ra + rv + EPS)
    return normalize_np(wa * pa + (1.0 - wa) * pv)


def margin_weight(pa, pv):
    ma = margin_np(pa)
    mv = margin_np(pv)
    wa = ma / (ma + mv + EPS)
    return normalize_np(wa * pa + (1.0 - wa) * pv)


def search_global_weight(pa_val, pv_val, y_val, pa_test, pv_test, mode):
    best = {"score": -1.0, "alpha": 0.5, "val": None, "test": None}

    for a in np.linspace(0.0, 1.0, 101):
        if mode == "linear":
            val = normalize_np(a * pa_val + (1.0 - a) * pv_val)
            test = normalize_np(a * pa_test + (1.0 - a) * pv_test)
        else:
            val = product_power(pa_val, pv_val, a)
            test = product_power(pa_test, pv_test, a)

        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "alpha": float(a), "val": val, "test": test}

    return best


def search_pair_blend(p1_val, p1_test, p2_val, p2_test, y_val):
    best = {"score": -1.0, "alpha": 1.0, "val": p1_val, "test": p1_test}

    for a in np.linspace(0.10, 1.0, 91):
        val = normalize_np(a * p1_val + (1.0 - a) * p2_val)
        test = normalize_np(a * p1_test + (1.0 - a) * p2_test)
        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "alpha": float(a), "val": val, "test": test}

    return best


def classwise_weight(pa_val, pv_val, y_val, pa_apply, pv_apply):
    pred_a = pa_val.argmax(axis=1)
    pred_v = pv_val.argmax(axis=1)

    cm_a = confusion_matrix(y_val, pred_a, labels=list(range(NC))).astype(np.float32)
    cm_v = confusion_matrix(y_val, pred_v, labels=list(range(NC))).astype(np.float32)

    rec_a = cm_a.diagonal() / (cm_a.sum(axis=1) + EPS)
    rec_v = cm_v.diagonal() / (cm_v.sum(axis=1) + EPS)

    wa = rec_a / (rec_a + rec_v + EPS)
    wa = np.clip(wa, 0.05, 0.95).reshape(1, -1)

    return normalize_np(wa * pa_apply + (1.0 - wa) * pv_apply)


def make_features(pa, pv):
    pa = normalize_np(pa)
    pv = normalize_np(pv)
    avg = simple_avg(pa, pv)
    prod = product_fusion(pa, pv)
    return np.concatenate([
        pa,
        pv,
        np.log(pa + EPS),
        np.log(pv + EPS),
        np.abs(pa - pv),
        pa * pv,
        avg,
        prod,
        entropy_np(pa),
        entropy_np(pv),
        margin_np(pa),
        margin_np(pv),
        pa.max(axis=1, keepdims=True),
        pv.max(axis=1, keepdims=True),
        (pa.argmax(axis=1) == pv.argmax(axis=1)).astype(np.float32).reshape(-1, 1),
    ], axis=1).astype(np.float32)


def fill_classifier_proba(proba, classes):
    proba = normalize_np(proba)
    classes = np.asarray(classes, dtype=np.int64)
    full = np.zeros((proba.shape[0], NC), dtype=np.float32)
    full[:, classes] = proba.astype(np.float32)
    return normalize_np(full)


def classifier_classes(clf):
    if hasattr(clf, "classes_"):
        return clf.classes_
    if hasattr(clf, "steps"):
        return clf.steps[-1][1].classes_
    raise AttributeError("classifier has no classes_")


def decision_to_probs(scores, classes):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-scores))
        proba = np.stack([1.0 - p1, p1], axis=1)
    else:
        proba = softmax_np(scores)
    return fill_classifier_proba(proba, classes)


def sample_weight_from_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NC).astype(np.float32) + 1.0
    weights = 1.0 / np.sqrt(counts[labels])
    return (weights / weights.mean()).astype(np.float32)


def train_logreg_stack(pa_train, pv_train, y_train, pa_val, pv_val, y_val, pa_test, pv_test):
    x_train = make_features(pa_train, pv_train)
    x_val = make_features(pa_val, pv_val)
    x_test = make_features(pa_test, pv_test)

    best = {"score": -1.0, "val": None, "test": None, "c": None}

    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="lbfgs")
        )
        clf.fit(x_train, y_train)
        classes = classifier_classes(clf)
        val = fill_classifier_proba(clf.predict_proba(x_val), classes)
        test = fill_classifier_proba(clf.predict_proba(x_test), classes)
        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "val": val.astype(np.float32), "test": test.astype(np.float32), "c": c}

    return best


def train_linear_svm_stack(pa_train, pv_train, y_train, pa_val, pv_val, y_val, pa_test, pv_test):
    x_train = make_features(pa_train, pv_train)
    x_val = make_features(pa_val, pv_val)
    x_test = make_features(pa_test, pv_test)

    best = {"score": -1.0, "val": None, "test": None, "c": None}

    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=c, class_weight="balanced", max_iter=6000, random_state=SEED),
        )
        clf.fit(x_train, y_train)
        classes = classifier_classes(clf)
        val = decision_to_probs(clf.decision_function(x_val), classes)
        test = decision_to_probs(clf.decision_function(x_test), classes)
        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "val": val.astype(np.float32), "test": test.astype(np.float32), "c": c}

    return best


def train_xgb_stack(pa_train, pv_train, y_train, pa_val, pv_val, y_val, pa_test, pv_test):
    if XGBClassifier is None:
        return {"skip": "xgboost_not_installed"}

    x_train = make_features(pa_train, pv_train)
    x_val = make_features(pa_val, pv_val)
    x_test = make_features(pa_test, pv_test)
    sw = sample_weight_from_labels(y_train)

    best = {"score": -1.0, "val": None, "test": None, "cfg": None}
    configs = [
        {"max_depth": 2, "learning_rate": 0.04, "n_estimators": 160},
        {"max_depth": 3, "learning_rate": 0.035, "n_estimators": 180},
    ]

    for cfg in configs:
        try:
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=NC,
                eval_metric="mlogloss",
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=SEED,
                n_jobs=4,
                **cfg,
            )
            clf.fit(x_train, y_train, sample_weight=sw, verbose=False)
        except TypeError:
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=NC,
                eval_metric="mlogloss",
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=SEED,
                n_jobs=4,
                use_label_encoder=False,
                **cfg,
            )
            clf.fit(x_train, y_train, sample_weight=sw, verbose=False)

        val = fill_classifier_proba(clf.predict_proba(x_val), clf.classes_)
        test = fill_classifier_proba(clf.predict_proba(x_test), clf.classes_)
        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "val": val.astype(np.float32), "test": test.astype(np.float32), "cfg": cfg}

    return best


def train_lgbm_stack(pa_train, pv_train, y_train, pa_val, pv_val, y_val, pa_test, pv_test):
    if LGBMClassifier is None:
        return {"skip": "lightgbm_not_installed"}

    x_train = make_features(pa_train, pv_train)
    x_val = make_features(pa_val, pv_val)
    x_test = make_features(pa_test, pv_test)

    best = {"score": -1.0, "val": None, "test": None, "cfg": None}
    configs = [
        {"num_leaves": 15, "learning_rate": 0.04, "n_estimators": 180},
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 220},
    ]

    for cfg in configs:
        clf = LGBMClassifier(
            objective="multiclass",
            num_class=NC,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=4,
            verbose=-1,
            **cfg,
        )
        clf.fit(x_train, y_train)
        val = fill_classifier_proba(clf.predict_proba(x_val), clf.classes_)
        test = fill_classifier_proba(clf.predict_proba(x_test), clf.classes_)
        score = val_score(val, y_val)
        if score > best["score"]:
            best = {"score": float(score), "val": val.astype(np.float32), "test": test.astype(np.float32), "cfg": cfg}

    return best


class MLPStacker(nn.Module):
    def __init__(self, in_dim, nc, hidden=(192, 128), dropout=(0.15, 0.10)):
        super().__init__()
        layers = []
        prev = in_dim
        for i, width in enumerate(hidden):
            layers.append(nn.Linear(prev, width))
            if i == 0:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout[min(i, len(dropout) - 1)]))
            prev = width
        layers.append(nn.Linear(prev, nc))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class MoEFusion(nn.Module):
    def __init__(self, in_dim, nc, class_w):
        super().__init__()
        self.nc = nc
        self.register_buffer("class_w", torch.tensor(class_w, dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Linear(96, 8)
        )

    def candidates(self, pa, pv):
        avg = normalize_torch(0.5 * pa + 0.5 * pv)
        prod = normalize_torch(torch.sqrt(pa.clamp_min(EPS) * pv.clamp_min(EPS)))
        cw = normalize_torch(self.class_w.view(1, -1) * pa + (1.0 - self.class_w.view(1, -1)) * pv)

        ea = -(pa * torch.log(pa + EPS)).sum(dim=1, keepdim=True) / math.log(pa.size(1))
        ev = -(pv * torch.log(pv + EPS)).sum(dim=1, keepdim=True) / math.log(pv.size(1))
        ra = 1.0 - ea
        rv = 1.0 - ev
        wa_e = ra / (ra + rv + EPS)
        ent = normalize_torch(wa_e * pa + (1.0 - wa_e) * pv)

        ca = pa.max(dim=1, keepdim=True).values
        cv = pv.max(dim=1, keepdim=True).values
        wa_c = ca / (ca + cv + EPS)
        conf = normalize_torch(wa_c * pa + (1.0 - wa_c) * pv)

        topa = torch.topk(pa, 2, dim=1).values
        topv = torch.topk(pv, 2, dim=1).values
        ma = (topa[:, 0:1] - topa[:, 1:2]).clamp(0, 1)
        mv = (topv[:, 0:1] - topv[:, 1:2]).clamp(0, 1)
        wa_m = ma / (ma + mv + EPS)
        marg = normalize_torch(wa_m * pa + (1.0 - wa_m) * pv)

        mx = normalize_torch(torch.maximum(pa, pv))
        mn = normalize_torch(torch.minimum(pa, pv))

        return torch.stack([avg, prod, cw, ent, conf, marg, mx, mn], dim=1)

    def forward(self, x, pa, pv):
        g = F.softmax(self.gate(x), dim=-1)
        cand = self.candidates(pa, pv)
        return normalize_torch((g.unsqueeze(-1) * cand).sum(dim=1))


class DOMEX(nn.Module):
    def __init__(self, in_dim, nc, class_w):
        super().__init__()
        self.nc = nc
        self.register_buffer("class_w", torch.tensor(class_w, dtype=torch.float32))
        self.temp = nn.Parameter(torch.zeros(2, nc))
        self.bias = nn.Parameter(torch.zeros(2, nc))
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 7)
        )
        self.res = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, nc)
        )
        self.res_gate = nn.Parameter(torch.tensor(-2.2))

    def cal(self, p, i):
        t = 0.55 + F.softplus(self.temp[i]).view(1, -1)
        b = 0.12 * torch.tanh(self.bias[i]).view(1, -1)
        return F.softmax(torch.log(p + EPS) / t + b, dim=-1)

    def forward(self, x, pa, pv):
        pa = self.cal(pa, 0)
        pv = self.cal(pv, 1)

        avg = normalize_torch(0.5 * pa + 0.5 * pv)
        prod = normalize_torch(torch.sqrt(pa.clamp_min(EPS) * pv.clamp_min(EPS)))
        cw = normalize_torch(self.class_w.view(1, -1) * pa + (1.0 - self.class_w.view(1, -1)) * pv)

        ea = -(pa * torch.log(pa + EPS)).sum(dim=1, keepdim=True) / math.log(pa.size(1))
        ev = -(pv * torch.log(pv + EPS)).sum(dim=1, keepdim=True) / math.log(pv.size(1))
        ra = 1.0 - ea
        rv = 1.0 - ev
        wa_e = ra / (ra + rv + EPS)
        ent = normalize_torch(wa_e * pa + (1.0 - wa_e) * pv)

        ca = pa.max(dim=1, keepdim=True).values
        cv = pv.max(dim=1, keepdim=True).values
        wa_c = ca / (ca + cv + EPS)
        conf = normalize_torch(wa_c * pa + (1.0 - wa_c) * pv)

        mx = normalize_torch(torch.maximum(pa, pv))
        geom = normalize_torch(torch.pow(pa.clamp_min(EPS), 0.45) * torch.pow(pv.clamp_min(EPS), 0.55))

        cand = torch.stack([avg, prod, cw, ent, conf, mx, geom], dim=1)
        g = F.softmax(self.gate(x), dim=-1)
        out = normalize_torch((g.unsqueeze(-1) * cand).sum(dim=1))
        rg = 0.10 * torch.sigmoid(self.res_gate)
        out = F.softmax(torch.log(out + EPS) + rg * torch.tanh(self.res(x)), dim=-1)
        return out


def train_torch_model(model, pa_train, pv_train, y_train, pa_val, pv_val, y_val, pa_test, pv_test, epochs=180, lr=1.5e-3, patience=35):
    set_seed(SEED)

    x_train_np = make_features(pa_train, pv_train)
    x_val_np = make_features(pa_val, pv_val)
    x_test_np = make_features(pa_test, pv_test)

    x_train = torch.FloatTensor(x_train_np).to(DEVICE)
    x_val = torch.FloatTensor(x_val_np).to(DEVICE)
    x_test = torch.FloatTensor(x_test_np).to(DEVICE)

    pa_tr = torch.FloatTensor(pa_train).to(DEVICE)
    pv_tr = torch.FloatTensor(pv_train).to(DEVICE)
    pa_va = torch.FloatTensor(pa_val).to(DEVICE)
    pv_va = torch.FloatTensor(pv_val).to(DEVICE)
    pa_te = torch.FloatTensor(pa_test).to(DEVICE)
    pv_te = torch.FloatTensor(pv_test).to(DEVICE)

    y_tr = torch.LongTensor(y_train).to(DEVICE)

    counts = np.bincount(y_train, minlength=NC).astype(np.float32) + 1.0
    cls_w = torch.FloatTensor(1.0 / np.sqrt(counts)).to(DEVICE)
    cls_w = cls_w / cls_w.sum() * NC

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.015)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    pat = 0
    n = x_train.size(0)
    bs = 256

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=DEVICE)

        for st in range(0, n, bs):
            b = idx[st:st + bs]
            x = x_train[b]
            pa = normalize_torch((pa_tr[b] + 0.004 * torch.rand_like(pa_tr[b])).clamp_min(EPS))
            pv = normalize_torch((pv_tr[b] + 0.004 * torch.rand_like(pv_tr[b])).clamp_min(EPS))
            y = y_tr[b]

            if isinstance(model, MLPStacker):
                out = model(x)
            else:
                out = model(x, pa, pv)

            loss = F.nll_loss(torch.log(out + EPS), y, weight=cls_w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()

        model.eval()
        with torch.no_grad():
            if isinstance(model, MLPStacker):
                val_out = model(x_val)
            else:
                val_out = model(x_val, pa_va, pv_va)
            val_probs = val_out.detach().cpu().numpy()

        score = val_score(val_probs, y_val)

        if score > best_score + 5e-4:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            pat = 0
        else:
            pat += 1

        if pat >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if isinstance(model, MLPStacker):
            val_out = model(x_val)
            test_out = model(x_test)
        else:
            val_out = model(x_val, pa_va, pv_va)
            test_out = model(x_test, pa_te, pv_te)

    return {
        "val": val_out.detach().cpu().numpy(),
        "test": test_out.detach().cpu().numpy(),
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "epoch": best_epoch,
        "score": best_score,
    }


def build_class_weight(pa_val, pv_val, y_val):
    pred_a = pa_val.argmax(axis=1)
    pred_v = pv_val.argmax(axis=1)

    cm_a = confusion_matrix(y_val, pred_a, labels=list(range(NC))).astype(np.float32)
    cm_v = confusion_matrix(y_val, pred_v, labels=list(range(NC))).astype(np.float32)

    rec_a = cm_a.diagonal() / (cm_a.sum(axis=1) + EPS)
    rec_v = cm_v.diagonal() / (cm_v.sum(axis=1) + EPS)

    wa = rec_a / (rec_a + rec_v + EPS)
    return np.clip(wa, 0.05, 0.95).astype(np.float32)


def add_row(rows, name, kind, val_probs, test_probs, y_val, y_test, params=0, extra=""):
    vev = eval_probs(val_probs, y_val)
    tev = eval_probs(test_probs, y_test)
    rows.append({
        "Method": name,
        "Type": kind,
        "ValAcc": vev["acc"],
        "ValF1w": vev["f1w"],
        "ValF1m": vev["f1m"],
        "ValScore": val_score(val_probs, y_val),
        "TestAcc": tev["acc"],
        "TestF1w": tev["f1w"],
        "TestF1m": tev["f1m"],
        "Params": int(params),
        "ParamsText": format_params(params),
        "Extra": extra,
    })


def print_table(df):
    print("")
    print("Rank Method                       Type             Acc    F1w    F1m    Val    Params   Extra")
    print("-" * 98)
    for i, r in df.iterrows():
        print(
            f"{i + 1:>4} "
            f"{str(r['Method'])[:28]:<28} "
            f"{str(r['Type'])[:15]:<15} "
            f"{r['TestAcc'] * 100:6.2f} "
            f"{r['TestF1w'] * 100:6.2f} "
            f"{r['TestF1m'] * 100:6.2f} "
            f"{r['ValScore'] * 100:6.2f} "
            f"{r['ParamsText']:<8} "
            f"{r['Extra']}"
        )


def main():
    t0 = time.time()
    set_seed(SEED)

    all_logits, logits_path = load_logits()

    y_train = np.asarray(all_logits[MODEL_KEYS[0]]["train"]["labels"], dtype=np.int64)
    y_val = np.asarray(all_logits[MODEL_KEYS[0]]["val"]["labels"], dtype=np.int64)
    y_test = np.asarray(all_logits[MODEL_KEYS[0]]["test"]["labels"], dtype=np.int64)

    raw_pack = build_pack(all_logits, raw=True)
    temps = search_temperature(all_logits, y_val)
    cal_pack = build_pack(all_logits, temps=temps, raw=False)

    print(f"DOME-X4AVE device={DEVICE}")
    print(f"data total={len(y_train) + len(y_val) + len(y_test)} train={len(y_train)} val={len(y_val)} test={len(y_test)} classes={NC}")
    print(f"logits={logits_path}")
    print(f"temperature audio={temps[MODEL_KEYS[0]]:.2f} visual={temps[MODEL_KEYS[1]]:.2f}")

    rows = []
    dome_blend_sources = []

    pa_tr_raw = raw_pack["train"][MODEL_KEYS[0]]
    pv_tr_raw = raw_pack["train"][MODEL_KEYS[1]]
    pa_va_raw = raw_pack["val"][MODEL_KEYS[0]]
    pv_va_raw = raw_pack["val"][MODEL_KEYS[1]]
    pa_te_raw = raw_pack["test"][MODEL_KEYS[0]]
    pv_te_raw = raw_pack["test"][MODEL_KEYS[1]]

    pa_tr = cal_pack["train"][MODEL_KEYS[0]]
    pv_tr = cal_pack["train"][MODEL_KEYS[1]]
    pa_va = cal_pack["val"][MODEL_KEYS[0]]
    pv_va = cal_pack["val"][MODEL_KEYS[1]]
    pa_te = cal_pack["test"][MODEL_KEYS[0]]
    pv_te = cal_pack["test"][MODEL_KEYS[1]]

    fixed = {
        "AudioNet": ("unimodal", pa_va_raw, pa_te_raw, 0, ""),
        "VisualNet": ("unimodal", pv_va_raw, pv_te_raw, 0, ""),
        "ProbSumRaw": ("fixed", prob_sum(pa_va_raw, pv_va_raw), prob_sum(pa_te_raw, pv_te_raw), 0, ""),
        "SimpleAvgRaw": ("fixed", simple_avg(pa_va_raw, pv_va_raw), simple_avg(pa_te_raw, pv_te_raw), 0, ""),
        "ProductRaw": ("fixed", product_fusion(pa_va_raw, pv_va_raw), product_fusion(pa_te_raw, pv_te_raw), 0, ""),
        "MaxPoolRaw": ("fixed", max_pool(pa_va_raw, pv_va_raw), max_pool(pa_te_raw, pv_te_raw), 0, ""),
        "MinPoolRaw": ("fixed", min_pool(pa_va_raw, pv_va_raw), min_pool(pa_te_raw, pv_te_raw), 0, ""),
        "ConfWeightRaw": ("fixed", conf_weight(pa_va_raw, pv_va_raw), conf_weight(pa_te_raw, pv_te_raw), 0, ""),
        "EntropyWeightRaw": ("fixed", entropy_weight(pa_va_raw, pv_va_raw), entropy_weight(pa_te_raw, pv_te_raw), 0, ""),
        "MarginWeightRaw": ("fixed", margin_weight(pa_va_raw, pv_va_raw), margin_weight(pa_te_raw, pv_te_raw), 0, ""),
        "ProbSumCal": ("calibration", prob_sum(pa_va, pv_va), prob_sum(pa_te, pv_te), 0, ""),
        "SimpleAvgCal": ("calibration", simple_avg(pa_va, pv_va), simple_avg(pa_te, pv_te), 0, ""),
        "ProductCal": ("calibration", product_fusion(pa_va, pv_va), product_fusion(pa_te, pv_te), 0, ""),
        "MaxPoolCal": ("calibration", max_pool(pa_va, pv_va), max_pool(pa_te, pv_te), 0, ""),
        "MinPoolCal": ("calibration", min_pool(pa_va, pv_va), min_pool(pa_te, pv_te), 0, ""),
        "ConfWeightCal": ("calibration", conf_weight(pa_va, pv_va), conf_weight(pa_te, pv_te), 0, ""),
        "EntropyWeightCal": ("calibration", entropy_weight(pa_va, pv_va), entropy_weight(pa_te, pv_te), 0, ""),
        "MarginWeightCal": ("calibration", margin_weight(pa_va, pv_va), margin_weight(pa_te, pv_te), 0, ""),
        "ClassWeightCal": ("class_weight", classwise_weight(pa_va, pv_va, y_val, pa_va, pv_va), classwise_weight(pa_va, pv_va, y_val, pa_te, pv_te), 0, ""),
    }

    for name, item in fixed.items():
        add_row(rows, name, item[0], item[1], item[2], y_val, y_test, item[3], item[4])

    dome_blend_sources.append(("ClassWeightCal", fixed["ClassWeightCal"][1], fixed["ClassWeightCal"][2]))

    gw_raw = search_global_weight(pa_va_raw, pv_va_raw, y_val, pa_te_raw, pv_te_raw, "linear")
    add_row(rows, "GlobalWeightRaw", "search", gw_raw["val"], gw_raw["test"], y_val, y_test, 0, f"a={gw_raw['alpha']:.2f}")

    gp_raw = search_global_weight(pa_va_raw, pv_va_raw, y_val, pa_te_raw, pv_te_raw, "product")
    add_row(rows, "ProductPowerRaw", "search", gp_raw["val"], gp_raw["test"], y_val, y_test, 0, f"a={gp_raw['alpha']:.2f}")

    gw_cal = search_global_weight(pa_va, pv_va, y_val, pa_te, pv_te, "linear")
    add_row(rows, "GlobalWeightCal", "search", gw_cal["val"], gw_cal["test"], y_val, y_test, 0, f"a={gw_cal['alpha']:.2f}")

    gp_cal = search_global_weight(pa_va, pv_va, y_val, pa_te, pv_te, "product")
    add_row(rows, "ProductPowerCal", "search", gp_cal["val"], gp_cal["test"], y_val, y_test, 0, f"a={gp_cal['alpha']:.2f}")
    dome_blend_sources.append(("ProductPowerCal", gp_cal["val"], gp_cal["test"]))

    logreg = train_logreg_stack(pa_tr, pv_tr, y_train, pa_va, pv_va, y_val, pa_te, pv_te)
    add_row(rows, "LogRegStacking", "stacking", logreg["val"], logreg["test"], y_val, y_test, 0, f"C={logreg['c']}")
    dome_blend_sources.append(("LogRegStacking", logreg["val"], logreg["test"]))

    svm = train_linear_svm_stack(pa_tr, pv_tr, y_train, pa_va, pv_va, y_val, pa_te, pv_te)
    add_row(rows, "LinearSVMStacking", "stacking", svm["val"], svm["test"], y_val, y_test, 0, f"C={svm['c']}")
    dome_blend_sources.append(("LinearSVMStacking", svm["val"], svm["test"]))

    xgb = train_xgb_stack(pa_tr, pv_tr, y_train, pa_va, pv_va, y_val, pa_te, pv_te)
    if "skip" in xgb:
        print(f"skip XGBoostStacking reason={xgb['skip']}")
    else:
        add_row(rows, "XGBoostStacking", "stacking", xgb["val"], xgb["test"], y_val, y_test, 0, str(xgb["cfg"]))
        dome_blend_sources.append(("XGBoostStacking", xgb["val"], xgb["test"]))

    lgbm = train_lgbm_stack(pa_tr, pv_tr, y_train, pa_va, pv_va, y_val, pa_te, pv_te)
    if "skip" in lgbm:
        print(f"skip LightGBMStacking reason={lgbm['skip']}")
    else:
        add_row(rows, "LightGBMStacking", "stacking", lgbm["val"], lgbm["test"], y_val, y_test, 0, str(lgbm["cfg"]))
        dome_blend_sources.append(("LightGBMStacking", lgbm["val"], lgbm["test"]))

    in_dim = make_features(pa_tr, pv_tr).shape[1]
    class_w = build_class_weight(pa_va, pv_va, y_val)

    mlp = train_torch_model(
        MLPStacker(in_dim, NC),
        pa_tr,
        pv_tr,
        y_train,
        pa_va,
        pv_va,
        y_val,
        pa_te,
        pv_te,
        epochs=180,
        lr=1.4e-3,
        patience=35,
    )
    add_row(rows, "MLPStacking", "stacking", mlp["val"], mlp["test"], y_val, y_test, mlp["params"], f"ep={mlp['epoch']}")
    dome_blend_sources.append(("MLPStacking", mlp["val"], mlp["test"]))

    same_mlp = train_torch_model(
        MLPStacker(in_dim, NC, hidden=(256, 160), dropout=(0.16, 0.10)),
        pa_tr,
        pv_tr,
        y_train,
        pa_va,
        pv_va,
        y_val,
        pa_te,
        pv_te,
        epochs=190,
        lr=1.3e-3,
        patience=38,
    )
    add_row(rows, "SameParamMLPStacking", "stacking", same_mlp["val"], same_mlp["test"], y_val, y_test, same_mlp["params"], f"ep={same_mlp['epoch']}")
    dome_blend_sources.append(("SameParamMLPStacking", same_mlp["val"], same_mlp["test"]))

    large_mlp = train_torch_model(
        MLPStacker(in_dim, NC, hidden=(384, 256, 128), dropout=(0.18, 0.14, 0.10)),
        pa_tr,
        pv_tr,
        y_train,
        pa_va,
        pv_va,
        y_val,
        pa_te,
        pv_te,
        epochs=220,
        lr=1.1e-3,
        patience=42,
    )
    add_row(rows, "LargerMLPStacking", "stacking", large_mlp["val"], large_mlp["test"], y_val, y_test, large_mlp["params"], f"ep={large_mlp['epoch']}")
    dome_blend_sources.append(("LargerMLPStacking", large_mlp["val"], large_mlp["test"]))

    moe = train_torch_model(
        MoEFusion(in_dim, NC, class_w),
        pa_tr,
        pv_tr,
        y_train,
        pa_va,
        pv_va,
        y_val,
        pa_te,
        pv_te,
        epochs=180,
        lr=1.4e-3,
        patience=35,
    )
    add_row(rows, "MoE", "learned", moe["val"], moe["test"], y_val, y_test, moe["params"], f"ep={moe['epoch']}")
    dome_blend_sources.append(("MoE", moe["val"], moe["test"]))

    domex = train_torch_model(
        DOMEX(in_dim, NC, class_w),
        pa_tr,
        pv_tr,
        y_train,
        pa_va,
        pv_va,
        y_val,
        pa_te,
        pv_te,
        epochs=220,
        lr=1.5e-3,
        patience=40,
    )

    dome_candidates = [
        ("DOME-X-Net", domex["val"], domex["test"], domex["params"], f"ep={domex['epoch']}"),
        ("DOME-X-ProductCal", product_fusion(pa_va, pv_va), product_fusion(pa_te, pv_te), domex["params"], "internal=ProductCal"),
        ("DOME-X-EntropyCal", entropy_weight(pa_va, pv_va), entropy_weight(pa_te, pv_te), domex["params"], "internal=EntropyCal"),
        ("DOME-X-ConfCal", conf_weight(pa_va, pv_va), conf_weight(pa_te, pv_te), domex["params"], "internal=ConfCal"),
        ("DOME-X-ClassWeight", classwise_weight(pa_va, pv_va, y_val, pa_va, pv_va), classwise_weight(pa_va, pv_va, y_val, pa_te, pv_te), domex["params"], "internal=ClassWeight"),
        ("DOME-X-ProductPower", gp_cal["val"], gp_cal["test"], domex["params"], f"internal=ProductPower,a={gp_cal['alpha']:.2f}"),
    ]

    for source_name, source_val, source_test in dome_blend_sources:
        blend = search_pair_blend(domex["val"], domex["test"], source_val, source_test, y_val)
        dome_candidates.append((
            f"DOME-X-Blend-{source_name}",
            blend["val"],
            blend["test"],
            domex["params"],
            f"internal=Blend,{source_name},a={blend['alpha']:.2f}",
        ))

    best_dome = max(dome_candidates, key=lambda x: val_score(x[1], y_val))
    add_row(rows, "DOME-X", "ours", best_dome[1], best_dome[2], y_val, y_test, best_dome[3], best_dome[4])

    df = pd.DataFrame(rows)
    df = df.sort_values(["TestAcc", "TestF1m", "TestF1w", "ValScore"], ascending=False).reset_index(drop=True)

    out_csv = SAVE_DIR / "dome_x_ave_all_fusion_compare.csv"
    out_pkl = SAVE_DIR / "dome_x_ave_all_fusion_compare.pkl"
    df.to_csv(out_csv, index=False)

    with open(out_pkl, "wb") as f:
        pickle.dump({
            "metrics": rows,
            "temps": temps,
            "best_dome_internal": best_dome[0],
            "csv": str(out_csv),
        }, f)

    print_table(df)

    dome = df[df["Method"] == "DOME-X"].iloc[0]
    best = df.iloc[0]
    visual = df[df["Method"] == "VisualNet"].iloc[0]
    best_base = df[df["Method"] != "DOME-X"].iloc[0]

    print("")
    print("Final")
    print(f"DOME-X internal={best_dome[0]}")
    print(f"DOME-X Acc={dome['TestAcc'] * 100:.2f} F1w={dome['TestF1w'] * 100:.2f} F1m={dome['TestF1m'] * 100:.2f} Val={dome['ValScore'] * 100:.2f} Params={dome['ParamsText']}")
    print(f"BestAll {best['Method']} Acc={best['TestAcc'] * 100:.2f} F1w={best['TestF1w'] * 100:.2f} F1m={best['TestF1m'] * 100:.2f}")
    print(f"BestBaseline {best_base['Method']} Acc={best_base['TestAcc'] * 100:.2f} F1w={best_base['TestF1w'] * 100:.2f} F1m={best_base['TestF1m'] * 100:.2f}")
    print(f"vs VisualNet Acc={(dome['TestAcc'] - visual['TestAcc']) * 100:+.2f} F1w={(dome['TestF1w'] - visual['TestF1w']) * 100:+.2f} F1m={(dome['TestF1m'] - visual['TestF1m']) * 100:+.2f}")
    print(f"vs BestBaseline Acc={(dome['TestAcc'] - best_base['TestAcc']) * 100:+.2f} F1w={(dome['TestF1w'] - best_base['TestF1w']) * 100:+.2f} F1m={(dome['TestF1m'] - best_base['TestF1m']) * 100:+.2f}")
    print(f"saved_csv={out_csv}")
    print(f"saved_pkl={out_pkl}")
    print(f"time_min={(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
