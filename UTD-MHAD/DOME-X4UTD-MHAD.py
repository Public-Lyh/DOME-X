from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PLACEHOLDER_ROOT = Path("your path")
PROJECT_ROOT = Path(os.environ.get("DOME_X_PROJECT_ROOT", "your path")).expanduser()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "Code").is_dir()),
        PLACEHOLDER_ROOT,
    )
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import dome_x_staged_runner as staged  # type: ignore[import-not-found]


POSTERIORS = (
    CODE_ROOT
    / "UTD-MHAD/results/canonical_rost_paired_continuation_v1/posteriors.npz"
)
CONTINUATION_MANIFEST = (
    CODE_ROOT
    / "UTD-MHAD/results/canonical_rost_paired_continuation_v1/manifest.json"
)


def load_canonical_utd() -> staged.ArtifactBundle:
    bundle = staged._load_joint_artifact(
        "UTD-MHAD canonical paired continuation",
        POSTERIORS,
        ("skeleton", "inertial", "rgb"),
        "test_folds",
        (
            "Odd/even cross-subject test protocol. Both posterior banks are "
            "fixed 15-epoch continuations from each identical v7 CE fold "
            "checkpoint; the only expert-objective difference is canonical ROST. "
            "Whole odd development subjects define the four staged fitting roles, "
            "and even test subjects remain isolated."
        ),
    )
    with np.load(POSTERIORS, allow_pickle=False) as artifact:
        development_subjects = np.asarray(
            artifact["development_subjects"], dtype=np.int64
        )
        test_subjects = np.asarray(artifact["test_subjects"], dtype=np.int64)
        if len(development_subjects) != len(bundle.labels):
            raise RuntimeError("development subject metadata is not row-aligned")
        if len(test_subjects) != len(bundle.test_labels):
            raise RuntimeError("test subject metadata is not row-aligned")
    if set(np.unique(development_subjects)) != {1, 3, 5, 7}:
        raise RuntimeError("unexpected UTD development subjects")
    if set(np.unique(test_subjects)) != {2, 4, 6, 8}:
        raise RuntimeError("unexpected UTD test subjects")
    bundle.development_groups = development_subjects
    bundle.test_groups = test_subjects
    bundle.sources += (CONTINUATION_MANIFEST,)
    return bundle


if __name__ == "__main__":
    staged.LOADERS["utd"] = load_canonical_utd
    if sys.argv[1:] == ["--check"]:
        bundle = load_canonical_utd()
        staged.validate_bundle(bundle)
        print(
            f"UTD-MHAD check passed: development={len(bundle.labels)} "
            f"test={len(bundle.test_labels)} regimes={sorted(bundle.regimes)}"
        )
    else:
        staged.main("utd")
