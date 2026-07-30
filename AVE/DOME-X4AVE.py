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


CONTINUATION_ROOT = CODE_ROOT / "AVE/results/canonical_rost_paired_continuation_v1"
POSTERIORS = CONTINUATION_ROOT / "posteriors.npz"
CONTINUATION_MANIFEST = CONTINUATION_ROOT / "manifest.json"


def load_canonical_ave() -> staged.ArtifactBundle:
    bundle = staged._load_joint_artifact(
        "AVE canonical paired continuation",
        POSTERIORS,
        ("audio", "visual"),
        "test_folds",
        (
            "Official AVE train+val development and isolated test protocol. Both "
            "posterior banks are fixed 15-epoch continuations from identical v1 CE "
            "fold checkpoints. Staged fitting retains the adopted AVE sample-stratified "
            "role contract; repeated test video IDs are grouped for uncertainty only."
        ),
    )
    with np.load(POSTERIORS, allow_pickle=False) as artifact:
        development_ids = np.asarray(artifact["development_video_ids"])
        test_ids = np.asarray(artifact["test_video_ids"])
    if len(development_ids) != len(bundle.labels):
        raise RuntimeError("development video IDs are not row-aligned")
    if len(test_ids) != len(bundle.test_labels):
        raise RuntimeError("test video IDs are not row-aligned")
    bundle.development_groups = None
    bundle.test_groups = test_ids
    bundle.sources += (CONTINUATION_MANIFEST,)
    return bundle


if __name__ == "__main__":
    staged.LOADERS["ave"] = load_canonical_ave
    if sys.argv[1:] == ["--check"]:
        bundle = load_canonical_ave()
        staged.validate_bundle(bundle)
        print(
            f"AVE check passed: development={len(bundle.labels)} "
            f"test={len(bundle.test_labels)} regimes={sorted(bundle.regimes)}"
        )
    else:
        staged.main("ave")
