import importlib
import os
import sys
from pathlib import Path


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


def load_runner():
    return importlib.import_module("dome_x_staged_runner")


def check_environment() -> None:
    runner = load_runner()
    bundle = runner.LOADERS["modelnet"]()
    runner.validate_bundle(bundle)
    print(
        f"ModelNet40 check passed: development={len(bundle.labels)} "
        f"test={len(bundle.test_labels)} regimes={sorted(bundle.regimes)}"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check_environment()
    else:
        load_runner().main("modelnet")
