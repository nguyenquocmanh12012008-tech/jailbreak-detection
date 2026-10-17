from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_src_is_importable_as_a_package():
    assert find_spec("src") is not None


def test_baseline_dependencies_are_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()

    for dependency in ("pandas", "numpy", "scikit-learn"):
        assert dependency in requirements


def test_pytest_is_declared_as_a_development_dependency():
    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()

    assert "pytest" in requirements


def test_local_data_and_generated_files_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in (".venv/", "/data/", "/outputs/", "submission.csv"):
        assert pattern in gitignore
