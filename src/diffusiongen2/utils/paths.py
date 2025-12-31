from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def root_path(*subpaths) -> Path:
    return PROJECT_ROOT.joinpath(*subpaths)

