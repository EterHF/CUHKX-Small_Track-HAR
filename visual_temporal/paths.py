"""Repository paths shared by visual-temporal entry points."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints/visual_temporal"
RESULT_DIR = REPO_ROOT / "results/visual_temporal"
DATA_DIR = REPO_ROOT / "data"
