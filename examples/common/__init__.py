"""Shared helpers for the NPU showcase examples: device compile/timing, HUD overlays, media recording."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MEDIA_DIR = REPO_ROOT / "media"
CLIPS_DIR = REPO_ROOT / "assets" / "clips"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
