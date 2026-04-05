"""Shared helpers for all step entry points."""

import argparse
import json
import os
import sys
from pathlib import Path

import SimpleITK as sitk


def parse_step_args() -> argparse.Namespace:
    """Standard arg parsing: working_dir, --options (JSON), --config (path)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", type=Path, help="Working directory with inputs/outputs")
    parser.add_argument("--options", type=str, default="{}", help="JSON string of step-specific options")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config.json")
    args = parser.parse_args()
    args.options = json.loads(args.options)
    args.config = load_config(args.config)
    return args


def load_config(config_path=None) -> dict:
    """Load pipeline config.json.

    Resolution order:
    1. Explicit config_path argument
    2. KNEEPIPELINE_CONFIG environment variable
    3. config.json in the kneepipeline directory (next to this file's parent)
    """
    if config_path is None:
        config_path = os.environ.get(
            "KNEEPIPELINE_CONFIG",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
        )
    with open(config_path) as f:
        return json.load(f)


def load_segmentation(working_dir: Path) -> sitk.Image:
    """Load *_all-labels.nii.gz from working_dir. Raises if not found."""
    path = find_file(working_dir, "*_all-labels.nii.gz")
    return sitk.ReadImage(str(path))


def load_subregions(working_dir: Path) -> sitk.Image:
    """Load *_subregions-labels.nii.gz from working_dir. Raises if not found."""
    path = find_file(working_dir, "*_subregions-labels.nii.gz")
    return sitk.ReadImage(str(path))


def emit_progress(percent: int, message: str):
    """Print [PROGRESS] line to stdout for orchestrator consumption."""
    print(f"[PROGRESS] {percent}% {message}", flush=True)


def find_file(working_dir: Path, pattern: str) -> Path:
    """Glob for a single file matching pattern. Raises if 0 or >1 matches."""
    working_dir = Path(working_dir)
    matches = list(working_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '{pattern}' in {working_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matching '{pattern}' in {working_dir}: {matches}")
    return matches[0]
