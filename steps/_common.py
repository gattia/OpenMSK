"""Shared helpers for all step entry points."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import SimpleITK as sitk

# Inter-step image handoff format.
#
# NIfTI stores the image affine as float32, so every .nii.gz write->read
# quantises the direction cosine matrix. Measured on data/anthonys_knee.nrrd:
# the direction moves 1.354e-08 through a NIfTI roundtrip and 0.0 through an
# NRRD one (NRRD writes the space directions as full-precision decimal text).
#
# 1e-08 sounds ignorable and is not. It moves marching-cubes vertices by
# ~1.5e-05 mm, which is enough for pyacvd -- deterministic in itself -- to land
# on a different clustering, changing even the vertex count, and regional
# cartilage thickness with it (0.007-0.02 mm). The monolith never showed this
# because it holds one sitk.Image in memory for the whole run and never reads
# one back; the steps hand images to each other through files, so from the first
# handoff on, every step worked in a slightly different frame than segmentation
# produced.
#
# So: steps READ the .nrrd. They still WRITE both -- the .nii.gz is what users
# download and what the research archive keeps -- but nothing reads it back.
LOSSLESS_EXT = ".nrrd"
NIFTI_EXT = ".nii.gz"
SEG_STEM = "_all-labels"
SUBREGIONS_STEM = "_subregions-labels"


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


def find_image_file(working_dir: Path, stem: str) -> Path:
    """Path of the image named ``*<stem>``, preferring the lossless NRRD.

    Every step writes its label image as both .nrrd and .nii.gz; this is the one
    place that decides which one the next step reads, so a new step cannot
    quietly reintroduce the float32 affine quantisation described at the top of
    this module.

    Falls back to the .nii.gz when there is no .nrrd -- working directories
    written before both formats existed, and jobs resumed from them, are still
    worth finishing -- but says so at WARNING level. A silent fallback would put
    the precision loss back invisibly, which is exactly how it got here.

    Raises FileNotFoundError if neither exists, so callers that treat a missing
    image as "degrade gracefully" (generate_meshes, t2_mapping on subregions)
    keep working unchanged.
    """
    working_dir = Path(working_dir)
    try:
        return find_file(working_dir, f"*{stem}{LOSSLESS_EXT}")
    except FileNotFoundError:
        pass

    try:
        path = find_file(working_dir, f"*{stem}{NIFTI_EXT}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No file matching '*{stem}{LOSSLESS_EXT}' or '*{stem}{NIFTI_EXT}' in {working_dir}"
        ) from exc

    logging.warning(
        "No *%s%s in %s; falling back to %s. NIfTI stores the affine as float32, "
        "so this image's direction cosines are quantised (~1e-08) relative to the "
        "step that wrote them, and meshes built from it will differ slightly from "
        "a run that had the NRRD.",
        stem, LOSSLESS_EXT, working_dir, path.name,
    )
    return path


def find_segmentation(working_dir: Path) -> Path:
    """Path of the canonical *_all-labels image. Raises if not found."""
    return find_image_file(working_dir, SEG_STEM)


def find_subregions(working_dir: Path) -> Path:
    """Path of the *_subregions-labels image. Raises if not found."""
    return find_image_file(working_dir, SUBREGIONS_STEM)


def image_prefix(path) -> str:
    """The output-filename prefix a label image implies.

    ``<prefix>_all-labels.nrrd`` and ``<prefix>_all-labels.nii.gz`` both give
    ``<prefix>``. Every output filename and the merged results file are named
    from this, so it must not depend on which format a step happened to read --
    a wrong prefix silently renames the user's results.
    """
    name = Path(path).name
    for ext in (NIFTI_EXT, LOSSLESS_EXT, ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    for stem in (SEG_STEM, SUBREGIONS_STEM):
        if name.endswith(stem):
            return name[: -len(stem)]
    return name


def load_segmentation(working_dir: Path) -> sitk.Image:
    """Load the canonical *_all-labels image from working_dir. Raises if not found."""
    return sitk.ReadImage(str(find_segmentation(working_dir)))


def load_subregions(working_dir: Path) -> sitk.Image:
    """Load the *_subregions-labels image from working_dir. Raises if not found."""
    return sitk.ReadImage(str(find_subregions(working_dir)))


def emit_progress(percent: int, message: str):
    """Print [PROGRESS] line to stdout for orchestrator consumption."""
    print(f"[PROGRESS] {percent}% {message}", flush=True)


STEP_RESULT_FILENAME = "_step_result.json"


def write_step_result(working_dir: Path, result: dict) -> None:
    """Write step result dict to working_dir/_step_result.json."""
    (Path(working_dir) / STEP_RESULT_FILENAME).write_text(json.dumps(result))


def find_file(working_dir: Path, pattern: str) -> Path:
    """Glob for a single file matching pattern. Raises if 0 or >1 matches."""
    working_dir = Path(working_dir)
    matches = list(working_dir.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '{pattern}' in {working_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matching '{pattern}' in {working_dir}: {matches}")
    return matches[0]
