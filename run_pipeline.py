"""Run the full knee MRI analysis pipeline.

Chains all steps in-process. Replaces dosma_knee_seg.py as the standalone
entry point.

Usage:
    python run_pipeline.py /path/to/image /path/to/output/ [model_name]
    python run_pipeline.py /path/to/image /path/to/output/ --config /path/to/config.json
"""

import argparse
import os
from pathlib import Path

from steps._common import load_config
from steps.segment import run as segment
from steps.label_remap import run as label_remap


def run_all(working_dir, model_name=None, config=None):
    """Run the full pipeline: segment -> remap -> meshes -> t2 -> nsm -> bscore.

    Steps that are not yet implemented are noted with TODOs and will be
    added as Phase 2 and Phase 3 land.
    """
    working_dir = Path(working_dir)

    # Step 1: Segmentation
    seg_result = segment(working_dir, options={"model": model_name}, config=config)

    # Step 2: Label remapping
    remap_table = _get_remap_table(seg_result["model_name"], config)
    if remap_table:
        label_remap(working_dir, options={"remap_table": remap_table}, config=config)

    # Step 3: Mesh generation + cartilage thickness
    # TODO: Phase 2 — from steps.generate_meshes import run as generate_meshes

    # Step 4: T2 mapping (only for qDESS input)
    # TODO: Phase 2 — from steps.t2_mapping import run as t2_mapping

    # Step 5: NSM fitting
    # TODO: Phase 3 — from steps.run_nsm import run as run_nsm

    # Step 6: BScore computation
    # TODO: Phase 3 — from steps.compute_bscore import run as compute_bscore


def _get_remap_table(model_name, config):
    """Get the label remap table for a given model.

    Returns None if no remapping is needed (model already uses canonical labels).

    TODO: Once remap tables are validated against actual model outputs,
    this should pull from the website repo's model_registry or from config.
    For now, DOSMA and nnU-Net models share the same native label scheme.
    """
    # DOSMA-native -> canonical remap
    # Native: 1=pat_cart, 2=fem_cart, 3=med_tib_cart, 4=lat_tib_cart,
    #         7=femur, 8=tibia, 9=patella
    # Canonical: 1=femur, 2=tibia, 3=patella, 4=fem_cart,
    #            5=med_tib_cart, 6=lat_tib_cart, 7=pat_cart
    DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}

    # All current models (DOSMA and nnU-Net) use the same native label scheme
    return DOSMA_REMAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knee MRI Analysis Pipeline")
    parser.add_argument("path_image", help="Path to input MRI (DICOM dir, NIfTI, NRRD)")
    parser.add_argument("path_save", help="Output directory")
    parser.add_argument("model_name", nargs="?", default=None,
                        help="Segmentation model name (default: from config)")
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.path_save, exist_ok=True)

    # If input is a file (not directory), we need it accessible in working_dir.
    # For DICOM directories, the orchestrator/user points path_image at the dir.
    # For files, we assume they're already in path_save or we copy them.
    # For now, pass the image path directly — _find_input_image handles both cases.
    #
    # The simplest approach: if path_image != path_save, copy/symlink the input
    # into path_save so all steps find it there.
    input_path = Path(args.path_image)
    save_path = Path(args.path_save)

    if input_path.resolve() != save_path.resolve() and not input_path.resolve().is_relative_to(save_path.resolve()):
        # Symlink the input into the working directory
        link_name = save_path / input_path.name
        if not link_name.exists():
            link_name.symlink_to(input_path.resolve())

    run_all(save_path, model_name=args.model_name, config=config)
