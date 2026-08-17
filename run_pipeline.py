"""Run the full knee MRI analysis pipeline.

Chains the `steps/` modules. The standalone / dev entry point; the website
drives the same steps through its own orchestrator.

This is a second orchestrator and nothing watches it, so keep it thin. Every
piece of pipeline knowledge that lives here is a place it can drift from the
website's plan -- which it has done three times already (a missing `subregions`
step after the D7b split, an unchecked `label_remap` skip, and
`_get_remap_table()` below).

Usage:
    python run_pipeline.py /path/to/image /path/to/output/ [model_name]
    python run_pipeline.py /path/to/image /path/to/output/ --config /path/to/config.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from steps._common import load_config


def _run_step_subprocess(module_name, working_dir, options=None, config_path=None):
    """Run a step module as a subprocess and return its JSON result.

    Used for steps that load GPU frameworks (TF, PyTorch) to ensure
    GPU memory is fully released when the subprocess exits.
    """

    cmd = [sys.executable, "-m", module_name, str(working_dir)]
    if options:
        cmd.extend(["--options", json.dumps(options)])
    if config_path:
        cmd.extend(["--config", str(config_path)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        raise RuntimeError(
            f"{module_name} failed (exit code {result.returncode}):\n{result.stderr[-1000:]}"
        )

    # Print step's stdout (progress/logging)
    if result.stdout.strip():
        print(result.stdout)

    # Read result from file written by step
    from steps._common import STEP_RESULT_FILENAME
    result_path = working_dir / STEP_RESULT_FILENAME
    if not result_path.exists():
        raise RuntimeError(f"{module_name} did not produce a result file")
    step_result = json.loads(result_path.read_text())
    result_path.unlink()
    return step_result


def _free_gpu_memory():
    """Release GPU memory from TF and PyTorch."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import tensorflow as tf
        # TF doesn't have a clean way to release GPU memory short of
        # clearing the session. This at least frees cached tensors.
        from tensorflow.python.eager import context
        if context._context is not None:
            context._context._clear_caches()
    except (ImportError, AttributeError):
        pass


def run_all(working_dir, model_name=None, config=None, config_path=None):
    """Run the full pipeline: segment -> remap -> meshes -> t2 -> nsm -> bscore."""
    working_dir = Path(working_dir)

    # Step 1: Segmentation (subprocess — TF/PyTorch grab GPU memory and
    # don't release it, so we need process isolation for later CUDA steps)
    seg_result = _run_step_subprocess(
        "steps.segment", working_dir,
        options={"model": model_name},
        config_path=config_path,
    )

    # Step 2: Label remapping
    from steps.label_remap import run as label_remap
    remap_table = _get_remap_table(seg_result["model_name"], config)
    if remap_table:
        remap_result = label_remap(working_dir, options={"remap_table": remap_table}, config=config)
        # Inspect the result, not just the absence of an exception. The step can
        # return successfully having declined to do anything -- the usual cause
        # is a working directory that already holds a *-native.nii.gz, i.e. this
        # ran before -- and every downstream step would then measure native
        # labels while believing they are canonical: a "femur" mesh built from
        # patellar cartilage, with no error anywhere. The website orchestrator
        # treats a skip here as fatal for the same reason; this path is the one
        # nothing else is watching.
        if remap_result.get("skipped"):
            raise RuntimeError(
                f"label_remap declined to run ({remap_result.get('reason')}). "
                f"Refusing to continue on labels that may not be canonical."
            )
        if remap_result.get("unmapped_labels"):
            raise RuntimeError(
                f"label_remap deleted native labels {remap_result['unmapped_labels']} "
                f"-- they have no entry in the remap table for this model."
            )

    # Step 3: Femur cartilage subregions.
    # Its own step since D7b -- it is pure image processing on the canonical
    # labels and never needed a mesh, which is why t2_mapping was able to depend
    # on "meshes" for something meshes did not provide. It has to run here, and
    # not inside generate_meshes, or BOTH consumers silently degrade: femoral
    # thickness collapses from five subregions to one, and T2 loses labels 11-15.
    # Non-fatal by design -- both consumers fall back -- so a skip is logged and
    # the run continues.
    from steps.subregions import run as subregions
    subregions_result = subregions(working_dir, config=config)
    if subregions_result.get("skipped"):
        logging.warning(
            "Subregion labelling skipped (%s); femoral regional thickness and "
            "depth-resolved T2 will not be produced.",
            subregions_result.get("reason"),
        )

    # Step 4: Mesh generation + cartilage thickness
    from steps.generate_meshes import run as generate_meshes
    generate_meshes(working_dir, config=config)

    # Step 5: T2 mapping (only for qDESS input)
    if seg_result["is_qdess"]:
        from steps.t2_mapping import run as t2_mapping
        t2_mapping(working_dir, config=config)

    # Step 6 & 7: NSM fitting (subprocess for fresh CUDA) + BScore
    if config.get("perform_bone_and_cart_nsm") or config.get("perform_bone_only_nsm"):
        nsm_type = _get_nsm_type(config)
        from steps.run_nsm import run as run_nsm
        from steps.compute_bscore import run as compute_bscore
        run_nsm(working_dir, options={"nsm_type": nsm_type}, config=config, config_path=config_path)
        compute_bscore(working_dir, options={"bscore_type": nsm_type}, config=config)


def _get_nsm_type(config):
    """Determine NSM type from config flags."""
    both = config.get("perform_bone_and_cart_nsm") and config.get("perform_bone_only_nsm")
    if both:
        return "both"
    elif config.get("perform_bone_and_cart_nsm"):
        return "bone_and_cart"
    else:
        return "bone_only"


def _get_remap_table(model_name, config):
    """Get the label remap table for a given model.

    Returns None if no remapping is needed (model already uses canonical labels).

    Validated against actual model outputs on 2026-08-15: DOSMA and nnU-Net do
    share one native label scheme. nnU-Net's own dataset.json declares it, the
    same for the fullres and cascade configurations, and its inference wrapper
    writes those labels through without remapping.
    """
    # DOSMA-native -> canonical remap
    # Native: 1=pat_cart, 2=fem_cart, 3=med_tib_cart, 4=lat_tib_cart,
    #         5=med_meniscus, 6=lat_meniscus, 7=femur, 8=tibia, 9=patella
    # Canonical: 1=femur, 2=tibia, 3=patella, 4=fem_cart, 5=med_tib_cart,
    #            6=lat_tib_cart, 7=pat_cart, 8=med_meniscus, 9=lat_meniscus
    #
    # 5 and 6 were missing here until 2026-08-15. label_remap builds its output
    # with np.zeros_like and writes only mapped values, so leaving them out did
    # not leave the menisci alone -- it deleted them, silently, on every scan
    # that had them (181 of 190 archived dosma_ananya jobs carry all nine).
    # Must stay equal to DOSMA_NATIVE_LABEL_MAP in the website repo's
    # model_registry.py; its TestLabelMapIntegrity is what notices if it does not.
    DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 1, 8: 2, 9: 3}

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

    config_path = args.config or os.environ.get(
        "KNEEPIPELINE_CONFIG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
    )
    config = load_config(config_path)

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

    run_all(save_path, model_name=args.model_name, config=config, config_path=config_path)
