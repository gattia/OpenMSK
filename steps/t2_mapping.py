"""Step 4: T2 relaxation mapping from qDESS DICOM input.

Computes T2 maps and per-region T2 statistics. Optionally computes
depth-dependent T2 metrics if bone meshes are available.

Precondition: Input must be a two-echo qDESS DICOM series. The orchestrator
should check the is_qdess flag from segmentation and skip this step if false;
this step also declines (``skipped: True``) rather than failing if the input
turns out not to be loadable as qDESS.

The qDESS spoiler private tags (GL area 0x001910B6, TG 0x001910B7) are NOT a
precondition. They are routinely stripped by DICOM anonymisation, and DOSMA
falls back to the Sveinsson low-spoiling equations (6 & 7) without them. The
two estimators disagree — low-spoiling reads low, by ~1.5% at 10-20 ms rising
to ~5.4% at 60-80 ms, so it is not a constant factor and cannot be corrected
after the fact — so the step reports which one ran as ``t2_method``:
``"spoiled"`` or ``"low_spoiling"``.

Neither the `subregions` step nor the `meshes` step is a precondition (D7).
Missing subregions costs the five femur subregion metrics (labels 11-15) and
leaves the whole-region ones (canonical labels 4-7, which live in the
segmentation itself); missing meshes costs the depth-resolved (deep and
superficial) metrics. Each is reported as ``has_subregions`` /
``has_depth_dependent`` rather than raising.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import (
    emit_progress,
    find_segmentation,
    image_prefix,
    load_segmentation,
    load_subregions,
    parse_step_args,
    write_step_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# T2 analysis constants
T2_MIN_VALID = 0
T2_MAX_VALID = 80
DEPTH_THRESHOLD = 0.5
DEEP_OFFSET = 100
SUPERFICIAL_OFFSET = 200

# Region names for T2 statistics. Uses canonical + subregion labels.
REGION_NAMES = {
    4: "fem_cart",
    5: "med_tib_cart",
    6: "lat_tib_cart",
    7: "pat_cart",
    11: "ant_fem_cart",
    12: "med_wb_fem_cart",
    13: "lat_wb_fem_cart",
    14: "med_post_fem_cart",
    15: "lat_post_fem_cart",
}

# Bone config for depth-dependent T2 (canonical labels)
BONE_CONFIG_T2 = {
    "femur": {
        "list_cart_labels": [4],     # CANONICAL femur_cart
    },
    "tibia": {
        "list_cart_labels": [5, 6],  # CANONICAL tibia_cart_med, tibia_cart_lat
    },
    "patella": {
        "list_cart_labels": [7],     # CANONICAL patella_cart
    },
}


def read_spoiler_parameter(qdess, tag):
    """Read a qDESS spoiler private tag (GL area or TG) as a float.

    Returns None when there is no usable value, which is how an anonymised
    scan normally looks. Callers turn that into the low-spoiling fallback.

    Public, and worth keeping public: the spoiled/low-spoiling decision decides
    which T2 estimator a scan gets, and the two are not comparable after the
    fact, so it is tested directly (tests/test_steps/test_t2_mapping.py) rather
    than only through the step. Anything that needs the same decision must call
    this rather than reimplement it: while a second implementation of the
    pipeline existed, its private copy of this logic had to be fixed separately,
    and until it was, the same scan got a different estimator depending on which
    path ran it.

    Three ways to have no usable value, all treated alike:
      - tag absent (anonymisation stripped it);
      - tag present but zero, which is DOSMA's own "no spoiler parameters"
        sentinel (qdess.py:202) — labelling that "spoiled" would be a lie;
      - tag present but not castable to float, e.g. an untyped (VR "UN") value
        from an export that dropped the private creator. DOSMA casts the raw
        tag values it reads itself (qdess.py:200-201) and would raise on these;
        low-spoiling T2 beats no T2 at all.
    """
    value = qdess.get_metadata(tag, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        logging.warning("qDESS spoiler tag %s is present but not numeric: %r", tag, value)
        return None
    return value if value != 0 else None


def run(working_dir, options=None, config=None):
    """Run T2 mapping step.

    Args:
        working_dir: Directory containing DICOM input and segmentation, and
            optionally the subregion segmentation and bone meshes.
        options: Dict (no user-configurable options currently).
        config: Pipeline config dict (unused currently).

    Returns:
        Dict with metrics, has_subregions and has_depth_dependent flags, and
        t2_method ("spoiled" or "low_spoiling"). If the input is not a two-echo
        qDESS series, returns {"skipped": True, "reason": ...} instead — that is
        the only case in which T2 genuinely cannot be computed.
    """
    from dosma.scan_sequences import QDess
    from dosma.tissues import FemoralCartilage
    import pymskt as mskt

    working_dir = Path(working_dir)
    options = options or {}

    emit_progress(0, "Loading qDESS data")
    qdess = _load_qdess(working_dir)
    if qdess is None:
        logging.info("Input is not a two-echo qDESS series. Skipping T2 computation.")
        return {
            "metrics": {},
            "has_subregions": False,
            "has_depth_dependent": False,
            "skipped": True,
            "reason": "not qDESS input",
        }

    # Spoiler amplitude (GL area) and duration (TG) are qDESS private tags that
    # DICOM anonymisation routinely strips. Their absence is NOT a reason to skip:
    # DOSMA falls back to the Sveinsson low-spoiling equations (6 & 7). Record
    # which estimator ran, because the two do not agree (see module docstring).
    gl_area = read_spoiler_parameter(qdess, QDess.__GL_AREA_TAG__)
    tg = read_spoiler_parameter(qdess, QDess.__TG_TAG__)
    spoiled = gl_area is not None and tg is not None
    t2_method = "spoiled" if spoiled else "low_spoiling"
    if not spoiled:
        logging.warning(
            "GL and TG tags not present (or zero). Computing T2 with the "
            "low-spoiling approximation (Sveinsson eqs. 6 & 7). NOTE: these are "
            "private tags and may have been removed in the DICOM anonymization "
            "process. Low-spoiling T2 reads slightly low relative to the spoiled "
            "solution; the step reports t2_method='low_spoiling'."
        )

    emit_progress(20, "Computing T2 map")
    cart = FemoralCartilage()
    t2map = qdess.generate_t2_map(
        cart, suppress_fat=False, suppress_fluid=False,
        # 0 is DOSMA's own sentinel for "no spoiler parameters" (qdess.py:202). It is
        # passed rather than left None only because DOSMA would otherwise dereference
        # the absent tags before reaching its fallback.
        gl_area=gl_area if spoiled else 0, tg=tg if spoiled else 0,
        spoiling=spoiled,
    )
    sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation="sagittal")

    # Determine filename prefix (independent of which format the segmentation
    # was read from, so output names never move)
    filename_prefix = image_prefix(find_segmentation(working_dir))

    # Save T2 map
    sitk.WriteImage(sitk_t2map, str(working_dir / f"{filename_prefix}_t2map.nii.gz"), useCompression=False)
    sitk.WriteImage(sitk_t2map, str(working_dir / f"{filename_prefix}_t2map.nrrd"), useCompression=False)

    # Load the label image the regional T2 statistics are computed over.
    emit_progress(40, "Computing T2 statistics")
    try:
        sitk_seg_regions = load_subregions(working_dir)
        has_subregions = True
    except FileNotFoundError:
        # No subregions step in this job. Whole-region cartilage labels (4-7) live in
        # the segmentation itself; only the femur subregions (11-15) are lost.
        logging.info(
            "No *_subregions-labels image in %s. Computing whole-region T2 only; "
            "the femur subregion metrics (labels 11-15) are not available.",
            working_dir,
        )
        sitk_seg_regions = load_segmentation(working_dir)
        has_subregions = False
    seg_array = sitk.GetArrayFromImage(sitk_seg_regions)

    t2_array = sitk.GetArrayFromImage(sitk_t2map)
    t2_array[t2_array >= T2_MAX_VALID] = np.nan
    t2_array[t2_array <= T2_MIN_VALID] = np.nan

    dict_results = {}

    # Global T2 metrics per region
    for cart_idx, region_name in REGION_NAMES.items():
        if cart_idx in seg_array:
            region_t2 = t2_array[seg_array == cart_idx]
            dict_results[f"{region_name}_t2_ms_mean"] = float(np.nanmean(region_t2))
            dict_results[f"{region_name}_t2_ms_std"] = float(np.nanstd(region_t2))
            dict_results[f"{region_name}_t2_ms_median"] = float(np.nanmedian(region_t2))

    # Depth-dependent T2 (requires bone meshes from generate_meshes step)
    has_depth_dependent = False
    bone_meshes_available = all(
        (working_dir / f"{bone}_mesh.vtk").exists() for bone in BONE_CONFIG_T2
    )

    # combine_depth_region_segs() genuinely needs the subregion image, so the
    # depth branch states that requirement here rather than at the top of the
    # step: without meshes OR without subregions, the whole-region metrics above
    # still stand (D7).
    if has_subregions and bone_meshes_available:
        emit_progress(60, "Computing depth-dependent T2")
        try:
            sitk_seg = load_segmentation(working_dir)
            depth_segs = []

            for bone_name, bone_t2_config in BONE_CONFIG_T2.items():
                bone_mesh_path = working_dir / f"{bone_name}_mesh.vtk"
                bone_mesh = mskt.mesh.io.read_vtk(str(bone_mesh_path))

                # Reconstruct BoneMesh with cartilage info for depth splitting
                bone_mesh_obj = mskt.mesh.BoneMesh(mesh=bone_mesh)
                bone_mesh_obj.list_cartilage_labels = bone_t2_config["list_cart_labels"]
                bone_mesh_obj.seg_image = sitk_seg

                bone_new_seg, bone_rel_depth = bone_mesh_obj.break_cartilage_into_superficial_deep(
                    rel_depth_thresh=DEPTH_THRESHOLD,
                    return_rel_depth=True,
                    resample_cartilage_surface=10_000,
                )
                depth_segs.append(bone_new_seg)

            new_seg_combined = mskt.image.cartilage_processing.combine_depth_region_segs(
                sitk_seg_regions, depth_segs
            )
            sitk.WriteImage(
                new_seg_combined,
                str(working_dir / f"{filename_prefix}_depth_seg.nrrd"),
                useCompression=True,
            )

            seg_array_depth = sitk.GetArrayFromImage(new_seg_combined)
            for cart_idx, region_name in REGION_NAMES.items():
                for depth_offset, depth_name in [(DEEP_OFFSET, "deep"), (SUPERFICIAL_OFFSET, "superficial")]:
                    cart_idx_depth = cart_idx + depth_offset
                    if cart_idx_depth in seg_array_depth:
                        region_t2 = t2_array[seg_array_depth == cart_idx_depth]
                        dict_results[f"{region_name}_{depth_name}_t2_ms_mean"] = float(np.nanmean(region_t2))
                        dict_results[f"{region_name}_{depth_name}_t2_ms_std"] = float(np.nanstd(region_t2))
                        dict_results[f"{region_name}_{depth_name}_t2_ms_median"] = float(np.nanmedian(region_t2))

            has_depth_dependent = True

        except Exception:
            logging.error("Depth-dependent T2 failed, returning global T2 only", exc_info=True)

    elif not bone_meshes_available:
        logging.info("Bone meshes not found, skipping depth-dependent T2")
    else:
        logging.info("Subregion segmentation not found, skipping depth-dependent T2")

    # Save T2 results
    emit_progress(90, "Saving T2 results")
    if dict_results:
        with open(str(working_dir / f"{filename_prefix}_t2_results.json"), "w") as f:
            # t2_method goes in the file as well as the step result: run_pipeline.py
            # discards the step result, so for standalone runs the file is the only
            # record of which estimator produced these numbers.
            json.dump({**dict_results, "t2_method": t2_method}, f, indent=4)

    emit_progress(100, "T2 mapping complete")
    return {
        "metrics": dict_results,
        "has_subregions": has_subregions,
        "has_depth_dependent": has_depth_dependent,
        "t2_method": t2_method,
    }


def _load_qdess(working_dir):
    """Load the working directory's DICOM input as a two-echo qDESS scan.

    Returns:
        QDess, or None if the input is not a two-echo qDESS series (including
        the case where there is no DICOM directory at all — e.g. a NIfTI job).
        A None return means T2 genuinely cannot be computed; it says nothing
        about the GL/TG spoiler tags.
    """
    from dosma.scan_sequences import QDess

    try:
        dicom_dir = _find_dicom_dir(Path(working_dir))
    except FileNotFoundError:
        return None

    try:
        try:
            return QDess.from_dicom(str(dicom_dir))
        except KeyError:
            return QDess.from_dicom(str(dicom_dir), group_by="EchoTime")
    except (ValueError, TypeError, FileNotFoundError, KeyError):
        # Same set steps.segment tolerates when deciding is_qdess: not two echoes,
        # unreadable as DICOM by DOSMA, or extensionless files DOSMA cannot see.
        logging.info("Could not load input as qDESS", exc_info=True)
        return None


def _find_dicom_dir(working_dir):
    """Find a DICOM directory in working_dir."""
    working_dir = Path(working_dir)
    for item in sorted(working_dir.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            # Check if it contains DICOM-like files
            files = list(item.iterdir())
            if files:
                return item
    raise FileNotFoundError(f"No DICOM directory found in {working_dir}")


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
