"""Step 4: T2 relaxation mapping from qDESS DICOM input.

Computes T2 maps and per-region T2 statistics. Optionally computes
depth-dependent T2 metrics if bone meshes are available.

Extracts from seg_thick_t2_pipeline.py lines 433-503.

Precondition: Input must be qDESS DICOM with GL/TG private tags.
The orchestrator should check the is_qdess flag from segmentation
and skip this step if false.
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
    find_file,
    load_segmentation,
    load_subregions,
    parse_step_args,
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


def run(working_dir, options=None, config=None):
    """Run T2 mapping step.

    Args:
        working_dir: Directory containing DICOM input, segmentation, subregion
            segmentation, and optionally bone meshes.
        options: Dict (no user-configurable options currently).
        config: Pipeline config dict (unused currently).

    Returns:
        Dict with metrics and has_depth_dependent flag.
    """
    from dosma.scan_sequences import QDess
    from dosma.tissues import FemoralCartilage
    import pymskt as mskt

    working_dir = Path(working_dir)
    options = options or {}

    emit_progress(0, "Loading qDESS data")
    dicom_dir = _find_dicom_dir(working_dir)
    try:
        qdess = QDess.from_dicom(str(dicom_dir))
    except KeyError:
        qdess = QDess.from_dicom(str(dicom_dir), group_by="EchoTime")

    # Check for required private tags
    include_required_tags = (
        (qdess.get_metadata(qdess.__GL_AREA_TAG__, None) is not None)
        and (qdess.get_metadata(qdess.__TG_TAG__, None) is not None)
    )
    if not include_required_tags:
        logging.warning(
            "GL and TG tags not present. Skipping T2 computation. "
            "NOTE: These are private tags and may have been removed "
            "in the DICOM anonymization process."
        )
        return {"metrics": {}, "has_depth_dependent": False, "skipped": True}

    emit_progress(20, "Computing T2 map")
    cart = FemoralCartilage()
    t2map = qdess.generate_t2_map(cart, suppress_fat=False, suppress_fluid=False)
    sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation="sagittal")

    # Determine filename prefix
    seg_path = find_file(working_dir, "*_all-labels.nii.gz")
    filename_prefix = seg_path.name.replace("_all-labels.nii.gz", "")

    # Save T2 map
    sitk.WriteImage(sitk_t2map, str(working_dir / f"{filename_prefix}_t2map.nii.gz"), useCompression=False)
    sitk.WriteImage(sitk_t2map, str(working_dir / f"{filename_prefix}_t2map.nrrd"), useCompression=False)

    # Load subregion segmentation for regional T2 statistics
    emit_progress(40, "Computing T2 statistics")
    sitk_seg_subregions = load_subregions(working_dir)
    seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

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

    if bone_meshes_available:
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
                sitk_seg_subregions, depth_segs
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

    else:
        logging.info("Bone meshes not found, skipping depth-dependent T2")

    # Save T2 results
    emit_progress(90, "Saving T2 results")
    if dict_results:
        with open(str(working_dir / f"{filename_prefix}_t2_results.json"), "w") as f:
            json.dump(dict_results, f, indent=4)

    emit_progress(100, "T2 mapping complete")
    return {"metrics": dict_results, "has_depth_dependent": has_depth_dependent}


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
    json.dump(result, sys.stdout)
