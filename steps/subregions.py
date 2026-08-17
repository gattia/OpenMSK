"""Step 3a: Femur cartilage subregion labelling.

Takes the canonical-label segmentation and splits the femoral cartilage into
five subregions (anterior, medial/lateral weight-bearing, medial/lateral
posterior), written alongside the canonical labels as
``*_subregions-labels.{nii.gz,nrrd}``.

Split out of ``generate_meshes.py`` (D7b). It is pure image processing on the
canonical labels and never needed a mesh -- it lived in the mesh step only
because that is where the code happened to be, which coupled regional T2 to
mesh generation and made "T2 without meshes" a crash rather than a degraded
run.

Nothing hard-depends on this step; it *enriches* two others, and each degrades
on its own terms when the file is absent:

- ``generate_meshes`` -- regional (subregion) cartilage thickness; falls back
  to whole-region cartilage labels.
- ``t2_mapping`` -- regional T2 for labels 11-15; falls back to the
  whole-region cartilage labels 4-7 in the segmentation itself.

``run_nsm`` used to read this file too, but only ever needed labels 5/6, which
are canonical; it now loads the segmentation directly.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import (
    emit_progress,
    find_segmentation,
    image_prefix,
    load_segmentation,
    parse_step_args,
    write_step_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Canonical labels this step reads.
FEM_CART_LABEL = 4        # CANONICAL femur_cart -- what gets subdivided
FEMUR_LABEL = 1           # CANONICAL femur_bone
MED_TIB_CART_LABEL = 5    # CANONICAL tibia_cart_med
LAT_TIB_CART_LABEL = 6    # CANONICAL tibia_cart_lat
TIBIA_LABEL = 2           # CANONICAL tibia_bone

# Subregion labels this step writes. NOT canonical labels -- they are pymskt
# subregion indices, and they exist only in the file this step produces.
# generate_meshes and t2_mapping both key off these, so they are defined here,
# where they are created.
FEMUR_SUBREGION_LABELS = [11, 12, 13, 14, 15]
ANT_FEMUR_MASK, MED_WB_FEMUR_MASK, LAT_WB_FEMUR_MASK, MED_POST_FEMUR_MASK, LAT_POST_FEMUR_MASK = (
    FEMUR_SUBREGION_LABELS
)

# Size of the weight-bearing region, as a fraction of the distance from the
# notch to the posterior of the condyles.
WB_REGION_PERCENT_DIST = 0.6

# Medial/lateral axis of a sagittal knee acquisition, in numpy array ordering.
ML_AXIS = 0


def run(working_dir, options=None, config=None):
    """Run femur cartilage subregion labelling.

    Args:
        working_dir: Directory containing the canonical-label segmentation.
        options: Dict (no options -- the step takes the canonical segmentation
            and needs nothing else).
        config: Pipeline config dict (unused -- labels are canonical).

    Returns:
        Dict with the written ``subregions_path`` and the subregion labels
        actually produced. On a segmentation with no femoral cartilage,
        ``{"skipped": True, "reason": ...}`` instead: subdividing cartilage
        that is not there is meaningless, and a bones-only segmentation model
        is a real case rather than a failure.
    """
    import pymskt as mskt

    working_dir = Path(working_dir)

    emit_progress(0, "Loading segmentation")
    sitk_seg = load_segmentation(working_dir)

    filename_prefix = image_prefix(find_segmentation(working_dir))

    # A bones-only model produces no cartilage at all, and
    # get_knee_segmentation_with_femur_subregions() has nothing to subdivide.
    # Declining is the honest answer; raising would fail a job over a file its
    # consumers each know how to do without.
    seg_array = sitk.GetArrayFromImage(sitk_seg)
    if not np.any(seg_array == FEM_CART_LABEL):
        logging.info(
            "No femoral cartilage (label %d) in the segmentation. Skipping subregion labelling.",
            FEM_CART_LABEL,
        )
        emit_progress(100, "No cartilage to subdivide")
        return {"skipped": True, "reason": "no cartilage labels in segmentation"}

    emit_progress(20, "Computing femur subregions")
    sitk_seg_subregions = mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions(
        sitk_seg,
        fem_cart_label_idx=FEM_CART_LABEL,
        wb_region_percent_dist=WB_REGION_PERCENT_DIST,
        femur_label=FEMUR_LABEL,
        med_tibia_label=MED_TIB_CART_LABEL,
        lat_tibia_label=LAT_TIB_CART_LABEL,
        ant_femur_mask=ANT_FEMUR_MASK,
        med_wb_femur_mask=MED_WB_FEMUR_MASK,
        lat_wb_femur_mask=LAT_WB_FEMUR_MASK,
        med_post_femur_mask=MED_POST_FEMUR_MASK,
        lat_post_femur_mask=LAT_POST_FEMUR_MASK,
        verify_med_lat_tib_cart=True,
        tibia_label=TIBIA_LABEL,
        ml_axis=ML_AXIS,
    )

    emit_progress(80, "Saving subregion segmentation")
    nrrd_path = working_dir / f"{filename_prefix}_subregions-labels.nrrd"
    nifti_path = working_dir / f"{filename_prefix}_subregions-labels.nii.gz"
    sitk.WriteImage(sitk_seg_subregions, str(nrrd_path), useCompression=True)
    sitk.WriteImage(sitk_seg_subregions, str(nifti_path), useCompression=True)

    subregion_array = sitk.GetArrayFromImage(sitk_seg_subregions)
    labels_written = sorted(
        int(label) for label in FEMUR_SUBREGION_LABELS if np.any(subregion_array == label)
    )

    emit_progress(100, "Subregion labelling complete")
    return {
        "subregions_path": str(nifti_path),
        "subregion_labels": labels_written,
    }


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
