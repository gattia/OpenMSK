"""Step 2: Remap segmentation labels to the canonical label set.

Takes a segmentation with model-native labels and remaps to canonical labels.
Backs up the native-label file before overwriting.

If remap_table is None or empty, this step is a no-op.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import emit_progress, find_file, parse_step_args, write_step_result


def run(working_dir, options=None, config=None):
    """Run label remapping step.

    Args:
        working_dir: Directory containing segmentation files.
        options: Dict with required key:
            - remap_table: dict mapping native label ints to canonical label ints.
              Example: {"1": 7, "2": 4, "7": 1, ...}
        config: Pipeline config dict (unused by this step).

    Returns:
        Dict with remapped flag and native_backup path.
    """
    working_dir = Path(working_dir)
    options = options or {}
    remap_table = options.get("remap_table")

    # No-op if no remap table provided
    if not remap_table:
        return {"remapped": False, "native_backup": None}

    emit_progress(0, "Loading segmentation")
    seg_path = find_file(working_dir, "*_all-labels.nii.gz")

    # Backup native labels
    native_backup = seg_path.with_name(seg_path.name.replace(".nii.gz", "-native.nii.gz"))
    shutil.copy2(seg_path, native_backup)

    emit_progress(30, "Remapping labels")
    sitk_seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(sitk_seg)
    remapped = np.zeros_like(arr)
    for src, dst in remap_table.items():
        remapped[arr == int(src)] = int(dst)

    sitk_remapped = sitk.GetImageFromArray(remapped)
    sitk_remapped.CopyInformation(sitk_seg)

    emit_progress(70, "Saving remapped segmentation")
    sitk.WriteImage(sitk_remapped, str(seg_path), useCompression=True)

    # Also update the .nrrd copy if it exists
    nrrd_matches = list(working_dir.glob("*_all-labels.nrrd"))
    if nrrd_matches:
        sitk.WriteImage(sitk_remapped, str(nrrd_matches[0]), useCompression=True)

    emit_progress(100, "Label remapping complete")
    return {"remapped": True, "native_backup": str(native_backup)}


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
