"""Step 2: Remap segmentation labels to the canonical label set.

Takes a segmentation with model-native labels and remaps to canonical labels.
Backs up the native-label file before overwriting.

It reads the `.nrrd` and writes both formats from it. The geometry of the file
it reads is the geometry it stamps on everything it writes, and NIfTI's affine
is float32 (see `_common`), so reading the `.nii.gz` here would quantise the
`.nrrd` as well -- and the `.nrrd` is what every later step reads.

The step is idempotent: the `*-native.nii.gz` backup is the marker that it has
already run, and a second call declines rather than remapping canonical labels a
second time (which would send femur 1 -> 7, i.e. into patellar cartilage, and
overwrite the backup that proves it happened).

Declining to work -- no remap table, or already remapped -- is reported as
`{"skipped": True, "reason": ...}`, the shared step-result convention. The
orchestrator only plans this step for models that have a label map, so a skip
there means something went wrong and it fails the job.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import (
    LOSSLESS_EXT,
    NIFTI_EXT,
    SEG_STEM,
    emit_progress,
    find_segmentation,
    image_prefix,
    parse_step_args,
    write_step_result,
)

# Appended to the prefix to name the pre-remap copy of the segmentation.
NATIVE_SUFFIX = "-native"


def run(working_dir, options=None, config=None):
    """Run label remapping step.

    Args:
        working_dir: Directory containing segmentation files.
        options: Dict with required key:
            - remap_table: dict mapping native label ints to canonical label ints.
              Example: {"1": 7, "2": 4, "7": 1, ...}
        config: Pipeline config dict (unused by this step).

    Returns:
        On success, a dict with `remapped: True`, the `native_backup` path, the
        `labels_present` in the written file, and the `unmapped_labels` -- native
        labels with no entry in the table, which this step therefore deleted from
        the segmentation. Otherwise `{"skipped": True, "reason": ...}`.
    """
    working_dir = Path(working_dir)
    options = options or {}
    remap_table = options.get("remap_table")

    if not remap_table:
        return {"skipped": True, "reason": "no remap table supplied"}

    emit_progress(0, "Loading segmentation")
    # Read the lossless copy when there is one. This step copies the geometry of
    # whatever it reads onto BOTH files it writes, so reading the .nii.gz would
    # stamp float32-quantised direction cosines onto the .nrrd too -- and every
    # later step reads the .nrrd.
    seg_path = find_segmentation(working_dir)

    # The two files this step maintains, named from the prefix rather than from
    # whichever one was read, so the pair stays the pair.
    prefix = image_prefix(seg_path)
    nifti_path = working_dir / f"{prefix}{SEG_STEM}{NIFTI_EXT}"
    nrrd_path = working_dir / f"{prefix}{SEG_STEM}{LOSSLESS_EXT}"

    # The backup keeps its documented `*_all-labels-native.nii.gz` name: it is a
    # byte copy of the .nii.gz the segmentation step wrote, and downstream
    # consumers (and the website's file listing) know it by that name. Both
    # possible names are checked below, because a directory that somehow has no
    # .nii.gz gets its backup as .nrrd instead -- the guard must not be
    # bypassable by which format happened to be present.
    native_backup = working_dir / f"{prefix}{SEG_STEM}{NATIVE_SUFFIX}{NIFTI_EXT}"
    native_backup_nrrd = working_dir / f"{prefix}{SEG_STEM}{NATIVE_SUFFIX}{LOSSLESS_EXT}"

    # The backup's existence is the marker that this step already ran. Remapping
    # again would apply the table to canonical labels and destroy both the
    # segmentation and the backup, so decline instead. Reachable as soon as a
    # task can be redelivered to a worker.
    if native_backup.exists() or native_backup_nrrd.exists():
        emit_progress(100, "Segmentation already remapped")
        return {"skipped": True, "reason": "already remapped"}

    emit_progress(30, "Remapping labels")
    sitk_seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(sitk_seg)

    # remap_table comes in as JSON, so its keys may be str or int depending on
    # the caller. Normalise before comparing against label values.
    table_keys = {int(src) for src in remap_table}

    remapped = np.zeros_like(arr)
    for src, dst in remap_table.items():
        remapped[arr == int(src)] = int(dst)

    sitk_remapped = sitk.GetImageFromArray(remapped)
    sitk_remapped.CopyInformation(sitk_seg)

    emit_progress(70, "Saving remapped segmentation")
    # Back up the native labels immediately before they are overwritten, so the
    # backup never claims a remap that did not finish. Copy the .nii.gz -- what
    # the segmentation step wrote and what the backup has always been -- unless
    # there isn't one, in which case the file that was read is what there is to
    # preserve.
    if nifti_path.exists():
        backup_path = native_backup
        shutil.copy2(nifti_path, backup_path)
    else:
        backup_path = native_backup_nrrd
        shutil.copy2(seg_path, backup_path)

    # Both formats stay in step: the .nii.gz because users download it, the .nrrd
    # because the next step reads it. Only files that already exist are rewritten
    # -- creating a .nrrd out of a .nii.gz-only directory would manufacture a
    # lossless-looking file from quantised geometry and silence the fallback
    # warning that says so.
    for path in (nifti_path, nrrd_path):
        if path.exists():
            sitk.WriteImage(sitk_remapped, str(path), useCompression=True)

    emit_progress(100, "Label remapping complete")
    # Report what was actually produced. A native label missing from the table is
    # not passed through -- it is deleted -- and this step is the only thing
    # positioned to notice, so it says so rather than leaving it to be inferred.
    return {
        "remapped": True,
        "native_backup": str(backup_path),
        "labels_present": sorted(int(v) for v in np.unique(remapped)),
        "unmapped_labels": sorted(
            int(v) for v in np.unique(arr) if v != 0 and int(v) not in table_keys
        ),
    }


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
