"""Tests for steps.label_remap."""

import numpy as np
import SimpleITK as sitk

from steps.label_remap import run


DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}


def test_remap_dosma_to_canonical(synthetic_native_segmentation):
    """DOSMA-native labels should be remapped to canonical labels."""
    working_dir = synthetic_native_segmentation

    result = run(working_dir, options={"remap_table": DOSMA_REMAP})

    assert result["remapped"] is True
    assert result["native_backup"] is not None

    # Read remapped segmentation
    sitk_seg = sitk.ReadImage(str(working_dir / "test_all-labels.nii.gz"))
    arr = sitk.GetArrayFromImage(sitk_seg)

    # Verify canonical labels are present with correct mapping
    unique = set(np.unique(arr))
    assert 1 in unique   # femur_bone (was DOSMA 7)
    assert 2 in unique   # tibia_bone (was DOSMA 8)
    assert 4 in unique   # femur_cart (was DOSMA 2)
    assert 5 in unique   # tibia_cart_med (was DOSMA 3)

    # Verify old DOSMA-only labels are gone (7, 8 had no canonical equivalent
    # that maps back to them; 3 maps to 5 so should not appear as 3)
    assert 7 not in unique  # DOSMA femur_bone -> canonical 1
    assert 8 not in unique  # DOSMA tibia_bone -> canonical 2
    assert 3 not in unique  # DOSMA med_tib_cart -> canonical 5


def test_remap_preserves_native_backup(synthetic_native_segmentation):
    """Native labels should be backed up before remapping."""
    working_dir = synthetic_native_segmentation

    result = run(working_dir, options={"remap_table": DOSMA_REMAP})

    # Read the backup
    backup_path = working_dir / "test_all-labels-native.nii.gz"
    assert backup_path.exists()

    sitk_backup = sitk.ReadImage(str(backup_path))
    arr_backup = sitk.GetArrayFromImage(sitk_backup)

    # Backup should still have DOSMA-native labels
    unique = set(np.unique(arr_backup))
    assert 7 in unique  # DOSMA femur_bone
    assert 8 in unique  # DOSMA tibia_bone


def test_remap_updates_nrrd_copy(synthetic_native_segmentation):
    """The .nrrd copy should also be remapped."""
    working_dir = synthetic_native_segmentation

    run(working_dir, options={"remap_table": DOSMA_REMAP})

    sitk_nrrd = sitk.ReadImage(str(working_dir / "test_all-labels.nrrd"))
    arr_nrrd = sitk.GetArrayFromImage(sitk_nrrd)

    unique = set(np.unique(arr_nrrd))
    assert 1 in unique   # canonical femur_bone
    assert 2 in unique   # canonical tibia_bone


def test_remap_noop_when_no_table(synthetic_native_segmentation):
    """No remapping should happen if remap_table is None or empty."""
    working_dir = synthetic_native_segmentation

    result = run(working_dir, options={})
    assert result["remapped"] is False

    result = run(working_dir, options={"remap_table": None})
    assert result["remapped"] is False


def test_remap_preserves_image_metadata(synthetic_native_segmentation):
    """Remapped image should preserve spacing, origin, direction."""
    working_dir = synthetic_native_segmentation

    sitk_before = sitk.ReadImage(str(working_dir / "test_all-labels.nii.gz"))
    spacing_before = sitk_before.GetSpacing()
    origin_before = sitk_before.GetOrigin()
    direction_before = sitk_before.GetDirection()

    run(working_dir, options={"remap_table": DOSMA_REMAP})

    sitk_after = sitk.ReadImage(str(working_dir / "test_all-labels.nii.gz"))
    assert sitk_after.GetSpacing() == spacing_before
    assert sitk_after.GetOrigin() == origin_before
    assert sitk_after.GetDirection() == direction_before


def test_remap_unmapped_labels_become_zero(synthetic_native_segmentation):
    """Labels not in the remap table should become 0 (background)."""
    working_dir = synthetic_native_segmentation

    # Remap only femur bone (7->1), leave others unmapped
    partial_remap = {7: 1}
    run(working_dir, options={"remap_table": partial_remap})

    sitk_seg = sitk.ReadImage(str(working_dir / "test_all-labels.nii.gz"))
    arr = sitk.GetArrayFromImage(sitk_seg)

    unique = set(np.unique(arr))
    # Only 0 (background) and 1 (remapped femur) should remain
    assert unique == {0, 1}
