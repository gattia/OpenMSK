"""Tests for steps.label_remap."""

import numpy as np
import SimpleITK as sitk

from steps.label_remap import run


# The complete DOSMA-native -> canonical table, including the menisci (5, 6).
DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 1, 8: 2, 9: 3}

NATIVE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

SEG_NAME = "test_all-labels.nii.gz"
BACKUP_NAME = "test_all-labels-native.nii.gz"


def _write_native_seg(working_dir, labels):
    """Write a synthetic native-label segmentation containing every label given.

    One cube per label, so each requested label is genuinely present in the
    volume. Mirrors the conftest fixtures, but lets a test choose its labels.
    """
    working_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 4 * len(labels)), dtype=np.uint8)
    for i, label in enumerate(labels):
        arr[1:3, 1:3, 4 * i + 1 : 4 * i + 3] = label
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(sitk_img, str(working_dir / SEG_NAME))
    sitk.WriteImage(sitk_img, str(working_dir / "test_all-labels.nrrd"))
    return working_dir


def _read(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


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


def test_remap_skipped_when_no_table(synthetic_native_segmentation):
    """No remap table means the step declines, and touches nothing."""
    working_dir = synthetic_native_segmentation
    before = _read(working_dir / SEG_NAME)

    for options in ({}, {"remap_table": None}, {"remap_table": {}}):
        result = run(working_dir, options=options)
        assert result["skipped"] is True
        assert result["reason"]
        assert "remapped" not in result

    assert np.array_equal(before, _read(working_dir / SEG_NAME))
    assert not (working_dir / BACKUP_NAME).exists()


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
    result = run(working_dir, options={"remap_table": partial_remap})

    sitk_seg = sitk.ReadImage(str(working_dir / "test_all-labels.nii.gz"))
    arr = sitk.GetArrayFromImage(sitk_seg)

    unique = set(np.unique(arr))
    # Only 0 (background) and 1 (remapped femur) should remain
    assert unique == {0, 1}

    # And the step says which native labels it dropped on the floor.
    assert result["unmapped_labels"] == [2, 3, 8]


def test_remap_twice_leaves_segmentation_identical(synthetic_native_segmentation):
    """A second run must decline, not remap the canonical labels again.

    Without the guard the second pass applies the DOSMA table to canonical
    labels -- femur 1 -> 7, every femur voxel becomes patellar cartilage -- and
    still exits 0 with a plausible-looking segmentation.
    """
    working_dir = synthetic_native_segmentation

    first = run(working_dir, options={"remap_table": DOSMA_REMAP})
    after_first = _read(working_dir / SEG_NAME)
    nrrd_after_first = _read(working_dir / "test_all-labels.nrrd")

    second = run(working_dir, options={"remap_table": DOSMA_REMAP})
    after_second = _read(working_dir / SEG_NAME)

    assert first["remapped"] is True
    assert second["skipped"] is True
    assert second["reason"] == "already remapped"
    assert "remapped" not in second

    assert np.array_equal(after_first, after_second)
    assert np.array_equal(nrrd_after_first, _read(working_dir / "test_all-labels.nrrd"))
    # The specific corruption: canonical femur (1) sent to patellar cart (7).
    assert 7 not in set(np.unique(after_second))
    assert (after_second == 1).sum() == (after_first == 1).sum()


def test_remap_twice_does_not_overwrite_native_backup(synthetic_native_segmentation):
    """The backup must keep the ORIGINAL native labels, not the canonical ones."""
    working_dir = synthetic_native_segmentation
    native_before = _read(working_dir / SEG_NAME)

    run(working_dir, options={"remap_table": DOSMA_REMAP})
    backup_after_first = _read(working_dir / BACKUP_NAME)

    run(working_dir, options={"remap_table": DOSMA_REMAP})
    backup_after_second = _read(working_dir / BACKUP_NAME)

    assert np.array_equal(native_before, backup_after_first)
    assert np.array_equal(backup_after_first, backup_after_second)
    # DOSMA-native bone labels, not canonical 1/2.
    assert {7, 8}.issubset(set(np.unique(backup_after_second)))


def test_remap_reports_labels_present(synthetic_native_segmentation):
    """labels_present should be the canonical labels actually written."""
    working_dir = synthetic_native_segmentation

    result = run(working_dir, options={"remap_table": DOSMA_REMAP})

    # Fixture holds native 7, 8, 2, 3 -> canonical 1, 2, 4, 5 (plus background).
    assert result["labels_present"] == [0, 1, 2, 4, 5]
    assert set(result["labels_present"]) == set(
        int(v) for v in np.unique(_read(working_dir / SEG_NAME))
    )


def test_remap_unmapped_labels_empty_for_complete_table(tmp_path):
    """A table covering all nine native labels leaves nothing unmapped."""
    working_dir = _write_native_seg(tmp_path / "job", NATIVE_LABELS)

    result = run(working_dir, options={"remap_table": DOSMA_REMAP})

    assert result["unmapped_labels"] == []
    assert result["labels_present"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_remap_reports_label_missing_from_table_and_deletes_it(tmp_path):
    """A native label with no table entry is reported -- and really is gone."""
    working_dir = _write_native_seg(tmp_path / "job", NATIVE_LABELS)
    # Drop the medial meniscus (native 5 -> canonical 8), the entry that was
    # missing from the real table until 2026-08-15.
    short_table = {k: v for k, v in DOSMA_REMAP.items() if k != 5}

    result = run(working_dir, options={"remap_table": short_table})

    assert result["unmapped_labels"] == [5]
    arr = _read(working_dir / SEG_NAME)
    # Not passed through unremapped, not left as 5 -- deleted.
    assert 8 not in set(np.unique(arr))
    assert result["labels_present"] == [0, 1, 2, 3, 4, 5, 6, 7, 9]
    native = _read(working_dir / BACKUP_NAME)
    assert (native == 5).any()
    assert (arr[native == 5] == 0).all()


def test_remap_string_and_int_keyed_tables_behave_identically(tmp_path):
    """remap_table arrives as JSON, so its keys may be strings."""
    int_dir = _write_native_seg(tmp_path / "int_keys", NATIVE_LABELS)
    str_dir = _write_native_seg(tmp_path / "str_keys", NATIVE_LABELS)

    int_result = run(int_dir, options={"remap_table": DOSMA_REMAP})
    str_result = run(
        str_dir, options={"remap_table": {str(k): v for k, v in DOSMA_REMAP.items()}}
    )

    for key in ("remapped", "labels_present", "unmapped_labels"):
        assert int_result[key] == str_result[key]
    assert np.array_equal(_read(int_dir / SEG_NAME), _read(str_dir / SEG_NAME))


def test_remap_unmapped_labels_correct_for_string_keyed_table(tmp_path):
    """String keys must not make every native label look unmapped."""
    working_dir = _write_native_seg(tmp_path / "job", NATIVE_LABELS)
    short_table = {str(k): v for k, v in DOSMA_REMAP.items() if k != 5}

    result = run(working_dir, options={"remap_table": short_table})

    assert result["unmapped_labels"] == [5]
    assert result["labels_present"] == [0, 1, 2, 3, 4, 5, 6, 7, 9]
