"""Tests for steps.subregions.

pymskt is mocked, as it is in test_generate_meshes: the real
``get_knee_segmentation_with_femur_subregions()`` registers the segmentation
against a reference knee, which needs anatomy rather than a 20x20x20 block of
labels. What is under test here is the step contract around that call — what it
writes, what it passes down, and when it declines to run at all.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import SimpleITK as sitk
import pytest

from steps.subregions import FEMUR_SUBREGION_LABELS, run


def _make_segmentation(tmp_path, cartilage=True):
    """Write a canonical-label segmentation. ``cartilage=False`` is a
    bones-only model's output — the case that must skip rather than raise."""
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[5:10, 5:10, 5:10] = 1     # femur_bone
    arr[12:17, 5:10, 5:10] = 2    # tibia_bone
    if cartilage:
        arr[5:10, 12:17, 5:10] = 4    # femur_cart
        arr[12:17, 12:17, 2:8] = 5    # med_tib_cart
        arr[12:17, 12:17, 12:18] = 6  # lat_tib_cart
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((0.5, 0.5, 0.5))
    sitk.WriteImage(sitk_img, str(tmp_path / "test_all-labels.nii.gz"))
    sitk.WriteImage(sitk_img, str(tmp_path / "test_all-labels.nrrd"))
    return arr


def _mock_pymskt(subregion_array):
    """A pymskt whose subregion call returns the given labelled array."""
    mock_mskt = MagicMock()
    sitk_sub = sitk.GetImageFromArray(subregion_array)
    sitk_sub.SetSpacing((0.5, 0.5, 0.5))
    mock_mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions.return_value = (
        sitk_sub
    )
    return mock_mskt


def _subregion_array(base):
    """The segmentation with its femoral cartilage split into labels 11-15."""
    arr = base.copy()
    fem_cart = np.argwhere(arr == 4)
    for i, label in enumerate(FEMUR_SUBREGION_LABELS):
        chunk = fem_cart[i::len(FEMUR_SUBREGION_LABELS)]
        arr[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = label
    return arr


class TestWritesSubregionLabels:
    """The step's whole output: one label image, in two formats."""

    def test_writes_both_formats_with_subregion_labels(self, tmp_path):
        base = _make_segmentation(tmp_path)
        mock_mskt = _mock_pymskt(_subregion_array(base))

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            result = run(tmp_path)

        assert (tmp_path / "test_subregions-labels.nii.gz").exists()
        assert (tmp_path / "test_subregions-labels.nrrd").exists()

        written = sitk.GetArrayFromImage(
            sitk.ReadImage(str(tmp_path / "test_subregions-labels.nii.gz"))
        )
        for label in FEMUR_SUBREGION_LABELS:
            assert np.any(written == label), f"subregion label {label} missing from output"
        assert result["subregion_labels"] == FEMUR_SUBREGION_LABELS
        assert result.get("skipped") is not True

    def test_result_points_at_the_file_it_wrote(self, tmp_path):
        base = _make_segmentation(tmp_path)
        mock_mskt = _mock_pymskt(_subregion_array(base))

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            result = run(tmp_path)

        assert result["subregions_path"] == str(tmp_path / "test_subregions-labels.nii.gz")

    def test_passes_canonical_labels_to_pymskt(self, tmp_path):
        """This step runs after label_remap, so every index it names is canonical."""
        base = _make_segmentation(tmp_path)
        mock_mskt = _mock_pymskt(_subregion_array(base))

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            run(tmp_path)

        kwargs = (
            mock_mskt.image.cartilage_processing
            .get_knee_segmentation_with_femur_subregions.call_args.kwargs
        )
        assert kwargs["fem_cart_label_idx"] == 4   # femur_cart
        assert kwargs["femur_label"] == 1          # femur_bone
        assert kwargs["med_tibia_label"] == 5      # tibia_cart_med
        assert kwargs["lat_tibia_label"] == 6      # tibia_cart_lat
        assert kwargs["tibia_label"] == 2          # tibia_bone
        assert [
            kwargs["ant_femur_mask"], kwargs["med_wb_femur_mask"], kwargs["lat_wb_femur_mask"],
            kwargs["med_post_femur_mask"], kwargs["lat_post_femur_mask"],
        ] == FEMUR_SUBREGION_LABELS


class TestSkipsWithoutCartilage:
    """A bones-only segmentation has nothing to subdivide (D7b).

    Skipping is not a failure: nothing hard-depends on this step, and both
    consumers degrade on their own terms without the file. Raising would fail a
    job over a file neither of them needs.
    """

    def test_returns_skipped_rather_than_raising(self, tmp_path):
        _make_segmentation(tmp_path, cartilage=False)
        mock_mskt = _mock_pymskt(np.zeros((20, 20, 20), dtype=np.uint8))

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            result = run(tmp_path)

        assert result["skipped"] is True
        assert "cartilage" in result["reason"]

    def test_does_not_call_pymskt_or_write_a_file(self, tmp_path):
        _make_segmentation(tmp_path, cartilage=False)
        mock_mskt = _mock_pymskt(np.zeros((20, 20, 20), dtype=np.uint8))

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            run(tmp_path)

        (mock_mskt.image.cartilage_processing
         .get_knee_segmentation_with_femur_subregions.assert_not_called())
        assert list(tmp_path.glob("*_subregions-labels.*")) == []
