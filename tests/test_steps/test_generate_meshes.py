"""Tests for steps.generate_meshes.

These tests mock pymskt since it requires real 3D data for mesh generation.
Integration tests with real data belong in Phase 4.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import SimpleITK as sitk
import pytest

from steps.generate_meshes import (
    BONE_CONFIG,
    FEMUR_SUBREGION_LABELS,
    CANONICAL_REGION_NAMES,
)


class TestBoneConfig:
    """Verify the canonical bone configuration is correct."""

    def test_femur_uses_canonical_labels(self):
        assert BONE_CONFIG["femur"]["tissue_idx"] == 1
        assert BONE_CONFIG["femur"]["list_cart_labels"] == [4]

    def test_tibia_uses_canonical_labels(self):
        assert BONE_CONFIG["tibia"]["tissue_idx"] == 2
        assert BONE_CONFIG["tibia"]["list_cart_labels"] == [5, 6]

    def test_patella_uses_canonical_labels(self):
        assert BONE_CONFIG["patella"]["tissue_idx"] == 3
        assert BONE_CONFIG["patella"]["list_cart_labels"] == [7]

    def test_patella_no_crop(self):
        assert BONE_CONFIG["patella"]["crop_percent"] is None

    def test_femur_subregion_labels(self):
        assert FEMUR_SUBREGION_LABELS == [11, 12, 13, 14, 15]


class TestCanonicalRegionNames:
    """Verify all expected regions have names."""

    def test_all_cart_labels_have_names(self):
        for bone_config in BONE_CONFIG.values():
            for label in bone_config["list_cart_labels"]:
                assert label in CANONICAL_REGION_NAMES

    def test_all_subregion_labels_have_names(self):
        for label in FEMUR_SUBREGION_LABELS:
            assert label in CANONICAL_REGION_NAMES


class TestRunMocked:
    """Test run() with mocked pymskt calls."""

    def _make_segmentation(self, tmp_path, canonical=True):
        """Create a synthetic canonical-label segmentation."""
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        if canonical:
            arr[5:10, 5:10, 5:10] = 1   # femur_bone
            arr[12:17, 5:10, 5:10] = 2  # tibia_bone
            arr[5:10, 12:17, 5:10] = 4  # femur_cart
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((0.5, 0.5, 0.5))
        sitk.WriteImage(sitk_img, str(tmp_path / "test_all-labels.nii.gz"))
        sitk.WriteImage(sitk_img, str(tmp_path / "test_all-labels.nrrd"))
        return arr

    def _setup_mock_mskt(self):
        """Create a mock pymskt module with common setup."""
        mock_mskt = MagicMock()

        mock_seg_sub = sitk.GetImageFromArray(np.zeros((20, 20, 20), dtype=np.uint8))
        mock_seg_sub.SetSpacing((0.5, 0.5, 0.5))
        mock_mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions.return_value = mock_seg_sub

        mock_bone_mesh = MagicMock()
        mock_bone_mesh.get_scalar.return_value = np.zeros(100)
        mock_bone_mesh.list_cartilage_meshes = [MagicMock()]
        mock_bone_mesh.copy.return_value = MagicMock()
        mock_mskt.mesh.BoneMesh.return_value = mock_bone_mesh

        return mock_mskt, mock_bone_mesh

    def test_run_saves_subregion_files(self, tmp_path):
        """run() should save subregion segmentation files."""
        self._make_segmentation(tmp_path)
        mock_mskt, _ = self._setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        assert (tmp_path / "test_subregions-labels.nrrd").exists()
        assert (tmp_path / "test_subregions-labels.nii.gz").exists()
        assert result["bones_processed"] == ["femur", "tibia", "patella"]

    def test_run_saves_raw_femur_mesh(self, tmp_path):
        """run() should save the raw femur mesh for NSM fitting."""
        self._make_segmentation(tmp_path)
        mock_mskt, mock_bone_mesh = self._setup_mock_mskt()
        mock_raw_copy = MagicMock()
        mock_bone_mesh.copy.return_value = mock_raw_copy

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        mock_raw_copy.save_mesh.assert_called_once_with(
            str(tmp_path / "femur_mesh_raw.vtk")
        )

    def test_run_skips_thickness_when_disabled(self, tmp_path):
        """run() should skip thickness computation when disabled."""
        self._make_segmentation(tmp_path)
        mock_mskt, mock_bone_mesh = self._setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        mock_bone_mesh.calc_cartilage_thickness.assert_not_called()
        assert result["thickness_computed"] is False
        assert result["metrics"] == {}
