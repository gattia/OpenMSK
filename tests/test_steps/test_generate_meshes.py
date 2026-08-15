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


def _make_segmentation(tmp_path, canonical=True):
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


def _write_subregions(tmp_path):
    """Write the file the subregions step produces, with labels 11-15."""
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[5:10, 5:10, 5:10] = 1
    arr[12:17, 5:10, 5:10] = 2
    for i, label in enumerate(FEMUR_SUBREGION_LABELS):
        arr[5 + i, 12:17, 5:10] = label
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((0.5, 0.5, 0.5))
    sitk.WriteImage(sitk_img, str(tmp_path / "test_subregions-labels.nii.gz"))
    return arr


def _setup_mock_mskt():
    """Create a mock pymskt module with common setup."""
    mock_mskt = MagicMock()

    mock_bone_mesh = MagicMock()
    # Every region this step can ask about is represented, so a thickness
    # metric that comes back NaN means the step looked for a label that is
    # not in the image it chose.
    labels = np.array(list(CANONICAL_REGION_NAMES) * 10)
    thickness = np.ones(labels.size)
    mock_bone_mesh.get_scalar.side_effect = (
        lambda name: thickness if name == "thickness (mm)" else labels
    )
    mock_bone_mesh.list_cartilage_meshes = [MagicMock()]
    mock_bone_mesh.copy.return_value = MagicMock()
    mock_mskt.mesh.BoneMesh.return_value = mock_bone_mesh

    return mock_mskt, mock_bone_mesh


class TestRunMocked:
    """Test run() with mocked pymskt calls."""

    def test_run_does_not_compute_subregions(self, tmp_path):
        """The subregion labels are the subregions step's job now (D7b)."""
        _make_segmentation(tmp_path)
        mock_mskt, _ = _setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        (mock_mskt.image.cartilage_processing
         .get_knee_segmentation_with_femur_subregions.assert_not_called())
        assert list(tmp_path.glob("*_subregions-labels.*")) == []
        assert result["bones_processed"] == ["femur", "tibia", "patella"]

    def test_run_saves_raw_femur_mesh(self, tmp_path):
        """run() should save the raw femur mesh for NSM fitting."""
        _make_segmentation(tmp_path)
        mock_mskt, mock_bone_mesh = _setup_mock_mskt()
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
        _make_segmentation(tmp_path)
        mock_mskt, mock_bone_mesh = _setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        mock_bone_mesh.calc_cartilage_thickness.assert_not_called()
        assert result["thickness_computed"] is False
        assert result["metrics"] == {}


class TestWithoutSubregions:
    """The subregions file is loaded lazily, and its absence is survivable.

    D7b moved the subregion labelling into its own step, which can skip itself
    (a bones-only model) or fail. Neither may cost the meshes: they are what
    NSM and BScore are built from.
    """

    def test_bones_only_job_needs_no_subregions_at_all(self, tmp_path):
        """Thickness off + no subregions file: meshes still get made."""
        _make_segmentation(tmp_path)
        mock_mskt, mock_bone_mesh = _setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": False})

        saved = [call.args[0] for call in mock_bone_mesh.save_mesh.call_args_list]
        assert saved == [
            str(tmp_path / "femur_mesh.vtk"),
            str(tmp_path / "tibia_mesh.vtk"),
            str(tmp_path / "patella_mesh.vtk"),
        ]
        assert result["has_subregions"] is False

    def test_thickness_falls_back_to_whole_femoral_cartilage(self, tmp_path):
        """With thickness on but no subregions, femoral thickness is reported
        for label 4 rather than for the five subregions — the columns degrade,
        the step does not."""
        _make_segmentation(tmp_path)
        mock_mskt, _ = _setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": True})

        assert result["has_subregions"] is False
        assert result["metrics"]["fem_cart_mm_mean"] == pytest.approx(1.0)
        assert not any(key.startswith("ant_fem_cart") for key in result["metrics"])
        # The tibia and patella never used the subregion labels, so they are
        # unaffected either way.
        assert result["metrics"]["med_tib_cart_mm_mean"] == pytest.approx(1.0)
        assert result["metrics"]["pat_cart_mm_mean"] == pytest.approx(1.0)

    def test_subregions_are_used_when_present(self, tmp_path):
        """The file is loaded, not computed, and drives femoral regions."""
        _make_segmentation(tmp_path)
        _write_subregions(tmp_path)
        mock_mskt, mock_bone_mesh = _setup_mock_mskt()

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            from importlib import reload
            import steps.generate_meshes
            reload(steps.generate_meshes)
            result = steps.generate_meshes.run(tmp_path, options={"compute_thickness": True})

        (mock_mskt.image.cartilage_processing
         .get_knee_segmentation_with_femur_subregions.assert_not_called())
        assert result["has_subregions"] is True
        assert mock_bone_mesh.list_cartilage_labels == FEMUR_SUBREGION_LABELS
        for label in FEMUR_SUBREGION_LABELS:
            key = f"{CANONICAL_REGION_NAMES[label]}_mm_mean"
            assert result["metrics"][key] == pytest.approx(1.0)
        assert "fem_cart_mm_mean" not in result["metrics"]
