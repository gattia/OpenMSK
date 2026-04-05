"""Tests for steps.segment."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import SimpleITK as sitk
import pytest

from steps.segment import _load_image, _find_input_image, run


class TestFindInputImage:
    """Tests for _find_input_image."""

    def test_finds_nifti(self, tmp_path):
        # Create a NIfTI file
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(sitk_img, str(tmp_path / "scan.nii.gz"))

        result = _find_input_image(tmp_path)
        assert result.name == "scan.nii.gz"

    def test_finds_nrrd(self, tmp_path):
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(sitk_img, str(tmp_path / "scan.nrrd"))

        result = _find_input_image(tmp_path)
        assert result.name == "scan.nrrd"

    def test_ignores_label_files(self, tmp_path):
        """Should not pick up segmentation label files as input."""
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        # Create a label file and a real input
        sitk.WriteImage(sitk_img, str(tmp_path / "scan_all-labels.nii.gz"))
        sitk.WriteImage(sitk_img, str(tmp_path / "scan.nii.gz"))

        result = _find_input_image(tmp_path)
        assert result.name == "scan.nii.gz"

    def test_raises_when_no_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_input_image(tmp_path)

    def test_raises_when_only_label_files(self, tmp_path):
        """Should raise if only label/output files exist."""
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(sitk_img, str(tmp_path / "scan_all-labels.nii.gz"))
        sitk.WriteImage(sitk_img, str(tmp_path / "scan_t2map.nrrd"))

        with pytest.raises(FileNotFoundError):
            _find_input_image(tmp_path)


class TestRun:
    """Tests for the segment run() entry point."""

    def test_run_saves_segmentation_files(self, tmp_path):
        """run() should save both .nii.gz and .nrrd segmentation files."""
        # Create a dummy NIfTI input
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(sitk_img, str(tmp_path / "scan.nii.gz"))

        # Create a mock segmentation result
        seg_arr = np.zeros((5, 5, 5), dtype=np.uint8)
        seg_arr[1:3, 1:3, 1:3] = 7  # femur bone
        mock_seg = sitk.GetImageFromArray(seg_arr)

        mock_volume = MagicMock()
        config = {"default_seg_model": "goyal_sagittal", "batch_size": 4, "models": {}}

        with patch("steps.segment._load_image", return_value=(mock_volume, False, "scan")), \
             patch("steps.segment.segment_image_dosma", return_value=mock_seg):
            result = run(tmp_path, options={"model": "goyal_sagittal"}, config=config)

        assert result["is_qdess"] is False
        assert result["model_name"] == "goyal_sagittal"
        assert Path(result["seg_path"]).exists()

        # Both output files should exist
        assert (tmp_path / "scan_all-labels.nii.gz").exists()
        assert (tmp_path / "scan_all-labels.nrrd").exists()

    def test_run_dispatches_nnunet(self, tmp_path):
        """run() should call nnunet segmentation for nnunet model names."""
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(sitk_img, str(tmp_path / "scan.nii.gz"))

        mock_seg = sitk.GetImageFromArray(np.zeros((5, 5, 5), dtype=np.uint8))
        mock_volume = MagicMock()
        config = {
            "default_seg_model": "nnunet_knee",
            "batch_size": 4,
            "nnunet": {"type": "cascade"},
            "models": {},
        }

        with patch("steps.segment._load_image", return_value=(mock_volume, False, "scan")), \
             patch("steps.segment.segment_image_nnunet", return_value=mock_seg) as mock_nn:
            result = run(tmp_path, options={"model": "nnunet_knee"}, config=config)

        mock_nn.assert_called_once()
        assert result["model_name"] == "nnunet_knee"
