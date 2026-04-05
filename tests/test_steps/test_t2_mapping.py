"""Tests for steps.t2_mapping.

These tests mock dosma and pymskt since they require real DICOM data.
Integration tests with real data belong in Phase 4.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import SimpleITK as sitk
import pytest

from steps.t2_mapping import (
    REGION_NAMES,
    T2_MIN_VALID,
    T2_MAX_VALID,
    DEEP_OFFSET,
    SUPERFICIAL_OFFSET,
    _find_dicom_dir,
)


class TestConstants:
    """Verify T2 analysis constants."""

    def test_t2_range(self):
        assert T2_MIN_VALID == 0
        assert T2_MAX_VALID == 80

    def test_depth_offsets(self):
        assert DEEP_OFFSET == 100
        assert SUPERFICIAL_OFFSET == 200

    def test_region_names_use_canonical_labels(self):
        assert 4 in REGION_NAMES   # fem_cart
        assert 5 in REGION_NAMES   # med_tib_cart
        assert 6 in REGION_NAMES   # lat_tib_cart
        assert 7 in REGION_NAMES   # pat_cart
        assert 11 in REGION_NAMES  # ant_fem_cart


class TestFindDicomDir:
    """Tests for _find_dicom_dir."""

    def test_finds_dicom_subdir(self, tmp_path):
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        (dicom_dir / "0001.dcm").touch()

        result = _find_dicom_dir(tmp_path)
        assert result == dicom_dir

    def test_ignores_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "file.dcm").touch()

        real_dir = tmp_path / "dicoms"
        real_dir.mkdir()
        (real_dir / "0001.dcm").touch()

        result = _find_dicom_dir(tmp_path)
        assert result == real_dir

    def test_raises_when_no_dicom_dir(self, tmp_path):
        # Only files, no subdirectories
        (tmp_path / "scan.nii.gz").touch()

        with pytest.raises(FileNotFoundError):
            _find_dicom_dir(tmp_path)
