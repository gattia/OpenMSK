"""Shared fixtures for step tests."""

import numpy as np
import SimpleITK as sitk
import pytest


@pytest.fixture
def synthetic_segmentation(tmp_path):
    """Create a small synthetic segmentation with canonical labels."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1   # femur_bone (canonical)
    arr[6:8, 2:5, 2:5] = 2   # tibia_bone (canonical)
    arr[2:5, 6:8, 2:5] = 4   # femur_cart (canonical)
    arr[6:8, 6:8, 2:5] = 5   # tibia_cart_med (canonical)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    path = tmp_path / "test_all-labels.nii.gz"
    sitk.WriteImage(sitk_img, str(path))
    # Also write .nrrd copy
    nrrd_path = tmp_path / "test_all-labels.nrrd"
    sitk.WriteImage(sitk_img, str(nrrd_path))
    return tmp_path


@pytest.fixture
def synthetic_native_segmentation(tmp_path):
    """Create a synthetic segmentation with DOSMA-native labels (pre-remap)."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 7   # DOSMA femur_bone
    arr[6:8, 2:5, 2:5] = 8   # DOSMA tibia_bone
    arr[2:5, 6:8, 2:5] = 2   # DOSMA femur_cart
    arr[6:8, 6:8, 2:5] = 3   # DOSMA med_tib_cart
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    path = tmp_path / "test_all-labels.nii.gz"
    sitk.WriteImage(sitk_img, str(path))
    nrrd_path = tmp_path / "test_all-labels.nrrd"
    sitk.WriteImage(sitk_img, str(nrrd_path))
    return tmp_path
