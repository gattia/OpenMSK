"""Tests for steps.run_nsm."""

import numpy as np
import SimpleITK as sitk
import pytest

from steps.run_nsm import determine_knee_side


class TestDetermineKneeSide:
    """Tests for knee side detection with canonical labels."""

    def _make_seg_with_cartilage(self, med_x_idx, lat_x_idx):
        """Create a segmentation with medial and lateral tibial cartilage
        at specified x positions (in ijk space)."""
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        # Place medial tibial cartilage (canonical label 5)
        arr[8:12, 8:12, med_x_idx:med_x_idx+3] = 5
        # Place lateral tibial cartilage (canonical label 6)
        arr[8:12, 8:12, lat_x_idx:lat_x_idx+3] = 6

        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        # Identity direction -- ijk maps directly to xyz
        sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return arr, sitk_img

    def test_right_knee_detection(self):
        """Medial cartilage at higher x -> right knee."""
        # In xyz space with identity direction, higher k index = higher x
        arr, sitk_img = self._make_seg_with_cartilage(med_x_idx=14, lat_x_idx=3)
        side = determine_knee_side(arr, sitk_img)
        assert side == "right"

    def test_left_knee_detection(self):
        """Medial cartilage at lower x -> left knee."""
        arr, sitk_img = self._make_seg_with_cartilage(med_x_idx=3, lat_x_idx=14)
        side = determine_knee_side(arr, sitk_img)
        assert side == "left"

    def test_custom_label_indices(self):
        """Should work with non-default label indices."""
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:12, 14:17] = 3  # custom med label at high x
        arr[8:12, 8:12, 3:6] = 4    # custom lat label at low x

        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        side = determine_knee_side(arr, sitk_img, med_tib_cart_label=3, lat_tib_cart_label=4)
        assert side == "right"

    def test_raises_on_equal_positions(self):
        """Should raise ValueError when med/lat have same x coordinate."""
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        # Both at same x position
        arr[8:12, 8:12, 10:13] = 5
        arr[8:12, 2:5, 10:13] = 6  # different y, same x

        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        with pytest.raises(ValueError, match="Unable to determine knee side"):
            determine_knee_side(arr, sitk_img)
