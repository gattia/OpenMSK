"""Tests for steps.run_nsm."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import SimpleITK as sitk
import pytest

from steps.run_nsm import _prepare_meshes, determine_knee_side


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

    def test_only_reads_the_two_tibial_cartilage_labels(self):
        """Why NSM never needed the subregions file (D7b).

        Labels 5 and 6 are canonical and live in *_all-labels.nii.gz; the
        subregions file merely happens to contain them too. A segmentation
        holding nothing else is enough to answer the question.
        """
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[8:12, 8:12, 14:17] = 5
        arr[8:12, 8:12, 3:6] = 6

        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        assert determine_knee_side(arr, sitk_img) == "right"

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


def _write_seg(path, med_k, lat_k, name="scan_all-labels.nii.gz"):
    """A canonical segmentation whose med/lat tibial cartilage sit at the
    given positions along the medial/lateral axis."""
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[8:12, 8:12, med_k:med_k + 3] = 5
    arr[8:12, 8:12, lat_k:lat_k + 3] = 6
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    sitk.WriteImage(sitk_img, str(path / name))


def _mock_pymskt_mesh():
    """pymskt.mesh with a Mesh whose points can be mirrored arithmetically."""
    mock_pymskt = MagicMock()
    mock_mesh = MagicMock()
    mock_mesh.point_coords = np.zeros((10, 3))
    mock_pymskt.mesh.Mesh.return_value = mock_mesh
    return {"pymskt": mock_pymskt, "pymskt.mesh": mock_pymskt.mesh}


class TestPrepareMeshesWithoutSubregions:
    """NSM reads knee side off the canonical segmentation, not the subregions
    file (D7b).

    It only ever needed labels 5 and 6, which are canonical. Loading
    *_subregions-labels.nii.gz for them meant NSM would raise FileNotFoundError
    for a file it does not use on exactly the jobs where the subregions step
    skipped — a bones-only model — which is D7's bug in a third step.
    """

    def test_works_with_no_subregions_file_present(self, tmp_path):
        _write_seg(tmp_path, med_k=14, lat_k=3)
        assert list(tmp_path.glob("*_subregions-labels.*")) == []

        with patch.dict(sys.modules, _mock_pymskt_mesh()):
            side = _prepare_meshes(tmp_path, "femur", config={"clip_femur_top": False})

        assert side == "right"

    def test_mirrors_a_left_knee_with_no_subregions_file(self, tmp_path):
        _write_seg(tmp_path, med_k=3, lat_k=14)

        modules = _mock_pymskt_mesh()
        with patch.dict(sys.modules, modules):
            side = _prepare_meshes(tmp_path, "femur", config={"clip_femur_top": False})
            mesh = modules["pymskt"].mesh.Mesh.return_value

        assert side == "left"
        mesh.save_mesh.assert_called_once_with(str(tmp_path / "femur_mesh_NSM_orig.vtk"))

    def test_the_segmentation_decides_even_when_a_subregions_file_exists(self, tmp_path):
        """Pins which file is read: the subregions copy here disagrees."""
        _write_seg(tmp_path, med_k=14, lat_k=3)
        _write_seg(tmp_path, med_k=3, lat_k=14, name="scan_subregions-labels.nii.gz")

        with patch.dict(sys.modules, _mock_pymskt_mesh()):
            side = _prepare_meshes(tmp_path, "femur", config={"clip_femur_top": False})

        assert side == "right"
