"""Diagnostic tests for TensorFlow/PyTorch CUDA conflict.

These tests isolate whether importing TF and torch in the same process
corrupts MedicalVolume data or nnU-Net inference. Run with:

    conda run -n kneepipeline python -m pytest tests/integration/test_tf_torch_conflict.py -v -s
"""

import os
import sys
import json
import subprocess
import numpy as np
import SimpleITK as sitk
import pytest

# Test data — skip if not available
TEST_NRRD = "data/anthonys_knee.nrrd"


def _have_test_data():
    return os.path.exists(TEST_NRRD)


def _have_config():
    return os.path.exists("config.json")


def _load_volume_directly():
    """Load the volume the plain way: sitk.ReadImage -> MedicalVolume.from_sitk.

    Deliberately does NOT go through steps.segment._load_image, so it can be
    imported without pulling in dosma (and therefore TensorFlow). That is the
    whole point of the comparisons below: this is the control, and _load_image
    is the thing under test.
    """
    from dosma import MedicalVolume
    image = sitk.ReadImage(TEST_NRRD)
    return MedicalVolume.from_sitk(image)


def _volume_to_array(volume):
    """Convert dosma MedicalVolume to numpy array via sagittal sitk."""
    sitk_img = volume.to_sitk(image_orientation="sagittal")
    return sitk.GetArrayFromImage(sitk_img)


def _is_tf_loaded():
    """Check if TensorFlow has been imported into this process."""
    return "tensorflow" in sys.modules or "tensorflow.python" in sys.modules


# ---------------------------------------------------------------------------
# Test 1: Is the data correct when loaded without any ML framework?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_test_data(), reason="Test data not available")
class TestBaselineLoading:
    """Verify data loads correctly without ML frameworks interfering."""

    def test_sitk_read_has_data(self):
        """SimpleITK should read non-zero data from the NRRD."""
        img = sitk.ReadImage(TEST_NRRD)
        arr = sitk.GetArrayFromImage(img)
        assert arr.max() > 0, f"NRRD data is all zeros (shape {arr.shape})"

    def test_dosma_volume_has_data(self):
        """MedicalVolume.from_sitk should preserve non-zero data."""
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, f"MedicalVolume data is all zeros after to_sitk"

    def test_modular_load_image_has_data(self):
        """steps.segment._load_image should return non-zero data for NRRD.

        NOTE: _load_image imports dosma.scan_sequences.QDess which pulls in
        TensorFlow. TF 2.11 corrupts numpy's C runtime in this environment,
        causing numpy reductions (.min(), .max()) to segfault or return wrong
        values on large arrays. Individual element access still works.

        If this test segfaults, it confirms the TF/numpy conflict.
        """
        sys.path.insert(0, ".")
        from steps.segment import _load_image
        vol, _, _ = _load_image(TEST_NRRD)

        tf_loaded = _is_tf_loaded()

        # Individual element access works even with TF loaded
        assert vol.A[100, 100, 50] > 0, (
            f"_load_image returned zero at known-nonzero location. "
            f"TF loaded: {tf_loaded}"
        )

        # numpy reductions may segfault or return wrong values if TF corrupted numpy
        # This is the actual test — if it segfaults, the TF/numpy conflict is confirmed
        arr = _volume_to_array(vol)
        assert arr.max() > 0, (
            f"_load_image returned all-zero volume via to_sitk. "
            f"TF loaded: {tf_loaded}. "
            f"This likely indicates TF 2.11 corrupted numpy's C runtime."
        )

    def test_load_image_matches_a_plain_load(self):
        """_load_image and a plain sitk load should produce identical arrays.

        _load_image imports dosma (and so TensorFlow) before touching the data;
        the plain load does not. Identical arrays mean the TF import has not
        changed what gets read.
        """
        sys.path.insert(0, ".")
        from steps.segment import _load_image

        vol_step, _, _ = _load_image(TEST_NRRD)
        vol_plain = _load_volume_directly()

        arr_step = _volume_to_array(vol_step)
        arr_plain = _volume_to_array(vol_plain)

        assert arr_step.shape == arr_plain.shape
        assert np.array_equal(arr_step, arr_plain), (
            f"Data mismatch: _load_image min/max={arr_step.min()}/{arr_step.max()}, "
            f"plain load min/max={arr_plain.min()}/{arr_plain.max()}"
        )


# ---------------------------------------------------------------------------
# Test 2: Does importing dosma (which pulls TF) corrupt numpy reductions?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_test_data(), reason="Test data not available")
class TestNumpyAfterDosmaImport:
    """Isolate whether dosma's TF import corrupts numpy operations."""

    def test_numpy_reduction_before_dosma(self):
        """numpy .min()/.max() should work before dosma import."""
        arr = np.random.rand(100, 100, 100).astype(np.int16)
        arr[50, 50, 50] = 5000
        assert arr.max() == 5000

    def test_numpy_reduction_after_dosma_qdess_import(self):
        """numpy .min()/.max() should work after dosma QDess import.

        dosma.scan_sequences.QDess import triggers TF loading.
        If numpy reductions break after this, the TF/numpy conflict
        is confirmed.
        """
        from dosma.scan_sequences import QDess  # This loads TF

        # Test with a synthetic array first (no SimpleITK)
        arr = np.random.rand(100, 100, 100).astype(np.int16)
        arr[50, 50, 50] = 5000
        assert arr.max() == 5000, (
            f"numpy .max() returned {arr.max()} after QDess import. "
            "TF may have corrupted numpy's C runtime."
        )

    def test_sitk_array_reduction_after_dosma_import(self):
        """numpy reductions on SimpleITK arrays should work after dosma import.

        This tests the exact pattern that fails in _load_image:
        import dosma → sitk.ReadImage → GetArrayFromImage → .min()/.max()
        """
        from dosma.scan_sequences import QDess  # This loads TF

        img = sitk.ReadImage(TEST_NRRD)
        arr = sitk.GetArrayFromImage(img)
        # If this segfaults, the conflict is in SimpleITK array + TF numpy
        assert arr.max() > 0, (
            f"sitk array .max() returned {arr.max()} after dosma import"
        )

    def test_medical_volume_reduction_after_dosma_import(self):
        """MedicalVolume.A.min()/max() should work after QDess import.

        This is the exact operation that segfaults in _load_image.
        """
        from dosma.scan_sequences import QDess
        from dosma import MedicalVolume

        img = sitk.ReadImage(TEST_NRRD)
        vol = MedicalVolume.from_sitk(img)

        # Element access should work
        assert vol.A[100, 100, 50] > 0, "Element access failed"

        # Reduction — this is what segfaults
        assert vol.A.max() > 0, (
            f"vol.A.max() returned {vol.A.max()} — numpy reduction corrupted"
        )


# ---------------------------------------------------------------------------
# Test 3: Does importing torch before loading corrupt the data?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_test_data(), reason="Test data not available")
class TestTorchImportEffect:
    """Check if importing torch before dosma/sitk corrupts data."""

    def test_import_torch_then_load(self):
        """Import torch first, then load volume — data should still be valid."""
        import torch
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, "Data corrupted after torch import"

    def test_import_torch_cuda_then_load(self):
        """Initialize torch CUDA, then load volume."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1).cuda()  # Force CUDA initialization
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, "Data corrupted after torch CUDA init"


# ---------------------------------------------------------------------------
# Test 3: Does importing TensorFlow before loading corrupt the data?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_test_data(), reason="Test data not available")
class TestTFImportEffect:
    """Check if importing TensorFlow before dosma/sitk corrupts data."""

    def test_import_tf_then_load(self):
        """Import TF first, then load volume — data should still be valid."""
        import tensorflow
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, "Data corrupted after TF import"


# ---------------------------------------------------------------------------
# Test 4: Does importing BOTH TF and torch corrupt things?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _have_test_data(), reason="Test data not available")
class TestBothFrameworks:
    """Check if having both TF and torch loaded causes corruption."""

    def test_import_tf_and_torch_then_load(self):
        """Import both TF and torch, then load volume."""
        import tensorflow
        import torch
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, (
            "Data corrupted with both TF+torch loaded. "
            f"Got min/max={arr.min()}/{arr.max()}"
        )

    def test_import_torch_then_tf_then_load(self):
        """Import torch first, then TF, then load volume."""
        import torch
        import tensorflow
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, "Data corrupted (torch → TF → load)"

    def test_torch_cuda_then_tf_then_load(self):
        """Init torch CUDA, import TF, then load volume."""
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1).cuda()
        import tensorflow
        vol = _load_volume_directly()
        arr = _volume_to_array(vol)
        assert arr.max() > 0, "Data corrupted (torch CUDA → TF → load)"


# ---------------------------------------------------------------------------
# Test 5: Does the actual nnU-Net inference produce valid segmentations?
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_have_test_data() and _have_config()),
    reason="Test data or config not available"
)
class TestNnunetInference:
    """Test nnU-Net segmentation output quality."""

    def test_nnunet_in_subprocess_has_labels(self):
        """nnU-Net via subprocess should produce real segmentation labels."""
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '.')
import SimpleITK as sitk
import numpy as np
from dosma import MedicalVolume

image = sitk.ReadImage('{TEST_NRRD}')
volume = MedicalVolume.from_sitk(image)

with open('config.json') as f:
    config = json.load(f)

from steps.segment import segment_image_nnunet
seg = segment_image_nnunet(volume, 'nnunet_knee', config)
arr = sitk.GetArrayFromImage(seg)
unique, counts = np.unique(arr, return_counts=True)
result = dict(zip(unique.tolist(), counts.tolist()))
print(json.dumps(result))
"""],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr[-500:]}"
        labels = json.loads(result.stdout.strip().split("\n")[-1])
        nonzero = {k: v for k, v in labels.items() if int(k) != 0}
        assert len(nonzero) >= 3, f"Expected >=3 tissue labels, got {nonzero}"
        total_tissue = sum(nonzero.values())
        assert total_tissue > 10000, f"Only {total_tissue} non-zero voxels — segmentation failed"

    def test_nnunet_inprocess_has_labels(self):
        """nnU-Net called in-process should produce real segmentation labels.

        If this fails but test_nnunet_in_subprocess_has_labels passes,
        the TF/torch conflict is confirmed.
        """
        sys.path.insert(0, ".")
        from dosma import MedicalVolume
        image = sitk.ReadImage(TEST_NRRD)
        volume = MedicalVolume.from_sitk(image)

        with open("config.json") as f:
            config = json.load(f)

        from steps.segment import segment_image_nnunet
        seg = segment_image_nnunet(volume, "nnunet_knee", config)
        arr = sitk.GetArrayFromImage(seg)
        unique = np.unique(arr)
        nonzero = unique[unique != 0]
        total_tissue = np.sum(arr > 0)
        assert total_tissue > 10000, (
            f"Only {total_tissue} non-zero voxels in-process. "
            f"Labels: {unique.tolist()}. "
            "If subprocess test passes, this confirms TF/torch CUDA conflict."
        )


# ---------------------------------------------------------------------------
# Test 6: Direct comparison — same function, subprocess vs in-process
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_have_test_data() and _have_config()),
    reason="Test data or config not available"
)
class TestSubprocessVsInprocess:
    """Compare identical function calls in-process vs subprocess."""

    def test_modular_segment_subprocess_vs_inprocess(self):
        """steps.segment functions should produce same result either way.

        Runs segment_image_nnunet from steps/segment.py both ways.
        """
        # Subprocess
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '.')
import SimpleITK as sitk
import numpy as np
from steps.segment import _load_image, segment_image_nnunet
from steps._common import load_config

config = load_config()
volume, _, _ = _load_image('{TEST_NRRD}')
seg = segment_image_nnunet(volume, 'nnunet_knee', config)
arr = sitk.GetArrayFromImage(seg)
unique, counts = np.unique(arr, return_counts=True)
print(json.dumps(dict(zip(unique.tolist(), counts.tolist()))))
"""],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr[-500:]}"
        sub_labels = json.loads(result.stdout.strip().split("\n")[-1])

        # In-process
        sys.path.insert(0, ".")
        from steps.segment import _load_image, segment_image_nnunet
        from steps._common import load_config
        config = load_config()
        volume, _, _ = _load_image(TEST_NRRD)
        seg = segment_image_nnunet(volume, "nnunet_knee", config)
        arr = sitk.GetArrayFromImage(seg)
        unique, counts = np.unique(arr, return_counts=True)
        inproc_labels = dict(zip(unique.tolist(), counts.tolist()))

        # Compare
        sub_nonzero = sum(v for k, v in sub_labels.items() if int(k) != 0)
        inp_nonzero = sum(v for k, v in inproc_labels.items() if int(k) != 0)

        print(f"Subprocess labels: {sub_labels}")
        print(f"In-process labels: {inproc_labels}")

        # If these differ significantly, the environment has a conflict
        if sub_nonzero > 10000 and inp_nonzero < 100:
            pytest.fail(
                f"Subprocess produced {sub_nonzero} tissue voxels, "
                f"in-process only {inp_nonzero}. "
                "This confirms a TF/torch in-process conflict. "
                "Segmentation must run as a subprocess."
            )
