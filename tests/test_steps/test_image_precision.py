"""Geometry has to survive the handoff between steps.

The steps pass images to each other as files. NIfTI stores the image affine as
float32, so a ``.nii.gz`` write->read quantises the direction cosine matrix, and
from the first handoff onward every step would work in a slightly different
frame than the one segmentation produced. The monolith never showed this: it
holds one ``sitk.Image`` in memory for the whole run and never reads one back.

1.354e-08 on a direction cosine sounds ignorable. It moves marching-cubes
vertices by ~1.5e-05 mm, which is enough for pyacvd -- deterministic in itself --
to land on a different clustering, changing even the vertex count (femur 19993
vs 19994), and with it the surface locations thickness is sampled at: regional
thickness moved 0.007-0.02 mm, which is what made the orchestrator-vs-monolith
comparison disagree on ``med_tib_cart_mm_mean``.

Every step already writes a ``.nrrd`` beside its ``.nii.gz``, and SimpleITK
writes the NRRD space directions as full-precision decimal text. So the reads
between steps go through the NRRD. These tests pin both halves: the upstream
behaviour being absorbed, and the fact that the loaders absorb it.
"""

import logging
import re
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from steps._common import (
    find_segmentation,
    find_subregions,
    image_prefix,
    load_segmentation,
    load_subregions,
)
from steps.label_remap import run as label_remap

# The geometry of data/anthonys_knee.nrrd: a real sagittal knee acquisition, and
# so nothing like axis-aligned. These are the numbers the defect was measured on
# -- this direction matrix moves 1.354e-08 through a NIfTI roundtrip and exactly
# 0 through an NRRD one.
OBLIQUE_DIRECTION = (
    0.27254711642223695, -0.07572251447737907, -0.959158052737476,
    0.9621424371317978, 0.021449904101004787, 0.2717017340469917,
    -9.024097696278517e-08, -0.9968981905968537, 0.0787019541352757,
)
SPACING = (0.31249999999999994, 0.3125000000000001, 1.0000247955322301)
ORIGIN = (-80.54039764404288, -119.24600219726595, 74.45159912109379)

# Measured: 1.3538636212118149e-08. Asserted as a band rather than the literal,
# because the point is the float32 scale of it, not the last digit.
NIFTI_DELTA_FLOOR = 1e-9
NIFTI_DELTA_CEILING = 1e-7

# DOSMA-native -> canonical, the table run_pipeline passes to label_remap.
DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 1, 8: 2, 9: 3}


def _labelled_array(labels=(7, 8, 2, 3)):
    """A small volume with one cube per label. Native labels by default."""
    arr = np.zeros((12, 14, 16), dtype=np.uint8)
    for i, label in enumerate(labels):
        arr[2:5, 2:5, 2 + 3 * i : 4 + 3 * i] = label
    return arr


def _oblique_image(labels=(7, 8, 2, 3)):
    img = sitk.GetImageFromArray(_labelled_array(labels))
    img.SetDirection(OBLIQUE_DIRECTION)
    img.SetSpacing(SPACING)
    img.SetOrigin(ORIGIN)
    return img


def _write_both(img, working_dir, stem="test_all-labels"):
    """Write the pair every step writes: the NIfTI users download, the NRRD the
    next step reads."""
    for ext in (".nii.gz", ".nrrd"):
        sitk.WriteImage(img, str(working_dir / f"{stem}{ext}"), useCompression=True)


def _direction_delta(image, reference=OBLIQUE_DIRECTION):
    return float(np.abs(np.array(image.GetDirection()) - np.array(reference)).max())


def _roundtrip(img, path):
    sitk.WriteImage(img, str(path), useCompression=True)
    return sitk.ReadImage(str(path))


class TestNiftiQuantisesTheAffine:
    """The upstream behaviour the fix exists to absorb."""

    def test_nifti_roundtrip_perturbs_the_direction_matrix(self, tmp_path):
        img = _oblique_image()

        delta = _direction_delta(_roundtrip(img, tmp_path / "scan.nii.gz"))

        assert delta > NIFTI_DELTA_FLOOR, (
            "NIfTI stopped quantising the affine -- if that is real the fix is "
            "no longer load-bearing, but check the writer before believing it"
        )
        assert delta < NIFTI_DELTA_CEILING  # float32 scale, ~1.35e-08 measured

    def test_nrrd_roundtrip_returns_the_direction_unchanged(self, tmp_path):
        img = _oblique_image()

        read_back = _roundtrip(img, tmp_path / "scan.nrrd")

        assert read_back.GetDirection() == OBLIQUE_DIRECTION
        assert read_back.GetSpacing() == SPACING
        assert read_back.GetOrigin() == ORIGIN

    def test_nifti_perturbs_spacing_and_origin_too(self, tmp_path):
        """Direction is what amplifies, but it is the whole affine that moves."""
        read_back = _roundtrip(_oblique_image(), tmp_path / "scan.nii.gz")

        assert read_back.GetSpacing() != SPACING
        assert read_back.GetOrigin() != ORIGIN

    def test_an_axis_aligned_direction_hides_the_defect(self, tmp_path):
        """Why the fixtures here are oblique.

        With an identity direction the NIfTI roundtrip is exact, so a synthetic
        axis-aligned test image -- like every other fixture in this suite --
        cannot see this bug at all.
        """
        img = sitk.GetImageFromArray(_labelled_array())
        img.SetSpacing((1.0, 1.0, 1.0))

        read_back = _roundtrip(img, tmp_path / "axis_aligned.nii.gz")

        assert read_back.GetDirection() == img.GetDirection()


class TestLoadersReadTheLosslessCopy:
    """What every step gets when it asks for its predecessor's image."""

    def test_load_segmentation_direction_is_exact(self, tmp_path):
        _write_both(_oblique_image(), tmp_path)

        assert load_segmentation(tmp_path).GetDirection() == OBLIQUE_DIRECTION

    def test_load_segmentation_geometry_is_exact(self, tmp_path):
        _write_both(_oblique_image(), tmp_path)

        loaded = load_segmentation(tmp_path)

        assert loaded.GetSpacing() == SPACING
        assert loaded.GetOrigin() == ORIGIN
        assert np.array_equal(
            sitk.GetArrayFromImage(loaded), _labelled_array()
        )

    def test_load_subregions_direction_is_exact(self, tmp_path):
        _write_both(_oblique_image(labels=(11, 12, 13, 14)), tmp_path,
                    stem="test_subregions-labels")

        assert load_subregions(tmp_path).GetDirection() == OBLIQUE_DIRECTION

    def test_the_nifti_beside_it_is_the_one_that_would_have_been_wrong(self, tmp_path):
        """The pair is written from one image; only the NIfTI comes back moved."""
        _write_both(_oblique_image(), tmp_path)

        from_nifti = sitk.ReadImage(str(tmp_path / "test_all-labels.nii.gz"))

        assert _direction_delta(from_nifti) > NIFTI_DELTA_FLOOR
        assert _direction_delta(load_segmentation(tmp_path)) == 0.0

    def test_the_nrrd_is_what_gets_chosen_when_both_exist(self, tmp_path):
        _write_both(_oblique_image(), tmp_path)

        assert find_segmentation(tmp_path).name == "test_all-labels.nrrd"

    def test_subregions_nrrd_is_chosen_when_both_exist(self, tmp_path):
        _write_both(_oblique_image(labels=(11, 12)), tmp_path,
                    stem="test_subregions-labels")

        assert find_subregions(tmp_path).name == "test_subregions-labels.nrrd"


class TestNiftiFallback:
    """Older working directories have no NRRD. Finish the job, but say so."""

    def test_nifti_only_directory_still_loads(self, tmp_path):
        sitk.WriteImage(_oblique_image(), str(tmp_path / "test_all-labels.nii.gz"))

        loaded = load_segmentation(tmp_path)

        assert np.array_equal(sitk.GetArrayFromImage(loaded), _labelled_array())
        # And it is the quantised geometry, which is precisely why it warns.
        assert _direction_delta(loaded) > NIFTI_DELTA_FLOOR

    def test_the_fallback_is_logged_not_silent(self, tmp_path, caplog):
        """A silent fallback would put the precision loss back invisibly."""
        sitk.WriteImage(_oblique_image(), str(tmp_path / "test_all-labels.nii.gz"))

        with caplog.at_level(logging.WARNING):
            load_segmentation(tmp_path)

        assert any(
            record.levelno >= logging.WARNING and "test_all-labels.nii.gz" in record.getMessage()
            for record in caplog.records
        ), caplog.text

    def test_no_warning_when_the_nrrd_is_there(self, tmp_path, caplog):
        _write_both(_oblique_image(), tmp_path)

        with caplog.at_level(logging.WARNING):
            load_segmentation(tmp_path)

        assert caplog.records == []

    def test_missing_both_formats_raises_naming_both(self, tmp_path):
        """Callers that degrade gracefully catch FileNotFoundError -- keep it."""
        with pytest.raises(FileNotFoundError) as excinfo:
            load_subregions(tmp_path)

        message = str(excinfo.value)
        assert re.search(r"\*_subregions-labels\.nrrd", message)
        assert re.search(r"\*_subregions-labels\.nii\.gz", message)


class TestFilenamePrefixIsFormatIndependent:
    """Every output filename, and the merged results file, is named from this.

    A prefix that changed with the read format would silently rename the user's
    results.
    """

    @pytest.mark.parametrize(
        "prefix",
        ["test", "scan", "IMG_0001", "knee.v2", "patient_01_SAG_3D_DESS", "a-b_c"],
    )
    def test_both_formats_give_the_prefix_the_nifti_name_always_gave(self, prefix):
        nifti_name = f"{prefix}_all-labels.nii.gz"
        # The expression every step used before the NRRD switch.
        legacy = nifti_name.replace("_all-labels.nii.gz", "")

        assert image_prefix(nifti_name) == legacy
        assert image_prefix(f"{prefix}_all-labels.nrrd") == legacy
        assert legacy == prefix

    def test_subregion_images_give_the_same_prefix(self):
        assert image_prefix("scan_subregions-labels.nrrd") == "scan"
        assert image_prefix("scan_subregions-labels.nii.gz") == "scan"

    def test_full_paths_work_not_just_names(self, tmp_path):
        _write_both(_oblique_image(), tmp_path)

        assert image_prefix(find_segmentation(tmp_path)) == "test"


class TestLabelRemapRoundTrip:
    """label_remap reads one file and stamps its geometry on both it writes."""

    def _prepare(self, tmp_path, both=True):
        img = _oblique_image()
        if both:
            _write_both(img, tmp_path)
        else:
            sitk.WriteImage(img, str(tmp_path / "test_all-labels.nii.gz"),
                            useCompression=True)
        return img

    def test_the_written_nrrd_keeps_the_exact_geometry(self, tmp_path):
        """The regression: reading the NIfTI here quantised the NRRD as well,
        and the NRRD is what every later step reads."""
        self._prepare(tmp_path)

        result = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert result["remapped"] is True
        written = sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd"))
        assert written.GetDirection() == OBLIQUE_DIRECTION
        assert written.GetSpacing() == SPACING
        assert written.GetOrigin() == ORIGIN

    def test_the_written_nifti_is_no_worse_than_one_roundtrip(self, tmp_path):
        """The .nii.gz cannot hold the exact affine -- but it must not compound."""
        self._prepare(tmp_path)
        reference = _roundtrip(_oblique_image(), tmp_path / "reference.nii.gz")

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        written = sitk.ReadImage(str(tmp_path / "test_all-labels.nii.gz"))
        assert written.GetDirection() == reference.GetDirection()

    def test_both_formats_carry_the_remapped_labels(self, tmp_path):
        self._prepare(tmp_path)

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        expected = {0, 1, 2, 4, 5}  # native 7, 8, 2, 3 -> canonical 1, 2, 4, 5
        for name in ("test_all-labels.nii.gz", "test_all-labels.nrrd"):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / name)))
            assert set(np.unique(arr)) == expected, name

    def test_the_two_formats_agree_voxel_for_voxel(self, tmp_path):
        self._prepare(tmp_path)

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        nifti = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "test_all-labels.nii.gz")))
        nrrd = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd")))
        assert np.array_equal(nifti, nrrd)

    def test_the_backup_is_the_native_nifti_byte_for_byte(self, tmp_path):
        """Same name, same bytes, native labels -- the backup did not change."""
        self._prepare(tmp_path)
        before = (tmp_path / "test_all-labels.nii.gz").read_bytes()

        result = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        backup = tmp_path / "test_all-labels-native.nii.gz"
        assert backup.exists()
        assert result["native_backup"] == str(backup)
        assert backup.read_bytes() == before
        native = sitk.GetArrayFromImage(sitk.ReadImage(str(backup)))
        assert {7, 8}.issubset(set(np.unique(native)))

    def test_a_second_run_declines(self, tmp_path):
        self._prepare(tmp_path)

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})
        after_first = sitk.GetArrayFromImage(
            sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd"))
        )
        second = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert second == {"skipped": True, "reason": "already remapped"}
        after_second = sitk.GetArrayFromImage(
            sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd"))
        )
        assert np.array_equal(after_first, after_second)

    def test_a_nifti_only_directory_keeps_working(self, tmp_path):
        """Fallback path: back up under the documented name, and do not
        manufacture an NRRD out of quantised geometry."""
        self._prepare(tmp_path, both=False)

        result = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert result["remapped"] is True
        assert (tmp_path / "test_all-labels-native.nii.gz").exists()
        assert not (tmp_path / "test_all-labels.nrrd").exists()
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "test_all-labels.nii.gz")))
        assert set(np.unique(arr)) == {0, 1, 2, 4, 5}

    def test_a_nifti_only_directory_is_still_idempotent(self, tmp_path):
        self._prepare(tmp_path, both=False)

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})
        second = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert second["skipped"] is True

    def test_an_nrrd_only_directory_is_still_idempotent(self, tmp_path):
        """The guard is the backup's existence, and the backup follows the file
        that exists. Changing formats must not give a second run a way in."""
        sitk.WriteImage(_oblique_image(), str(tmp_path / "test_all-labels.nrrd"),
                        useCompression=True)

        first = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})
        after_first = sitk.GetArrayFromImage(
            sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd"))
        )
        second = label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert first["remapped"] is True
        assert (tmp_path / "test_all-labels-native.nrrd").exists()
        assert first["native_backup"] == str(tmp_path / "test_all-labels-native.nrrd")
        assert second["skipped"] is True
        after_second = sitk.GetArrayFromImage(
            sitk.ReadImage(str(tmp_path / "test_all-labels.nrrd"))
        )
        assert np.array_equal(after_first, after_second)
        # The specific corruption a second remap would cause: canonical femur
        # (1) sent to patellar cartilage (7).
        assert 7 not in set(np.unique(after_second))

    def test_the_backup_is_not_mistaken_for_the_segmentation(self, tmp_path):
        """`*_all-labels-native.*` must not match the `*_all-labels.*` glob --
        find_file raises on two matches, so this would fail the next step."""
        self._prepare(tmp_path)

        label_remap(tmp_path, options={"remap_table": DOSMA_REMAP})

        assert find_segmentation(tmp_path).name == "test_all-labels.nrrd"
        assert set(np.unique(sitk.GetArrayFromImage(load_segmentation(tmp_path)))) == {
            0, 1, 2, 4, 5
        }


class TestStepsReceiveTheExactGeometry:
    """End to end for one consumer: what the step hands to pymskt.

    pymskt is mocked as it is in test_subregions -- the real subregion call
    registers against a reference knee and needs anatomy. What is under test is
    the geometry the step passes down.
    """

    def test_subregions_passes_the_unquantised_segmentation(self, tmp_path):
        from steps.subregions import run as subregions

        _write_both(_oblique_image(labels=(1, 2, 4, 5)), tmp_path)

        mock_mskt = MagicMock()
        subregion_image = _oblique_image(labels=(11, 12, 13, 14))
        mock_mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions.return_value = (
            subregion_image
        )

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            result = subregions(tmp_path)

        call = mock_mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions.call_args
        passed_image = call.args[0]
        assert passed_image.GetDirection() == OBLIQUE_DIRECTION
        assert passed_image.GetOrigin() == ORIGIN
        # And the prefix it named its outputs with did not move.
        assert result["subregions_path"] == str(tmp_path / "test_subregions-labels.nii.gz")
        assert (tmp_path / "test_subregions-labels.nrrd").exists()

    def test_the_subregion_image_it_writes_is_read_back_exactly(self, tmp_path):
        """The next consumer of that file (generate_meshes, t2_mapping) gets the
        geometry the step produced, not a float32 copy of it."""
        from steps.subregions import run as subregions

        _write_both(_oblique_image(labels=(1, 2, 4, 5)), tmp_path)

        mock_mskt = MagicMock()
        mock_mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions.return_value = (
            _oblique_image(labels=(11, 12, 13, 14))
        )

        with patch.dict(sys.modules, {"pymskt": mock_mskt}):
            subregions(tmp_path)

        assert load_subregions(tmp_path).GetDirection() == OBLIQUE_DIRECTION
