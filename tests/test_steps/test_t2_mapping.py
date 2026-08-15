"""Tests for steps.t2_mapping.

These tests mock dosma and pymskt since they require real DICOM data.
Integration tests with real data belong in Phase 4.

The exception is the spoiling tests, which build a tiny but genuine two-echo
qDESS series with pydicom and run DOSMA's real T2 solver over it: the whole
point of those tests is which DOSMA code path runs, so mocking DOSMA would
test nothing. The volumes are 8x8x8 and the maths is pure numpy, so they cost
no more than the rest of the file.
"""

import json
from pathlib import Path
from unittest.mock import patch

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
    _load_qdess,
    read_spoiler_parameter,
    run,
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


# ---------------------------------------------------------------------------
# Spoiling / t2_method (D6)
# ---------------------------------------------------------------------------

GL_AREA_TAG = 0x001910B6   # qDESS private tag: spoiler area
TG_TAG = 0x001910B7        # qDESS private tag: spoiler duration

VOLUME_SIZE = 8


def _write_qdess_series(directory, spoiler_tags=True, n_echoes=2, size=VOLUME_SIZE):
    """Write a minimal but genuine qDESS DICOM series DOSMA can load.

    Two echoes of constant signal, plus the standard timing tags the T2
    solution needs (TR/TE/flip angle). ``spoiler_tags=False`` reproduces
    anonymised data, where the private GL/TG tags have been stripped.
    ``n_echoes=1`` produces a series that is not qDESS at all.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian, generate_uid

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    echoes = [(1, 6.6), (2, 34.0)][:n_echoes]

    for echo_idx, (echo_number, echo_time) in enumerate(echoes):
        for i in range(size):
            ds = Dataset()
            ds.file_meta = FileMetaDataset()
            ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
            ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = series_uid
            ds.StudyInstanceUID = study_uid
            ds.Modality = "MR"
            ds.Rows, ds.Columns = size, size
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = 1.0
            ds.SpacingBetweenSlices = 1.0
            ds.ImagePositionPatient = [0.0, 0.0, float(i)]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.InstanceNumber = echo_idx * size + i + 1
            ds.EchoNumbers = echo_number
            ds.EchoTime = echo_time
            ds.RepetitionTime = 21.9
            ds.FlipAngle = 20.0
            if spoiler_tags:
                # Values measured on data/012_knee1.
                ds.add_new(GL_AREA_TAG, "DS", "3131.662354")
                ds.add_new(TG_TAG, "DS", "2472.000000")
            signal = 1000 if echo_number == 1 else 400
            ds.PixelData = np.full((size, size), signal, dtype=np.uint16).tobytes()
            ds.save_as(
                str(directory / f"E{echo_number}_I{i:04d}.dcm"), write_like_original=False
            )
    return directory


WHOLE_REGIONS = ("fem_cart", "med_tib_cart", "lat_tib_cart", "pat_cart")
FEMUR_SUBREGIONS = (11, 12, 13, 14, 15)


def _make_t2_working_dir(tmp_path, spoiler_tags=True, n_echoes=2, size=VOLUME_SIZE,
                         subregions=True, split_femur=False):
    """Working dir with a qDESS series plus the segmentations the step reads.

    ``subregions=False`` is a job with no subregions step — the D7 case, where
    the step must fall back to the canonical labels in the segmentation itself.
    ``split_femur=True`` makes the subregions file realistic: pymskt replaces
    femoral cartilage (4) with its five subregions (11-15), so label 4 is gone
    from that file and only whole-region tibial/patellar labels survive in both.
    """
    _write_qdess_series(tmp_path / "dicoms", spoiler_tags=spoiler_tags,
                        n_echoes=n_echoes, size=size)

    seg = np.zeros((size, size, size), dtype=np.uint8)
    seg[1:3, 1:4, 1:4] = 4   # fem_cart (canonical)
    seg[4:6, 1:4, 1:4] = 5   # med_tib_cart
    seg[4:6, 4:7, 1:4] = 6   # lat_tib_cart
    seg[1:3, 5:7, 1:4] = 7   # pat_cart
    sitk_seg = sitk.GetImageFromArray(seg)
    sitk.WriteImage(sitk_seg, str(tmp_path / "scan_all-labels.nii.gz"))

    if subregions:
        sub = seg.copy()
        if split_femur:
            fem_cart = np.argwhere(seg == 4)
            for i, label in enumerate(FEMUR_SUBREGIONS):
                chunk = fem_cart[i::len(FEMUR_SUBREGIONS)]
                sub[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = label
        sitk.WriteImage(sitk.GetImageFromArray(sub),
                        str(tmp_path / "scan_subregions-labels.nii.gz"))
    return tmp_path


def _depth_labelled_image(working_dir):
    """The image combine_depth_region_segs() would return: every cartilage
    voxel labelled deep (+100) or superficial (+200)."""
    sub = sitk.GetArrayFromImage(
        sitk.ReadImage(str(working_dir / "scan_subregions-labels.nii.gz"))
    )
    depth = np.zeros_like(sub, dtype=np.uint16)
    for i, (z, y, x) in enumerate(np.argwhere(np.isin(sub, list(REGION_NAMES)))):
        offset = DEEP_OFFSET if i % 2 == 0 else SUPERFICIAL_OFFSET
        depth[z, y, x] = sub[z, y, x] + offset
    return sitk.GetImageFromArray(depth)


def _mock_pymskt_for_depth(working_dir):
    """A pymskt whose depth splitting succeeds, so a test that still gets no
    depth metrics is being told something by the step, not by a broken mock."""
    from unittest.mock import MagicMock

    mock_mskt = MagicMock()
    mock_mskt.mesh.BoneMesh.return_value.break_cartilage_into_superficial_deep.return_value = (
        np.zeros((VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE), dtype=np.uint16), None,
    )
    mock_mskt.image.cartilage_processing.combine_depth_region_segs.return_value = (
        _depth_labelled_image(working_dir)
    )
    return mock_mskt


def _write_bone_meshes(working_dir):
    """The three files the step checks for before attempting depth T2."""
    for bone in ("femur", "tibia", "patella"):
        (working_dir / f"{bone}_mesh.vtk").touch()


class _MetadataOnlyQdess:
    """Just enough of a QDess to answer get_metadata()."""

    def __init__(self, metadata):
        self._metadata = metadata

    def get_metadata(self, key, default=None):
        return self._metadata.get(key, default)


class TestReadSpoilerParameter:
    """The one decision both pipeline paths share: spoiled or low-spoiling."""

    def test_reads_a_present_tag(self):
        qdess = _MetadataOnlyQdess({GL_AREA_TAG: "3131.662354"})
        assert read_spoiler_parameter(qdess, GL_AREA_TAG) == pytest.approx(3131.662354)

    def test_absent_tag_is_none(self):
        """The normal state of anonymised DICOM."""
        assert read_spoiler_parameter(_MetadataOnlyQdess({}), GL_AREA_TAG) is None

    def test_zero_is_none(self):
        """Zero is DOSMA's own 'no spoiler parameters' sentinel (qdess.py:202)."""
        qdess = _MetadataOnlyQdess({GL_AREA_TAG: 0.0})
        assert read_spoiler_parameter(qdess, GL_AREA_TAG) is None

    def test_uncastable_value_is_none_rather_than_raising(self):
        """An untyped (VR 'UN') tag would crash DOSMA's own float() cast.

        Low-spoiling T2, labelled as such, beats losing the T2 columns.
        """
        qdess = _MetadataOnlyQdess({GL_AREA_TAG: b"\x01\x02"})
        assert read_spoiler_parameter(qdess, GL_AREA_TAG) is None


class TestSpoilerTagsAreNotAPrecondition:
    """D6: anonymised qDESS keeps its T2, labelled with the estimator used.

    Regression: the step used to return ``skipped: True`` whenever the qDESS
    private GL/TG tags were absent — the normal state of anonymised DICOM — so
    every such job silently lost all 72 T2 columns. DOSMA falls back to the
    Sveinsson low-spoiling equations without those tags; it just has to be told
    to, because it dereferences the absent tags before reaching its own
    fallback (pinned by TestDosmaSpoilingFallback below).
    """

    def test_absent_tags_still_produce_t2(self, tmp_path):
        working_dir = _make_t2_working_dir(tmp_path, spoiler_tags=False)

        result = run(working_dir)

        assert result.get("skipped") is not True
        assert result["t2_method"] == "low_spoiling"
        assert result["metrics"], "T2 metrics must not be empty for anonymised qDESS"
        assert np.isfinite(result["metrics"]["fem_cart_t2_ms_mean"])
        assert (tmp_path / "scan_t2map.nii.gz").exists()
        assert (tmp_path / "scan_t2map.nrrd").exists()

    def test_absent_tags_label_reaches_the_results_file(self, tmp_path):
        """run_pipeline.py discards the step result, so the file must carry it."""
        working_dir = _make_t2_working_dir(tmp_path, spoiler_tags=False)

        run(working_dir)

        saved = json.loads((tmp_path / "scan_t2_results.json").read_text())
        assert saved["t2_method"] == "low_spoiling"
        assert "fem_cart_t2_ms_mean" in saved

    def test_present_tags_report_spoiled(self, tmp_path):
        working_dir = _make_t2_working_dir(tmp_path, spoiler_tags=True)

        result = run(working_dir)

        assert result.get("skipped") is not True
        assert result["t2_method"] == "spoiled"
        assert result["metrics"]
        saved = json.loads((tmp_path / "scan_t2_results.json").read_text())
        assert saved["t2_method"] == "spoiled"

    def test_the_two_estimators_disagree(self, tmp_path):
        """Why the method has to be recorded at all.

        Low-spoiling reads low, and the gap grows with T2, so it cannot be
        corrected after the fact. If this ever stops being true, the t2_method
        flag is no longer load-bearing.
        """
        spoiled = run(_make_t2_working_dir(tmp_path / "spoiled", spoiler_tags=True))
        low = run(_make_t2_working_dir(tmp_path / "low", spoiler_tags=False))

        spoiled_t2 = spoiled["metrics"]["fem_cart_t2_ms_mean"]
        low_t2 = low["metrics"]["fem_cart_t2_ms_mean"]
        assert low_t2 < spoiled_t2
        assert not np.isclose(low_t2, spoiled_t2)

    def test_zero_spoiler_tags_are_labelled_low_spoiling(self, tmp_path):
        """A present-but-zero tag is DOSMA's own 'no spoiler parameters' case.

        DOSMA drops to low-spoiling on a zero (qdess.py:202), so labelling it
        "spoiled" would make the flag lie about what ran.
        """
        working_dir = _make_t2_working_dir(tmp_path, spoiler_tags=True)
        qdess = _load_qdess(working_dir)
        qdess._metadata = {GL_AREA_TAG: 0.0, TG_TAG: 0.0}

        with patch("steps.t2_mapping._load_qdess", return_value=qdess):
            result = run(working_dir)

        assert result["t2_method"] == "low_spoiling"


class TestSkippedOnlyWhenNotQdess:
    """``skipped: True`` means T2 genuinely cannot be computed — nothing else."""

    def test_single_echo_series_is_skipped(self, tmp_path):
        working_dir = _make_t2_working_dir(tmp_path, n_echoes=1)

        result = run(working_dir)

        assert result["skipped"] is True
        assert result["metrics"] == {}
        assert result["has_depth_dependent"] is False
        assert "qdess" in result["reason"].lower()

    def test_no_dicom_directory_is_skipped(self, tmp_path):
        """A NIfTI/NRRD job has no DICOM to load — a skip, not a failure."""
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(arr), str(tmp_path / "scan.nii.gz"))

        result = run(tmp_path)

        assert result["skipped"] is True
        assert result["metrics"] == {}

    def test_load_qdess_returns_none_rather_than_raising(self, tmp_path):
        _write_qdess_series(tmp_path / "dicoms", n_echoes=1)

        assert _load_qdess(tmp_path) is None


def _depth_keys(metrics):
    return [k for k in metrics if "_deep_" in k or "_superficial_" in k]


class TestDegradesWithoutSubregionsOrMeshes:
    """D7: neither the subregions file nor the meshes is a precondition.

    Regression: the step called ``load_subregions()`` outside its try block, so
    a job without the mesh step (which used to write that file) did not degrade
    to whole-region T2 — it raised FileNotFoundError, and the user was told the
    input was not a two-echo qDESS scan, a false explanation of a real failure.
    """

    def test_no_subregions_file_still_returns_whole_region_t2(self, tmp_path):
        """The test that would have caught D7."""
        working_dir = _make_t2_working_dir(tmp_path, subregions=False)
        assert list(working_dir.glob("*_subregions-labels.*")) == []

        result = run(working_dir)

        assert result.get("skipped") is not True
        assert result["has_subregions"] is False
        assert result["has_depth_dependent"] is False
        for region in WHOLE_REGIONS:
            assert np.isfinite(result["metrics"][f"{region}_t2_ms_mean"]), region
        assert _depth_keys(result["metrics"]) == []
        assert (tmp_path / "scan_t2map.nii.gz").exists()
        assert json.loads((tmp_path / "scan_t2_results.json").read_text())["t2_method"]

    def test_no_subregions_loses_only_the_subregion_metrics(self, tmp_path):
        """Labels 11-15 live only in the subregions file; 4-7 do not."""
        working_dir = _make_t2_working_dir(tmp_path, subregions=False)

        metrics = run(working_dir)["metrics"]

        for label in FEMUR_SUBREGIONS:
            assert f"{REGION_NAMES[label]}_t2_ms_mean" not in metrics
        assert "fem_cart_t2_ms_mean" in metrics

    def test_meshes_without_subregions_do_not_reach_the_depth_branch(self, tmp_path):
        """combine_depth_region_segs() needs the subregion image, so the depth
        branch is gated on both — and says so where the requirement applies."""
        working_dir = _make_t2_working_dir(tmp_path, subregions=False)
        _write_bone_meshes(working_dir)
        mock_mskt = _mock_pymskt_for_depth(
            _make_t2_working_dir(tmp_path / "reference", split_femur=True)
        )

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            result = run(working_dir)

        assert result["has_subregions"] is False
        assert result["has_depth_dependent"] is False
        assert _depth_keys(result["metrics"]) == []
        mock_mskt.image.cartilage_processing.combine_depth_region_segs.assert_not_called()

    def test_subregions_without_meshes_gives_regional_but_not_depth_t2(self, tmp_path):
        working_dir = _make_t2_working_dir(tmp_path, split_femur=True)

        result = run(working_dir)

        assert result["has_subregions"] is True
        assert result["has_depth_dependent"] is False
        for label in FEMUR_SUBREGIONS:
            key = f"{REGION_NAMES[label]}_t2_ms_mean"
            assert np.isfinite(result["metrics"][key]), key
        for region in ("med_tib_cart", "lat_tib_cart", "pat_cart"):
            assert np.isfinite(result["metrics"][f"{region}_t2_ms_mean"]), region
        # pymskt splits femoral cartilage into 11-15, so 4 is not in that file.
        assert "fem_cart_t2_ms_mean" not in result["metrics"]
        assert _depth_keys(result["metrics"]) == []

    def test_subregions_and_meshes_give_all_three_metric_sets(self, tmp_path):
        working_dir = _make_t2_working_dir(tmp_path, split_femur=True)
        _write_bone_meshes(working_dir)
        mock_mskt = _mock_pymskt_for_depth(working_dir)

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            result = run(working_dir)

        assert result["has_subregions"] is True
        assert result["has_depth_dependent"] is True
        for label in FEMUR_SUBREGIONS:
            region = REGION_NAMES[label]
            for key in (f"{region}_t2_ms_mean",
                        f"{region}_deep_t2_ms_mean",
                        f"{region}_superficial_t2_ms_mean"):
                assert np.isfinite(result["metrics"][key]), key
        assert (tmp_path / "scan_depth_seg.nrrd").exists()

    def test_the_depth_branch_is_given_the_subregion_image(self, tmp_path):
        """Not the plain segmentation: the depth labels are combined with the
        subregion labels, which is the whole reason the branch needs them."""
        working_dir = _make_t2_working_dir(tmp_path, split_femur=True)
        _write_bone_meshes(working_dir)
        mock_mskt = _mock_pymskt_for_depth(working_dir)

        with patch.dict("sys.modules", {"pymskt": mock_mskt}):
            run(working_dir)

        passed_seg = (mock_mskt.image.cartilage_processing
                      .combine_depth_region_segs.call_args.args[0])
        passed = sitk.GetArrayFromImage(passed_seg)
        assert set(FEMUR_SUBREGIONS) <= set(np.unique(passed).tolist())


class TestDosmaSpoilingFallback:
    """Pins the upstream DOSMA behaviour the step's explicit 0s exist for.

    If DOSMA's fallback is ever finished (its zero-check moved above the tag
    dereference), the first assertion fails and the explicit ``gl_area=0`` /
    ``tg=0`` arguments can be reconsidered.
    """

    def test_dosma_dereferences_absent_tags_before_its_own_fallback(self, tmp_path):
        from dosma.tissues import FemoralCartilage

        qdess = _load_qdess(_make_t2_working_dir(tmp_path, spoiler_tags=False))

        with pytest.raises(KeyError):
            qdess.generate_t2_map(
                FemoralCartilage(), suppress_fat=False, suppress_fluid=False
            )

    def test_zero_selects_the_low_spoiling_path(self, tmp_path):
        from dosma.tissues import FemoralCartilage

        qdess = _load_qdess(_make_t2_working_dir(tmp_path, spoiler_tags=False))

        t2map = qdess.generate_t2_map(
            FemoralCartilage(), suppress_fat=False, suppress_fluid=False,
            gl_area=0, tg=0, spoiling=False,
        )

        values = t2map.volumetric_map.volume
        assert np.isfinite(values).all()
        assert np.nanmean(values) > 0
