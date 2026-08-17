"""Step 1: Image segmentation.

Loads an MRI image (DICOM, NIfTI, NRRD), runs segmentation using either
DOSMA or nnU-Net models, and saves the result as NIfTI and NRRD.

Output labels are in the model's native label scheme (not canonical).
Use steps.label_remap to convert to canonical labels afterward.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import emit_progress, parse_step_args, write_step_result

# Deterministic segmentation.
#
# cuDNN picks convolution algorithms by benchmarking them at runtime, and
# different algorithms sum floating point in different orders. The logits then
# differ in their last bits, and argmax flips at voxels where two classes are
# nearly tied -- always a handful of isolated voxels on a class boundary.
#
# Measured on the example scan: two runs of identical code on identical input
# differed by **2 voxels in 25M** without these, and by **0** with them. Small,
# but it is the first step and everything downstream inherits it: meshes,
# thickness, NSM, BScore. A research archive that cannot reproduce its own
# numbers is worth less than one that can, and NSM already goes to considerable
# trouble to be deterministic (seeded torch, subprocess isolation, the seed
# placed after `.cuda()`) -- it was inconsistent for the step feeding it not to.
#
# Must be set BEFORE TensorFlow initialises, which is why this is at module
# level and not inside run(): TF reads them once, at import. `setdefault` so an
# operator can still opt out.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(path_image):
    """Load an MRI image from DICOM dir, NIfTI, NRRD, or single enhanced DICOM.

    Returns:
        (volume, is_qdess, filename_prefix)
        - volume: dosma MedicalVolume
        - is_qdess: True if DOSMA could load the input as a two-echo qDESS
          scan (``qdess is not None``). It says nothing about the GL/TG
          spoiler private tags, which anonymisation routinely strips and which
          T2 mapping does not require — steps.t2_mapping falls back to the
          low-spoiling equations without them and labels which estimator ran.
        - filename_prefix: base name for output files (no extension)
    """
    from dosma.scan_sequences import QDess
    from dosma import MedicalVolume
    import dosma as dm

    path_image = str(path_image)

    if os.path.isdir(path_image):
        # Try loading as qDESS first (enables T2 mapping if successful)
        qdess = None
        try:
            try:
                qdess = QDess.from_dicom(path_image)
            except KeyError:
                qdess = QDess.from_dicom(path_image, group_by="EchoTime")
            volume = qdess.calc_rss()
        except (ValueError, TypeError, FileNotFoundError):
            # Not a qDESS scan -- fall back to generic DICOM loading.
            # FileNotFoundError: DOSMA only matches *.dcm, so extensionless DICOM
            # (Philips / PACS exports) looks like an empty directory to it. GDCM
            # below reads those fine, so fall through instead of failing the job.
            logging.info("Not a qDESS scan, loading as generic DICOM via SimpleITK...")
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path_image)
            if not dicom_names:
                raise ValueError(f"No DICOM files found in directory: {path_image}")
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            volume = MedicalVolume.from_sitk(image)
            volume._volume = volume._volume.copy()

        is_qdess = qdess is not None
        filename_prefix = os.path.basename(path_image)

    elif path_image.endswith(("nii", "nii.gz")):
        nr = dm.NiftiReader()
        volume = nr.load(path_image)
        is_qdess = False
        filename_prefix = os.path.basename(path_image).split(".nii")[0]

    elif path_image.endswith(("nrrd", "dcm")):
        image = sitk.ReadImage(path_image)
        volume = MedicalVolume.from_sitk(image)
        # MedicalVolume.from_sitk creates a zero-copy view of the SimpleITK
        # image data. If `image` goes out of scope the underlying C++ data is
        # freed, causing a use-after-free segfault. Force a copy so the volume
        # owns its own data.
        volume._volume = volume._volume.copy()
        is_qdess = False
        extension = ".nrrd" if path_image.endswith("nrrd") else ".dcm"
        filename_prefix = os.path.basename(path_image).replace(extension, "")

    else:
        raise ValueError("Image format not supported.")

    return volume, is_qdess, filename_prefix


# ---------------------------------------------------------------------------
# Segmentation: DOSMA
# ---------------------------------------------------------------------------

def segment_image_dosma(volume, model_name, config):
    """Segment image using DOSMA models.

    Args:
        volume: dosma MedicalVolume to segment
        model_name: Name of segmentation model to use
        config: Pipeline configuration dictionary

    Returns:
        sitk.Image: segmentation image in SimpleITK format
    """
    from dosma.models import (
        StanfordQDessBoneUNet2D, StanfordCubeBoneUNet2D,
        StanfordQDessBoneUNet2DSagittal,
        StanfordQDessBoneUNet2DCoronal,
        StanfordQDessBoneUNet2DAxial,
        StanfordQDessBoneUNet2DSTAPLE,
    )

    model_path = config["models"].get(model_name)
    if not model_path:
        raise ValueError(f"Model '{model_name}' not found in config")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.info("Loading Model...")
    if model_name == "staple":
        logging.info("Using Staple Model")
        model = StanfordQDessBoneUNet2DSTAPLE(
            config["models"]["goyal_sagittal"],
            config["models"]["goyal_coronal"],
            config["models"]["goyal_axial"],
        )
    else:
        orig_model_image_size = (512, 512)
        if "cube" in model_name:
            logging.info("Using Cube Model")
            model_class = StanfordCubeBoneUNet2D
        elif model_name == "goyal_sagittal":
            logging.info("Using Goyal Sagittal Model")
            model_class = StanfordQDessBoneUNet2DSagittal
            orig_model_image_size = None
        elif model_name == "goyal_coronal":
            logging.info("Using Goyal Coronal Model")
            model_class = StanfordQDessBoneUNet2DCoronal
            orig_model_image_size = None
        elif model_name == "goyal_axial":
            logging.info("Using Goyal Axial Model")
            model_class = StanfordQDessBoneUNet2DAxial
            orig_model_image_size = None
        else:
            model_class = StanfordQDessBoneUNet2D

        logging.info(f"Loading {model_name} model, orig_model_image_size: {orig_model_image_size}")
        model = model_class(config["models"][model_name], orig_model_image_size=orig_model_image_size)

    model.batch_size = int(config["batch_size"])

    logging.info("Segmenting Image...")
    seg = model.generate_mask(volume)
    sitk_seg = seg["all"].to_sitk(image_orientation="sagittal")
    return sitk_seg


# ---------------------------------------------------------------------------
# Segmentation: nnU-Net
# ---------------------------------------------------------------------------

def segment_image_nnunet(volume, model_name, config):
    """Segment image using nnU-Net models.

    Args:
        volume: dosma MedicalVolume to segment
        model_name: Name of nnunet model
        config: Pipeline configuration dictionary

    Returns:
        sitk.Image: segmentation image in SimpleITK format
    """
    from dosma.models.seg_model import fill_holes, get_connected_segments

    nnunet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "DEPENDENCIES", "nnunet_knee_inference")
    if nnunet_path not in sys.path:
        sys.path.append(nnunet_path)

    from scripts.inference import KneeSegmentationInference

    if not config:
        raise ValueError("Config dictionary is required but was None or empty")
    if "nnunet" not in config:
        raise ValueError("Config must contain 'nnunet' section. Check your config file.")
    if "type" not in config["nnunet"]:
        raise ValueError("Config['nnunet'] must contain 'type' key. Valid values: 'cascade' or 'fullres'")

    nnunet_type = config["nnunet"]["type"]
    if nnunet_type not in ["cascade", "fullres"]:
        raise ValueError(
            f"Invalid nnunet type '{nnunet_type}'. Must be 'cascade' or 'fullres'. "
            f"Update your config file."
        )

    logging.info(f"Loading nnU-Net model (type: {nnunet_type})")
    inference = KneeSegmentationInference(config=nnunet_type)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
        temp_input_path = temp_file.name

    try:
        sitk_input = volume.to_sitk(image_orientation="sagittal")
        sitk.WriteImage(sitk_input, temp_input_path)

        logging.info("Running nnU-Net segmentation...")
        sitk_seg = inference.predict(temp_input_path)

        logging.info("Applying post-processing (connected components and hole filling)...")
        seg_array = sitk.GetArrayFromImage(sitk_seg)
        seg_array = get_connected_segments(seg_array)

        bone_indices = [7, 8, 9]
        for bone_idx in bone_indices:
            if bone_idx in seg_array:
                mask_ = fill_holes(seg_array, label_idx=bone_idx)
                seg_array[mask_ == 1] = bone_idx

        seg_array = seg_array.astype(np.uint8)
        sitk_seg_processed = sitk.GetImageFromArray(seg_array)
        sitk_seg_processed.CopyInformation(sitk_seg)
        return sitk_seg_processed

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

def run(working_dir, options=None, config=None):
    """Run segmentation step.

    Args:
        working_dir: Directory containing input image and where outputs are saved.
        options: Dict with optional keys:
            - model: segmentation model name (default: from config)
            - batch_size: inference batch size (default: from config)
        config: Pipeline config dict.

    Returns:
        Dict with seg_path, is_qdess, filename_prefix, model_name.
    """
    working_dir = Path(working_dir)
    options = options or {}

    model_name = options.get("model") or config.get("default_seg_model", "goyal_sagittal")
    batch_size = options.get("batch_size", config.get("batch_size", 64))

    if not working_dir.exists():
        raise FileNotFoundError(f"Working directory not found: {working_dir}")

    emit_progress(0, "Loading image")
    # Find the input image -- could be a DICOM subdirectory or a file
    input_path = _find_input_image(working_dir)
    volume, is_qdess, filename_prefix = _load_image(input_path)

    emit_progress(10, "Running segmentation")
    if model_name.startswith("nnunet"):
        sitk_seg = segment_image_nnunet(volume, model_name, config)
    else:
        config["batch_size"] = batch_size
        sitk_seg = segment_image_dosma(volume, model_name, config)

    emit_progress(80, "Saving segmentation")
    seg_path_nii = working_dir / f"{filename_prefix}_all-labels.nii.gz"
    seg_path_nrrd = working_dir / f"{filename_prefix}_all-labels.nrrd"
    sitk.WriteImage(sitk_seg, str(seg_path_nii), useCompression=True)
    sitk.WriteImage(sitk_seg, str(seg_path_nrrd), useCompression=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    emit_progress(100, "Segmentation complete")
    result = {
        "seg_path": str(seg_path_nii),
        "is_qdess": is_qdess,
        "filename_prefix": filename_prefix,
        "model_name": model_name,
    }
    if not is_qdess:
        result["skip_steps"] = ["t2_mapping"]
    return result


def _find_input_image(working_dir):
    """Find the input image in working_dir.

    Looks for (in order):
    1. A subdirectory containing DICOM files
    2. A NIfTI file (*.nii or *.nii.gz) that is NOT a segmentation label file
    3. An NRRD file that is NOT a segmentation label file
    4. A single .dcm file
    """
    working_dir = Path(working_dir)

    # Check for DICOM subdirectory
    for item in working_dir.iterdir():
        if item.is_dir() and not item.name.startswith((".", "_")):
            dcm_files = list(item.glob("*.dcm")) + list(item.glob("*.DCM"))
            # Also check for DICOM files without extension (common in medical imaging)
            if dcm_files or any(not f.suffix and f.is_file() for f in item.iterdir()):
                return item

    # Check for NIfTI (excluding label files)
    for nifti in sorted(working_dir.glob("*.nii*")):
        if "labels" not in nifti.name and "t2map" not in nifti.name:
            return nifti

    # Check for NRRD (excluding label files)
    for nrrd in sorted(working_dir.glob("*.nrrd")):
        if "labels" not in nrrd.name and "t2map" not in nrrd.name and "depth_seg" not in nrrd.name:
            return nrrd

    # Check for single DCM file
    dcm_files = list(working_dir.glob("*.dcm")) + list(working_dir.glob("*.DCM"))
    if dcm_files:
        return dcm_files[0]

    raise FileNotFoundError(f"No input image found in {working_dir}")


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
