# CLAUDE.md - Knee MRI Analysis Pipeline

## What This Project Does

An automated pipeline for analyzing knee MRI scans. Given a knee MRI image (DICOM, NIfTI, or NRRD), it:

1. **Segments** bone and cartilage structures (femur, tibia, patella + their cartilage) using either DOSMA or nnU-Net deep learning models
2. **Creates 3D surface meshes** from the segmentations
3. **Computes cartilage thickness** metrics by region (anterior, medial/lateral weight-bearing, medial/lateral posterior femoral cartilage, plus tibial and patellar cartilage)
4. **Computes T2 relaxation maps** (only if the input is a qDESS DICOM scan with GL/TG private tags)
5. **Fits Neural Shape Models (NSM)** to the femur bone (and optionally cartilage) to get a latent shape representation
6. **Computes BScore** (osteoarthritis severity score) from the NSM latent vector

## Execution Flow

```
dosma_knee_seg.py  (orchestrator / entry point)
    |
    +--> seg_thick_t2_pipeline.py  (segmentation, meshing, thickness, T2 - runs as subprocess)
    |       uses: utils.py (clip_femur_top)
    |
    +--> NSM_analysis.py           (bone+cartilage NSM fitting - runs as subprocess, optional)
    |
    +--> NSM_analysis_bone_only.py (bone-only NSM fitting - runs as subprocess, optional)
```

Each step after segmentation runs as a **separate Python subprocess** launched by `dosma_knee_seg.py`. The NSM steps are optional, controlled by `config.json` flags.

## File Descriptions

### Core Pipeline Files

- **`dosma_knee_seg.py`** - Entry point / orchestrator. Reads config, launches `seg_thick_t2_pipeline.py` as subprocess, then optionally launches NSM analysis scripts. Takes `path_image`, `path_save`, and optional `model_name` as CLI args. Config can be overridden via `KNEEPIPELINE_CONFIG` env var.

- **`seg_thick_t2_pipeline.py`** - The main workhorse (~577 lines). Handles:
  - Image loading (DICOM directories, NIfTI, NRRD, single enhanced DICOMs)
  - Segmentation via DOSMA models or nnU-Net
  - Femoral cartilage subregion splitting (5 subregions)
  - 3D mesh generation for all bones/cartilage using pymskt
  - Cartilage thickness computation per region
  - T2 map generation and regional T2 statistics (full-thickness and depth-dependent)
  - NSM preprocessing (left-knee mirroring, femur clipping, mesh export)
  - Step tracking via `StepTracker` class (writes `_step_log.json` for web UI consumption)

- **`NSM_analysis.py`** - Fits the bone+cartilage NSM model. Takes paths to femur bone mesh and cartilage mesh + save directory. Outputs reconstructed meshes and `NSM_recon_params.json` containing: latent vector (512-d), BScore, registration parameters (ICP transform, center, scale), and ASSD error metrics.

- **`NSM_analysis_bone_only.py`** - Same as above but for bone-only NSM model. Outputs `NSM_bone_only_recon_params.json`. Nearly identical code to `NSM_analysis.py` with minor differences (1 input mesh instead of 2, different config keys).

- **`utils.py`** - Single utility function `clip_femur_top()` that clips the superior portion of femur meshes. Logic: if S-I dimension > 70% of M-L dimension, clip to 70% of M-L; otherwise clip to 95% of S-I.

- **`download_nsm_models.py`** - Standalone script to download NSM models from HuggingFace (`aagatti/ShapeMedKnee`). Requires authentication (gated repo). Uses argparse.

### Configuration

- **`config.json`** - Active configuration (not committed to git, derived from template). Contains:
  - `perform_bone_only_nsm` / `perform_bone_and_cart_nsm` - Flags to enable/disable NSM steps
  - `clip_femur_top` - Whether to clip superior femur before NSM
  - `default_seg_model` - Default segmentation model name
  - `batch_size` - Inference batch size
  - `models` - Paths to DOSMA `.h5` weight files
  - `nnunet` - nnU-Net configuration (type: "cascade" or "fullres")
  - `nsm` / `nsm_bone_only` - Paths to NSM model configs and state dicts
  - `bscore` / `bscore_bone_only` - Paths to BScore model folders
  - `regions` - Segmentation label index to region name mapping
  - `bones` - Per-bone mesh config (label index, cartilage labels, point count, crop percent)

- **`config_template.json`** - Template with placeholder paths. Copy to `config.json` and update paths.

### Non-Code Directories

- **`DOSMA_WEIGHTS/`** - DOSMA segmentation model `.h5` weight files (downloaded from `aagatti/dosma_bones` on HuggingFace)
- **`NSM_MODELS/`** - Neural Shape Model configs and state dicts (downloaded from `aagatti/ShapeMedKnee` on HuggingFace, gated repo)
- **`BSCORE_MODELS/`** - BScore prediction models (Python script + parameters, loaded via `sys.path` manipulation)
- **`DEPENDENCIES/`** - External libraries:
  - `nnunet_knee_inference/` - git submodule for nnU-Net-based knee segmentation

## Segmentation Label Map

| Index | Structure |
|-------|-----------|
| 1 | Patellar cartilage |
| 2 | Femoral cartilage |
| 3 | Medial tibial cartilage |
| 4 | Lateral tibial cartilage |
| 7 | Femur bone |
| 8 | Tibia bone |
| 9 | Patella bone |
| 11 | Anterior femoral cartilage (subregion) |
| 12 | Medial weight-bearing femoral cartilage (subregion) |
| 13 | Lateral weight-bearing femoral cartilage (subregion) |
| 14 | Medial posterior femoral cartilage (subregion) |
| 15 | Lateral posterior femoral cartilage (subregion) |

## Key Dependencies

- **DOSMA** (`dosma`) - MRI analysis framework (bone_seg branch of gattia/DOSMA fork)
- **pymskt** / `mskt` - Musculoskeletal toolkit for mesh generation and cartilage analysis
- **NSM** - Neural Shape Model library (gattia/nsm)
- **nnunetv2** - nnU-Net segmentation framework (optional, for `nnunet_knee` model)
- **PyTorch** - Deep learning (CUDA required for NSM)
- **TensorFlow 2.11** - Required for DOSMA `.h5` models (CUDA 11.x, NumPy < 2.0, Keras < 3)
- **VTK** - Mesh I/O and transforms
- **SimpleITK** - Medical image I/O and processing

### Environment Constraints

TensorFlow 2.11 and PyTorch coexist but have conflicting CUDA requirements. TF needs CUDA 11.x; PyTorch uses CUDA 12.x. Both work via conda-installed `cudatoolkit=11.8` + `cudnn=8.2`. NumPy must be 1.24.x (TF needs < 2.0; nnunet needs >= 1.24). See README.MD for full install instructions.

## How to Run

```bash
# Basic usage (uses default_seg_model from config)
python dosma_knee_seg.py /path/to/image /path/to/output/

# With specific model
python dosma_knee_seg.py /path/to/image /path/to/output/ goyal_sagittal

# With nnU-Net
python dosma_knee_seg.py /path/to/image /path/to/output/ nnunet_knee

# Override config location
KNEEPIPELINE_CONFIG=/path/to/custom_config.json python dosma_knee_seg.py ...
```

## Known Bugs

1. **`dosma_knee_seg.py:27` reads `sys.argv[3]` inside `main()`** - The `main()` function receives `config_path, path_image, path_save, path_seg_script` as parameters but reads `model_name` directly from `sys.argv[3]` instead of accepting it as a parameter. This makes `main()` impossible to call programmatically with a custom model name.

2. **Duplicate `import sys`** in `NSM_analysis.py:1,5` and `NSM_analysis_bone_only.py:1,5`.

3. **`NSM_analysis_bone_only.py` doesn't handle `None` ICP transform** - `NSM_analysis.py:166-169` has an `elif icp_transform is None:` branch that falls back to an identity matrix. `NSM_analysis_bone_only.py` lacks this case and will raise `ValueError` instead.

4. **`torch.load()` without `weights_only=True`** in both NSM scripts (lines 78/84) - Deprecated in modern PyTorch and a security risk (arbitrary code execution via pickle). Will emit warnings on PyTorch >= 2.6.

5. **`objects_per_decoder=2` hardcoded** in `NSM_analysis.py:112` instead of reading `config["objects_per_decoder"]` like `NSM_analysis_bone_only.py` does. This happens to be correct for bone+cartilage (2 objects) but is fragile.

6. **Lambda tuple in `dosma_knee_seg.py:58-60`** - The `pre_hook` lambda returns a tuple of side-effect calls. It works, but the return value is silently discarded. Confusing pattern.

7. **`time.sleep(5)` for GPU memory release** appears in 3 places (`seg_thick_t2_pipeline.py:562`, `NSM_analysis.py:199`, `NSM_analysis_bone_only.py:182`). This is a non-deterministic workaround; CUDA memory release doesn't require a sleep, and 5 seconds may be too long or too short depending on the system.

## Potential Improvements

1. **Refactor NSM scripts into functions** - Both `NSM_analysis.py` and `NSM_analysis_bone_only.py` execute everything at module level (no `if __name__ == "__main__":` guard). They cannot be imported without triggering the entire pipeline. Wrapping in `main()` functions would enable testing and reuse.

2. **Deduplicate NSM scripts** - ~80% of the code is identical between `NSM_analysis.py` and `NSM_analysis_bone_only.py` (model loading, config parsing, icp_transform handling, result saving). A shared function could handle both cases.

3. **Run NSM in-process instead of as subprocesses** - Currently each NSM analysis spawns a fresh Python process, re-importing PyTorch and re-initializing CUDA. Running them in-process (after refactoring into functions) would save significant startup time.

4. **Use `argparse` consistently** - `dosma_knee_seg.py` and `seg_thick_t2_pipeline.py` use raw `sys.argv`. Only `download_nsm_models.py` uses argparse.

5. **Config validation** - No schema validation on `config.json`. Missing or misspelled keys produce confusing KeyErrors deep in the pipeline. A validation step on startup would catch configuration issues early.

6. **No tests** - There are no test files in the repository. Unit tests for `utils.py`, config loading, and segmentation label logic would improve reliability.

7. **`sys.path.append` for BScore import** - Both NSM scripts manipulate `sys.path` at runtime to import `Bscore` from a configured folder. This is fragile. Consider making BScore a proper installable package or using `importlib`.

8. **`os.path.exists(...) == False`** in both NSM scripts - Should use idiomatic `not os.path.exists(...)`.

9. **No type hints** anywhere in the codebase.

10. **Step tracker only covers `seg_thick_t2_pipeline.py`** - The NSM steps launched by `dosma_knee_seg.py` are not tracked in `_step_log.json`, so the web UI cannot monitor NSM progress.
