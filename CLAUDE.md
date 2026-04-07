# CLAUDE.md - Knee MRI Analysis Pipeline

## What This Project Does

An automated pipeline for analyzing knee MRI scans. Given a knee MRI image (DICOM, NIfTI, or NRRD), it:

1. **Segments** bone and cartilage structures (femur, tibia, patella + their cartilage) using either DOSMA or nnU-Net deep learning models
2. **Remaps labels** from model-native to a canonical label set
3. **Creates 3D surface meshes** from the segmentations
4. **Computes cartilage thickness** metrics by region (anterior, medial/lateral weight-bearing, medial/lateral posterior femoral cartilage, plus tibial and patellar cartilage)
5. **Computes T2 relaxation maps** (only if the input is a qDESS DICOM scan with GL/TG private tags)
6. **Fits Neural Shape Models (NSM)** to the femur bone (and optionally cartilage) to get a latent shape representation
7. **Computes BScore** (osteoarthritis severity score) from the NSM latent vector

## Architecture

There are two pipeline implementations. The **modular pipeline** (`steps/`) is the active one. The **monolith** (`seg_thick_t2_pipeline.py`) is kept as a fallback until the website integration is verified.

### Modular Pipeline (active)

Each step is an independent module that works as both a Python import and a CLI entry point:

```
run_pipeline.py                    (standalone orchestrator — chains all steps)
    |
    +--> steps/segment.py          (segmentation — runs as subprocess for GPU isolation)
    +--> steps/label_remap.py      (canonical label remapping — in-process)
    +--> steps/generate_meshes.py  (mesh generation + thickness — in-process)
    +--> steps/t2_mapping.py       (T2 maps — in-process, skipped if not qDESS)
    +--> steps/run_nsm.py          (NSM fitting — each fit runs as subprocess)
    +--> steps/compute_bscore.py   (BScore from latent — in-process)
```

### Monolith (legacy, kept until website integration verified)

```
dosma_knee_seg.py  (orchestrator / entry point)
    |
    +--> seg_thick_t2_pipeline.py  (segmentation, meshing, thickness, T2)
    +--> NSM_analysis.py           (bone+cartilage NSM fitting)
    +--> NSM_analysis_bone_only.py (bone-only NSM fitting)
```

## Website / Orchestrator Integration Guide

The modular pipeline is designed for two calling patterns:

### Pattern 1: Standalone (replaces `dosma_knee_seg.py`)

```bash
python run_pipeline.py /path/to/image /path/to/output/ [model_name] [--config /path/to/config.json]
```

### Pattern 2: Website orchestrator (step-by-step control)

Each step can be called individually as a subprocess, giving the orchestrator per-step progress tracking, error handling, and the ability to skip/rerun steps.

**CLI interface** (every step follows this contract):
```bash
python -m steps.<module> <working_dir> [--options '<json>'] [--config '<path>']
```

- `working_dir`: directory containing inputs from prior steps + original input
- `--options`: JSON string with step-specific parameters
- `--config`: path to pipeline `config.json`
- **stdout**: progress lines (`[PROGRESS] <percent>% <message>`) followed by a JSON result as the last line
- **exit code**: 0 on success, non-zero on failure

**Python interface** (if calling in-process):
```python
from steps.<module> import run
result = run(working_dir, options=dict, config=dict)
```

**Step-by-step orchestration example:**

```python
import json
import subprocess
import sys

working_dir = "/path/to/job/directory"  # contains the input image
config_path = "/path/to/config.json"

# Step 1: Segmentation (MUST run as subprocess — TF grabs GPU memory)
result = subprocess.run(
    [sys.executable, "-m", "steps.segment", working_dir,
     "--options", json.dumps({"model": "nnunet_knee"}),
     "--config", config_path],
    capture_output=True, text=True, timeout=600,
    cwd="/path/to/kneepipeline",
)
seg_result = json.loads(result.stdout.strip().split("\n")[-1])
# seg_result = {"seg_path": "...", "is_qdess": false, "filename_prefix": "...", "model_name": "..."}

# Step 2: Label remapping
DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}
result = subprocess.run(
    [sys.executable, "-m", "steps.label_remap", working_dir,
     "--options", json.dumps({"remap_table": DOSMA_REMAP}),
     "--config", config_path],
    capture_output=True, text=True,
    cwd="/path/to/kneepipeline",
)

# Step 3: Mesh generation + cartilage thickness
result = subprocess.run(
    [sys.executable, "-m", "steps.generate_meshes", working_dir,
     "--options", json.dumps({"compute_thickness": True}),
     "--config", config_path],
    capture_output=True, text=True, timeout=600,
    cwd="/path/to/kneepipeline",
)

# Step 4: T2 mapping (only if qDESS)
if seg_result["is_qdess"]:
    result = subprocess.run(
        [sys.executable, "-m", "steps.t2_mapping", working_dir,
         "--config", config_path],
        capture_output=True, text=True, timeout=300,
        cwd="/path/to/kneepipeline",
    )

# Step 5: NSM fitting (each fit is internally a subprocess for GPU isolation)
result = subprocess.run(
    [sys.executable, "-m", "steps.run_nsm", working_dir,
     "--options", json.dumps({"nsm_type": "both"}),
     "--config", config_path],
    capture_output=True, text=True, timeout=600,
    cwd="/path/to/kneepipeline",
)

# Step 6: BScore
result = subprocess.run(
    [sys.executable, "-m", "steps.compute_bscore", working_dir,
     "--options", json.dumps({"bscore_type": "both"}),
     "--config", config_path],
    capture_output=True, text=True,
    cwd="/path/to/kneepipeline",
)
```

**Important notes for website integration:**
- Segmentation **must** run as a subprocess (TF grabs all GPU memory and doesn't release it)
- NSM fitting internally runs each fit as a subprocess (for CUDA state isolation and reproducibility)
- The `working_dir` must contain the input image (or a symlink to it)
- All step outputs are written into `working_dir`
- The `KNEEPIPELINE_CONFIG` env var is respected by all steps as a fallback config path
- For `LD_LIBRARY_PATH`: set to `$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` so TF can find CUDA 11.x

**Step options reference:**

| Step | Key Options |
|------|-------------|
| `segment` | `model`: model name (default from config), `batch_size`: int |
| `label_remap` | `remap_table`: `{native_int: canonical_int}` dict |
| `generate_meshes` | `compute_thickness`: bool (default True), `cartilage_smoothing`: float (default 0.3125) |
| `t2_mapping` | none currently |
| `run_nsm` | `nsm_type`: "bone_and_cart", "bone_only", or "both"; `nsm_bones`: list (default ["femur"]) |
| `compute_bscore` | `bscore_type`: "bone_and_cart", "bone_only", or "both"; `bscore_bones`: list |

**Output files produced per step:**

| Step | Files Written to working_dir |
|------|------------------------------|
| `segment` | `{prefix}_all-labels.nii.gz`, `{prefix}_all-labels.nrrd` |
| `label_remap` | Overwrites `*_all-labels.*` in-place; backs up to `*-native.nii.gz` |
| `generate_meshes` | `{prefix}_subregions-labels.{nii.gz,nrrd}`, `{bone}_mesh.vtk`, `{bone}_cart_{idx}_mesh.vtk`, `femur_mesh_raw.vtk`, `{prefix}_thickness_results.{csv,json}` |
| `t2_mapping` | `{prefix}_t2map.{nii.gz,nrrd}`, `{prefix}_depth_seg.nrrd`, `{prefix}_t2_results.json` |
| `run_nsm` | `femur_mesh_NSM_orig.vtk`, `fem_cart_mesh_NSM_orig.vtk`, `NSM_recon_*.vtk`, `NSM_recon_params.json`, `NSM_bone_only_recon_*.vtk`, `NSM_bone_only_recon_params.json` |
| `compute_bscore` | `bscore_results.json` |

## Canonical Label Set

The modular pipeline uses canonical labels. After segmentation, `label_remap` converts model-native labels to canonical. All downstream steps use canonical labels.

| Index | Structure | DOSMA Native Index |
|-------|-----------|--------------------|
| 0 | Background | 0 |
| 1 | Femur bone | 7 |
| 2 | Tibia bone | 8 |
| 3 | Patella bone | 9 |
| 4 | Femoral cartilage | 2 |
| 5 | Medial tibial cartilage | 3 |
| 6 | Lateral tibial cartilage | 4 |
| 7 | Patellar cartilage | 1 |
| 8 | Medial meniscus | — |
| 9 | Lateral meniscus | — |
| 11-15 | Femur cartilage subregions | 11-15 (generated by pymskt) |

The DOSMA-native to canonical remap table: `{1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}`

The monolith uses DOSMA-native labels throughout (no remapping step).

## File Descriptions

### Modular Pipeline (`steps/`)

- **`steps/__init__.py`** — Package init.

- **`steps/_common.py`** — Shared helpers: `parse_step_args()` (CLI arg parsing), `load_config()` (config resolution: explicit path → `KNEEPIPELINE_CONFIG` env → local `config.json`), `load_segmentation()`, `load_subregions()`, `emit_progress()`, `find_file()`.

- **`steps/segment.py`** — Image loading (DICOM/NIfTI/NRRD) and segmentation dispatch (DOSMA or nnU-Net). Outputs model-native labels. **Important**: `MedicalVolume.from_sitk()` creates a zero-copy view — the volume must own its data before the sitk image goes out of scope (see `_volume.copy()` fix).

- **`steps/label_remap.py`** — Pure numpy label remapping. Backs up native labels before overwriting. No-op if no remap table provided.

- **`steps/generate_meshes.py`** — Mesh generation using pymskt with canonical labels. Computes femur subregions, cartilage thickness per region, saves raw femur mesh for NSM. Uses `BONE_CONFIG` dict with canonical label indices.

- **`steps/t2_mapping.py`** — T2 relaxation maps from qDESS DICOM. Global and depth-dependent T2 metrics per region. Soft dependency on meshes (depth-dependent T2 skipped if meshes unavailable).

- **`steps/run_nsm.py`** — Unified NSM fitting (replaces both `NSM_analysis.py` and `NSM_analysis_bone_only.py`). Contains:
  - `determine_knee_side()` — parameterized with canonical label indices
  - `_load_nsm_model()` — model loading with `weights_only=True`
  - `_convert_icp_transform()` — handles all VTK types + None
  - `fit_nsm()` — the core fitting function, bone_only flag controls behavior
  - `_fit_nsm_subprocess()` — runs each fit in a fresh subprocess for CUDA state isolation
  - `_prepare_meshes()` — knee side detection, mirroring, clipping
  - **Seed ordering**: `torch.manual_seed(42)` must be set AFTER `model.cuda()`, not before. `model.cuda()` consumes CUDA random state.

- **`steps/compute_bscore.py`** — BScore from NSM latent vectors. Pure numpy, no GPU. Clears `sys.modules["Bscore"]` before import to handle different model paths.

### Standalone Entry Point

- **`run_pipeline.py`** — Chains all steps. Segmentation runs as subprocess (TF GPU isolation). NSM fits run as subprocesses (CUDA state isolation). Other steps run in-process. Uses lazy imports to avoid loading TF/torch at module level.

### Legacy Monolith (kept until website integration verified)

- **`dosma_knee_seg.py`** — Old orchestrator. Will be removed.
- **`seg_thick_t2_pipeline.py`** — Old monolithic workhorse (~577 lines). Will be removed.
- **`NSM_analysis.py`** — Old bone+cartilage NSM. Will be removed.
- **`NSM_analysis_bone_only.py`** — Old bone-only NSM. Will be removed.

### Other Files

- **`utils.py`** — `clip_femur_top()` utility. Used by `steps/run_nsm.py`.
- **`download_nsm_models.py`** — Downloads NSM models from HuggingFace.
- **`config.json`** / **`config_template.json`** — Pipeline configuration.

### Test Files

- **`tests/test_steps/`** — 37 unit tests covering all step modules.
- **`tests/integration/compare_pipelines.py`** — Compares monolith vs modular pipeline output on real data (segmentation, meshes, thickness, NSM, BScore).
- **`tests/integration/test_tf_torch_conflict.py`** — Diagnostic tests for TF/numpy/torch environment issues.

## Configuration

- **`config.json`** — Active configuration (not committed to git, derived from template). Contains:
  - `perform_bone_only_nsm` / `perform_bone_and_cart_nsm` — Flags to enable/disable NSM steps
  - `clip_femur_top` — Whether to clip superior femur before NSM
  - `default_seg_model` — Default segmentation model name
  - `batch_size` — Inference batch size
  - `models` — Paths to DOSMA `.h5` weight files
  - `nnunet` — nnU-Net configuration (type: "cascade" or "fullres")
  - `nsm` / `nsm_bone_only` — Paths to NSM model configs and state dicts
  - `bscore` / `bscore_bone_only` — Paths to BScore model folders
  - `regions` — Segmentation label index to region name mapping (DOSMA-native, used by monolith)
  - `bones` — Per-bone mesh config with DOSMA-native label indices (used by monolith)

## Key Dependencies

- **DOSMA** (`dosma`) — MRI analysis framework (bone_seg branch of gattia/DOSMA fork)
- **pymskt** / `mskt` — Musculoskeletal toolkit for mesh generation and cartilage analysis
- **NSM** — Neural Shape Model library (gattia/nsm)
- **nnunetv2** — nnU-Net segmentation framework (optional, for `nnunet_knee` model)
- **PyTorch** — Deep learning (CUDA required for NSM)
- **TensorFlow 2.11** — Required for DOSMA `.h5` models (CUDA 11.x, NumPy < 2.0, Keras < 3)
- **VTK** — Mesh I/O and transforms
- **SimpleITK** — Medical image I/O and processing

### Environment Constraints

TensorFlow 2.11 and PyTorch coexist but have conflicting CUDA requirements. TF needs CUDA 11.x; PyTorch uses CUDA 12.x. Both work via conda-installed `cudatoolkit=11.8` + `cudnn=8.2`. NumPy must be 1.24.x (TF needs < 2.0; nnunet needs >= 1.24). See README.MD for full install instructions.

**Critical for subprocess isolation**: TF grabs all GPU memory on init and never releases it. Segmentation and NSM fitting must run as separate subprocesses so GPU memory is freed between steps. Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` for TF GPU support.

## How to Run

```bash
# Modular pipeline (recommended)
python run_pipeline.py /path/to/image /path/to/output/ [model_name]
python run_pipeline.py /path/to/image /path/to/output/ --config /path/to/config.json

# Individual steps
python -m steps.segment /path/to/working_dir --options '{"model": "nnunet_knee"}' --config config.json
python -m steps.label_remap /path/to/working_dir --options '{"remap_table": {"1":7,"2":4,"3":5,"4":6,"7":1,"8":2,"9":3}}'
python -m steps.generate_meshes /path/to/working_dir --config config.json
python -m steps.run_nsm /path/to/working_dir --options '{"nsm_type": "both"}' --config config.json
python -m steps.compute_bscore /path/to/working_dir --options '{"bscore_type": "both"}' --config config.json

# Legacy monolith (still works, will be removed after website integration)
python dosma_knee_seg.py /path/to/image /path/to/output/ [model_name]

# Run tests
python -m pytest tests/test_steps/ -v          # unit tests (no GPU needed)
conda run -n kneepipeline python tests/integration/compare_pipelines.py data/anthonys_knee.nrrd --model acl_qdess_bone_july_2024 --keep-output  # integration test

# Override config location
KNEEPIPELINE_CONFIG=/path/to/custom_config.json python run_pipeline.py ...
```

## Known Issues

1. **TF GPU memory** — TensorFlow grabs all GPU memory and doesn't release it. Segmentation must run as a subprocess. If you see `Failed copying input tensor... Dst tensor is not initialized`, another process is holding GPU memory.

2. **MedicalVolume zero-copy** — `MedicalVolume.from_sitk()` creates a view, not a copy. If the SimpleITK image goes out of scope, the volume data becomes invalid (segfault or zeros). Fixed in `steps/segment.py` with `volume._volume = volume._volume.copy()`.

3. **NSM seed ordering** — `torch.manual_seed()` must be called AFTER `model.cuda()`, not before. `model.cuda()` consumes CUDA random state. Wrong ordering causes different optimization trajectories and ~0.08 BScore differences.

4. **VTK float32 precision** — VTK saves mesh points as float32. Meshes that go through a save/load roundtrip lose ~5e-9 precision per point. This causes ~0.004 BScore difference for bone+cart NSM (cartilage mesh roundtrip). Bone-only is unaffected because the bone mesh stays consistent. Not a bug — inherent VTK limitation.

5. **pymskt `fix_mesh` non-determinism** — `fix_mesh("pcu")` on cartilage meshes can produce different point counts across runs. This contributes to the ~0.004 bone+cart BScore variation between runs.

6. **Segmentation non-determinism** — TF and nnU-Net produce slightly different results across runs (1-2 voxels for DOSMA, ~84 for nnU-Net). This is GPU floating-point non-determinism, not a code bug.

7. **`sys.path.append` for BScore** — Both the old NSM scripts and `steps/compute_bscore.py` manipulate `sys.path` to import `Bscore` from a configured folder. Consider making BScore a proper installable package.

## Future Work

1. **Remove monolith** — Delete `seg_thick_t2_pipeline.py`, `dosma_knee_seg.py`, `NSM_analysis.py`, `NSM_analysis_bone_only.py` after website integration is verified working with the modular steps.

2. **Remove TF dependency** — DOSMA models use TF/Keras. Porting to PyTorch (or using ONNX) would eliminate the TF/PyTorch CUDA conflict and simplify the environment.

3. **Config validation** — No schema validation on `config.json`. A validation step on startup would catch misconfigurations early.

4. **Canonical labels in config** — The `regions` and `bones` sections of `config.json` still use DOSMA-native labels (used by the monolith). After monolith removal, update to canonical labels or remove these sections (the modular pipeline hardcodes bone config in `generate_meshes.py:BONE_CONFIG`).
