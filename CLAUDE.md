# CLAUDE.md - Knee MRI Analysis Pipeline

## What This Project Does

An automated pipeline for analyzing knee MRI scans. Given a knee MRI image (DICOM, NIfTI, or NRRD), it:

1. **Segments** bone and cartilage structures (femur, tibia, patella + their cartilage) using either DOSMA or nnU-Net deep learning models
2. **Remaps labels** from model-native to a canonical label set
3. **Creates 3D surface meshes** from the segmentations
4. **Computes cartilage thickness** metrics by region (anterior, medial/lateral weight-bearing, medial/lateral posterior femoral cartilage, plus tibial and patellar cartilage)
5. **Computes T2 relaxation maps** (only if the input is a **two-echo qDESS DICOM** scan). The GL/TG private spoiler tags are *not* required — when they are absent, as they are in most anonymised DICOM, DOSMA's Sveinsson low-spoiling equations are used instead. The step records which estimator ran as `t2_method` (`"spoiled"` / `"low_spoiling"`), because the two disagree by −1.4% at 10-20 ms rising to −5.4% at 60-80 ms and the difference is not a constant factor.
6. **Fits Neural Shape Models (NSM)** to the femur bone (and optionally cartilage) to get a latent shape representation
7. **Computes BScore** (osteoarthritis severity score) from the NSM latent vector

## Architecture

**There is one pipeline: the step modules in `steps/`.** Each step is an independent
module that works as both a Python import and a CLI entry point:

```
run_pipeline.py                    (standalone orchestrator — chains all steps)
    |
    +--> steps/segment.py          (segmentation — runs as subprocess for GPU isolation)
    +--> steps/label_remap.py      (canonical label remapping — in-process, idempotent)
    +--> steps/subregions.py       (femur cartilage subregions — pure image processing)
    +--> steps/generate_meshes.py  (mesh generation + thickness — in-process)
    +--> steps/t2_mapping.py       (T2 maps — in-process, skipped if not qDESS)
    +--> steps/run_nsm.py          (NSM fitting — each fit runs as subprocess)
    +--> steps/compute_bscore.py   (BScore from latent — in-process)
```

A single-process implementation (`seg_thick_t2_pipeline.py`, driven by
`dosma_knee_seg.py`, with `NSM_analysis.py` / `NSM_analysis_bone_only.py` for the NSM
fits) ran production until the website cut over to the steps on 2026-08-17, and was
**deleted on 2026-08-17**. Git history is the only copy; do not reintroduce a second
implementation. Carrying two cost three fixes that had to land twice, each silent if
the second copy was missed — the extensionless-DICOM `except` clause, the qDESS
spoiler-tag guard, and the cuDNN determinism env vars.

## Website / Orchestrator Integration Guide

The pipeline is designed for two calling patterns:

### Pattern 1: Standalone

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
- **stdout**: progress lines only (`[PROGRESS] <percent>% <message>`), plus free-form logging
- **the result**: written to `<working_dir>/_step_result.json` by `_common.write_step_result()`. **It is NOT the last line of stdout** — that was true once and has not been since the "Write step results to file instead of stdout JSON" commit, because parsing JSON out of a stream that also carries TensorFlow's logging was fragile. `run_pipeline.py:48` reads the file, and so must any orchestrator.
- **exit code**: 0 on success, non-zero on failure. **The exit code is not the whole answer**: a step can exit 0 having declined to do its work, and says so with `{"skipped": True, "reason": ...}` in its result. Read the result dict, not just the status. A skip from a step that had to run (`label_remap` on a model with a label map) means something went wrong.
- **`skip_steps: [names]`** in a result means a *later* step cannot apply to this input — `segment` returns `["t2_mapping"]` for anything that is not a two-echo qDESS. Not a failure, and deliberately not a warning: it fires for every NIfTI and NRRD upload, so treating it as one would tell most users their job partially failed when nothing did.

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
# The result is a FILE, not the last line of stdout. Read it and delete it, so a
# later step can never pick up its predecessor's result.
_result_path = Path(working_dir) / "_step_result.json"
seg_result = json.loads(_result_path.read_text())
_result_path.unlink()
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
| `compute_bscore` | `bscore_type`: **omit it** (or `"both"`) to score every NSM params file present — the orchestrated case, and what the directory already answers. Naming a single variant restricts scoring to it, for CLI use; if that variant's file is absent the step warns and omits the score rather than raising. `bscore_bones`: list |

**Output files produced per step:**

| Step | Files Written to working_dir |
|------|------------------------------|
| `segment` | `{prefix}_all-labels.nii.gz`, `{prefix}_all-labels.nrrd` |
| `label_remap` | Overwrites `*_all-labels.*` in-place; backs up to `*-native.nii.gz` |
| `subregions` | `{prefix}_subregions-labels.{nii.gz,nrrd}` — **moved here from `generate_meshes` (D7b)**. pymskt *replaces* femoral cartilage (canonical 4) with subregions 11-15, so 4 is absent from this file by design |
| `generate_meshes` | `{bone}_mesh.vtk`, `{bone}_cart_{idx}_mesh.vtk`, `femur_mesh_raw.vtk`, `{prefix}_thickness_results.{csv,json}` — **loads** the subregions file, lazily and only when `compute_thickness` is on |
| `t2_mapping` | `{prefix}_t2map.{nii.gz,nrrd}`, `{prefix}_depth_seg.nrrd`, `{prefix}_t2_results.json` |
| `run_nsm` | `femur_mesh_NSM_orig.vtk`, `fem_cart_mesh_NSM_orig.vtk`, `NSM_recon_*.vtk`, `NSM_recon_params.json`, `NSM_bone_only_recon_*.vtk`, `NSM_bone_only_recon_params.json` |
| `compute_bscore` | `bscore_results.json` |

## Canonical Label Set

After segmentation, `label_remap` converts model-native labels to canonical. All downstream steps use canonical labels.

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
| 8 | Medial meniscus | 5 |
| 9 | Lateral meniscus | 6 |
| 11-15 | Femur cartilage subregions | 11-15 (generated by pymskt) |

The DOSMA-native to canonical remap table: `{1: 7, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 1, 8: 2, 9: 3}`

⚠️ **Every archived job produced before the orchestrator cutover carries
DOSMA-native labels**, because the pipeline that produced them had no remapping
step. Reading one of those files with the canonical map paints every bone as
cartilage. The website's `model_registry.CANONICAL_LABELS` is the post-remap set
and is not what is on disk for those jobs.

**The menisci are real, and this table said they were not until 2026-08-15.**
Native 5 and 6 used to be listed as "—" here and were missing from
`_get_remap_table()`. `label_remap` builds its output with `np.zeros_like` and
writes only mapped values, so an absent entry does not pass a label through — it
deletes it. 181 of 190 archived `dosma_ananya` jobs carry all nine labels, so the
short table erased both menisci from almost every scan, silently and with exit 0.

**Both model families use this one scheme**, verified rather than assumed:

- nnU-Net declares it in its own trained
  `huggingface/models/Dataset500_KneeMRI/*/dataset.json`, identically for the
  `3d_fullres` and `3d_cascade_fullres` configurations, and
  `scripts/inference.py` deliberately writes those labels through unremapped.
- 237 archived segmentations across all five models contain labels 1-9.
- Laterality was checked, not inferred: on 220 scans, label 5's centroid is on
  the same side of the knee as the medial tibial cartilage every time.

⚠️ `DEPENDENCIES/nnunet_knee_inference/scripts/utils.py`'s `LABEL_NAMES` dict
disagrees — it has `8: femur_bone, 9: tibia_bone, 10: patella_bone` and a note
that "label 7 is skipped in original". It is headed *"Original label names for
reference"* and describes the upstream dataset **before** training, not the
model's output. **`dataset.json` is the authority.**

The website repo holds the same table as `model_registry.DOSMA_NATIVE_LABEL_MAP`;
its `TestLabelMapIntegrity` is what notices if the two repos drift apart.

## File Descriptions

### Pipeline steps (`steps/`)

- **`steps/__init__.py`** — Package init.

- **`steps/_common.py`** — Shared helpers: `parse_step_args()` (CLI arg parsing), `load_config()` (config resolution: explicit path → `KNEEPIPELINE_CONFIG` env → local `config.json`), `load_segmentation()`, `load_subregions()`, `find_segmentation()`, `find_subregions()`, `image_prefix()`, `emit_progress()`, `find_file()`.

  **Steps read each other's label images as `.nrrd`, via `find_image_file()`.** Both formats are still written — the `.nii.gz` is what users download — but nothing reads it back, because NIfTI's float32 affine quantises the direction cosines (Known Issue 8). New steps must go through these helpers rather than globbing `*_all-labels.nii.gz` themselves.

- **`steps/segment.py`** — Image loading (DICOM/NIfTI/NRRD) and segmentation dispatch (DOSMA or nnU-Net). Outputs model-native labels. **Important**: `MedicalVolume.from_sitk()` creates a zero-copy view — the volume must own its data before the sitk image goes out of scope (see `_volume.copy()` fix).

- **`steps/label_remap.py`** — Pure numpy label remapping. Backs up native labels before overwriting. No-op if no remap table provided.

- **`steps/generate_meshes.py`** — Mesh generation using pymskt with canonical labels. Computes femur subregions, cartilage thickness per region, saves raw femur mesh for NSM. Uses `BONE_CONFIG` dict with canonical label indices.

- **`steps/t2_mapping.py`** — T2 relaxation maps from qDESS DICOM. Global and depth-dependent T2 metrics per region. Soft dependency on meshes (depth-dependent T2 skipped if meshes unavailable).

- **`steps/run_nsm.py`** — NSM fitting, bone+cartilage and bone-only in one module (`bone_only` flag). Contains:
  - `determine_knee_side()` — parameterized with canonical label indices
  - `_load_nsm_model()` — model loading with `weights_only=True`
  - `_convert_icp_transform()` — handles all VTK types + None
  - `fit_nsm()` — the core fitting function, bone_only flag controls behavior
  - `_fit_nsm_subprocess()` — runs each fit in a fresh subprocess for CUDA state isolation
  - `_prepare_meshes()` — knee side detection, mirroring, clipping
  - **Seed ordering**: `torch.manual_seed(42)` must be set AFTER `model.cuda()`, not before. `model.cuda()` consumes CUDA random state.
  - **No thickness on the reconstruction**, deliberately: the reconstruction is a template fit, not patient anatomy. Thickness comes from `generate_meshes.py`, on the real meshes.

- **`steps/compute_bscore.py`** — BScore from NSM latent vectors. Pure numpy, no GPU. Clears `sys.modules["Bscore"]` before import to handle different model paths.

### Standalone Entry Point

- **`run_pipeline.py`** — Chains all steps. Segmentation runs as subprocess (TF GPU isolation). NSM fits run as subprocesses (CUDA state isolation). Other steps run in-process. Uses lazy imports to avoid loading TF/torch at module level.

### Other Files

- **`utils.py`** — `clip_femur_top()` utility. Used by `steps/run_nsm.py`.
- **`download_nsm_models.py`** — Downloads NSM models from HuggingFace.
- **`config.json`** / **`config_template.json`** — Pipeline configuration.

### Test Files

- **`tests/test_steps/`** — 124 unit tests covering all step modules. No GPU needed. This is the suite that must stay green.
- **`tests/integration/run_pipeline_test.py`** — Runs the steps on real data and validates the outputs. Driven by `tests/integration/run_all.sh`.
- **`tests/integration/test_tf_torch_conflict.py`** — Diagnostic tests for TF/numpy/torch environment issues.
- **`tests/integration/test_seg_quick.py`** — Smoke test: does DOSMA segmentation run at all.

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

  ⚠️ **`regions` and `bones` were removed on 2026-08-17** along with the pipeline that
  read them. They held DOSMA-native label indices and region names, and nothing in
  `steps/` or `run_pipeline.py` ever read them — the equivalents live in
  `generate_meshes.BONE_CONFIG` and `generate_meshes.CANONICAL_REGION_NAMES`, in
  canonical labels. `config.json` is generated per job by the website from this base
  file (`config_generator.generate_pipeline_config()` copies it and overrides named
  keys only), so re-adding them would just put dead DOSMA-native data back into every
  job config.

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
# Whole pipeline
python run_pipeline.py /path/to/image /path/to/output/ [model_name]
python run_pipeline.py /path/to/image /path/to/output/ --config /path/to/config.json

# Individual steps
python -m steps.segment /path/to/working_dir --options '{"model": "nnunet_knee"}' --config config.json
python -m steps.label_remap /path/to/working_dir --options '{"remap_table": {"1":7,"2":4,"3":5,"4":6,"5":8,"6":9,"7":1,"8":2,"9":3}}'
python -m steps.subregions /path/to/working_dir --config config.json
python -m steps.generate_meshes /path/to/working_dir --config config.json
python -m steps.t2_mapping /path/to/working_dir --config config.json
python -m steps.run_nsm /path/to/working_dir --options '{"nsm_type": "both"}' --config config.json
python -m steps.compute_bscore /path/to/working_dir --options '{"bscore_type": "both"}' --config config.json

# Run tests
python -m pytest tests/test_steps/ -v          # unit tests (no GPU needed), 124 passing
./tests/integration/run_all.sh                 # integration, needs GPU + data/

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

7. **`sys.path.append` for BScore** — `steps/compute_bscore.py` manipulates `sys.path` to import `Bscore` from a configured folder, and clears `sys.modules["Bscore"]` first so a second model folder is not shadowed by the first. Consider making BScore a proper installable package.

8. **NIfTI float32 affine — fixed 2026-08-16, do not undo** — NIfTI stores the image affine as float32, so a `.nii.gz` write→read moves the direction cosines by **1.354e-08** (measured on `data/anthonys_knee.nrrd`; NRRD roundtrips it at **0.0**). That is not cosmetic: marching-cubes vertices move ~1.5e-05 mm, pyacvd (deterministic in itself) lands on a different clustering — verified on a real segmentation, tibia vertices then differ by **77 mm** and vertex counts can change — and regional thickness shifts 0.007-0.02 mm. It was the whole reason the step-based pipeline disagreed with its single-process predecessor on `med_tib_cart_mm_mean` — that one held a single image in memory for the entire run and never read one back, so it never had a handoff to lose precision at. **Inter-step reads therefore go through the `.nrrd`** (`_common.find_image_file`). Reading a label image with a bare `sitk.ReadImage(...nii.gz)` in a new step reintroduces it. Pinned by `tests/test_steps/test_image_precision.py`.

## Future Work

1. **Remove TF dependency** — DOSMA models use TF/Keras. Porting to PyTorch (or using ONNX) would eliminate the TF/PyTorch CUDA conflict and simplify the environment.

2. **Config validation** — No schema validation on `config.json`. A validation step on startup would catch misconfigurations early.

3. **`run_pipeline.py` is a second orchestrator and nothing watches it** — the website has its own. It must stay thin, and it has been the operative risk three times: a missing `subregions` step after the D7b split, an unchecked `label_remap` skip, and `_get_remap_table()` being D1 living in this repo. Anything it learns that the website already knows is a place the two can drift.

*(Done 2026-08-17: the single-process pipeline — `seg_thick_t2_pipeline.py`,
`dosma_knee_seg.py`, `NSM_analysis.py`, `NSM_analysis_bone_only.py` — and the
`regions`/`bones` config sections it alone read.)*
