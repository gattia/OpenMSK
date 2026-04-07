# Plan A.3a: Step Entry Points

**Parent plan**: `kneepipeline_segmentaton_website/.claude/plans/extensible_pipeline_architecture.md`
**Companion**: A.3b (pipeline orchestrator, in the website repo) depends on this plan.

## Goal

Replace the monolithic pipeline with independent, composable step modules.
Each step is both a Python import (`from steps.X import run`) and a CLI
entry point (`python -m steps.X <working_dir>`).

Two calling patterns:
- **Standalone**: `run_pipeline.py` chains all steps in-process (replaces
  `dosma_knee_seg.py`)
- **Website orchestrator** (A.3b): calls each step individually as a
  subprocess, with per-step progress tracking and the ability to skip/rerun

Old scripts (`NSM_analysis.py`, `NSM_analysis_bone_only.py`,
`dosma_knee_seg.py`) are **replaced**, not preserved. `seg_thick_t2_pipeline.py`
is kept as reference during extraction, then removed once all steps pass
integration tests.

## Prerequisites (verify before starting)

- [ ] Working directory is `~/programming/kneepipeline/` (this repo)
- [ ] **A.1 is complete** in the website repo: `model_registry.py` exists with
  `CANONICAL_LABELS` and `MODEL_REGISTRY` with `label_map` entries
- [ ] Python environment has: `SimpleITK`, `numpy`, `pymskt`, `dosma`, `torch`

**If A.1 is not complete, stop.** The canonical label set and label_map
definitions come from A.1.

---

## Canonical Label Set (from A.1)

All downstream post-processing uses these labels. Remapping from model-native
labels happens as a dedicated step immediately after segmentation.

```python
CANONICAL_LABELS = {
    "background":      0,
    "femur_bone":      1,
    "tibia_bone":      2,
    "patella_bone":    3,
    "femur_cart":      4,
    "tibia_cart_med":  5,
    "tibia_cart_lat":  6,
    "patella_cart":    7,
    "meniscus_med":    8,
    "meniscus_lat":    9,
}
```

### Current model label schemes (for building remap tables)

**DOSMA models** output (used by `seg_thick_t2_pipeline.py` lines 351-425):
```
1=patella_cart, 2=femur_cart, 3=med_tib_cart, 4=lat_tib_cart,
7=femur_bone, 8=tibia_bone, 9=patella_bone
```

Remap table (DOSMA native -> canonical):
```python
DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}
```

**nnU-Net** outputs: Uses the same label indices as DOSMA after the
post-processing in `segment_image_nnunet()` (lines 164-250). Bone indices
[7, 8, 9] are referenced explicitly at line 233. **Verify during
implementation** by running nnU-Net inference and inspecting output labels.

**Note**: All models currently have `label_map: None` in the website repo's
`MODEL_REGISTRY`, meaning remap tables still need to be finalized. This is
an implementation task for the `label_remap` step -- run each model, inspect
outputs, build exact remap dicts, then update both this step and the website
repo's registry.

---

## Entry Point Contract

Every module in `steps/` is both a CLI entry point and a Python import:

```python
# CLI:
#   python -m steps.<module> <working_dir> [--options '<json>'] [--config '<path>']
#
# Python:
#   from steps.<module> import run
#   result = run(working_dir, options=dict, config=dict)
#
# Rules:
#   - working_dir contains outputs from prior steps + original input
#   - --options: JSON string of step-specific parameters
#   - --config: path to pipeline config.json (model weights, paths, region maps)
#   - Exit 0 on success, non-zero on failure
#   - Writes outputs into working_dir
#   - Stderr for errors
#   - Stdout: progress via "[PROGRESS] <percent>% <message>"
#   - On success: prints result JSON as last stdout line

def run(working_dir: Path, options: dict = None, config: dict = None) -> dict:
    """Execute this step. Returns result dict with output paths and metrics."""
    ...

if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    json.dump(result, sys.stdout)
```

---

## New Directory Structure

```
kneepipeline/
├── run_pipeline.py                  # NEW — standalone entry point, chains all steps
├── utils.py                          # PRESERVED
├── config.json                       # PRESERVED
├── steps/                            # NEW — modular entry points
│   ├── __init__.py
│   ├── _common.py                    # Shared: arg parsing, I/O, progress
│   ├── segment.py                    # Segmentation wrapper
│   ├── label_remap.py                # Canonical label remapping
│   ├── generate_meshes.py            # Mesh generation + cartilage thickness
│   ├── t2_mapping.py                 # T2 relaxation maps
│   ├── run_nsm.py                    # NSM mesh prep + fitting (unified)
│   └── compute_bscore.py            # BScore from NSM latents
├── tests/
│   └── test_steps/                   # NEW — per-step tests
│       ├── conftest.py               # Fixtures: synthetic segmentation outputs
│       ├── test_segment.py
│       ├── test_label_remap.py
│       ├── test_generate_meshes.py
│       ├── test_t2_mapping.py
│       ├── test_run_nsm.py
│       └── test_compute_bscore.py
│
│  # OLD FILES — removed after integration tests pass
├── seg_thick_t2_pipeline.py          # REFERENCE during extraction, then REMOVED
├── dosma_knee_seg.py                 # REPLACED by run_pipeline.py, then REMOVED
├── NSM_analysis.py                   # REPLACED by steps/run_nsm.py, then REMOVED
└── NSM_analysis_bone_only.py         # REPLACED by steps/run_nsm.py, then REMOVED
```

---

## Standalone Entry Point: `run_pipeline.py`

Replaces `dosma_knee_seg.py`. Chains all steps in-process.

```python
"""Run the full knee MRI analysis pipeline."""
import argparse
from pathlib import Path
from steps._common import load_config
from steps.segment import run as segment
from steps.label_remap import run as label_remap
from steps.generate_meshes import run as generate_meshes
from steps.t2_mapping import run as t2_mapping
from steps.run_nsm import run as run_nsm
from steps.compute_bscore import run as compute_bscore

def run_all(working_dir: Path, model_name: str = None, config: dict = None):
    seg_result = segment(working_dir, options={"model": model_name}, config=config)

    remap_table = _get_remap_table(seg_result["model_name"], config)
    if remap_table:
        label_remap(working_dir, options={"remap_table": remap_table}, config=config)

    mesh_result = generate_meshes(working_dir, config=config)

    if seg_result["is_qdess"]:
        t2_mapping(working_dir, config=config)

    if config.get("perform_bone_and_cart_nsm") or config.get("perform_bone_only_nsm"):
        nsm_type = "both" if (config.get("perform_bone_and_cart_nsm")
                              and config.get("perform_bone_only_nsm")) \
                   else "bone_and_cart" if config.get("perform_bone_and_cart_nsm") \
                   else "bone_only"
        run_nsm(working_dir, options={"nsm_type": nsm_type}, config=config)
        compute_bscore(working_dir, options={"bscore_type": nsm_type}, config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knee MRI Analysis Pipeline")
    parser.add_argument("path_image", help="Path to input MRI")
    parser.add_argument("path_save", help="Output directory")
    parser.add_argument("model_name", nargs="?", default=None)
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    run_all(Path(args.path_save), model_name=args.model_name, config=config)
```

---

## Shared Utilities: `steps/_common.py`

```python
"""Shared helpers for all step entry points."""

def parse_step_args() -> argparse.Namespace:
    """Standard arg parsing: working_dir, --options (JSON), --config (path)."""

def load_config(config_path: Path = None) -> dict:
    """Load pipeline config.json. Checks KNEEPIPELINE_CONFIG env var, then
    falls back to config.json in the kneepipeline directory."""

def load_segmentation(working_dir: Path) -> sitk.Image:
    """Load *_all-labels.nii.gz from working_dir. Raises if not found."""

def load_subregions(working_dir: Path) -> sitk.Image:
    """Load *_subregions-labels.nii.gz from working_dir. Raises if not found."""

def emit_progress(percent: int, message: str):
    """Print [PROGRESS] line to stdout for orchestrator consumption."""
    print(f"[PROGRESS] {percent}% {message}", flush=True)

def find_file(working_dir: Path, pattern: str) -> Path:
    """Glob for a single file matching pattern. Raises if 0 or >1 matches."""
```

---

## Step Specifications

### Step 1: `steps/segment.py`

Thin wrapper around existing segmentation code. Does NOT remap labels -- that's
a separate step so the orchestrator can control remapping.

**Inputs**: Raw MRI file (DICOM dir, NIfTI, NRRD) in `working_dir`
**Outputs**: `*_all-labels.nii.gz`, `*_all-labels.nrrd` (model-native labels)
**Options**: `{"model": "acl_qdess_bone_july_2024", "batch_size": 128}`

```python
def run(working_dir, options=None, config=None):
    model_name = options.get("model", config["default_seg_model"])
    batch_size = options.get("batch_size", config.get("batch_size", 64))

    input_path = _find_input_image(working_dir)

    # Reuse existing image loading logic from seg_thick_t2_pipeline.py lines 284-323
    # (DICOM qDESS detection -> generic DICOM fallback -> NIfTI -> NRRD -> single DCM)
    volume, is_qdess, filename_prefix = _load_image(input_path, working_dir)

    emit_progress(10, "Loading segmentation model")

    # Dispatch to existing functions (extracted into this module, not imported
    # from the monolith)
    if model_name.startswith("nnunet"):
        sitk_seg = segment_image_nnunet(volume, model_name, config)
    else:
        config["batch_size"] = batch_size
        sitk_seg = segment_image_dosma(volume, model_name, config)

    emit_progress(80, "Saving segmentation")

    seg_path_nii = working_dir / f"{filename_prefix}_all-labels.nii.gz"
    seg_path_nrrd = working_dir / f"{filename_prefix}_all-labels.nrrd"
    sitk.WriteImage(sitk_seg, str(seg_path_nii))
    sitk.WriteImage(sitk_seg, str(seg_path_nrrd))

    torch.cuda.empty_cache()

    return {
        "seg_path": str(seg_path_nii),
        "is_qdess": is_qdess,
        "filename_prefix": filename_prefix,
        "model_name": model_name,
    }
```

**Key decisions**:
- Image loading logic and segmentation functions are extracted from
  `seg_thick_t2_pipeline.py` into this module directly -- not imported from
  the monolith.
- The `is_qdess` flag is returned so the orchestrator can decide whether to
  run T2 mapping.
- GPU memory freed after segmentation to give headroom to later steps.

---

### Step 2: `steps/label_remap.py`

Remaps model-native segmentation labels to the canonical label set.

**Inputs**: `*_all-labels.nii.gz` in working_dir (model-native labels)
**Outputs**: Overwrites in-place with canonical labels; backup to `*_all-labels-native.nii.gz`
**Options**: `{"remap_table": {1: 7, 2: 4, ...}}` (native_int -> canonical_int)

The orchestrator converts the website repo's `label_map` format (`{int: str}`)
to `{int: int}` before passing: `{k: CANONICAL_LABELS[v] for k, v in label_map.items()}`

```python
def run(working_dir, options=None, config=None):
    remap_table = options["remap_table"]

    seg_path = find_file(working_dir, "*_all-labels.nii.gz")

    # Backup native labels
    native_backup = seg_path.with_name(seg_path.name.replace(".nii.gz", "-native.nii.gz"))
    shutil.copy2(seg_path, native_backup)

    # Remap
    sitk_seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(sitk_seg)
    remapped = np.zeros_like(arr)
    for src, dst in remap_table.items():
        remapped[arr == int(src)] = int(dst)

    sitk_remapped = sitk.GetImageFromArray(remapped)
    sitk_remapped.CopyInformation(sitk_seg)
    sitk.WriteImage(sitk_remapped, str(seg_path))

    # Also update the .nrrd copy
    nrrd_path = find_file(working_dir, "*_all-labels.nrrd")
    sitk.WriteImage(sitk_remapped, str(nrrd_path))

    return {"remapped": True, "native_backup": str(native_backup)}
```

**Key decisions**:
- Remap is a separate step (not baked into segmentation) so that:
  - The orchestrator can inspect native labels for debugging
  - New models only need a remap table entry, not code changes
  - Source of truth stays in the website repo's `model_registry.py`
- Native labels are backed up to avoid data loss
- If `remap_table` is None/empty, this step is a no-op (identity remap)

---

### Step 3: `steps/generate_meshes.py`

Mesh generation, cartilage thickness, and region assignment. Extracts the
largest chunk of `seg_thick_t2_pipeline.py` (lines 346-428).

**Inputs**: Canonical-label `*_all-labels.nii.gz`
**Outputs**: `*_subregions-labels.nii.gz`, `*_subregions-labels.nrrd`,
`{bone}_mesh.vtk`, `{bone}_cart_{idx}_mesh.vtk`, thickness metrics
**Options**: `{"compute_thickness": true, "cartilage_smoothing": 0.4}`

```python
# Bone config using CANONICAL labels (not DOSMA-native labels).
BONE_CONFIG = {
    "femur": {
        "tissue_idx": 1,        # CANONICAL femur_bone
        "list_cart_labels": [4], # CANONICAL femur_cart
        "n_points": 20000,
        "crop_percent": 0.8,
    },
    "tibia": {
        "tissue_idx": 2,        # CANONICAL tibia_bone
        "list_cart_labels": [5, 6], # CANONICAL tibia_cart_med, tibia_cart_lat
        "n_points": 20000,
        "crop_percent": 0.8,
    },
    "patella": {
        "tissue_idx": 3,        # CANONICAL patella_bone
        "list_cart_labels": [7], # CANONICAL patella_cart
        "n_points": 10000,
        "crop_percent": None,   # patella: never crop
    },
}

def run(working_dir, options=None, config=None):
    compute_thickness = options.get("compute_thickness", True)
    cartilage_smoothing = options.get("cartilage_smoothing", 0.4)

    emit_progress(0, "Loading segmentation")
    sitk_seg = load_segmentation(working_dir)

    # Generate femur subregions (pymskt)
    emit_progress(10, "Computing femur subregions")
    sitk_seg_subregions = mskt.image.cartilage_processing \
        .get_knee_segmentation_with_femur_subregions(
            sitk_seg,
            fem_cart_label_idx=4,       # CANONICAL femur_cart
            wb_region_percent_dist=0.6,
            femur_label=1,              # CANONICAL femur_bone
            tibia_label=2,              # CANONICAL tibia_bone
            patella_label=3,            # CANONICAL patella_bone
            med_tib_cart_label_idx=5,   # CANONICAL tibia_cart_med
            lat_tib_cart_label_idx=6,   # CANONICAL tibia_cart_lat
            patella_cart_label_idx=7,   # CANONICAL patella_cart
        )
    # Note: subregion labels 11-15 are generated by pymskt internally.
    # These are NOT canonical labels -- they're femur cartilage subregion
    # labels used for regional thickness analysis.

    # Save subregion segmentation
    # ... save .nii.gz and .nrrd

    # Per-bone mesh generation (mirrors lines 373-424)
    dict_results = {}
    for i, (bone_name, bone_config) in enumerate(BONE_CONFIG.items()):
        pct = 20 + int(60 * i / len(BONE_CONFIG))
        emit_progress(pct, f"Generating {bone_name} mesh")

        bone_mesh = mskt.mesh.BoneMesh(
            seg_image=sitk_seg,
            label_idx=bone_config["tissue_idx"],
            list_cartilage_labels=bone_config["list_cart_labels"],
            bone=bone_name,
            crop_percent=bone_config["crop_percent"],
        )
        bone_mesh.create_mesh(smooth_image_var=0.5)

        # Save raw mesh before pyacvd resampling (needed for NSM)
        if bone_name == "femur":
            raw_mesh = bone_mesh.copy()
            raw_mesh.save_mesh(str(working_dir / "femur_mesh_raw.vtk"))

        bone_mesh.resample_surface(clusters=bone_config["n_points"])
        bone_mesh.fix_mesh()

        if compute_thickness:
            bone_mesh.calc_cartilage_thickness(
                image_smooth_var_cart=cartilage_smoothing
            )
            bone_mesh.seg_image = sitk_seg_subregions
            for cart_mesh in bone_mesh.list_cartilage_meshes:
                cart_mesh.fix_mesh()
            bone_mesh.assign_cartilage_regions()

        # Save meshes
        bone_mesh.save_mesh(str(working_dir / f"{bone_name}_mesh.vtk"))
        if compute_thickness:
            for idx, cart_mesh in enumerate(bone_mesh.list_cartilage_meshes):
                cart_mesh.save_mesh(
                    str(working_dir / f"{bone_name}_cart_{idx}_mesh.vtk")
                )
            # Collect thickness metrics per region
            # ... region_name_mm_{mean,std,median}

    return {
        "metrics": dict_results,
        "bones_processed": list(BONE_CONFIG.keys()),
        "thickness_computed": compute_thickness,
    }
```

**Key decisions**:
- `BONE_CONFIG` uses CANONICAL label indices (not DOSMA-native). The current
  `seg_thick_t2_pipeline.py` uses DOSMA-native labels (femur=7, fem_cart=2)
  because it runs without remapping. This step runs AFTER `label_remap`.
- `cartilage_smoothing` maps to the existing `image_smooth_var_cart` parameter.
  The current monolithic script uses 0.3125 (line 392); the website's step
  registry default is 0.4, which is the correct value to use going forward.
- The raw (pre-pyacvd) femur mesh is saved as `femur_mesh_raw.vtk` for NSM
  fitting. Currently stored in-memory via `dict_bones['femur']['raw_mesh']`
  (line 385). Writing to disk is needed because NSM may run as a separate
  subprocess for GPU memory isolation.
- Subregion computation (femur subregions 11-15) stays in this step since
  it's needed for cartilage region assignment on the meshes.
- The `get_knee_segmentation_with_femur_subregions()` call needs updated
  parameter names -- the current code (line 351) uses `med_tibia_label` and
  `lat_tibia_label` kwargs that correspond to medial/lateral tibial cartilage
  labels. **Check the actual pymskt function signature during implementation.**

---

### Step 4: `steps/t2_mapping.py`

T2 relaxation mapping from qDESS DICOM input. Extracts from
`seg_thick_t2_pipeline.py` lines 433-503.

**Inputs**: Original DICOM directory, canonical-label segmentation, subregion
segmentation, bone meshes
**Outputs**: `*_t2map.nii.gz`, `*_t2map.nrrd`, `*_depth_seg.nrrd`, T2 metrics
**Options**: `{}` (no user-configurable options currently)

**Precondition**: Input must be qDESS DICOM. The orchestrator checks the
`is_qdess` flag from segmentation and skips this step if false.

```python
def run(working_dir, options=None, config=None):
    emit_progress(0, "Loading qDESS data")
    dicom_dir = _find_dicom_dir(working_dir)
    qdess = QDess.from_dicom(dicom_dir)

    emit_progress(20, "Computing T2 map")
    cart = FemoralCartilage()
    t2map = qdess.generate_t2_map(cart, suppress_fat=False, suppress_fluid=False)
    sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation='sagittal')

    # Clip to valid range [0, 80] ms (lines 465-466)
    # ...

    # Global T2 metrics per region (lines 468-472)
    emit_progress(50, "Computing T2 statistics")
    sitk_seg_subregions = load_subregions(working_dir)
    # ... per-region T2 mean/std/median

    # Depth-dependent T2 (requires bone meshes, lines 475-497)
    emit_progress(70, "Computing depth-dependent T2")
    # Soft dependency: if meshes weren't generated, skip depth-dependent T2
    # but still return global T2 metrics
    # ...

    return {"metrics": dict_results, "has_depth_dependent": bool}
```

**Key decisions**:
- qDESS detection happens at segmentation time -- the `is_qdess` flag from the
  segmentation result tells the orchestrator whether to run this step.
- **Soft dependency on meshes**: Depth-dependent T2 requires bone meshes from
  the mesh generation step. If meshes weren't generated, depth-dependent T2
  is skipped (global T2 metrics still computed). This is an internal check,
  NOT a registry-level dependency -- T2 mapping's `depends_on` is `[]` in
  the step registry because it produces useful output without meshes.
- Needs original DICOM files in the working directory. The orchestrator must
  preserve DICOM input (currently handled by the website's `file_handler.py`
  extracting zip contents into the job directory).
- Region label indices must use CANONICAL labels for the subregion lookup.
  The current code (lines 468-472) iterates `dict_regions['cart']` which uses
  DOSMA-native indices. This step must use the canonical subregion indices
  (or read them from the subregion segmentation directly).

---

### Step 5: `steps/run_nsm.py`

Unified NSM fitting. Replaces both `NSM_analysis.py` and
`NSM_analysis_bone_only.py` with a single module containing proper functions.

**Inputs**: `femur_mesh_raw.vtk` (pre-pyacvd), `femur_cart_0_mesh.vtk` from mesh generation
**Outputs**: `femur_mesh_NSM_orig.vtk`, `fem_cart_mesh_NSM_orig.vtk`,
`NSM_recon_*.vtk`, `NSM_recon_params.json` (and/or bone-only variants)
**Options**: `{"nsm_type": "bone_and_cart", "nsm_bones": ["femur"]}`

```python
def _load_nsm_model(config, bone_only=False):
    """Load NSM model from config. Returns (model, model_config)."""
    key = "nsm_bone_only" if bone_only else "nsm"
    path_model_config = config[key]["path_model_config"]
    path_model_state = config[key]["path_model_state"]

    with open(path_model_config) as f:
        model_config = json.load(f)

    params = {
        "latent_dim": model_config["latent_size"],
        "n_objects": model_config["objects_per_decoder"],
        "conv_hidden_dims": model_config["conv_hidden_dims"],
        "conv_deep_image_size": model_config["conv_deep_image_size"],
        "conv_norm": model_config["conv_norm"],
        "conv_norm_type": model_config["conv_norm_type"],
        "conv_start_with_mlp": model_config["conv_start_with_mlp"],
        "sdf_latent_size": model_config["sdf_latent_size"],
        "sdf_hidden_dims": model_config["sdf_hidden_dims"],
        "sdf_weight_norm": model_config["weight_norm"],
        "sdf_final_activation": model_config["final_activation"],
        "sdf_activation": model_config["activation"],
        "sdf_dropout_prob": model_config["dropout_prob"],
        "sum_sdf_features": model_config["sum_conv_output_features"],
        "conv_pred_sdf": model_config["conv_pred_sdf"],
    }

    model = TriplanarDecoder(**params)
    saved_state = torch.load(path_model_state, weights_only=True)
    model.load_state_dict(saved_state["model"])
    model = model.cuda()
    model.eval()
    return model, model_config


def _convert_icp_transform(icp_transform):
    """Convert ICP transform to numpy array, handling all VTK types + None."""
    if isinstance(icp_transform, (vtk.vtkIterativeClosestPointTransform,
                                   vtk.vtkTransform)):
        return get_linear_transform_matrix(icp_transform)
    elif isinstance(icp_transform, np.ndarray):
        return icp_transform
    elif isinstance(icp_transform, vtk.vtkMatrix4x4):
        matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                matrix[i, j] = icp_transform.GetElement(i, j)
        return matrix
    elif icp_transform is None:
        return np.eye(4)
    else:
        raise ValueError(f"icp_transform not a valid type: {type(icp_transform)}")


def fit_nsm(mesh_paths, save_dir, config, bone_only=False, calc_assd=True):
    """
    Fit NSM model to mesh(es). Unified replacement for NSM_analysis.py
    and NSM_analysis_bone_only.py.

    Args:
        mesh_paths: List of mesh file paths. [bone] for bone_only,
                    [bone, cartilage] for bone+cart.
        save_dir: Directory to save results.
        config: Pipeline config dict (already loaded).
        bone_only: If True, use bone-only model config.
        calc_assd: If True, compute ASSD metrics.

    Returns:
        dict with latent, icp_transform, center, scale, assd metrics.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    model, model_config = _load_nsm_model(config, bone_only)

    mesh_result = reconstruct_mesh(
        path=mesh_paths,
        decoders=model,
        latent_size=model_config["latent_size"],
        num_iterations=model_config["num_iterations_recon"],
        l2reg=model_config["l2reg_recon"],
        latent_reg_weight=model_config["l2reg_recon"],
        loss_type="l1",
        lr=model_config["lr_recon"],
        lr_update_factor=model_config["lr_update_factor_recon"],
        n_lr_updates=model_config["n_lr_updates_recon"],
        return_latent=True,
        register_similarity=True,
        scale_jointly=model_config["scale_jointly"],
        scale_all_meshes=True,
        objects_per_decoder=model_config["objects_per_decoder"],
        batch_size_latent_recon=model_config["batch_size_latent_recon"],
        get_rand_pts=model_config["get_rand_pts_recon"],
        n_pts_random=model_config["n_pts_random_recon"],
        sigma_rand_pts=model_config["sigma_rand_pts_recon"],
        n_samples_latent_recon=model_config["n_samples_latent_recon"],
        calc_assd=calc_assd,
        convergence=model_config["convergence_type_recon"],
        convergence_patience=model_config["convergence_patience_recon"],
        clamp_dist=model_config["clamp_dist_recon"],
        fix_mesh=model_config["fix_mesh_recon"],
        verbose=True,
        return_registration_params=True,
    )

    # Save reconstructed meshes
    os.makedirs(save_dir, exist_ok=True)
    prefix = "NSM_bone_only_recon_" if bone_only else "NSM_recon_"
    bone_mesh = BoneMesh(mesh_result["mesh"][0].mesh)
    bone_mesh.save_mesh(os.path.join(save_dir, f"{prefix}{os.path.basename(mesh_paths[0])}"))

    if not bone_only:
        cart_mesh = mesh_result["mesh"][1]
        cart_mesh.save_mesh(os.path.join(save_dir, f"{prefix}{os.path.basename(mesh_paths[1])}"))

    # Build results
    latent = mesh_result["latent"].detach().cpu().numpy().tolist()
    icp_transform = _convert_icp_transform(mesh_result["icp_transform"])

    dict_results = {
        "latent": latent,
        "icp_transform": icp_transform.tolist(),
        "center": mesh_result["center"].tolist(),
        "scale": mesh_result["scale"],
        "assd_bone_mm": mesh_result["assd_0"],
    }
    if not bone_only:
        dict_results["assd_cartilage_mm"] = mesh_result["assd_1"]

    params_filename = "NSM_bone_only_recon_params.json" if bone_only else "NSM_recon_params.json"
    with open(os.path.join(save_dir, params_filename), "w") as f:
        json.dump(dict_results, f, indent=4)

    # GPU cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return dict_results


def _prepare_meshes(working_dir, bone, config):
    """
    Prepare meshes for NSM fitting: mirror left knee, clip femur top.

    Extracts from seg_thick_t2_pipeline.py lines 512-548.
    Uses the raw (pre-pyacvd) mesh saved by generate_meshes as femur_mesh_raw.vtk.
    """
    sitk_seg = load_segmentation(working_dir)
    sitk_seg_subregions = load_subregions(working_dir)
    seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

    # determine_knee_side() — parameterized to accept canonical label indices
    side = determine_knee_side(
        seg_array, sitk_seg_subregions,
        med_tib_cart_label=5,  # CANONICAL
        lat_tib_cart_label=6,  # CANONICAL
    )

    femur_mesh = pv.read(str(working_dir / f"{bone}_mesh_raw.vtk"))
    fem_cart_mesh = pv.read(str(working_dir / f"{bone}_cart_0_mesh.vtk"))

    if side == "left":
        center = np.mean(femur_mesh.points, axis=0)[0]
        femur_mesh.points[:, 0] *= -1
        femur_mesh.points[:, 0] += 2 * center
        fem_cart_mesh.points[:, 0] *= -1
        fem_cart_mesh.points[:, 0] += 2 * center

    if config.get("clip_femur_top", True):
        femur_mesh = clip_femur_top(femur_mesh)  # from utils.py

    femur_mesh.save(str(working_dir / "femur_mesh_NSM_orig.vtk"))
    fem_cart_mesh.save(str(working_dir / "fem_cart_mesh_NSM_orig.vtk"))

    return side


def run(working_dir, options=None, config=None):
    """Step entry point. Preps meshes then runs NSM fitting in-process."""
    options = options or {}
    nsm_type = options.get("nsm_type", "bone_and_cart")
    nsm_bones = options.get("nsm_bones", ["femur"])

    results = {}

    for bone in nsm_bones:
        emit_progress(5, f"Preparing {bone} meshes for NSM")
        knee_side = _prepare_meshes(working_dir, bone, config)

        if nsm_type in ("bone_and_cart", "both"):
            emit_progress(20, f"Running bone+cartilage NSM for {bone}")
            mesh_paths = [
                str(working_dir / f"{bone}_mesh_NSM_orig.vtk"),
                str(working_dir / "fem_cart_mesh_NSM_orig.vtk"),
            ]
            params = fit_nsm(mesh_paths, str(working_dir), config, bone_only=False)
            results[f"{bone}_bone_and_cart"] = params

        if nsm_type in ("bone_only", "both"):
            emit_progress(60, f"Running bone-only NSM for {bone}")
            mesh_paths = [str(working_dir / f"{bone}_mesh_NSM_orig.vtk")]
            params = fit_nsm(mesh_paths, str(working_dir), config, bone_only=True)
            results[f"{bone}_bone_only"] = params

    return {"nsm_results": results, "knee_side": knee_side}
```

**Key decisions**:
- `fit_nsm()` is a proper function that unifies both old NSM scripts.
  Differences handled by `bone_only` flag.
- **No cartilage thickness on reconstructed mesh.** The old
  `NSM_analysis.py:130-134` computed thickness on the NSM reconstruction,
  which is a template-fitted approximation, not patient anatomy. Thickness
  is already computed on real meshes in `generate_meshes.py`. Removed.
- `_convert_icp_transform()` handles `None` (identity fallback) -- fixes
  the bug from `NSM_analysis_bone_only.py` that was missing this case.
- `torch.load()` uses `weights_only=True` -- fixes deprecation/security issue.
- `objects_per_decoder` read from config, not hardcoded -- fixes fragility bug.
- Runs in-process by default. If GPU memory isolation is needed, the caller
  (website orchestrator or `run_pipeline.py`) can invoke via subprocess using
  the CLI entry point: `python -m steps.run_nsm <working_dir> ...`.
  The step contract already supports this -- no special code needed.
- Currently only femur NSM exists. `nsm_bones` is forward-looking for when
  tibia/patella NSM models arrive.

---

### Step 6: `steps/compute_bscore.py`

Standalone BScore computation from NSM latent vectors.

**Inputs**: `NSM_recon_params.json` (contains latent vector)
**Outputs**: `bscore_results.json`
**Options**: `{"bscore_type": "bone_and_cart", "bscore_bones": ["femur"]}`

```python
def run(working_dir, options=None, config=None):
    bscore_type = options.get("bscore_type", "bone_and_cart")
    bscore_bones = options.get("bscore_bones", ["femur"])

    results = {}
    for bone in bscore_bones:
        if bscore_type == "bone_and_cart":
            params_file = working_dir / "NSM_recon_params.json"
            model_path = Path(config["bscore"]["path_model_folder"])
        else:
            params_file = working_dir / "NSM_bone_only_recon_params.json"
            model_path = Path(config["bscore_bone_only"]["path_model_folder"])

        params = json.loads(params_file.read_text())
        latent = params["latent"]

        # Load Bscore model -- currently imports from the model folder
        # (sys.path.append + from Bscore import Bscore)
        sys.path.insert(0, str(model_path))
        from Bscore import Bscore
        bscore = Bscore(latent)
        sys.path.pop(0)

        results[f"{bone}_{bscore_type}"] = float(np.squeeze(bscore))

    bscore_path = working_dir / "bscore_results.json"
    bscore_path.write_text(json.dumps(results, indent=2))
    return {"bscore_results": results}
```

**Key decisions**:
- BScore is an explicit orchestrator step -- clean separation from NSM.
- NSM = fitting + latent vectors; BScore = scoring from latents.
- `compute_bscore.py` uses the same `Bscore` class from the model folder,
  maintaining identical computation.
- BScore computation is pure numpy (no GPU) -- fast, no timeout concerns.

---

## Test Fixtures

Tests use **synthetic numpy arrays wrapped in SimpleITK images** -- no real
MRI data needed for unit tests.

```python
# tests/test_steps/conftest.py

import numpy as np
import SimpleITK as sitk
import pytest

@pytest.fixture
def synthetic_segmentation(tmp_path):
    """Create a small synthetic segmentation with canonical labels."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1   # femur_bone
    arr[6:8, 6:8, 6:8] = 4   # femur_cart
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    path = tmp_path / "test_all-labels.nii.gz"
    sitk.WriteImage(sitk_img, str(path))
    return path

@pytest.fixture
def synthetic_native_segmentation(tmp_path):
    """Create a synthetic segmentation with DOSMA-native labels (pre-remap)."""
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 7   # DOSMA femur_bone
    arr[6:8, 6:8, 6:8] = 2   # DOSMA femur_cart
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    path = tmp_path / "test_all-labels.nii.gz"
    sitk.WriteImage(sitk_img, str(path))
    return path
```

Steps that need mesh objects (generate_meshes, run_nsm, compute_bscore)
should **mock at the `pymskt` / NSM reconstruct level** rather than requiring
real mesh computations -- those are integration tests for Phase 4.

---

## Implementation Phases

### Phase 0: Bug fixes (before refactoring)

Quick fixes to existing code. These are independent of the modular refactor
but fix real bugs. Some are also fixed by the refactor itself (e.g., the NSM
bugs are fixed in `steps/run_nsm.py`), but fixing them in the old scripts
first means the monolithic path works correctly until it's removed.

1. `NSM_analysis_bone_only.py`: Add `elif icp_transform is None:` branch
   (crash bug -- raises ValueError instead of using identity matrix)
2. Both NSM scripts: Add `weights_only=True` to `torch.load()` calls
   (security + deprecation)
3. `NSM_analysis.py`: Read `objects_per_decoder` from config instead of
   hardcoding `2` (fragility bug)

### Phase 1: Skeleton + segment + label_remap + run_pipeline

1. Create `steps/` package, `__init__.py`, `_common.py`
2. Implement `steps/segment.py` -- extract `_load_image()`,
   `segment_image_dosma()`, `segment_image_nnunet()` from
   `seg_thick_t2_pipeline.py` into this module directly
3. Implement `steps/label_remap.py` -- pure numpy remapping
4. Create `run_pipeline.py` -- chains all steps (initially just segment +
   remap, extend as more steps land)
5. Tests: `test_segment.py` (mock segmentation functions), `test_label_remap.py`
   (synthetic arrays, verify remap correctness)

### Phase 2: generate_meshes + t2_mapping

1. Implement `steps/generate_meshes.py` -- extract mesh logic from lines 346-428
2. Implement `steps/t2_mapping.py` -- extract T2 logic from lines 433-503
3. Resolve `determine_knee_side()` label parameterization (needed by Phase 3)
4. Tests: mock `pymskt` calls for mesh tests, mock `dosma.QDess` for T2 tests

### Phase 3: run_nsm + compute_bscore (unified)

1. Implement `steps/run_nsm.py` with `fit_nsm()` as a proper function --
   unifies `NSM_analysis.py` and `NSM_analysis_bone_only.py`
2. Implement `steps/compute_bscore.py` -- extract BScore logic
3. Wire into `run_pipeline.py`
4. Tests: mock NSM reconstruct for fitting tests, mock Bscore class for
   BScore tests

### Phase 4: Integration testing + cleanup

1. Run full sequence: segment -> remap -> meshes -> t2 -> nsm -> bscore
2. Compare outputs against monolithic pipeline outputs on a known input
3. Verify identical results (or document acceptable differences with rationale)
4. Remove old files: `NSM_analysis.py`, `NSM_analysis_bone_only.py`,
   `dosma_knee_seg.py`, `seg_thick_t2_pipeline.py`

---

## Files Touched Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `run_pipeline.py` | New | 60 |
| `steps/__init__.py` | New | 10 |
| `steps/_common.py` | New | 60 |
| `steps/segment.py` | New | 150 |
| `steps/label_remap.py` | New | 50 |
| `steps/generate_meshes.py` | New | 120 |
| `steps/t2_mapping.py` | New | 100 |
| `steps/run_nsm.py` | New | 200 |
| `steps/compute_bscore.py` | New | 50 |
| `tests/test_steps/conftest.py` | New | 40 |
| `tests/test_steps/test_segment.py` | New | 60 |
| `tests/test_steps/test_label_remap.py` | New | 50 |
| `tests/test_steps/test_generate_meshes.py` | New | 60 |
| `tests/test_steps/test_t2_mapping.py` | New | 50 |
| `tests/test_steps/test_run_nsm.py` | New | 70 |
| `tests/test_steps/test_compute_bscore.py` | New | 50 |
| `NSM_analysis.py` | Phase 0 bug fixes, then **removed** in Phase 4 | — |
| `NSM_analysis_bone_only.py` | Phase 0 bug fixes, then **removed** in Phase 4 | — |
| `dosma_knee_seg.py` | **Removed** in Phase 4 | — |
| `seg_thick_t2_pipeline.py` | Reference only, **removed** in Phase 4 | — |
| `utils.py` | **Preserved** | — |

---

## Open Questions (to resolve during implementation)

1. **Remap tables**: Need to be validated against actual model outputs.
   Run each model, inspect output labels, build exact remap dicts. Start
   with DOSMA models (the most commonly used).

2. **nnU-Net label scheme**: The nnU-Net post-processing in
   `segment_image_nnunet()` (lines 231-234) references bone_indices [7, 8, 9].
   Verify whether nnU-Net outputs the same label scheme as DOSMA after
   post-processing.

3. **`determine_knee_side()` label indices**: The function (line 73-98) uses
   hardcoded indices 3 and 4 for medial/lateral tibial cartilage. After
   label_remap, these become canonical 5 and 6. Decision: parameterize the
   function to accept label indices. Must be resolved in Phase 2 before
   Phase 3 needs it.

4. **`get_knee_segmentation_with_femur_subregions()` kwargs**: The current
   call (line 351-366) uses `med_tibia_label=3, lat_tibia_label=4` -- these
   are DOSMA medial/lateral tibial cartilage indices. The CANONICAL equivalents
   are 5 and 6. **Check the pymskt function signature** -- the parameter names
   suggest they expect tibial cartilage labels, not tibial bone labels.

5. **Raw mesh persistence**: The current pipeline stores the raw (pre-pyacvd)
   femur mesh in-memory (`dict_bones['femur']['raw_mesh']`, line 385). The
   modular pipeline needs to write it to disk as `femur_mesh_raw.vtk` so
   `run_nsm.py` can read it. Verify the raw mesh round-trips through
   VTK save/load without losing information relevant to NSM fitting.

6. **GPU memory isolation**: `steps/run_nsm.py` runs in-process by default.
   If GPU memory isn't fully released between segmentation and NSM (or between
   bone+cart and bone-only NSM), the website orchestrator can call
   `python -m steps.run_nsm` as a subprocess instead -- the CLI entry point
   already supports this via the step contract. Test in-process first; only
   add subprocess isolation if needed.

## Resolved Decisions

7. **`cartilage_thickness` as a step** -- Merged into `meshes` step. Users
   control via `compute_thickness: bool` option.

8. **`nsm_prep` as a step** -- Folded into `run_nsm.py` as internal
   `_prepare_meshes()`.

9. **BScore separation from NSM** -- BScore is always a separate step.
   `fit_nsm()` does not compute BScore.

10. **Preserve vs replace old scripts** -- Old scripts (`NSM_analysis.py`,
    `NSM_analysis_bone_only.py`, `dosma_knee_seg.py`) are replaced, not
    preserved alongside. `seg_thick_t2_pipeline.py` kept as reference during
    extraction, removed after integration tests pass. No dual maintenance.

11. **NSM deduplication** -- `NSM_analysis.py` and `NSM_analysis_bone_only.py`
    are unified into a single `fit_nsm()` function with a `bone_only` flag.
    Done in Phase 3, not deferred.

12. **Cartilage thickness on NSM reconstruction** -- Removed. The old
    `NSM_analysis.py:130-134` computed thickness on the template-fitted
    reconstruction. Thickness should only be computed on patient anatomy
    meshes (done in `generate_meshes.py`).

13. **In-process vs subprocess for NSM** -- In-process by default. The step
    contract's CLI entry point provides subprocess isolation for free if
    GPU memory issues arise. No special subprocess wrapper code needed.

---

## Phase 5: Validation — keep old pipeline and compare results

Do NOT remove the old pipeline scripts until new pipeline outputs have been
compared against old pipeline outputs on real data.

### Approach

1. **Keep old scripts intact** throughout Phases 1–3. The old entry point
   (`dosma_knee_seg.py` → `seg_thick_t2_pipeline.py` → `NSM_analysis*.py`)
   must remain runnable alongside the new modular pipeline.

2. **Run both pipelines on the same input(s)**. For at least one real knee MRI
   (ideally one qDESS DICOM and one non-qDESS input):
   ```bash
   # Old pipeline
   python dosma_knee_seg.py /path/to/image /tmp/old_output/

   # New pipeline
   python run_pipeline.py /path/to/image /tmp/new_output/
   ```

3. **Compare outputs**:
   - Segmentation labels: voxel-wise diff of `*_all-labels.nii.gz` — should
     be identical after remap (old pipeline uses DOSMA-native labels, new uses
     canonical, so compare post-remap new vs old directly).
   - Meshes: vertex count, ASSD between old/new meshes for each bone.
   - Cartilage thickness metrics: compare JSON metrics — allow small float
     tolerance (1e-6) for reordering/float arithmetic differences.
   - T2 metrics: compare per-region T2 mean/std/median values.
   - NSM latent vectors: element-wise diff — should be identical (same seeds,
     same model, same input meshes).
   - BScore: should match exactly (deterministic numpy computation from latent).

4. **Document any differences** with rationale. Acceptable differences:
   - Label indices differ (DOSMA-native vs canonical) — expected, just verify
     the mapping is correct.
   - `cartilage_smoothing` default changed from 0.3125 to 0.4 — document this
     as intentional.
   - Cartilage thickness NOT computed on NSM reconstruction — intentional removal.

5. **Only after validation passes**, remove old scripts in Phase 4 cleanup.

### What this changes in the phasing

Phase 4 in the original plan ("Integration testing + cleanup") is split:
- **Phase 4a**: Integration testing — run both pipelines, compare outputs.
- **Phase 4b**: Cleanup — remove old scripts only after Phase 4a passes.

---

## Phase 6: qDESS detection and handling as a proper module

### Problem

qDESS handling is currently entangled with image loading in `_load_image()`.
The function does two things at once:

1. **Detects** whether input is qDESS (by trying `QDess.from_dicom()` and
   catching exceptions on failure)
2. **Pre-processes** the image differently for qDESS: calls `qdess.calc_rss()`
   to combine two echo volumes via root-sum-of-squares, producing a single
   volume for segmentation

For non-qDESS DICOM, the volume is loaded directly via SimpleITK. For NIfTI
and NRRD inputs, qDESS is assumed to be already RSS-combined / post-processed.

This means the segmentation model always receives a single-volume input, but
the *source* of that volume differs depending on qDESS status.

### Current flow (entangled)

```
_load_image()
├── DICOM dir?
│   ├── Try QDess.from_dicom() → success → qdess.calc_rss() → volume, is_qdess=True
│   └── Exception → sitk DICOM reader → volume, is_qdess=False
├── NIfTI? → load directly (assumed already RSS) → is_qdess=False
└── NRRD/DCM? → sitk.ReadImage() → is_qdess=False
```

The `qdess` object from loading is then kept alive until much later when
T2 mapping calls `qdess.generate_t2_map()`. This couples image loading to
T2 mapping through a long-lived object.

### Proposed changes

#### 1. Extract qDESS detection into a standalone function

```python
# In steps/segment.py (or steps/_common.py if T2 mapping also needs it)

def detect_qdess(path_image):
    """Check if a DICOM directory contains qDESS data.

    Returns:
        QDess object if qDESS detected, None otherwise.
        Only works on DICOM directories — returns None for NIfTI/NRRD.
    """
    if not os.path.isdir(path_image):
        return None

    from dosma.scan_sequences import QDess

    try:
        try:
            return QDess.from_dicom(str(path_image))
        except KeyError:
            return QDess.from_dicom(str(path_image), group_by="EchoTime")
    except (ValueError, TypeError):
        return None
```

#### 2. Simplify `_load_image()` to use `detect_qdess()`

```python
def _load_image(path_image):
    path_image = str(path_image)

    qdess = detect_qdess(path_image)

    if os.path.isdir(path_image):
        if qdess is not None:
            # qDESS: RSS-combine two echo volumes into single volume
            volume = qdess.calc_rss()
        else:
            # Generic DICOM
            volume = _load_dicom_sitk(path_image)
        filename_prefix = os.path.basename(path_image)

    elif path_image.endswith(("nii", "nii.gz")):
        # NIfTI assumed already RSS / post-processed
        volume = dm.NiftiReader().load(path_image)
        filename_prefix = os.path.basename(path_image).split(".nii")[0]

    elif path_image.endswith(("nrrd", "dcm")):
        volume = MedicalVolume.from_sitk(sitk.ReadImage(path_image))
        ext = ".nrrd" if path_image.endswith("nrrd") else ".dcm"
        filename_prefix = os.path.basename(path_image).replace(ext, "")

    else:
        raise ValueError("Image format not supported.")

    return volume, qdess is not None, filename_prefix
```

#### 3. T2 mapping step loads `QDess` object (not re-detecting)

The T2 mapping step only runs when the orchestrator already knows it's qDESS
(from `seg_result["is_qdess"]`). So T2 mapping does NOT need to re-detect —
it just loads the `QDess` object to access both echo volumes for T2
computation.

However, it still needs to **validate GL/TG tags** before computing T2.
A scan can be structurally valid qDESS (two echoes, loads fine with
`QDess.from_dicom()`) but have private tags stripped during DICOM
anonymization, making T2 computation impossible.

```python
def run(working_dir, options=None, config=None):
    dicom_dir = _find_dicom_dir(working_dir)

    # Load QDess object — we already know this is qDESS,
    # the orchestrator wouldn't call this step otherwise
    qdess = _load_qdess(dicom_dir)

    # Validate GL/TG tags (may be stripped by anonymization)
    has_gl = qdess.get_metadata(qdess.__GL_AREA_TAG__, None) is not None
    has_tg = qdess.get_metadata(qdess.__TG_TAG__, None) is not None
    if not (has_gl and has_tg):
        logging.warning(
            "GL and TG tags not present — skipping T2 computation. "
            "These are private tags and may have been removed during "
            "DICOM anonymization."
        )
        return {"skipped": True, "reason": "missing_gl_tg_tags"}

    # Compute T2 map using both echo volumes
    t2map = qdess.generate_t2_map(...)
```

#### 4. Orchestrator flow for qDESS

The orchestrator (`run_pipeline.py`) uses the `is_qdess` flag from the
segmentation result to decide whether to run T2 mapping:

```python
seg_result = segment(working_dir, ...)

if seg_result["is_qdess"]:
    t2_mapping(working_dir, ...)
```

Detection happens once (in `segment`). T2 mapping trusts that decision and
just loads the `QDess` object directly. The only extra check is GL/TG tag
validation — a qDESS scan without these tags can be segmented (RSS still
works) but can't produce T2 maps.

### Key points about qDESS image handling

- **qDESS acquires two echoes** — the raw DICOM contains interleaved echo
  volumes
- **For segmentation**: the two echoes are combined via **RSS
  (root-sum-of-squares)** into a single volume (`qdess.calc_rss()`)
- **For T2 mapping**: both echo volumes are needed separately — the T2 map
  is computed from the ratio of echo signals plus scanner parameters (GL area
  and TG tags)
- **NIfTI/NRRD inputs** are assumed to already be RSS-combined or otherwise
  post-processed — qDESS detection is skipped for these formats
- This RSS step is why qDESS detection is coupled with image loading: the
  detection determines *how* the volume for segmentation is constructed
