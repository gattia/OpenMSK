"""Integration test runner for the knee MRI pipeline.

Runs the new modular pipeline step-by-step on real data, validates that
expected outputs exist and are reasonable, and prints a summary.

Usage:
    # NRRD input with default (ananya) model:
    python tests/integration/run_pipeline_test.py data/anthonys_knee.nrrd /tmp/test_ananya

    # NRRD input with nnU-Net:
    python tests/integration/run_pipeline_test.py data/anthonys_knee.nrrd /tmp/test_nnunet --model nnunet_knee

    # qDESS DICOM (tests T2 mapping):
    python tests/integration/run_pipeline_test.py data/012_knee1 /tmp/test_qdess

    # Skip NSM/BScore (no GPU or just testing seg+mesh):
    python tests/integration/run_pipeline_test.py data/anthonys_knee.nrrd /tmp/test_fast --skip-nsm
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from glob import glob
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from steps._common import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StepResult:
    """Tracks result of a single pipeline step."""

    def __init__(self, name):
        self.name = name
        self.status = "pending"  # pending, pass, fail, skip
        self.duration = 0.0
        self.result = None
        self.error = None
        self.checks = []  # list of (description, passed)

    def check(self, description, condition):
        self.checks.append((description, bool(condition)))
        if not condition:
            print(f"    FAIL: {description}")
        else:
            print(f"    OK:   {description}")
        return condition


def file_exists(working_dir, pattern):
    """Check if at least one file matching pattern exists in working_dir."""
    matches = glob(str(Path(working_dir) / pattern))
    return len(matches) > 0


def file_size_ok(working_dir, pattern, min_bytes=100):
    """Check file exists and is larger than min_bytes."""
    matches = glob(str(Path(working_dir) / pattern))
    if not matches:
        return False
    return os.path.getsize(matches[0]) > min_bytes


def count_labels(working_dir, pattern):
    """Read a segmentation NIfTI and return the set of unique labels."""
    import SimpleITK as sitk
    import numpy as np

    matches = glob(str(Path(working_dir) / pattern))
    if not matches:
        return set()
    img = sitk.ReadImage(matches[0])
    arr = sitk.GetArrayFromImage(img)
    return set(np.unique(arr).tolist())


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step_segment(working_dir, options, config, result):
    from steps.segment import run as segment
    result.result = segment(working_dir, options=options, config=config)

    result.check("seg_path in result", "seg_path" in result.result)
    result.check("is_qdess in result", "is_qdess" in result.result)
    result.check("*_all-labels.nii.gz exists", file_exists(working_dir, "*_all-labels.nii.gz"))
    result.check("*_all-labels.nrrd exists", file_exists(working_dir, "*_all-labels.nrrd"))
    result.check("seg file > 100 bytes", file_size_ok(working_dir, "*_all-labels.nii.gz"))

    labels = count_labels(working_dir, "*_all-labels.nii.gz")
    result.check(f"seg has >1 label (got {labels})", len(labels) > 1)
    result.check("seg has background (0)", 0 in labels)

    print(f"    is_qdess={result.result['is_qdess']}, model={result.result['model_name']}")
    print(f"    labels found: {sorted(labels)}")


def run_step_label_remap(working_dir, remap_table, config, result):
    from steps.label_remap import run as label_remap
    result.result = label_remap(working_dir, options={"remap_table": remap_table}, config=config)

    result.check("remapped=True", result.result.get("remapped"))
    result.check("native backup exists", file_exists(working_dir, "*-native.nii.gz"))

    labels = count_labels(working_dir, "*_all-labels.nii.gz")
    # Canonical labels: 0=bg, 1=femur, 2=tibia, 3=patella, 4=fem_cart,
    #                   5=med_tib_cart, 6=lat_tib_cart, 7=pat_cart
    canonical = {0, 1, 2, 3, 4, 5, 6, 7}
    result.check(f"all labels are canonical (got {sorted(labels)})",
                 labels.issubset(canonical))
    result.check("has bone labels (1,2,3)", {1, 2, 3}.issubset(labels))

    print(f"    canonical labels: {sorted(labels)}")


def run_step_subregions(working_dir, config, result):
    """Its own stage since D7b — generate_meshes no longer produces this file.

    Without a stage here the runner would still pass while the file was never
    written, because both consumers now degrade rather than fail: femoral
    thickness would quietly collapse from five subregions to one, and T2 would
    lose labels 11-15. That is the check, not the file's mere existence.
    """
    from steps.subregions import run as subregions
    result.result = subregions(working_dir, config=config)

    result.check("not skipped", not result.result.get("skipped"))
    result.check("*_subregions-labels.nii.gz exists",
                 file_exists(working_dir, "*_subregions-labels.nii.gz"))
    result.check("*_subregions-labels.nrrd exists",
                 file_exists(working_dir, "*_subregions-labels.nrrd"))

    labels = count_labels(working_dir, "*_subregions-labels.nii.gz")
    # pymskt REPLACES femoral cartilage (canonical 4) with subregions 11-15,
    # so 4 is expected to be absent here — verified against 60 archived jobs.
    result.check("femur subregions 11-15 present",
                 {11, 12, 13, 14, 15} <= set(labels))
    print(f"    subregion labels: {sorted(labels)}")


def run_step_generate_meshes(working_dir, config, result):
    from steps.generate_meshes import run as generate_meshes
    result.result = generate_meshes(working_dir, config=config)

    result.check("femur_mesh.vtk exists", file_exists(working_dir, "femur_mesh.vtk"))
    result.check("tibia_mesh.vtk exists", file_exists(working_dir, "tibia_mesh.vtk"))
    result.check("patella_mesh.vtk exists", file_exists(working_dir, "patella_mesh.vtk"))
    result.check("femur_mesh_raw.vtk exists", file_exists(working_dir, "femur_mesh_raw.vtk"))
    result.check("femur_cart_0_mesh.vtk exists", file_exists(working_dir, "femur_cart_0_mesh.vtk"))
    result.check("used the subregions file", result.result.get("has_subregions", False))
    result.check("thickness_results.json exists",
                 file_exists(working_dir, "*_thickness_results.json"))

    result.check("metrics in result", "metrics" in result.result)
    result.check("thickness_computed", result.result.get("thickness_computed", False))

    if result.result.get("metrics"):
        n_metrics = len(result.result["metrics"])
        print(f"    {n_metrics} thickness metrics computed")
        # Print a few
        for k, v in list(result.result["metrics"].items())[:5]:
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")


def run_step_t2_mapping(working_dir, config, result):
    from steps.t2_mapping import run as t2_mapping
    result.result = t2_mapping(working_dir, config=config)

    if result.result.get("skipped"):
        print(f"    T2 mapping skipped: {result.result.get('reason', 'unknown')}")
        result.status = "skip"
        return

    result.check("*_t2map.nii.gz exists", file_exists(working_dir, "*_t2map.nii.gz"))
    result.check("*_t2map.nrrd exists", file_exists(working_dir, "*_t2map.nrrd"))
    result.check("*_t2_results.json exists", file_exists(working_dir, "*_t2_results.json"))
    result.check("metrics in result", "metrics" in result.result)

    if result.result.get("metrics"):
        n_metrics = len(result.result["metrics"])
        print(f"    {n_metrics} T2 metrics computed")
        print(f"    has_depth_dependent: {result.result.get('has_depth_dependent')}")
        for k, v in list(result.result["metrics"].items())[:5]:
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")


def run_step_nsm(working_dir, config, result):
    from steps.run_nsm import run as run_nsm

    nsm_type = "both" if (config.get("perform_bone_and_cart_nsm")
                          and config.get("perform_bone_only_nsm")) \
               else "bone_and_cart" if config.get("perform_bone_and_cart_nsm") \
               else "bone_only"

    result.result = run_nsm(working_dir, options={"nsm_type": nsm_type}, config=config)

    result.check("femur_mesh_NSM_orig.vtk exists",
                 file_exists(working_dir, "femur_mesh_NSM_orig.vtk"))
    result.check("fem_cart_mesh_NSM_orig.vtk exists",
                 file_exists(working_dir, "fem_cart_mesh_NSM_orig.vtk"))

    if nsm_type in ("bone_and_cart", "both"):
        result.check("NSM_recon_params.json exists",
                     file_exists(working_dir, "NSM_recon_params.json"))
        params_path = Path(working_dir) / "NSM_recon_params.json"
        if params_path.exists():
            params = json.loads(params_path.read_text())
            result.check("latent vector present", "latent" in params)
            result.check(f"latent dim={len(params.get('latent', []))}",
                         len(params.get("latent", [])) > 0)
            print(f"    bone+cart ASSD: {params.get('assd_bone_mm', 'N/A')} mm")

    if nsm_type in ("bone_only", "both"):
        result.check("NSM_bone_only_recon_params.json exists",
                     file_exists(working_dir, "NSM_bone_only_recon_params.json"))
        params_path = Path(working_dir) / "NSM_bone_only_recon_params.json"
        if params_path.exists():
            params = json.loads(params_path.read_text())
            print(f"    bone-only ASSD: {params.get('assd_bone_mm', 'N/A')} mm")

    print(f"    knee_side: {result.result.get('knee_side', 'N/A')}")


def run_step_bscore(working_dir, config, result):
    from steps.compute_bscore import run as compute_bscore

    nsm_type = "both" if (config.get("perform_bone_and_cart_nsm")
                          and config.get("perform_bone_only_nsm")) \
               else "bone_and_cart" if config.get("perform_bone_and_cart_nsm") \
               else "bone_only"

    result.result = compute_bscore(working_dir, options={"bscore_type": nsm_type}, config=config)

    result.check("bscore_results.json exists", file_exists(working_dir, "bscore_results.json"))
    result.check("bscore_results in result", "bscore_results" in result.result)

    if result.result.get("bscore_results"):
        for k, v in result.result["bscore_results"].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}

STEPS = [
    ("segment", run_step_segment),
    ("label_remap", run_step_label_remap),
    # subregions must run BEFORE generate_meshes: since D7b it produces
    # *_subregions-labels.*, which generate_meshes and t2_mapping both consume.
    # Omitting it does not fail loudly -- both consumers fall back -- it just
    # silently costs femoral regional thickness and the depth-resolved T2.
    ("subregions", run_step_subregions),
    ("generate_meshes", run_step_generate_meshes),
    ("t2_mapping", run_step_t2_mapping),
    ("nsm", run_step_nsm),
    ("bscore", run_step_bscore),
]


def main():
    parser = argparse.ArgumentParser(description="Integration test for knee MRI pipeline")
    parser.add_argument("input_path", help="Path to input MRI (DICOM dir, NIfTI, NRRD)")
    parser.add_argument("output_dir", help="Output directory (will be created)")
    parser.add_argument("--model", default=None,
                        help="Segmentation model name (default: from config)")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--skip-nsm", action="store_true",
                        help="Skip NSM and BScore steps")
    parser.add_argument("--skip-t2", action="store_true",
                        help="Skip T2 mapping even for qDESS input")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = load_config(args.config)

    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)

    # Set up working directory
    os.makedirs(output_dir, exist_ok=True)

    # Symlink or copy input into working dir
    if input_path.is_dir():
        # DICOM directory — symlink contents
        for f in os.listdir(input_path):
            link = output_dir / f
            if not link.exists():
                link.symlink_to(input_path / f)
    else:
        link = output_dir / input_path.name
        if not link.exists():
            link.symlink_to(input_path)

    print("=" * 70)
    print("KNEE MRI PIPELINE INTEGRATION TEST")
    print("=" * 70)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  Model:  {args.model or config.get('default_seg_model', '(default)')}")
    print(f"  Skip NSM: {args.skip_nsm}")
    print(f"  Skip T2:  {args.skip_t2}")
    print()

    results = []
    is_qdess = False

    for step_name, step_fn in STEPS:
        # Skip logic
        if step_name == "t2_mapping" and (args.skip_t2 or not is_qdess):
            sr = StepResult(step_name)
            sr.status = "skip"
            reason = "not qDESS" if not is_qdess else "user --skip-t2"
            print(f"  [{step_name}] SKIP ({reason})")
            results.append(sr)
            continue

        if step_name in ("nsm", "bscore") and args.skip_nsm:
            sr = StepResult(step_name)
            sr.status = "skip"
            print(f"  [{step_name}] SKIP (user --skip-nsm)")
            results.append(sr)
            continue

        sr = StepResult(step_name)
        print(f"  [{step_name}] Running...")
        t0 = time.time()
        try:
            if step_name == "segment":
                step_fn(output_dir, {"model": args.model}, config, sr)
                is_qdess = sr.result.get("is_qdess", False)
            elif step_name == "label_remap":
                step_fn(output_dir, DOSMA_REMAP, config, sr)
            elif step_name == "t2_mapping":
                step_fn(output_dir, config, sr)
            elif step_name in ("subregions", "generate_meshes", "nsm", "bscore"):
                step_fn(output_dir, config, sr)
            else:
                # A step listed in STEPS but missing from this chain would call
                # nothing, record nothing, and still be reported as a pass --
                # which is exactly how `subregions` went unrun after D7b split
                # it out. Fail loudly instead.
                raise RuntimeError(
                    f"step {step_name!r} is in STEPS but has no dispatch branch"
                )

            sr.duration = time.time() - t0
            failed_checks = [desc for desc, ok in sr.checks if not ok]
            if failed_checks:
                sr.status = "fail"
            elif sr.status != "skip":
                sr.status = "pass"
        except Exception as e:
            sr.duration = time.time() - t0
            sr.status = "fail"
            sr.error = str(e)
            print(f"    ERROR: {e}")
            traceback.print_exc()
            # Continue to next steps only if segmentation succeeded
            if step_name == "segment":
                print("    Segmentation failed — aborting remaining steps")
                results.append(sr)
                break

        results.append(sr)
        status_icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[sr.status]
        print(f"  [{step_name}] {status_icon} ({sr.duration:.1f}s)\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for sr in results:
        icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP", "pending": "----"}[sr.status]
        time_str = f"{sr.duration:.1f}s" if sr.duration else ""
        err_str = f"  ({sr.error})" if sr.error else ""
        failed = [d for d, ok in sr.checks if not ok]
        fail_str = f"  failed: {failed}" if failed else ""
        print(f"  {icon}  {sr.name:<20s} {time_str:>8s}{err_str}{fail_str}")

    total_time = sum(sr.duration for sr in results)
    n_pass = sum(1 for sr in results if sr.status == "pass")
    n_fail = sum(1 for sr in results if sr.status == "fail")
    n_skip = sum(1 for sr in results if sr.status == "skip")
    print(f"\n  Total: {n_pass} pass, {n_fail} fail, {n_skip} skip ({total_time:.1f}s)")

    # List output files
    print(f"\n  Output files in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        fp = output_dir / f
        if fp.is_file() and not fp.is_symlink():
            sz = os.path.getsize(fp)
            print(f"    {f}  ({sz:,} bytes)")

    # Save results JSON
    results_json = {
        "input": str(input_path),
        "model": args.model or config.get("default_seg_model"),
        "steps": {
            sr.name: {
                "status": sr.status,
                "duration_s": round(sr.duration, 2),
                "error": sr.error,
                "checks": {desc: ok for desc, ok in sr.checks},
            }
            for sr in results
        },
    }
    results_path = output_dir / "_test_results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    print(f"\n  Results saved to {results_path}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
