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

    # Also run old pipeline for comparison:
    python tests/integration/run_pipeline_test.py data/anthonys_knee.nrrd /tmp/test_compare --compare-old
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


def run_step_generate_meshes(working_dir, config, result):
    from steps.generate_meshes import run as generate_meshes
    result.result = generate_meshes(working_dir, config=config)

    result.check("femur_mesh.vtk exists", file_exists(working_dir, "femur_mesh.vtk"))
    result.check("tibia_mesh.vtk exists", file_exists(working_dir, "tibia_mesh.vtk"))
    result.check("patella_mesh.vtk exists", file_exists(working_dir, "patella_mesh.vtk"))
    result.check("femur_mesh_raw.vtk exists", file_exists(working_dir, "femur_mesh_raw.vtk"))
    result.check("femur_cart_0_mesh.vtk exists", file_exists(working_dir, "femur_cart_0_mesh.vtk"))
    result.check("*_subregions-labels.nii.gz exists",
                 file_exists(working_dir, "*_subregions-labels.nii.gz"))
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
# Old pipeline runner (for comparison)
# ---------------------------------------------------------------------------

def run_old_pipeline(input_path, output_dir, model_name, config_path):
    """Run the old monolithic pipeline via subprocess."""
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "dosma_knee_seg.py"),
        str(input_path),
        str(output_dir),
    ]
    if model_name:
        cmd.append(model_name)

    env = os.environ.copy()
    if config_path:
        env["KNEEPIPELINE_CONFIG"] = str(config_path)

    print(f"\n  Running old pipeline: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"  Old pipeline FAILED (exit {proc.returncode})")
        print(f"  stderr: {proc.stderr[-2000:]}")
        return False
    print("  Old pipeline completed successfully")
    return True


def compare_outputs(new_dir, old_dir):
    """Compare key outputs between new and old pipeline."""
    import SimpleITK as sitk
    import numpy as np

    print("\n" + "=" * 70)
    print("COMPARISON: new vs old pipeline")
    print("=" * 70)

    # Compare segmentation labels
    new_segs = glob(str(Path(new_dir) / "*_all-labels.nii.gz"))
    old_segs = glob(str(Path(old_dir) / "*_all-labels.nii.gz"))

    if new_segs and old_segs:
        new_img = sitk.ReadImage(new_segs[0])
        old_img = sitk.ReadImage(old_segs[0])
        new_arr = sitk.GetArrayFromImage(new_img)
        old_arr = sitk.GetArrayFromImage(old_img)

        print(f"\n  Segmentation shapes: new={new_arr.shape}, old={old_arr.shape}")
        print(f"  New labels: {sorted(np.unique(new_arr))}")
        print(f"  Old labels: {sorted(np.unique(old_arr))}")

        if new_arr.shape == old_arr.shape:
            # Old uses DOSMA-native labels, new uses canonical after remap.
            # Map old labels to canonical for comparison.
            DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}
            old_remapped = np.zeros_like(old_arr)
            for src, dst in DOSMA_REMAP.items():
                old_remapped[old_arr == src] = dst

            match = np.sum(new_arr == old_remapped)
            total = new_arr.size
            pct = 100.0 * match / total
            print(f"  Voxel match (after remap): {match}/{total} ({pct:.2f}%)")
            if pct < 99.99:
                diff_mask = new_arr != old_remapped
                diff_labels_new = np.unique(new_arr[diff_mask])
                diff_labels_old = np.unique(old_remapped[diff_mask])
                print(f"  Differing voxels: new labels={sorted(diff_labels_new)}, "
                      f"old labels={sorted(diff_labels_old)}")
        else:
            print("  SHAPES DIFFER - cannot compare voxel-wise")

    # Compare thickness results
    new_thick = glob(str(Path(new_dir) / "*_thickness_results.json"))
    old_thick = glob(str(Path(old_dir) / "*_thickness_results.json"))
    if new_thick and old_thick:
        new_metrics = json.loads(Path(new_thick[0]).read_text())
        old_metrics = json.loads(Path(old_thick[0]).read_text())
        print("\n  Thickness comparison:")
        all_keys = sorted(set(list(new_metrics.keys()) + list(old_metrics.keys())))
        for k in all_keys:
            nv = new_metrics.get(k, "MISSING")
            ov = old_metrics.get(k, "MISSING")
            if isinstance(nv, (int, float)) and isinstance(ov, (int, float)):
                diff = abs(nv - ov)
                flag = " ***" if diff > 1e-4 else ""
                print(f"    {k}: new={nv:.4f}  old={ov:.4f}  diff={diff:.6f}{flag}")
            else:
                print(f"    {k}: new={nv}  old={ov}")

    # Compare NSM params
    for params_file in ["NSM_recon_params.json", "NSM_bone_only_recon_params.json"]:
        new_p = Path(new_dir) / params_file
        old_p = Path(old_dir) / params_file
        if new_p.exists() and old_p.exists():
            new_params = json.loads(new_p.read_text())
            old_params = json.loads(old_p.read_text())
            print(f"\n  {params_file}:")
            for key in ["assd_bone_mm", "assd_cartilage_mm", "scale"]:
                if key in new_params and key in old_params:
                    diff = abs(new_params[key] - old_params[key])
                    print(f"    {key}: new={new_params[key]:.4f}  "
                          f"old={old_params[key]:.4f}  diff={diff:.6f}")
            if "latent" in new_params and "latent" in old_params:
                new_lat = np.array(new_params["latent"]).flatten()
                old_lat = np.array(old_params["latent"]).flatten()
                if len(new_lat) == len(old_lat):
                    l2 = np.linalg.norm(new_lat - old_lat)
                    print(f"    latent L2 diff: {l2:.6f}")

    # Compare BScore
    new_bs = Path(new_dir) / "bscore_results.json"
    old_bs_files = glob(str(Path(old_dir) / "*bscore*"))
    if new_bs.exists() and old_bs_files:
        print(f"\n  BScore: new={json.loads(new_bs.read_text())}")

    # List files in each
    print("\n  Files in new output:")
    for f in sorted(os.listdir(new_dir)):
        sz = os.path.getsize(Path(new_dir) / f)
        print(f"    {f}  ({sz:,} bytes)")
    print("\n  Files in old output:")
    for f in sorted(os.listdir(old_dir)):
        sz = os.path.getsize(Path(old_dir) / f)
        print(f"    {f}  ({sz:,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}

STEPS = [
    ("segment", run_step_segment),
    ("label_remap", run_step_label_remap),
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
    parser.add_argument("--compare-old", action="store_true",
                        help="Also run old pipeline and compare outputs")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = load_config(args.config)
    config_path = args.config

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
    print(f"  Compare old: {args.compare_old}")
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
            elif step_name in ("generate_meshes", "nsm", "bscore"):
                step_fn(output_dir, config, sr)

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

    # Run old pipeline for comparison if requested
    if args.compare_old:
        old_output = str(output_dir) + "_old"
        print(f"\nRunning old pipeline for comparison -> {old_output}")
        old_ok = run_old_pipeline(input_path, old_output, args.model, config_path)
        if old_ok:
            compare_outputs(str(output_dir), old_output)

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
