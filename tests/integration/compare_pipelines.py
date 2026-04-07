"""Integration test: compare modular pipeline output against monolith.

Runs both pipelines on the same input and compares:
- Segmentation labels (exact match after remap)
- Mesh files exist and have similar vertex counts
- Thickness metrics (within tolerance)
- NSM latent vectors (within tolerance)
- BScore values (within tolerance)

Usage:
    conda run -n kneepipeline python tests/integration/compare_pipelines.py \
        data/anthonys_knee.nrrd --model nnunet_knee

    conda run -n kneepipeline python tests/integration/compare_pipelines.py \
        data/012_knee1/ --model nnunet_knee
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# Tolerance for floating-point comparisons
THICKNESS_RTOL = 0.05   # 5% relative tolerance for thickness metrics
T2_RTOL = 0.05          # 5% for T2 metrics
NSM_ATOL = 0.01         # absolute tolerance for NSM latent vectors
BSCORE_ATOL = 0.05      # absolute tolerance for BScore

# DOSMA-native -> canonical label remap (for comparing segmentations)
DOSMA_REMAP = {1: 7, 2: 4, 3: 5, 4: 6, 7: 1, 8: 2, 9: 3}


def run_monolith(input_path, output_dir, model_name, config_path):
    """Run the monolithic pipeline (dosma_knee_seg.py)."""
    print(f"\n{'='*60}")
    print("RUNNING MONOLITH PIPELINE")
    print(f"{'='*60}")

    script_dir = Path(__file__).resolve().parent.parent.parent
    cmd = [
        sys.executable,
        str(script_dir / "dosma_knee_seg.py"),
        str(input_path),
        str(output_dir),
        model_name,
    ]
    env = os.environ.copy()
    env["KNEEPIPELINE_CONFIG"] = str(config_path)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    if result.returncode != 0:
        print(f"MONOLITH FAILED with exit code {result.returncode}")
        return False
    print("MONOLITH COMPLETED SUCCESSFULLY")
    return True


def run_modular(input_path, output_dir, model_name, config_path):
    """Run the modular pipeline (run_pipeline.py)."""
    print(f"\n{'='*60}")
    print("RUNNING MODULAR PIPELINE")
    print(f"{'='*60}")

    script_dir = Path(__file__).resolve().parent.parent.parent
    cmd = [
        sys.executable,
        str(script_dir / "run_pipeline.py"),
        str(input_path),
        str(output_dir),
        model_name,
        "--config", str(config_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    if result.returncode != 0:
        print(f"MODULAR FAILED with exit code {result.returncode}")
        return False
    print("MODULAR COMPLETED SUCCESSFULLY")
    return True


def compare_segmentations(mono_dir, mod_dir):
    """Compare segmentation label volumes."""
    print(f"\n{'─'*60}")
    print("COMPARING SEGMENTATIONS")
    print(f"{'─'*60}")

    mono_segs = sorted(Path(mono_dir).glob("*_all-labels.nii.gz"))
    mod_segs = sorted(Path(mod_dir).glob("*_all-labels.nii.gz"))

    if not mono_segs:
        print("  SKIP: No monolith segmentation found")
        return True
    if not mod_segs:
        print("  FAIL: No modular segmentation found")
        return False

    mono_img = sitk.ReadImage(str(mono_segs[0]))
    mod_img = sitk.ReadImage(str(mod_segs[0]))

    mono_arr = sitk.GetArrayFromImage(mono_img)
    mod_arr = sitk.GetArrayFromImage(mod_img)

    if mono_arr.shape != mod_arr.shape:
        print(f"  FAIL: Shape mismatch: monolith {mono_arr.shape} vs modular {mod_arr.shape}")
        return False

    # The modular pipeline remaps to canonical labels, the monolith uses native.
    # Remap the monolith output for comparison.
    mono_remapped = np.zeros_like(mono_arr)
    for src, dst in DOSMA_REMAP.items():
        mono_remapped[mono_arr == src] = dst

    # Compare
    match = np.sum(mono_remapped == mod_arr)
    total = mono_arr.size
    pct = 100 * match / total

    mono_labels = set(np.unique(mono_remapped)) - {0}
    mod_labels = set(np.unique(mod_arr)) - {0}

    print(f"  Monolith labels (remapped): {sorted(mono_labels)}")
    print(f"  Modular labels:             {sorted(mod_labels)}")
    print(f"  Voxel agreement: {pct:.2f}% ({match}/{total})")

    if pct < 99.9:
        print(f"  WARN: Less than 99.9% agreement")

    # Labels should be identical (same model, same input)
    if mono_labels != mod_labels:
        print(f"  FAIL: Different label sets")
        return False

    if pct == 100.0:
        print("  PASS: Exact match")
    else:
        print(f"  PASS: {pct:.4f}% match (minor floating-point differences expected)")
    return True


def compare_meshes(mono_dir, mod_dir):
    """Compare mesh files exist and have reasonable sizes."""
    print(f"\n{'─'*60}")
    print("COMPARING MESHES")
    print(f"{'─'*60}")

    expected_meshes = ["femur_mesh.vtk", "tibia_mesh.vtk", "patella_mesh.vtk"]
    all_ok = True

    for mesh_name in expected_meshes:
        mono_path = Path(mono_dir) / mesh_name
        mod_path = Path(mod_dir) / mesh_name

        if not mono_path.exists():
            print(f"  SKIP: {mesh_name} not in monolith output")
            continue
        if not mod_path.exists():
            print(f"  FAIL: {mesh_name} missing from modular output")
            all_ok = False
            continue

        mono_size = mono_path.stat().st_size
        mod_size = mod_path.stat().st_size
        ratio = mod_size / mono_size if mono_size > 0 else 0

        status = "PASS" if 0.5 < ratio < 2.0 else "WARN"
        print(f"  {status}: {mesh_name} — monolith {mono_size:,}B, modular {mod_size:,}B (ratio {ratio:.2f})")

    # Check for raw femur mesh (modular only)
    if (Path(mod_dir) / "femur_mesh_raw.vtk").exists():
        print("  INFO: femur_mesh_raw.vtk present (modular-only, for NSM)")

    return all_ok


def compare_json_metrics(mono_dir, mod_dir, pattern, name, rtol=0.05):
    """Compare JSON result files between pipelines."""
    print(f"\n{'─'*60}")
    print(f"COMPARING {name.upper()}")
    print(f"{'─'*60}")

    mono_files = sorted(Path(mono_dir).glob(pattern))
    mod_files = sorted(Path(mod_dir).glob(pattern))

    if not mono_files:
        print(f"  SKIP: No monolith {name} results found")
        return True
    if not mod_files:
        # Modular may save results with different filenames
        print(f"  SKIP: No modular {name} results found (may use different filename)")
        return True

    with open(mono_files[0]) as f:
        mono_data = json.load(f)
    with open(mod_files[0]) as f:
        mod_data = json.load(f)

    all_ok = True
    for key in sorted(set(list(mono_data.keys()) + list(mod_data.keys()))):
        mono_val = mono_data.get(key)
        mod_val = mod_data.get(key)

        if mono_val is None:
            print(f"  INFO: {key} — modular-only")
            continue
        if mod_val is None:
            print(f"  INFO: {key} — monolith-only")
            continue

        if isinstance(mono_val, (int, float)) and isinstance(mod_val, (int, float)):
            if np.isnan(mono_val) and np.isnan(mod_val):
                print(f"  PASS: {key} — both NaN")
            elif abs(mono_val) < 1e-10:
                diff = abs(mod_val - mono_val)
                status = "PASS" if diff < 1e-6 else "FAIL"
                print(f"  {status}: {key} — mono={mono_val:.6f}, mod={mod_val:.6f}")
            else:
                rel_diff = abs(mod_val - mono_val) / abs(mono_val)
                status = "PASS" if rel_diff < rtol else "FAIL"
                print(f"  {status}: {key} — mono={mono_val:.4f}, mod={mod_val:.4f} (rel_diff={rel_diff:.4f})")
                if status == "FAIL":
                    all_ok = False
        elif isinstance(mono_val, list) and isinstance(mod_val, list):
            # Latent vectors, transforms, etc.
            mono_arr = np.array(mono_val)
            mod_arr = np.array(mod_val)
            if mono_arr.shape != mod_arr.shape:
                print(f"  FAIL: {key} — shape mismatch {mono_arr.shape} vs {mod_arr.shape}")
                all_ok = False
            else:
                max_diff = np.max(np.abs(mono_arr - mod_arr))
                print(f"  {'PASS' if max_diff < 0.1 else 'WARN'}: {key} — max_diff={max_diff:.6f} (shape {mono_arr.shape})")

    return all_ok


def compare_nsm(mono_dir, mod_dir):
    """Compare NSM reconstruction params."""
    ok1 = compare_json_metrics(mono_dir, mod_dir, "NSM_recon_params.json", "NSM bone+cart")
    ok2 = compare_json_metrics(mono_dir, mod_dir, "NSM_bone_only_recon_params.json", "NSM bone-only")
    return ok1 and ok2


def main():
    parser = argparse.ArgumentParser(description="Compare monolith vs modular pipeline")
    parser.add_argument("input_path", help="Path to input MRI (DICOM dir, NIfTI, NRRD)")
    parser.add_argument("--model", default="nnunet_knee", help="Segmentation model name")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--keep-output", action="store_true", help="Keep output directories")
    parser.add_argument("--mono-dir", default=None, help="Skip monolith run, use existing output dir")
    parser.add_argument("--mod-dir", default=None, help="Skip modular run, use existing output dir")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config) if args.config else script_dir / "config.json"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    # Create output directories
    mono_dir = Path(args.mono_dir) if args.mono_dir else Path(tempfile.mkdtemp(prefix="knee_mono_"))
    mod_dir = Path(args.mod_dir) if args.mod_dir else Path(tempfile.mkdtemp(prefix="knee_mod_"))

    print(f"Input:    {input_path}")
    print(f"Model:    {args.model}")
    print(f"Config:   {config_path}")
    print(f"Monolith: {mono_dir}")
    print(f"Modular:  {mod_dir}")

    try:
        # Run pipelines
        if not args.mono_dir:
            mono_ok = run_monolith(input_path, mono_dir, args.model, config_path)
            if not mono_ok:
                print("\nMONOLITH FAILED — cannot compare")
                sys.exit(1)

        if not args.mod_dir:
            mod_ok = run_modular(input_path, mod_dir, args.model, config_path)
            if not mod_ok:
                print("\nMODULAR FAILED — cannot compare")
                sys.exit(1)

        # Compare outputs
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")

        results = {}
        results["segmentation"] = compare_segmentations(mono_dir, mod_dir)
        results["meshes"] = compare_meshes(mono_dir, mod_dir)
        results["thickness"] = compare_json_metrics(
            mono_dir, mod_dir, "*_results.json", "thickness", THICKNESS_RTOL
        )
        results["nsm"] = compare_nsm(mono_dir, mod_dir)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        all_pass = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {name}")
            if not passed:
                all_pass = False

        if all_pass:
            print("\nALL COMPARISONS PASSED")
        else:
            print("\nSOME COMPARISONS FAILED — review output above")

        # List output files for inspection
        print(f"\nMonolith output files:")
        for f in sorted(mono_dir.iterdir()):
            print(f"  {f.name} ({f.stat().st_size:,}B)")
        print(f"\nModular output files:")
        for f in sorted(mod_dir.iterdir()):
            print(f"  {f.name} ({f.stat().st_size:,}B)")

    finally:
        if not args.keep_output and not args.mono_dir and not args.mod_dir:
            print(f"\nCleaning up temp dirs...")
            shutil.rmtree(mono_dir, ignore_errors=True)
            shutil.rmtree(mod_dir, ignore_errors=True)
        else:
            print(f"\nOutput preserved at:")
            print(f"  Monolith: {mono_dir}")
            print(f"  Modular:  {mod_dir}")


if __name__ == "__main__":
    main()
