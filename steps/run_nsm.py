"""Step 5: NSM mesh preparation and fitting.

Prepares meshes (knee side detection, left-knee mirroring, femur clipping)
then fits the Neural Shape Model. Unified replacement for NSM_analysis.py
and NSM_analysis_bone_only.py.

Outputs reconstructed meshes and NSM_recon_params.json with latent vectors,
registration parameters, and ASSD error metrics.
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from steps._common import (
    emit_progress,
    load_segmentation,
    parse_step_args,
    write_step_result,
)
from utils import clip_femur_top

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Knee side detection
# ---------------------------------------------------------------------------

def determine_knee_side(seg_array, sitk_seg, med_tib_cart_label=5, lat_tib_cart_label=6):
    """Determine if knee is left or right based on cartilage positions.

    Uses the medial/lateral tibial cartilage centroids transformed into
    physical (xyz) space to determine knee laterality.

    Args:
        seg_array: numpy array of the segmentation
        sitk_seg: SimpleITK image (for direction/rotation matrix)
        med_tib_cart_label: label index for medial tibial cartilage (canonical: 5)
        lat_tib_cart_label: label index for lateral tibial cartilage (canonical: 6)

    Returns:
        "left" or "right"
    """
    loc_med_cart = np.mean(np.where(seg_array == med_tib_cart_label), axis=1)
    loc_lat_cart = np.mean(np.where(seg_array == lat_tib_cart_label), axis=1)

    rotation_matrix = np.array(sitk_seg.GetDirection()).reshape(3, 3)

    # Flip ijk (SimpleITK ordering) then apply rotation to get xyz
    loc_med_cart_xyz = rotation_matrix @ loc_med_cart[::-1]
    loc_lat_cart_xyz = rotation_matrix @ loc_lat_cart[::-1]

    loc_med_cart_x = loc_med_cart_xyz[0]
    loc_lat_cart_x = loc_lat_cart_xyz[0]

    if loc_med_cart_x > loc_lat_cart_x:
        return "right"
    elif loc_med_cart_x < loc_lat_cart_x:
        return "left"
    else:
        raise ValueError(
            "Unable to determine knee side: medial and lateral cartilage have same x-coordinate"
        )


# ---------------------------------------------------------------------------
# NSM model loading
# ---------------------------------------------------------------------------

def _load_nsm_model(config, bone_only=False):
    """Load NSM model from config.

    Returns:
        (model, model_config) tuple
    """
    import torch
    from NSM.models import TriplanarDecoder

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


# ---------------------------------------------------------------------------
# ICP transform conversion
# ---------------------------------------------------------------------------

def _convert_icp_transform(icp_transform):
    """Convert ICP transform to numpy array, handling all VTK types + None."""
    import vtk
    from pymskt.mesh.meshTransform import get_linear_transform_matrix

    if isinstance(icp_transform, (vtk.vtkIterativeClosestPointTransform, vtk.vtkTransform)):
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
        logging.warning("icp_transform is None, using identity matrix")
        return np.eye(4)
    else:
        raise ValueError(f"icp_transform not a valid type: {type(icp_transform)}")


# ---------------------------------------------------------------------------
# NSM fitting
# ---------------------------------------------------------------------------

def fit_nsm(mesh_paths, save_dir, config, bone_only=False, calc_assd=True, seed=42):
    """Fit NSM model to mesh(es).

    Unified replacement for NSM_analysis.py and NSM_analysis_bone_only.py.

    Args:
        mesh_paths: List of mesh file paths. [bone] for bone_only,
                    [bone, cartilage] for bone+cart.
        save_dir: Directory to save results.
        config: Pipeline config dict (already loaded).
        bone_only: If True, use bone-only model config.
        calc_assd: If True, compute ASSD metrics.
        seed: Random seed for reproducibility (default: 42).
              Set to None to skip seeding (non-deterministic).

    Returns:
        dict with latent, icp_transform, center, scale, assd metrics.
    """
    import torch
    from pymskt.mesh import BoneMesh
    from NSM.reconstruct import reconstruct_mesh

    os.environ["LOC_SDF_CACHE"] = ""

    model, model_config = _load_nsm_model(config, bone_only)

    # Reproducibility — seed AFTER model loading to match the original
    # NSM scripts. model.cuda() consumes CUDA random state, so seeding
    # before it shifts the random sequence seen by reconstruct_mesh.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    if not bone_only and len(mesh_result["mesh"]) > 1:
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

    # Print results (excluding latent)
    for key, val in dict_results.items():
        if key != "latent":
            logging.info(f"{key}: {val}")

    params_filename = "NSM_bone_only_recon_params.json" if bone_only else "NSM_recon_params.json"
    with open(os.path.join(save_dir, params_filename), "w") as f:
        json.dump(dict_results, f, indent=4)

    # GPU cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return dict_results


# ---------------------------------------------------------------------------
# Mesh preparation
# ---------------------------------------------------------------------------

def _prepare_meshes(working_dir, bone, config):
    """Prepare meshes for NSM fitting: mirror left knee, clip femur top.

    Uses the raw (pre-pyacvd) mesh saved by generate_meshes as femur_mesh_raw.vtk
    and the cartilage mesh from generate_meshes.

    Returns:
        knee side ("left" or "right")
    """
    working_dir = Path(working_dir)

    # The canonical segmentation, not *_subregions-labels.nii.gz: knee side is
    # read off the medial/lateral tibial cartilage (canonical labels 5 and 6),
    # which are in the segmentation itself. This step used to load the subregions
    # file, which happens to contain them too -- and would then have failed for
    # want of a file it does not use on any job where subregions skipped (D7b).
    sitk_seg = load_segmentation(working_dir)
    seg_array = sitk.GetArrayFromImage(sitk_seg)

    side = determine_knee_side(seg_array, sitk_seg)

    from pymskt.mesh import Mesh

    femur_mesh = Mesh(str(working_dir / f"{bone}_mesh_raw.vtk"))

    # Load cartilage mesh if available
    fem_cart_path = working_dir / f"{bone}_cart_0_mesh.vtk"
    fem_cart_mesh = Mesh(str(fem_cart_path)) if fem_cart_path.exists() else None

    if side == "left":
        center = np.mean(femur_mesh.point_coords, axis=0)[0]
        femur_mesh.point_coords = femur_mesh.point_coords * [-1, 1, 1]
        femur_mesh.point_coords = femur_mesh.point_coords + [2 * center, 0, 0]
        if fem_cart_mesh is not None:
            fem_cart_mesh.point_coords = fem_cart_mesh.point_coords * [-1, 1, 1]
            fem_cart_mesh.point_coords = fem_cart_mesh.point_coords + [2 * center, 0, 0]

    if config.get("clip_femur_top", True):
        femur_mesh = clip_femur_top(femur_mesh)

    femur_mesh.save_mesh(str(working_dir / "femur_mesh_NSM_orig.vtk"))
    if fem_cart_mesh is not None:
        fem_cart_mesh.save_mesh(str(working_dir / "fem_cart_mesh_NSM_orig.vtk"))

    return side


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

def _fit_nsm_subprocess(mesh_paths, save_dir, config_path, bone_only=False, calc_assd=True, seed=42):
    """Run fit_nsm in a subprocess for clean CUDA state.

    Each NSM fit gets a fresh process so CUDA memory layout and cuBLAS
    workspace don't carry over between fits, ensuring reproducibility.
    """
    import subprocess

    cmd = [
        sys.executable, "-c",
        f"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('{__file__}'))))
from steps.run_nsm import fit_nsm
from steps._common import load_config

config = load_config({config_path!r})
mesh_paths = {mesh_paths!r}
result = fit_nsm(mesh_paths, {save_dir!r}, config, bone_only={bone_only!r}, calc_assd={calc_assd!r}, seed={seed!r})
print(json.dumps(result, default=str))
"""
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"fit_nsm subprocess failed:\n{result.stderr[-1000:]}")

    # Parse JSON from the last line of stdout
    stdout_lines = result.stdout.strip().split("\n")
    return json.loads(stdout_lines[-1])


def run(working_dir, options=None, config=None, config_path=None):
    """Run NSM fitting step.

    Args:
        working_dir: Directory containing mesh files from generate_meshes.
        options: Dict with optional keys:
            - nsm_type: "bone_and_cart", "bone_only", or "both" (default: "bone_and_cart")
            - nsm_bones: list of bone names (default: ["femur"])
            - seed: Random seed for NSM fitting (default: 42).
                    Set to None to skip seeding (non-deterministic).
        config: Pipeline config dict.
        config_path: Path to config.json file (needed for subprocess calls).

    Returns:
        Dict with nsm_results and knee_side.
    """
    working_dir = Path(working_dir)
    options = options or {}
    nsm_type = options.get("nsm_type", "bone_and_cart")
    nsm_bones = options.get("nsm_bones", ["femur"])
    seed = options.get("seed", 42)

    # Resolve config_path for subprocess calls
    if config_path is None:
        config_path = os.environ.get(
            "KNEEPIPELINE_CONFIG",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
        )

    results = {}
    knee_side = None

    for bone in nsm_bones:
        emit_progress(5, f"Preparing {bone} meshes for NSM")
        knee_side = _prepare_meshes(working_dir, bone, config)

        if nsm_type in ("bone_and_cart", "both"):
            emit_progress(20, f"Running bone+cartilage NSM for {bone}")
            mesh_paths = [
                str(working_dir / "femur_mesh_NSM_orig.vtk"),
                str(working_dir / "fem_cart_mesh_NSM_orig.vtk"),
            ]
            params = _fit_nsm_subprocess(
                mesh_paths, str(working_dir), config_path, bone_only=False, seed=seed,
            )
            results[f"{bone}_bone_and_cart"] = params

        if nsm_type in ("bone_only", "both"):
            emit_progress(60, f"Running bone-only NSM for {bone}")
            mesh_paths = [str(working_dir / "femur_mesh_NSM_orig.vtk")]
            params = _fit_nsm_subprocess(
                mesh_paths, str(working_dir), config_path, bone_only=True, seed=seed,
            )
            results[f"{bone}_bone_only"] = params

    emit_progress(100, "NSM fitting complete")
    return {"nsm_results": results, "knee_side": knee_side}


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
