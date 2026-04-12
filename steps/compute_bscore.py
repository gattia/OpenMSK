"""Step 6: BScore computation from NSM latent vectors.

Loads the latent vector from NSM_recon_params.json and computes the BScore
(osteoarthritis severity score) using the BScore model.

BScore computation is pure numpy (no GPU required).
"""

import json
import sys
from pathlib import Path

import numpy as np

from steps._common import emit_progress, parse_step_args, write_step_result


def run(working_dir, options=None, config=None):
    """Run BScore computation step.

    Args:
        working_dir: Directory containing NSM_recon_params.json.
        options: Dict with optional keys:
            - bscore_type: "bone_and_cart", "bone_only", or "both" (default: "bone_and_cart")
            - bscore_bones: list of bone names (default: ["femur"])
        config: Pipeline config dict (needs bscore/bscore_bone_only paths).

    Returns:
        Dict with bscore_results.
    """
    working_dir = Path(working_dir)
    options = options or {}
    bscore_type = options.get("bscore_type", "bone_and_cart")
    bscore_bones = options.get("bscore_bones", ["femur"])

    results = {}

    for bone in bscore_bones:
        if bscore_type in ("bone_and_cart", "both"):
            emit_progress(10, f"Computing bone+cart BScore for {bone}")
            params_file = working_dir / "NSM_recon_params.json"
            model_path = Path(config["bscore"]["path_model_folder"])
            score = _compute_bscore(params_file, model_path)
            results[f"{bone}_bone_and_cart"] = score

        if bscore_type in ("bone_only", "both"):
            emit_progress(50, f"Computing bone-only BScore for {bone}")
            params_file = working_dir / "NSM_bone_only_recon_params.json"
            model_path = Path(config["bscore_bone_only"]["path_model_folder"])
            score = _compute_bscore(params_file, model_path)
            results[f"{bone}_bone_only"] = score

    # Save results
    bscore_path = working_dir / "bscore_results.json"
    bscore_path.write_text(json.dumps(results, indent=2))

    emit_progress(100, "BScore computation complete")
    return {"bscore_results": results}


def _compute_bscore(params_file, model_path):
    """Load latent vector and compute BScore.

    Args:
        params_file: Path to NSM_recon_params.json containing the latent vector.
        model_path: Path to BScore model folder (contains Bscore.py).

    Returns:
        float BScore value.
    """
    params = json.loads(params_file.read_text())
    latent = params["latent"]

    # Import Bscore from the model folder.
    # Remove any cached Bscore module first so we always load from model_path.
    sys.modules.pop("Bscore", None)
    sys.path.insert(0, str(model_path))
    try:
        from Bscore import Bscore
        bscore = Bscore(latent)
    finally:
        sys.path.pop(0)

    return float(np.squeeze(bscore))


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
