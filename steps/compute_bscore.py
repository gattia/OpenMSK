"""Step 6: BScore computation from NSM latent vectors.

Loads the latent vector from each NSM params file `run_nsm` left in the working
directory and computes the BScore (osteoarthritis severity score) with the
matching BScore model.

BScore computation is pure numpy (no GPU required).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from steps._common import emit_progress, parse_step_args, write_step_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# NSM variant -> (params file `run_nsm` writes for it, config key holding its
# BScore model folder, progress percent). The filenames are the whole reason
# this step does not need to be told which variant ran: they are what NSM
# produced, so their presence *is* the answer.
VARIANTS = {
    "bone_and_cart": ("NSM_recon_params.json", "bscore", 10),
    "bone_only": ("NSM_bone_only_recon_params.json", "bscore_bone_only", 50),
}


def run(working_dir, options=None, config=None):
    """Run BScore computation step.

    By default the step scores **every NSM params file present** in
    `working_dir` instead of being told which variant to score. The directory
    already answers the question, and unlike an option it cannot disagree with
    what NSM actually ran: the orchestrator passes no options at all, and the
    old default of "bone_and_cart" therefore crashed on bone-only jobs and
    silently scored half of "both" jobs.

    Args:
        working_dir: Directory containing the NSM params file(s).
        options: Dict with optional keys:
            - bscore_type: "bone_and_cart", "bone_only", or "both". Omit it (the
              orchestrated case) to score whatever NSM produced; "both" means
              the same thing. Naming a single variant restricts scoring to that
              one, for standalone CLI use. If a variant is named explicitly but
              its params file is absent, the step logs a warning and omits that
              score rather than raising: BScore is derived from outputs that are
              already safely on disk, so it should not fail a job over the
              absence of one, and the caller sees the missing key in the result
              dict (a step can exit 0 having done nothing, so inspect the
              result, not just the exit code).
            - bscore_bones: list of bone names (default: ["femur"])
        config: Pipeline config dict (needs bscore/bscore_bone_only paths).

    Returns:
        Dict with bscore_results.
    """
    working_dir = Path(working_dir)
    options = options or {}
    config = config or {}
    bscore_type = options.get("bscore_type")
    bscore_bones = options.get("bscore_bones", ["femur"])

    explicit = bscore_type in VARIANTS
    if explicit:
        requested = [bscore_type]
    else:
        if bscore_type not in (None, "both"):
            logging.warning(
                f"Unrecognised bscore_type {bscore_type!r} — scoring every NSM params "
                f"file present instead. Valid values: {sorted(VARIANTS)} or 'both'."
            )
        requested = list(VARIANTS)

    results = {}

    for bone in bscore_bones:
        for variant in requested:
            filename, config_key, percent = VARIANTS[variant]
            params_file = working_dir / filename

            if not params_file.exists():
                if explicit:
                    logging.warning(
                        f"bscore_type={variant!r} was requested but {filename} is not in "
                        f"{working_dir} — NSM did not produce it, so no {variant} BScore."
                    )
                continue

            if config_key not in config:
                logging.warning(
                    f"{filename} is present but config has no {config_key!r} model "
                    f"folder — skipping the {variant} BScore for {bone}."
                )
                continue

            emit_progress(percent, f"Computing {variant.replace('_', ' ')} BScore for {bone}")
            model_path = Path(config[config_key]["path_model_folder"])
            results[f"{bone}_{variant}"] = _compute_bscore(params_file, model_path)

    # Save results
    bscore_path = working_dir / "bscore_results.json"
    bscore_path.write_text(json.dumps(results, indent=2))

    emit_progress(100, "BScore computation complete")
    return {"bscore_results": results}


def _compute_bscore(params_file, model_path):
    """Load latent vector, compute BScore, and write it back into the params file.

    The score is stored under the "Bscore" key of the params file it was computed
    from, alongside the latent vector, so the NSM params file stays
    self-describing (this is where the monolithic pipeline put it, and where
    downstream readers still look). `bscore_results.json` remains the
    authoritative output, so a params file that cannot be rewritten is logged
    and ignored rather than failing the step.

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

    score = float(np.squeeze(bscore))

    params["Bscore"] = score
    try:
        params_file.write_text(json.dumps(params, indent=4))
    except Exception:
        logging.warning(f"Could not write Bscore back to {params_file}", exc_info=True)

    return score


if __name__ == "__main__":
    args = parse_step_args()
    result = run(args.working_dir, args.options, args.config)
    write_step_result(args.working_dir, result)
