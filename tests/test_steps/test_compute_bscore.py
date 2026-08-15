"""Tests for steps.compute_bscore."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from steps.compute_bscore import run, _compute_bscore


def _write_bscore_model(model_dir, value):
    """Write a fake BScore model folder that always returns `value`."""
    model_dir.mkdir(exist_ok=True)
    (model_dir / "Bscore.py").write_text(
        "import numpy as np\n"
        "def Bscore(latent):\n"
        f"    return np.array([{value}])\n"
    )
    return model_dir


class TestComputeBscore:
    """Tests for _compute_bscore helper."""

    def test_loads_latent_and_computes(self, tmp_path):
        """Should load latent from params file and compute BScore."""
        # Create a fake params file
        latent = list(np.random.randn(512))
        params = {"latent": latent, "icp_transform": np.eye(4).tolist()}
        params_file = tmp_path / "NSM_recon_params.json"
        params_file.write_text(json.dumps(params))

        # Create a fake Bscore module in a temp directory
        model_dir = tmp_path / "bscore_model"
        model_dir.mkdir()
        bscore_py = model_dir / "Bscore.py"
        bscore_py.write_text(
            "import numpy as np\n"
            "def Bscore(latent):\n"
            "    return np.array([0.42])\n"
        )

        result = _compute_bscore(params_file, model_dir)
        assert isinstance(result, float)
        assert result == pytest.approx(0.42)


class TestRun:
    """Tests for the compute_bscore run() entry point."""

    def test_run_bone_and_cart(self, tmp_path):
        """Should compute BScore for bone+cart type."""
        # Create params file
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))

        # Create mock Bscore module
        model_dir = tmp_path / "bscore_model"
        model_dir.mkdir()
        (model_dir / "Bscore.py").write_text(
            "import numpy as np\n"
            "def Bscore(latent):\n"
            "    return np.array([1.23])\n"
        )

        config = {
            "bscore": {"path_model_folder": str(model_dir)},
            "bscore_bone_only": {"path_model_folder": str(model_dir)},
        }

        result = run(tmp_path, options={"bscore_type": "bone_and_cart"}, config=config)

        assert "femur_bone_and_cart" in result["bscore_results"]
        assert result["bscore_results"]["femur_bone_and_cart"] == pytest.approx(1.23)

        # Results should be saved to disk
        saved = json.loads((tmp_path / "bscore_results.json").read_text())
        assert "femur_bone_and_cart" in saved

    def test_run_bone_only(self, tmp_path):
        """Should compute BScore for bone-only type."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        model_dir = tmp_path / "bscore_model"
        model_dir.mkdir()
        (model_dir / "Bscore.py").write_text(
            "import numpy as np\n"
            "def Bscore(latent):\n"
            "    return np.array([0.77])\n"
        )

        config = {
            "bscore": {"path_model_folder": str(model_dir)},
            "bscore_bone_only": {"path_model_folder": str(model_dir)},
        }

        result = run(tmp_path, options={"bscore_type": "bone_only"}, config=config)

        assert "femur_bone_only" in result["bscore_results"]
        assert result["bscore_results"]["femur_bone_only"] == pytest.approx(0.77)

    def test_run_both(self, tmp_path):
        """Should compute BScore for both types when nsm_type is 'both'."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        model_dir = tmp_path / "bscore_model"
        model_dir.mkdir()
        (model_dir / "Bscore.py").write_text(
            "import numpy as np\n"
            "def Bscore(latent):\n"
            "    return np.array([0.5])\n"
        )

        config = {
            "bscore": {"path_model_folder": str(model_dir)},
            "bscore_bone_only": {"path_model_folder": str(model_dir)},
        }

        result = run(tmp_path, options={"bscore_type": "both"}, config=config)

        assert "femur_bone_and_cart" in result["bscore_results"]
        assert "femur_bone_only" in result["bscore_results"]


class TestScoresWhateverNsmProduced:
    """D5: with no `bscore_type`, score every NSM params file that is present.

    The orchestrator passes no options, so the step has to work out for itself
    what NSM ran. The directory it is standing in already says.
    """

    def test_bone_only_params_alone_are_scored(self, tmp_path):
        """A bone-only NSM job: no options, no NSM_recon_params.json, no crash."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        result = run(tmp_path, config=config)

        assert result["bscore_results"] == {"femur_bone_only": pytest.approx(-0.75)}
        saved = json.loads((tmp_path / "bscore_results.json").read_text())
        assert saved == {"femur_bone_only": pytest.approx(-0.75)}

    def test_bone_and_cart_params_alone_are_scored(self, tmp_path):
        """A bone+cart NSM job: only that variant is scored."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        result = run(tmp_path, config=config)

        assert result["bscore_results"] == {"femur_bone_and_cart": pytest.approx(1.5)}

    def test_both_present_each_scored_from_its_own_file(self, tmp_path):
        """Each variant's score comes from its own params file and its own model.

        `_write_bscore_model()` interpolates its `value` into the module body, so
        passing an expression gives a fake model that echoes the latent it was
        handed — which is how the file each score was computed from is pinned
        down, not just the model that computed it.
        """
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps({"latent": [11.0, 0.0]}))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(
            json.dumps({"latent": [22.0, 0.0]})
        )

        config = {
            "bscore": {
                "path_model_folder": str(
                    _write_bscore_model(tmp_path / "bc", "latent[0] + 0.5")
                )
            },
            "bscore_bone_only": {
                "path_model_folder": str(
                    _write_bscore_model(tmp_path / "bo", "latent[0] + 0.25")
                )
            },
        }

        result = run(tmp_path, config=config)

        # 11.5 could only come from the bone+cart file through the bone+cart
        # model; 22.25 only from the bone-only file through the bone-only model.
        assert result["bscore_results"]["femur_bone_and_cart"] == pytest.approx(11.5)
        assert result["bscore_results"]["femur_bone_only"] == pytest.approx(22.25)

    def test_no_params_files_at_all(self, tmp_path):
        """NSM did not run (or was disabled): empty results, exit 0, no crash."""
        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        result = run(tmp_path, config=config)

        assert result["bscore_results"] == {}
        assert json.loads((tmp_path / "bscore_results.json").read_text()) == {}

    def test_explicit_bscore_type_still_restricts_scoring(self, tmp_path):
        """Standalone CLI use: an explicit variant is honoured, not overridden."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        result = run(tmp_path, options={"bscore_type": "bone_only"}, config=config)

        assert result["bscore_results"] == {"femur_bone_only": pytest.approx(-0.75)}
        # The bone+cart params file was left alone entirely.
        assert "Bscore" not in json.loads((tmp_path / "NSM_recon_params.json").read_text())

    def test_explicit_variant_with_no_params_file_warns_instead_of_raising(
        self, tmp_path, caplog
    ):
        """An explicitly requested variant NSM never produced: warn, do not raise.

        BScore is derived from outputs that are already safe on disk, so it must
        not fail the job; the caller sees the absent key in the result dict.
        """
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        with caplog.at_level(logging.WARNING):
            result = run(tmp_path, options={"bscore_type": "bone_and_cart"}, config=config)

        assert result["bscore_results"] == {}
        assert json.loads((tmp_path / "bscore_results.json").read_text()) == {}
        assert "NSM_recon_params.json" in caplog.text

    def test_explicit_both_scores_whichever_half_exists(self, tmp_path):
        """`run_pipeline.py` passes the NSM type through; half of it may be missing."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        result = run(tmp_path, options={"bscore_type": "both"}, config=config)

        assert result["bscore_results"] == {"femur_bone_only": pytest.approx(-0.75)}

    def test_unconfigured_model_folder_is_skipped_not_raised(self, tmp_path):
        """A params file with no BScore model configured for it is not fatal."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
        }

        result = run(tmp_path, config=config)

        assert result["bscore_results"] == {"femur_bone_and_cart": pytest.approx(1.5)}


class TestBscoreWrittenBackToParams:
    """The score is stored in the NSM params file it was computed from (D2)."""

    def test_params_file_gains_bscore_and_keeps_everything_else(self, tmp_path):
        """Params file gets `Bscore`; latent and the rest survive the rewrite."""
        latent = list(np.random.randn(512))
        params = {
            "latent": latent,
            "icp_transform": np.eye(4).tolist(),
            "center": [1.0, 2.0, 3.0],
            "scale": 1.5,
            "assd_bone_mm": 0.25,
        }
        params_file = tmp_path / "NSM_recon_params.json"
        params_file.write_text(json.dumps(params))

        model_dir = _write_bscore_model(tmp_path / "bscore_model", 1.23)
        config = {
            "bscore": {"path_model_folder": str(model_dir)},
            "bscore_bone_only": {"path_model_folder": str(model_dir)},
        }

        result = run(tmp_path, options={"bscore_type": "bone_and_cart"}, config=config)

        written = json.loads(params_file.read_text())
        assert written["Bscore"] == pytest.approx(1.23)

        # bscore_results.json is still written, and the two agree.
        saved = json.loads((tmp_path / "bscore_results.json").read_text())
        assert saved["femur_bone_and_cart"] == pytest.approx(written["Bscore"])
        assert result["bscore_results"]["femur_bone_and_cart"] == pytest.approx(1.23)

        # Nothing already in the params file was lost or altered.
        assert written["latent"] == latent
        assert written["icp_transform"] == params["icp_transform"]
        assert written["center"] == params["center"]
        assert written["scale"] == params["scale"]
        assert written["assd_bone_mm"] == params["assd_bone_mm"]

    def test_each_variant_gets_the_score_computed_from_it(self, tmp_path):
        """Bone+cart and bone-only params files get their own scores."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        run(tmp_path, options={"bscore_type": "both"}, config=config)

        bone_and_cart = json.loads((tmp_path / "NSM_recon_params.json").read_text())
        bone_only = json.loads((tmp_path / "NSM_bone_only_recon_params.json").read_text())
        assert bone_and_cart["Bscore"] == pytest.approx(1.5)
        assert bone_only["Bscore"] == pytest.approx(-0.75)

        saved = json.loads((tmp_path / "bscore_results.json").read_text())
        assert saved["femur_bone_and_cart"] == pytest.approx(bone_and_cart["Bscore"])
        assert saved["femur_bone_only"] == pytest.approx(bone_only["Bscore"])

    def test_write_back_survives_the_d5_discovery_loop(self, tmp_path):
        """Every variant D5 discovers on disk still gets its score written back."""
        params = {"latent": list(np.random.randn(512))}
        (tmp_path / "NSM_recon_params.json").write_text(json.dumps(params))
        (tmp_path / "NSM_bone_only_recon_params.json").write_text(json.dumps(params))

        config = {
            "bscore": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bc", 1.5))
            },
            "bscore_bone_only": {
                "path_model_folder": str(_write_bscore_model(tmp_path / "bo", -0.75))
            },
        }

        # No options at all -- the orchestrated call.
        run(tmp_path, config=config)

        bone_and_cart = json.loads((tmp_path / "NSM_recon_params.json").read_text())
        bone_only = json.loads((tmp_path / "NSM_bone_only_recon_params.json").read_text())
        assert bone_and_cart["Bscore"] == pytest.approx(1.5)
        assert bone_only["Bscore"] == pytest.approx(-0.75)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_unwritable_params_file_does_not_fail_the_step(self, tmp_path):
        """bscore_results.json is authoritative; a failed rewrite is logged only."""
        params = {"latent": list(np.random.randn(512))}
        params_file = tmp_path / "NSM_recon_params.json"
        params_file.write_text(json.dumps(params))
        params_file.chmod(0o444)

        model_dir = _write_bscore_model(tmp_path / "bscore_model", 0.9)
        config = {
            "bscore": {"path_model_folder": str(model_dir)},
            "bscore_bone_only": {"path_model_folder": str(model_dir)},
        }

        try:
            result = run(tmp_path, options={"bscore_type": "bone_and_cart"}, config=config)
        finally:
            params_file.chmod(0o644)

        assert result["bscore_results"]["femur_bone_and_cart"] == pytest.approx(0.9)
        saved = json.loads((tmp_path / "bscore_results.json").read_text())
        assert saved["femur_bone_and_cart"] == pytest.approx(0.9)

        # The params file is intact — no partial write.
        assert json.loads(params_file.read_text()) == params
