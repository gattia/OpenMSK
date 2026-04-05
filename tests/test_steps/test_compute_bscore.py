"""Tests for steps.compute_bscore."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from steps.compute_bscore import run, _compute_bscore


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
