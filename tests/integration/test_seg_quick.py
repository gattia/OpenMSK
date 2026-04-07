"""Quick test: does DOSMA segmentation work on the NRRD at all?"""
import json
import sys
sys.path.insert(0, ".")

config = json.load(open("config.json"))
config["batch_size"] = 8
print(f"Testing with batch_size={config['batch_size']}")

from steps.segment import _load_image, segment_image_dosma

vol, is_qdess, prefix = _load_image("data/anthonys_knee.nrrd")
print(f"Loaded: shape={vol.shape}, is_qdess={is_qdess}, prefix={prefix}")

print("Running segmentation...")
seg = segment_image_dosma(vol, "acl_qdess_bone_july_2024", config)
print(f"Segmentation done: {seg.GetSize()}")
