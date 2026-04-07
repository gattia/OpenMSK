"""Quick test: does the OLD pipeline segmentation work?"""
import sys
sys.path.insert(0, ".")

from seg_thick_t2_pipeline import segment_image_dosma
import json

config = json.load(open("config.json"))
config["batch_size"] = 8

from dosma import MedicalVolume
import SimpleITK as sitk
img = sitk.ReadImage("data/anthonys_knee.nrrd")
volume = MedicalVolume.from_sitk(img)
print(f"Loaded: shape={volume.shape}")

print("Running old segmentation...")
seg = segment_image_dosma(volume, "acl_qdess_bone_july_2024", config)
print(f"Done: {seg.GetSize()}")
