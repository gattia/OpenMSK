### Environment Creation

- Dependencies are: 
    - DOSMA (bone_seg branch)
    - pymskt
    - NSM
    - nnunetv2
```
# create environment
conda create -n knee_pipeline python=3.8
conda activate knee_pipeline

# install non-python dependcies
conda install -c conda-forge cmake
conda install -c conda-forge lit

# make directory to download and install dependencies
mkdir DEPENDENCIES
cd DEPENDENCIES

# Install NSM
git clone https://github.com/gattia/nsm
cd nsm
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

make requirements
pip install .

# to download NSM models, install huggingface API
pip install huggingface-hub

# Install CUDA 11.8 and cuDNN 8.2 for TensorFlow GPU support
# CRITICAL: TensorFlow 2.11 requires CUDA 11.x and cuDNN 8.x
# PyTorch uses CUDA 12 but both can coexist
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.2 -y

# Install DOSMA
git clone https://github.com/gattia/DOSMA
cd DOSMA
git checkout bone_seg
pip install '.[ai]'

# Install pymskt - this is superfluous becuase its already included in NSM
pip install mskt

# Install nnU-Net inference
# For new clones or after pulling submodule changes:
git submodule update --init --recursive
cd DEPENDENCIES/nnunet_knee_inference
# Follow install instructions in that repo's README for nnU-Net setup

# IF HAVING ISSUES WITH TF, INSTALL VERSION 2.11 (pip install tensorflow==2.11)
# If TensorFlow still can't find GPU, ensure LD_LIBRARY_PATH includes conda lib:
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Known Compatibility Issues: TensorFlow/Keras vs nnU-Net

This environment requires both **TensorFlow/Keras** (for DOSMA segmentation models) and **nnU-Net** (for alternative segmentation).

#### CUDA Version Requirements (Critical for GPU Support)

**TensorFlow 2.11 requires CUDA 11.x**, while PyTorch typically uses CUDA 12.x. Both can coexist in the same conda environment:
- Install `cudatoolkit=11.8` and `cudnn=8.2` (see installation steps above)
- Set `LD_LIBRARY_PATH` to include conda's lib directory when running the pipeline
- For systemd services, add: `Environment="LD_LIBRARY_PATH=/path/to/conda/envs/kneepipeline/lib"`

**Symptoms of CUDA mismatch:**
```
Could not load dynamic library 'libcudart.so.11.0'
Could not load dynamic library 'libcudnn.so.8'
The Conv2D op currently only supports the NHWC tensor format on the CPU
```

These packages also have conflicting NumPy version requirements:

| Package | NumPy Requirement | Notes |
|---------|-------------------|-------|
| TensorFlow 2.11 | < 2.0 (compiled for NumPy 1.x) | Required for DOSMA `.h5` model loading |
| Keras < 3 | < 2.0 | Needed to avoid `groups` argument errors in Conv2DTranspose |
| nnunetv2 | >= 1.24 | Modern deep learning segmentation |
| scikit-image | >= 1.24 | Image processing utilities |
| PyTorch 2.x | 1.x or 2.x | Supports both versions |

#### Symptoms of NumPy Incompatibility

If you see errors like:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
AttributeError: _ARRAY_API not found
ImportError: numpy.core._multiarray_umath failed to import
```

This means your NumPy version is too new (2.x) for TensorFlow 2.11.

#### Symptoms of Keras 2/3 Incompatibility

If you see errors like:
```
ValueError: Unrecognized keyword arguments passed to Conv2DTranspose: {'groups': 1}
```

This means your Keras version is too new (Keras 3) for the saved `.h5` model weights.

#### Recommended Solution

Use **NumPy 1.24.x** as a compromise that satisfies most packages:

```bash
conda activate kneepipeline_310

# Install compatible TensorFlow and Keras versions
pip install tensorflow==2.11
pip install "keras<3"

# Install NumPy 1.24.x (works with TF 2.11 and satisfies nnunetv2/scikit-image)
pip install numpy==1.24.3

# Verify all packages import correctly
python -c "import numpy; import torch; import tensorflow; print(f'NumPy: {numpy.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'TensorFlow: {tensorflow.__version__}')"
```

**Note:** You may still see pip warnings about `blosc2` requiring `numpy>=1.26`, but this typically doesn't affect functionality.

#### TensorRT Warnings (Safe to Ignore)

Warnings about missing `libnvinfer.so.7` are related to optional TensorRT GPU optimization and do not affect model inference:
```
Could not load dynamic library 'libnvinfer.so.7'
TF-TRT Warning: Cannot dlopen some TensorRT libraries
```

### Download models from huggingface

Download NSM HF Models using instructions on: https://github.com/gattia/ShapeMedKnee/
Also, instructions are below: 
```bash
huggingface-cli login
```

Once you run the above, you will then input your access token from huggingface.
You will need to have/create a hugginface account, and then can get the access
token by:
- login to huggingface
- click on user icon in top right
- click settings
- click "access tokens" on the left 
- click "Create new token" on left (topish)
- copy the token and input into the commandline prompt

```python
# in python script: 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="aagatti/ShapeMedKnee", local_dir='./NSM_MODELS')
```
That will download the NSM models into the current directory in a folder
called: `NSM_MODELS`. This will contain both the neural shape model itself
and it will also contain a function and parameters needed to compute a BScore
from the fitted NSM latent vector. 

Download the DOSMA weights so you can perform the segmentation:

```python
# in python script: 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="aagatti/dosma_bones", local_dir='./DOSMA_WEIGHTS')
```

### Update the config file to point to the above files

Update the `config_template.json` so that it includes the paths to your
downloaded files - RESAVE IT AS `config.json`:
    - models/acl_qdess_bone_july_2024: Update to path where `"Goyal_Bone_Cart_July_2024_best_model.h5` saved
    - nsm/path_model_config: update to path of the nsm `model_config.json`
    - nsm/path_model_state: update to path of the nsm model state (`2000.pth`)
    - bscore/path_model_folder: update to folder where bscore json and .py script saved

There is currently a parameter in the config.json that 
determines if the pipeline will perform the NSM analyses. 
```
"perform_bone_only_nsm": false,
"perform_bone_and_cart_nsm": true,
```

By default, it fits the NSM to the bone+cartilage, but does
not do it for the bone only model. The bone only model is
not currently public - please reach out if it is of interest.. 

If you do not want to do any NSM analyses, this can be turned
off, by setting `perform_bone_and_cart_nsm` to `false`.

### Key Configuration Options

Additional configuration parameters in `config.json`:
- `clip_femur_top`: (true/false) - Clips the superior portion of femur meshes during NSM preprocessing 
- `batch_size`: (default: 32) - Batch size for segmentation model inference
- `default_seg_model`: (default: "goyal_sagittal") - Default segmentation model when none specified

### Run script
To run script: 
```bash
python /path/to/dosma_knee_seg.py <path_to_qdess_image> <path_to_save_results> [model_name]
```

**Examples:**
```bash
# Use default DOSMA model
python dosma_knee_seg.py /path/to/image.nii.gz /path/to/results/

# Use specific DOSMA model  
python dosma_knee_seg.py /path/to/image.nii.gz /path/to/results/ goyal_sagittal

# Use nnU-Net segmentation
python dosma_knee_seg.py /path/to/image.nii.gz /path/to/results/ nnunet_knee
```

**Input Format Support:**
- DICOM directories (for full T2 analysis)
- NIfTI files (`.nii`, `.nii.gz`) - RSS/preprocessed images
- NRRD files (`.nrrd`)

**Model Selection:**
The pipeline supports multiple segmentation methods:
- **DOSMA models**: `goyal_sagittal` (default), `goyal_coronal`, `goyal_axial`, `staple`
- **nnU-Net models**: `nnunet_knee` - high-performance deep learning segmentation

The pipeline automatically detects the segmentation method based on the model name prefix. nnU-Net models provide state-of-the-art performance but require additional setup (see installation above).

**T2 Analysis Requirements:**
- T2 mapping only works with DICOM input that contains GL (Gradient Linearity) and TG (Transmit Gain) private tags
- These tags may be removed during DICOM anonymization processes
- T2 values are automatically clipped to the physiological range of 0-80ms
- Analysis includes both full-thickness cartilage and depth-dependent layers (superficial/deep cartilage)

### Results

All results will be saved in <path_to_save_results>. This will include
- segmentations in .nii.gz and .nrrd format
- results.csv and results.json files with the thickness and T2 results by region
- meshes for the bones and cartilage: 
    - femur, tibia, patella, fem cart, med tib cart (0), lat tib cart (1), pat cart
- femur meshes pre-processed for the NSM:
    - femur_mesh_NSM_orig.vtk
    - fem_cart_mesh_NSM_orig.vtk
- Results from the NSM analysis:
    - reconstructed meshes using NSM:
        - NSM_recon_femur_mesh_NSM_orig.vtk
        - NSM_recon_fem_cart_mesh_NSM_orig.vtk
    - NSM fitted parameters in NSM_recon_params.json
        - NSM fitted latent vector (length 512)
        - BScore: osteoarthritis severity score derived from shape patterns learned on the Osteoarthritis Initiative dataset
        - registration parameters used to align femur in NSM space
        - assd metrics for the error in the reconstructed bone/cart surfaces. 




If doing work in notebooks to test then follow below: 

Do work in notebook, need to start job on node and connect to node to run notebook. 
```
ON SERVER: 
salloc -c 4 --gres=gpu:2080ti:1 --mem=24gb --time=1-00

LOCALLY: 
ssh -L 1523:amalfi:22 aagatti@torino
ssh -L 1525:siena:22 aagatti@torino
ssh -L 1524:vigata:22 aagatti@torino

# connect new vscode remotely to the appropriate server (amalfi/vigata/siena)

```