import sys
import torch
import json
import os
import vtk
import numpy as np
import gc
import time

os.environ['LOC_SDF_CACHE'] = ''

from pymskt.mesh import BoneMesh
from pymskt.mesh.meshTransform import get_linear_transform_matrix

from NSM.models import TriplanarDecoder
from NSM.reconstruct import reconstruct_mesh


# append 'LOC_SDF_CACHE' to environment variables


# system arguments
# path to the femur bone and cartilage meshes
path_meshes = [sys.argv[1]]
# locaton to save results
loc_save_recons = sys.argv[2]
# print reconstruction error metrics
if len(sys.argv) > 3:
    CALC_ASSD = sys.argv[3].lower() == 'true'
else:
    CALC_ASSD = True

# path config is in the same directory as this script
path_script = os.path.dirname(os.path.abspath(__file__))
path_config = os.path.join(path_script, 'config.json')

with open(path_config) as f:
    general_config = json.load(f)


# Setup file paths - relative to this script
path_model_config = general_config['nsm_bone_only']['path_model_config']
path_model_state = general_config['nsm_bone_only']['path_model_state']

# Get BScore model path information
path_bscore_folder = general_config['bscore_bone_only']['path_model_folder']

# append bscore_folder to sys.path and load BScore from Bscore.py
sys.path.append(path_bscore_folder)
from Bscore import Bscore

# Load nsm model config
with open(path_model_config, 'r') as f:
    config = json.load(f)

# define params as needed to be input into network 
params = {
    'latent_dim': config['latent_size'],
    'n_objects': config['objects_per_decoder'],
    'conv_hidden_dims': config['conv_hidden_dims'],
    'conv_deep_image_size': config['conv_deep_image_size'],
    'conv_norm': config['conv_norm'], 
    'conv_norm_type': config['conv_norm_type'],
    'conv_start_with_mlp': config['conv_start_with_mlp'],
    'sdf_latent_size': config['sdf_latent_size'],
    'sdf_hidden_dims': config['sdf_hidden_dims'],
    'sdf_weight_norm': config['weight_norm'],
    'sdf_final_activation': config['final_activation'],
    'sdf_activation': config['activation'],
    'sdf_dropout_prob': config['dropout_prob'],
    'sum_sdf_features': config['sum_conv_output_features'],
    'conv_pred_sdf': config['conv_pred_sdf'],
}

# build the model 
model = TriplanarDecoder(**params)
saved_model_state = torch.load(path_model_state, weights_only=True)
model.load_state_dict(saved_model_state["model"])
model = model.cuda()
model.eval()


# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Reconstruct the meshes (bone/cartilage) using the NSM model
mesh_result = reconstruct_mesh(
    path=path_meshes,
    decoders=model,
    num_iterations=config['num_iterations_recon'],
    register_similarity=True,   
    latent_size=config['latent_size'],
    lr=config['lr_recon'],
    l2reg=config['l2reg_recon'],
    clamp_dist=config['clamp_dist_recon'],
    n_lr_updates=config['n_lr_updates_recon'],
    lr_update_factor=config['lr_update_factor_recon'],
    calc_assd=CALC_ASSD,
    calc_symmetric_chamfer=False, #model_config['chamfer'],
    calc_emd=False, #model_config['emd'],
    convergence=config['convergence_type_recon'], 
    convergence_patience=config['convergence_patience_recon'],
    verbose=True,
    objects_per_decoder=config["objects_per_decoder"],
    batch_size_latent_recon=config['batch_size_latent_recon'],
    get_rand_pts=config['get_rand_pts_recon'],
    n_pts_random=config['n_pts_random_recon'],
    sigma_rand_pts=config['sigma_rand_pts_recon'],
    n_samples_latent_recon=config['n_samples_latent_recon'],
    scale_all_meshes=True,
    func=None,
    scale_jointly=config['scale_jointly'],
    fix_mesh=config['fix_mesh_recon'],
     
    latent_reg_weight=config['l2reg_recon'],
    loss_type='l1',
    return_latent=True,
    return_registration_params=True,
)

# get meshes from results, convert bone to a "BoneMesh" so cartilage thickness can be calculated
bone_mesh = BoneMesh(mesh_result['mesh'][0].mesh)

# get latent - we're not doing anything with this now, but it can be used for downstream predictions
latent = mesh_result['latent'].detach().cpu().numpy().tolist()

# Compute the BScore.... 
bscore = Bscore(latent)

# save the reconstructed meshes 
if not os.path.exists(loc_save_recons):
    os.makedirs(loc_save_recons, exist_ok=True)

bone_mesh.save_mesh(os.path.join(loc_save_recons, f'NSM_bone_only_recon_{os.path.basename(path_meshes[0])}'))

# SAVE THE GENERAL PARAMETERS:
# first, convert the icp_transform to a numpy array if it is a vtk object
icp_transform = mesh_result['icp_transform']
if isinstance(icp_transform, vtk.vtkIterativeClosestPointTransform):
    icp_transform = get_linear_transform_matrix(icp_transform)
elif isinstance(icp_transform, np.ndarray):
    pass
elif isinstance(icp_transform, vtk.vtkTransform):
    icp_transform = get_linear_transform_matrix(icp_transform)
elif isinstance(icp_transform, vtk.vtkMatrix4x4):
    # Convert vtkMatrix4x4 to numpy array
    matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            matrix[i, j] = icp_transform.GetElement(i, j)
    icp_transform = matrix
elif icp_transform is None:
    # Handle case where icp_transform is None
    print("WARNING: icp_transform is None, using identity matrix")
    icp_transform = np.eye(4)
else:
    raise ValueError(f'icp_transform not a valid type: {type(icp_transform)}')

# then save the latent, registration params used to fit the model,
# and the ASSD of the reconstructed meshes vs. the original meshes.  
dict_results = {
    'latent': latent,
    'Bscore': bscore.squeeze().tolist(),
    'icp_transform': icp_transform.tolist(),
    'center': mesh_result['center'].tolist(),
    'scale': mesh_result['scale'],
    'assd_bone_mm': mesh_result['assd_0'],
}

# print all of the dict results, except latent:
for key, val in dict_results.items():
    if key != 'latent':
        print(f'{key}: {val}')

with open(os.path.join(loc_save_recons, 'NSM_bone_only_recon_params.json'), 'w') as f:
    json.dump(dict_results, f, indent=4)
    

# delete the model from memory & clear GPU memory cache
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
time.sleep(5)  # Give CUDA time to actually release memory

