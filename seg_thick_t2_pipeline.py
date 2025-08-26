#!/usr/bin/env python
import sys
import os
import json
from dosma.scan_sequences import QDess
from dosma import MedicalVolume
import dosma as dm
from dosma.models import (
    StanfordQDessBoneUNet2D, StanfordCubeBoneUNet2D, 
    StanfordQDessBoneUNet2DSagittal, 
    StanfordQDessBoneUNet2DCoronal, 
    StanfordQDessBoneUNet2DAxial,
    StanfordQDessBoneUNet2DSTAPLE
)
from dosma.tissues import FemoralCartilage
import SimpleITK as sitk
import pymskt as mskt
import numpy as np
import pandas as pd
import warnings
import torch
import gc
import time
from utils import clip_femur_top

# T2 analysis constants
T2_MIN_VALID = 0
T2_MAX_VALID = 80
DEPTH_THRESHOLD = 0.5
DEEP_OFFSET = 100
SUPERFICIAL_OFFSET = 200

def determine_knee_side(seg_array, sitk_seg_subregions):
    """Determine if knee is left or right based on cartilage positions."""
    # figure out if right or left leg. Use the medial/lateral tibial cartilage to determine side
    loc_med_cart = np.mean(np.where(seg_array == 3), axis=1)
    loc_lat_cart = np.mean(np.where(seg_array == 4), axis=1)

    # get rotation matrix
    rotation_matrix = np.array(sitk_seg_subregions.GetDirection()).reshape(3,3)

    # flip the ijk becuase simpleitk returns them in the opposite order of the image space
    # then apply rotation matrix to get them in the correct xyz space orientation
    # not worrying about translations - only care about relative position along x (med/lat) axis.
    loc_med_cart_xyz = rotation_matrix @ loc_med_cart[::-1]
    loc_lat_cart_xyz = rotation_matrix @ loc_lat_cart[::-1]

    loc_med_cart_x = loc_med_cart_xyz[0]
    loc_lat_cart_x = loc_lat_cart_xyz[0]

    # get the xyz location, not the ijk index, for the medial and lateral tibial cartilages
    # then use this to determine the side of the knee.
    if loc_med_cart_x > loc_lat_cart_x:
        return 'right'
    elif loc_med_cart_x < loc_lat_cart_x:
        return 'left'

def main(path_image, path_save, path_config, model_name='goyal_sagittal'):
    print('Loading Inputs and Configurations...')
        
    print('Path to image analyzing:', path_image)



    # READ IN CONFIGURATION STUFF
    with open(path_config) as f:
        config = json.load(f)
    # get parameters for bone/cartilage reconstructions
    dict_bones = config['bones']
    # get the lists of the region names and their seg labels
    dict_regions_ = config['regions']
    # convert the regions to integers
    dict_regions = {}
    for tissue, tissue_dict in dict_regions_.items():
        dict_regions[tissue] = {}
        for region_num, region_name in tissue_dict.items():
            dict_regions[tissue][int(region_num)] = region_name


    print('Loading Image...')
    # figure out if the image is dicom, or nifti, or nrrd & load / get filename
    # as appropriate
    if os.path.isdir(path_image):
        # read in dicom image using DOSMA
        try:
            qdess = QDess.from_dicom(path_image)
        except KeyError:
            qdess = QDess.from_dicom(path_image, group_by='EchoTime')
        volume = qdess.calc_rss()
        filename_save = os.path.basename(path_image)
    elif path_image.endswith(('nii', 'nii.gz')):
        # read in nifti using DOSMA
        # ASSUMING NIFTI SAVED AS RSS /POSTPROCESSED ALREADY
        qdess = None
        nr = dm.NiftiReader()
        volume = nr.load(path_image)
        filename_save = os.path.basename(path_image).split('.nii')[0]
    elif path_image.endswith('nrrd'):
        # read in using SimpleITK, then convert to DOSMA
        qdess = None
        image = sitk.ReadImage(path_image)
        volume = MedicalVolume.from_sitk(image)
        filename_save = os.path.basename(path_image).split('.nrrd')[0]
    else:
        raise ValueError('Image format not supported.')


    print('Loading Model...')
    # load the appropriate segmentation model & its weights
    # set the actual model class being used
    if model_name == 'staple':
        print('Using Staple Model')
        model = StanfordQDessBoneUNet2DSTAPLE(
            config['models']['goyal_sagittal'],
            config['models']['goyal_coronal'],
            config['models']['goyal_axial']
        )
    else:
        orig_model_image_size = (512,512)
        if 'cube' in model_name:
            print('Using Cube Model')
            model_class = StanfordCubeBoneUNet2D
        elif model_name == 'goyal_sagittal':
            print('Using Goyal Sagittal Model')
            model_class = StanfordQDessBoneUNet2DSagittal
            orig_model_image_size = None
        elif model_name == 'goyal_coronal':
            print('Using Goyal Coronal Model')
            model_class = StanfordQDessBoneUNet2DCoronal
            orig_model_image_size = None
        elif model_name == 'goyal_axial':
            print('Using Goyal Axial Model')
            model_class = StanfordQDessBoneUNet2DAxial
            orig_model_image_size = None
        
        else:
            model_class = StanfordQDessBoneUNet2D
        # load the model. 
        print(f'Loading {model_name} model, orig_model_image_size: {orig_model_image_size}')
        model = model_class(config['models'][model_name], orig_model_image_size=orig_model_image_size)
    
    model.batch_size = int(config['batch_size'])


    print('Segmenting Image...')
    # SEGMENT THE MRI
    seg = model.generate_mask(volume)

    # save the segmentation as nifti
    nw = dm.NiftiWriter()
    nw.save(seg['all'], os.path.join(path_save, filename_save + '_all-labels.nii.gz'))

    # convert seg to sitk format for pymskt processing
    sitk_seg = seg['all'].to_sitk(image_orientation='sagittal')
    # save the segmentation as nrrd
    sitk.WriteImage(sitk_seg, os.path.join(path_save, filename_save + '_all-labels.nrrd'), useCompression=True)


    print('Creating Meshes and Computing Cartilage Thickness...')
    # break segmentation into subregions
    sitk_seg_subregions = mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions(
        sitk_seg,
        fem_cart_label_idx=2,
        wb_region_percent_dist=0.6,
        femur_label=7,
        med_tibia_label=3,
        lat_tibia_label=4,
        ant_femur_mask=11,
        med_wb_femur_mask=12,
        lat_wb_femur_mask=13,
        med_post_femur_mask=14,
        lat_post_femur_mask=15,
        verify_med_lat_tib_cart=True,
        tibia_label=8,
        ml_axis=0,
    )
    
    # save the subregions to disk
    sitk.WriteImage(sitk_seg_subregions, os.path.join(path_save, f'{filename_save}_subregions-labels.nrrd'), useCompression=True)
    sitk.WriteImage(sitk_seg_subregions, os.path.join(path_save, f'{filename_save}_subregions-labels.nii.gz'), useCompression=True)

    # create 3D surfaces w/ cartilage thickness & compute thickness metrics
    dict_results = {}
    for bone_name, bone_config in dict_bones.items():
        # create bone mesh and crop as appropriate
        bone_mesh = mskt.mesh.BoneMesh(
            seg_image=sitk_seg,
            label_idx=bone_config['tissue_idx'],
            list_cartilage_labels=bone_config['list_cart_labels'],
            bone=bone_name,
            crop_percent=bone_config['crop_percent'],
        )
        bone_mesh.create_mesh(smooth_image_var=0.5)
        bone_mesh.resample_surface(clusters=bone_config['n_points'])
        
        # fix bone mesh
        bone_mesh.fix_mesh()
        
        # compute cartilage thickness metrics - on surfaces
        bone_mesh.calc_cartilage_thickness(image_smooth_var_cart=0.3125)
        bone_mesh.seg_image = sitk_seg_subregions
        
        # fix cartilage surface
        for cart_mesh in bone_mesh.list_cartilage_meshes:
            cart_mesh.fix_mesh()
        
        # get labels to compute thickness metrics
        if bone_name == 'femur':
            cart_labels = [11, 12, 13, 14, 15]
            bone_mesh.list_cartilage_labels=cart_labels
        else:
            cart_labels = bone_config['list_cart_labels']
        # assign labels to bone surface
        bone_mesh.assign_cartilage_regions()
        
        # store this mesh in dict for later use
        dict_bones[bone_name]['mesh'] = bone_mesh
        
        # get thickness and region for each bone vertex
        thickness = np.array(bone_mesh.get_scalar('thickness (mm)'))
        regions = np.array(bone_mesh.get_scalar('labels'))
        
        # for each region, compute thicknes statics. 
        for region in cart_labels:
            dict_results[f"{dict_regions['cart'][region]}_mm_mean"] = np.nanmean(thickness[regions == region])
            dict_results[f"{dict_regions['cart'][region]}_mm_std"] = np.nanstd(thickness[regions == region])
            dict_results[f"{dict_regions['cart'][region]}_mm_median"] = np.nanmedian(thickness[regions == region])
        
        
        # save the bone and cartilage meshes. 
        bone_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_mesh.vtk'))
        # iterate over the cartilage meshes associated with the bone_mesh and save: 
        for cart_idx, cart_mesh in enumerate(bone_mesh.list_cartilage_meshes):
            cart_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_cart_{cart_idx}_mesh.vtk'))


    print('Computing T2 Maps and Metrics...')
    # need seg_array for preprocessing related to NSM fitting. 
    seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)
    
    if (qdess is not None):
        include_required_tags = (
            (qdess.get_metadata(qdess.__GL_AREA_TAG__, None) is not None)
            and (qdess.get_metadata(qdess.__TG_TAG__, None) is not None)
        )
        if include_required_tags:
            # See if gl and tg private tags are present, if not, 32Askip T2 computation
            # create T2 map and clip values
            cart = FemoralCartilage()
            t2map = qdess.generate_t2_map(cart, suppress_fat=False, suppress_fluid=False)

            # convert to sitk for mean T2 computation
            sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation='sagittal')
            
            # save the t2 map
            sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nii.gz'), useCompression=False)
            sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nrrd'), useCompression=False)

            seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

            # get T2 as array and set values outside of expected/reasonable range to nan
            t2_array = sitk.GetArrayFromImage(sitk_t2map)
            t2_array[t2_array >= T2_MAX_VALID] = np.nan
            t2_array[t2_array <= T2_MIN_VALID] = np.nan
            
            

            # compute T2 metrics for each region & store in results dictionary
            for cart_idx, cart_region in dict_regions['cart'].items():
                if cart_idx in seg_array:
                    mean_t2 = np.nanmean(t2_array[seg_array == cart_idx])
                    std_t2 = np.nanstd(t2_array[seg_array == cart_idx])
                    median_t2 = np.nanmedian(t2_array[seg_array == cart_idx])
                    dict_results[f'{cart_region}_t2_ms_mean'] = mean_t2
                    dict_results[f'{cart_region}_t2_ms_std'] = std_t2
                    dict_results[f'{cart_region}_t2_ms_median'] = median_t2
            
            # convert segmentation into simple depth dependent version of the segmentation.
            for bone_name, bone_config in dict_bones.items():
                bone_mesh = bone_config['mesh']
                # update bone_mesh list_cartilage_labels to be the original ones
                # this is only really needed for the femur, but we do it for all bones... just in case. 
                bone_mesh.list_cartilage_labels = bone_config['list_cart_labels']
                # assign the segmentation mask to be the original one.. 
                bone_mesh.seg_image = sitk_seg
                bone_new_seg, bone_rel_depth = bone_mesh.break_cartilage_into_superficial_deep(rel_depth_thresh=DEPTH_THRESHOLD, return_rel_depth=True, resample_cartilage_surface=10_000)
                bone_config['bone_new_seg'] = bone_new_seg
                bone_config['bone_rel_depth'] = bone_rel_depth
            new_seg_combined = mskt.image.cartilage_processing.combine_depth_region_segs(
                sitk_seg_subregions,
                [x['bone_new_seg'] for x in dict_bones.values()],
            )
            
            # save the depth dependent segmentation to disk 
            sitk.WriteImage(new_seg_combined, os.path.join(path_save, f'{filename_save}_depth_seg.nrrd'), useCompression=True)
            
            # compute T2 metrics for each region & store in results dictionary
            # store as superficial / deep T2 maps. 
            seg_array_depth = sitk.GetArrayFromImage(new_seg_combined)
            for cart_idx, cart_region in dict_regions['cart'].items():
                for depth_idx, depth_name in [(DEEP_OFFSET, 'deep'), (SUPERFICIAL_OFFSET, 'superficial')]:
                    cart_idx_depth = cart_idx + depth_idx
                    if cart_idx_depth in seg_array_depth:
                        mean_t2 = np.nanmean(t2_array[seg_array_depth == cart_idx_depth])
                        std_t2 = np.nanstd(t2_array[seg_array_depth == cart_idx_depth])
                        median_t2 = np.nanmedian(t2_array[seg_array_depth == cart_idx_depth])
                        dict_results[f'{cart_region}_{depth_name}_t2_ms_mean'] = mean_t2
                        dict_results[f'{cart_region}_{depth_name}_t2_ms_std'] = std_t2
                        dict_results[f'{cart_region}_{depth_name}_t2_ms_median'] = median_t2
        else:
            warnings.warn(
                'GL and TG tags not present. Skipping T2 computation. '+
                'NOTE: These are private tags and may have been removed ' +
                'in the DICOM anonymization process.')
    else:
        print('Not a qdess image. Skipping T2 computation.')

    print('Saving Results...')
    # SAVE THICKNESS & T2 METRICS
    # save as csv
    df = pd.DataFrame([dict_results])
    df.to_csv(os.path.join(path_save, f'{filename_save}_results.csv'), index=False)

    # save as json
    with open(os.path.join(path_save, f'{filename_save}_results.json'), 'w') as f:
        json.dump(dict_results, f, indent=4)
    
    print('Saving Meshes for NSM Fitting...')
    
    # look at the config  says to analyze nsm bone or bone and cartilage... 
    if config['perform_bone_only_nsm'] or config['perform_bone_and_cart_nsm']:
        # Determine knee side by comparing medial vs lateral tibial cartilage positions
        # in world coordinates after applying rotation matrix
        side = determine_knee_side(seg_array, sitk_seg_subregions)

        # if side is left, flip the mesh to be a right knee
        if side == 'left':
            femur = dict_bones['femur']['mesh'] # in the future, copy this when BoneMesh has own copy method
            # get the center of the mesh - so we can translate it back to have the same "center"
            center = np.mean(femur.point_coords, axis=0)[0]
            femur.point_coords = femur.point_coords * [-1, 1, 1]
            # move the mesh back so the center is the same as before the flip along x-axis. 
            femur.point_coords = femur.point_coords + [2*center, 0, 0]
            # apply transformation to the cartilage mesh
            fem_cart = femur.list_cartilage_meshes[0].copy()
            fem_cart.point_coords = fem_cart.point_coords * [-1, 1, 1]
            fem_cart.point_coords = fem_cart.point_coords + [2*center, 0, 0]
        else:
            femur = dict_bones['femur']['mesh']
            fem_cart = femur.list_cartilage_meshes[0]
        
        if config['clip_femur_top']:
            # this will clip the femur to be 70% of width, or 95% of height, whichever is larger.
            femur = clip_femur_top(femur)
   
        # save the femur and cartilage meshes - this is in "NSM" format
        femur.save_mesh(os.path.join(path_save, 'femur_mesh_NSM_orig.vtk'))
        fem_cart.save_mesh(os.path.join(path_save, 'fem_cart_mesh_NSM_orig.vtk'))

    # Memory cleanup
    print('Cleaning up memory...')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

if __name__ == "__main__":
    # Read command line arguments
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'goyal_sagittal'
    
    # set path to config as config.json in current directory
    path_config = os.path.join(os.path.dirname(__file__), 'config.json')
    
    main(path_image, path_save, path_config, model_name)
