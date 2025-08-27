import sys
import os
import json
import subprocess
import torch
import gc
import time

def run_subprocess(command, process_name):
    """Run a subprocess and handle output/error logging."""
    print(f'Running {process_name}...')
    result = subprocess.run(command, capture_output=True, text=True)
    
    print(f"{process_name} Output:", result.stdout)
    if result.stderr:
        print(f"{process_name} Error:", result.stderr)
    if result.returncode != 0:
        print(f"{process_name} Failed with Exit Code:", result.returncode)
    
    return result.returncode

def main(config_path, path_image, path_save, path_seg_script):
    # Load configuration with default model name
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = sys.argv[3] if len(sys.argv) > 3 else config.get('default_seg_model', 'goyal_sagittal')

    print(f"Running Segmentation Pipeline with Model: {model_name}")
    
    # Run main segmentation pipeline
    seg_command = [
        'python',
        path_seg_script,
        path_image,
        path_save,
        model_name
    ]
    
    if run_subprocess(seg_command, "Segmentation Pipeline") != 0:
        sys.exit("Aborting due to segmentation pipeline failure")

    # Configure NSM analyses
    nsm_analyses = []
    if config.get('perform_bone_and_cart_nsm', False):
        nsm_analyses.append({
            'script': 'NSM_analysis.py',
            'args': [
                os.path.join(path_save, 'femur_mesh_NSM_orig.vtk'),
                os.path.join(path_save, 'fem_cart_mesh_NSM_orig.vtk')
            ]
        })
        
    if config.get('perform_bone_only_nsm', False):
        nsm_analyses.append({
            'script': 'NSM_analysis_bone_only.py',
            'args': [os.path.join(path_save, 'femur_mesh_NSM_orig.vtk')],
            'pre_hook': lambda: (torch.cuda.empty_cache() if torch.cuda.is_available() else None,
                                 gc.collect(),
                                 time.sleep(5))
        })

    # Run NSM analyses
    for analysis in nsm_analyses:
        if 'pre_hook' in analysis:
            analysis['pre_hook']()
            
        command = [
            'python',
            os.path.join(os.path.dirname(__file__), analysis["script"]),
            *analysis['args'],
            path_save
        ]
        
        if run_subprocess(command, analysis["script"]) != 0:
            print(f"Warning: {analysis['script']} failed, continuing with other analyses")

if __name__ == "__main__":
    print('Loading Inputs and Configurations...')
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get config path as config.json in current directory
    config_path = os.path.join(current_dir, 'config.json')
    
    # get path seg script   
    path_seg_script = os.path.join(current_dir, 'seg_thick_t2_pipeline.py')
    
    main(config_path, path_image, path_save, path_seg_script,)