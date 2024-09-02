import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from collections import OrderedDict

def list_nii_paths(directory):
    """Generator function to iterate over all nii files in a given directory.

    Args:
        directory: Directory path to search for nii files.

    Returns:
        Sorted array of file paths for each nii file found.
    """
    file_paths = glob.glob(f'{directory}/**/*.nii', recursive=True)
    return np.array(sorted(file_paths))

def list_prostate_paths(directory):
    """Generator function to iterate over all prostate and lesion mask files in a given directory.

    Args:
        directory: Directory path to search for mask files.

    Returns:
        Sorted array of file paths for lesion and prostate mask files found.
    """
    lesion_paths = glob.glob(f'{directory}/**/lesion_mask.npy', recursive=True)
    prostate_paths = glob.glob(f'{directory}/**/prostate_mask.npy', recursive=True)
    return np.array([sorted(lesion_paths), sorted(prostate_paths)])

def nib_to_numpy(directory):
    """Load an image using nibabel, and convert it to a numpy array.

    Args:
        directory: Directory path to convert nib file to numpy array.

    Returns:
        Numpy array of type uint8.
    """
    img = nib.load(directory).get_fdata()
    img = np.nan_to_num(img)
    img_np = np.array(img).astype(np.uint8)
    img_f = img_np.astype(float)
    img_f32 = img_f.astype(np.float32)
    
    img_linear_window = [img_f32.min(), img_f32.max()]

    img_clip = np.clip(img_f32, *img_linear_window)
    
    norm_img = (img_clip - img_linear_window[0]) / (
        img_linear_window[1] - img_linear_window[0]
    )
    
    return norm_img
    
def load_weights(model, directory):
    """Load in pre-trained model weights.

    Args:
        model: Model architecture to load weights.
        directory: Directory path to model weights.

    Returns:
        Model with weights loaded.
    """
    pretrained_dict = torch.load(directory, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    new_state_dict = OrderedDict()

    for k, v in pretrained_dict.items():
        if 'conv' in k and v.dim() == 5:
            middle_index = v.shape[2] // 2
            adapted_v = v[:, :, middle_index, :, :] 
            if adapted_v.shape == model.state_dict()[k].shape:
                new_state_dict[k] = adapted_v
        elif k in model.state_dict() and v.shape == model.state_dict()[k].shape:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    
    return model