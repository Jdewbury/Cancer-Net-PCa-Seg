import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def list_nii_paths(directory):
    """Generator function to iterate over all nii files in a given directory.

    Args:
        directory: Directory path to search for nii files.

    Yields:
        Sorted array of file paths for each nii file found.
    """
    file_paths = glob.glob(f'{directory}/**/*.nii', recursive=True)
    return np.array(sorted(file_paths))

def list_prostate_paths(directory):
    """Generator function to iterate over all prostate and lesion mask files in a given directory.

    Args:
        directory: Directory path to search for mask files.

    Yields:
        Sorted array of file paths for lesion and prostate mask files found.
    """
    lesion_paths = glob.glob(f'{directory}/**/lesion_mask.npy', recursive=True)
    prostate_paths = glob.glob(f'{directory}/**/prostate_mask.npy', recursive=True)
    return np.array([sorted(lesion_paths), sorted(prostate_paths)])

def nib_to_numpy(path):
    """
    Load an image using nibabel, and convert it to a numpy array of type uint8.
    """
    image = nib.load(path).dataobj
    return np.array(image).astype(np.uint8)

def normalize_image(img):
    """Normalize images to [0, 1].

    Args:
        img: Image array.

    Yields:
        Normalized image array.
    """
    return img / img.max() if img.max() != 0 else img

def resize_image(img, img_size=(256, 256)):
    """Resize images to desired size.

    Args:
        img: Image array.
        img_size: Desire image array resize.

    Yields:
        Resized image array.
    """
    return cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

def visualize_sample(img_tensor, label_tensor, img_size=(7, 3), pred_tensor=None):
    img = img_tensor.numpy().squeeze()
    label = label_tensor.numpy().squeeze()
    
    if pred_tensor is not None:
        pred = pred_tensor.numpy().squeeze()
        fig, axes = plt.subplots(1, 3, figsize=img_size)
        titles = ['Image', 'True Mask', 'Predicted Mask']
        images = [img, label, pred]
    else:
        fig, axes = plt.subplots(1, 2, figsize=img_size) 
        titles = ['Image', 'True Mask']
        images = [img, label]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off') 

    plt.tight_layout()
    plt.show()