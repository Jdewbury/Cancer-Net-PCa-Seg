import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def list_nii_paths(directory):
    """Generator function to iterate over all nii files in a given directory.

    Args:
        directory: Directory path to search for nii files.

    Yields:
        Sorted array of file paths for each nii file found.
    """
    file_paths = glob.glob(f'{directory}/**/*.nii', recursive=True)
    return np.array(sorted(file_paths))

def normalize_image(img):
    """Normalize images to [0, 1].

    Args:
        img: Image array.

    Yields:
        Normalized image array.
    """
    if img.max() == 0:
        return img
    return img / img.max()

def resize_image(img, img_size=(256, 256)):
    """Resize images to desired size.

    Args:
        img: Image array.
        img_size: Desire image array resize.

    Yields:
        Resized image array.
    """
    return cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

def visualize_sample(img_tensor, mask_tensor):
    img = img_tensor.numpy().squeeze()
    mask = mask_tensor.numpy().squeeze()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    plt.show()