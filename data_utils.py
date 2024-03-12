import os
import glob
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

def nib_to_numpy(directory):
    """Load an image using nibabel, and convert it to a numpy array.

    Args:
        directory: Directory path to convert nib file to numpy array.

    Yields:
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

def visualize_sample(img_tensor, label_tensor, img_size=(7, 3), pred_tensor=None):
    """Visualizes set of samples to compare image, label, and predicted image (optional).

    Args:
        img_tensor: Tensor of image. 
        img_tensor: Tensor of label image. 
        img_size: Desired size of figure plot. 
        pred_tensor: Tensor of predicted image. 
    
    Yields:
        Single row plot of images.
    """
    
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

def plot_single_metric(loss, dice, label='Training'):
    """Plot the loss and dice scores across each epoch side by side.

    Args:
        loss: Array of training loss values.
        dice: Array of training dice scores.
        label: Label for the plot.

    Yields:
        Single row plot of loss and dice metrics.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    epochs = len(loss)

    axs[0].plot(range(1, epochs + 1), loss)
    axs[0].set_title(f'{label} Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    axs[1].plot(range(1, epochs + 1), dice)
    axs[1].set_title(f'{label} Dice Score')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Dice Score')

    plt.tight_layout()
    plt.show()

def plot_metrics(directory):
    """Plot the loss and dice scores across train and validation set.

    Args:
        directory: Directory path to folder containing metric arrays.

    Yields:
        Two single row plot of training and validation loss and dice metrics.
    """
    train_loss_path = os.path.join(directory, 'train-loss.npy')
    train_dice_path = os.path.join(directory, 'train-dice.npy')
    val_loss_path = os.path.join(directory, 'val-loss.npy')
    val_dice_path = os.path.join(directory, 'val-dice.npy')
    
    train_loss = np.load(train_loss_path)
    train_dice = np.load(train_dice_path)
    val_loss = np.load(val_loss_path)
    val_dice = np.load(val_dice_path)

    plot_single_metric(train_loss, train_dice, label='Training')
    plot_single_metric(val_loss, val_dice, label='Validation')