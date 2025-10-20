"""
File containing utility functions.
Mostly to do with preprocessing of images before feeding to model.
E.g. clipping, cropping, and converting to SAM input format.
"""

import numpy as np
from torch.nn import functional as F
from torchvision.transforms.functional import resize

# Function to take MRI image, clip the pixel values to specified upper
# bound, and normalise between zero and this bound.
# If upper bound not provided, simply rescale
def clip_and_norm(image, lower_bound=0, upper_bound=None):
    # Clip intensity values
    if upper_bound==None:
        upper_bound=np.max(image)
    
    image = np.clip(image, lower_bound, upper_bound)

    # Normalize the image to the range [0, 1]
    norm = (image - lower_bound) / (upper_bound - lower_bound)

    return norm

# Function to perform z-score normalisation on an image
def z_score_norm(image: np.ndarray) -> np.ndarray:
    mean = np.mean(image)
    std = np.std(image)
    norm = (image - mean) / std

    return norm

# This function will crop the OAI MRI images to a pre-chosen size.
# May alter to allow range as an argument.
# Assumes input image is in yxz axis order, but can take other
# permutations if specified. Will return cropped image in yxz order.
def crop_im(image, axis_order='yxz'):
    """Function to crop MRI image to chosen size.
    Args:
        image (np.ndarray): Input MRI image.
        axis_order (str): Axis order of input image. Must be a permutation of 'yxz'.
    Returns:
        np.ndarray: Cropped MRI image in yxz axis order.
    """

    axis_order = axis_order.lower()

    # Ensure image is in yxz order
    # first check that axis order is some permutation of yxz
    assert set(axis_order) == set('yxz'), "axis_order must be a permutation of 'yxz'"

    if axis_order != 'yxz':
        axis_map = {axis: i for i, axis in enumerate(axis_order)}
        image = np.transpose(image, (axis_map['y'], axis_map['x'], axis_map['z']))

    # Cropping indices
    dim1_lower, dim1_upper = 120, 320
    dim2_lower, dim2_upper = 70, 326

    cropped = image[dim1_lower:dim1_upper, dim2_lower:dim2_upper, :]

    return cropped

def undo_crop(cropped_mask):
    """Function to pad the cropped mask in/outputs back to full size

    Args:
        cropped_mask (np.ndarray): Either a previously cropped mask,
        or an outputted prediction of size (200, 256, 160)

    Returns:
        np.ndarray: padded mask of size (384, 384, 160)
    """
    # Original dimensions
    original_shape = (384, 384, 160)
    
    # Cropping indices from the crop_im function
    dim1_lower, dim1_upper = 120, 320
    dim2_lower, dim2_upper = 70, 326
    
    # Initialize a zero array with the original shape
    padded_image = np.zeros(original_shape, dtype=cropped_mask.dtype)
    
    # Place the cropped image in the correct location within the zero-padded array
    padded_image[dim1_lower:dim1_upper, dim2_lower:dim2_upper, :] = cropped_mask
    
    return padded_image

# This function will pad an image upto a square of a give size
def pad_to_square(x, size):
        h, w = x.shape[-2:]
        padh = size - h
        padw = size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

# Function that reads in txt file with each line in format x=y
# and converts to hyperparam dictionary
def read_hyperparams(path):
    hyperparams = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            # Convert to float if possible, else leave as string
            try:
                value = float(value)
            except ValueError:
                pass
            hyperparams[key] = value

    return hyperparams