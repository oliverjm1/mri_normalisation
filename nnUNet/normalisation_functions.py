# Here are functions for normalisation schemes that did not appear in the default nnunet selection
# Need to add these to nnunetv2/preprocessing/normalization/default_normalization_schemes.py
# Z-score and Rescale already exist in default_normalization_schemes.py

import numpy as np
import torch
import torchio as tio
import os
from skimage import exposure
from sklearn.mixture import GaussianMixture

from nnunetv2.paths import nnUNet_preprocessed

# Function to normalise an image using Gaussian Mixture Model (GMM) normalisation.
# A GMM with 4 components is fitted to the intensities of the image. The 3rd peak is used to normalise the image.
def GMM_norm(image, n_components=4, peak_index=2, sample=False):

    # extract nonzero intensities
    intensities = image[image > 0].flatten()

    # sample intensities for faster computation
    if sample:
        intensities = np.random.choice(intensities, 1000000)

    # Fit GMM with n components
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(intensities.reshape(-1, 1))

    # Get Gaussian means (intensity peaks)
    intensity_peaks = np.sort(gmm.means_.flatten())

    # use 3rd peak to normalise the image
    normalisation_peak = intensity_peaks[peak_index]
    normalised = image / normalisation_peak

    return normalised


class ClipAndRescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # Image first clipped to 1-99th Percentiles to ignore outliers

        # Calculate the 1st and 99th percentiles
        lower_percentile = np.percentile(image, 1)
        upper_percentile = np.percentile(image, 99)
        
        # Clip the image
        image = np.clip(image, lower_percentile, upper_percentile)

        # Image then Rescaled to between 0 and 1
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image
    

class HistogramEqualizationNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # Image first clipped to 0-99.8th Percentiles to ignore extreme outliers

        # Calculate the 0 and 99.8th percentiles
        lower_percentile = np.percentile(image, 0)
        upper_percentile = np.percentile(image, 99.8)

        # Clip the image
        image = np.clip(image, lower_percentile, upper_percentile)

        # Image rescaled to between 0 and 1
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)

        # Image then histogram equalized
        image = exposure.equalize_hist(image)

        return image


class CLAHENormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # Image first clipped to 0-99.8th Percentiles to ignore extreme outliers

        # Calculate the 0 and 99.8th percentiles
        lower_percentile = np.percentile(image, 0)
        upper_percentile = np.percentile(image, 99.8)

        # Clip the image
        image = np.clip(image, lower_percentile, upper_percentile)

        # Image rescaled to between 0 and 1
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)

        # CLAHE normalization
        # kernel size to be 1/10th of the image size
        kernel_size = (image.shape[0] // 10, image.shape[1] // 10, image.shape[2] // 10)
        image = exposure.equalize_adapthist(image, clip_limit=0.01, kernel_size=kernel_size)

        return image
    

class NyulHistogramStandardization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # Load the precomputed histogram landmarks from file
        template_landmarks_path = os.path.join(nnUNet_preprocessed, 'Dataset361_Menisci', 'nyul_landmarks.npy')
        landmarks = np.load(template_landmarks_path, allow_pickle=True)

        # Create a dictionary of landmarks (required for the HistogramStandardization transform)
        landmarks_dict = {'mri': landmarks}
        histogram_transform = tio.HistogramStandardization(landmarks_dict)

        # Convert image to a torchio subject
        tio_image = tio.Subject(
            mri=tio.ScalarImage(tensor=np.expand_dims(image, axis=0))
        )

        # Apply the Nyúl & Udupa histogram standardization
        standard = histogram_transform(tio_image)

        # Convert back to NumPy
        standard_image = standard.mri.data.numpy()

        # ZScore Normalization
        mean = standard_image.mean()
        std = standard_image.std()
        standard_image -= mean
        standard_image /= (max(std, 1e-8))

        return standard_image
    

class GMMStandardization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # Scale to between 0 and 1
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)

        # Apply GMM Normalization
        image = GMM_norm(image, sample=True)

        return image
