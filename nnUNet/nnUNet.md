# Folder for nnU-Net

This is the folder where nnUNet should be cloned locally.

Other than z-score and [0, 1] rescaling, which are already present in the nnUNet default normalisation schemes, all normalisation functions used in this study are found in `normalisation_functions.py`. These should be added to the `nnunetv2/preprocessing/normalization/default_normalization_schemes.py` file. The `nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py` file can also be edited, adding to the `channel_name_to_normalization_mapping` dictionary to allow for normalisation scheme names to be specified when running the nnU-Net experiment planner pipeline.
