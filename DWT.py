import cv2
import os
import numpy as np
import pywt  # Library for wavelet transformations
from scipy.ndimage import gaussian_filter  # For Gaussian filtering

# Define paths
train_folder = r'D:\face recoginization\archive (4)\split_data\train'
processed_test_folder = r'D:\face recoginization\archive (4)\processed_data\test'

# Output directories for processed images
final_train_folder = r'D:\face recoginization\archive (4)\final_data\train'
final_test_folder = r'D:\face recoginization\archive (4)\final_data\test'

# Create directories if not exist
os.makedirs(final_train_folder, exist_ok=True)
os.makedirs(final_test_folder, exist_ok=True)

# Function to apply DWT, Gaussian filter, and IDWT
def apply_dwt_gaussian_idwt(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Perform DWT (Haar wavelet used here)
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Apply Gaussian filter to the LL (low-frequency) sub-band
    LL_filtered = gaussian_filter(LL, sigma=1)  # Adjust sigma as needed

    # Reconstruct the image using IDWT
    coeffs2_filtered = (LL_filtered, (LH, HL, HH))
    reconstructed_image = pywt.idwt2(coeffs2_filtered, 'haar')

    # Clip values to valid range [0, 255]
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    # Save the reconstructed image
    cv2.imwrite(output_path, np.uint8(reconstructed_image))
    print(f"Saved reconstructed image: {output_path}")

# Process train images
for folder_name in os.listdir(train_folder):
    folder_path = os.path.join(train_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing train folder: {folder_name}")
        
        # Create subfolder in final train folder
        final_train_subfolder = os.path.join(final_train_folder, folder_name)
        os.makedirs(final_train_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(final_train_subfolder, file_name)
                apply_dwt_gaussian_idwt(file_path, output_path)

# Process MICE processed test images
for folder_name in os.listdir(processed_test_folder):
    folder_path = os.path.join(processed_test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing test folder: {folder_name}")
        
        # Create subfolder in final test folder
        final_test_subfolder = os.path.join(final_test_folder, folder_name)
        os.makedirs(final_test_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(final_test_subfolder, file_name)
                apply_dwt_gaussian_idwt(file_path, output_path)

print("DWT, Gaussian filtering, and IDWT applied to train and test images!")
