import os
import cv2
import numpy as np
from sklearn.decomposition import PCA

# Paths to mean-centered images
mean_centered_train_folder = r'D:\face recoginization\archive (4)\mean_centered_data\train'
mean_centered_test_folder = r'D:\face recoginization\archive (4)\mean_centered_data\test'

# Output directories for PCA compressed images
pca_train_folder = r'D:\face recoginization\archive (4)\pca_data\train'
pca_test_folder = r'D:\face recoginization\archive (4)\pca_data\test'

# Create directories for saving PCA-compressed images if not exist
os.makedirs(pca_train_folder, exist_ok=True)
os.makedirs(pca_test_folder, exist_ok=True)

# Function to apply PCA to an RGB image
def apply_pca_rgb(image_path, output_path, n_components=200):
    # Read the image in RGB
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Split the image into R, G, B channels
    blue, green, red = cv2.split(image)

    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Apply PCA and reconstruct each channel
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)

    # Stack the reconstructed channels back into an RGB image
    img_pca_compressed = (np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)

    # Save the PCA-compressed image
    cv2.imwrite(output_path, img_pca_compressed)
    print(f"Saved PCA-compressed image: {output_path}")

# Process train images
for folder_name in os.listdir(mean_centered_train_folder):
    folder_path = os.path.join(mean_centered_train_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing train folder: {folder_name}")
        
        # Create subfolder in PCA train folder
        pca_train_subfolder = os.path.join(pca_train_folder, folder_name)
        os.makedirs(pca_train_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(pca_train_subfolder, file_name)
                apply_pca_rgb(file_path, output_path, n_components=200)

# Process test images
for folder_name in os.listdir(mean_centered_test_folder):
    folder_path = os.path.join(mean_centered_test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing test folder: {folder_name}")
        
        # Create subfolder in PCA test folder
        pca_test_subfolder = os.path.join(pca_test_folder, folder_name)
        os.makedirs(pca_test_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(pca_test_subfolder, file_name)
                apply_pca_rgb(file_path, output_path, n_components=200)

print("PCA applied to train and test RGB images!")
