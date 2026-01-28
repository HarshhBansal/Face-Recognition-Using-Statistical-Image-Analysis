import os
import cv2
import numpy as np
from scipy.linalg import svd  # Use SciPy for SVD

# Paths to PCA-compressed images
pca_train_folder = r'D:\face recoginization\archive (4)\pca_data\train'
pca_test_folder = r'D:\face recoginization\archive (4)\pca_data\test'

# Output directories for SVD-compressed images
svd_train_folder = r'D:\face recoginization\archive (4)\svd_data\train'
svd_test_folder = r'D:\face recoginization\archive (4)\svd_data\test'

# Create directories for saving SVD-compressed images if not exist
os.makedirs(svd_train_folder, exist_ok=True)
os.makedirs(svd_test_folder, exist_ok=True)

# Function to apply SVD to an image
def apply_svd(image_path, output_path, num_singular_values=100):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Perform SVD
    U, S, Vt = svd(image, full_matrices=False)
    
    # Retain only the top num_singular_values singular values
    S[num_singular_values:] = 0

    # Reconstruct the image using the reduced SVD
    img_svd_compressed = np.dot(U, np.dot(np.diag(S), Vt))

    # Clip and convert the reconstructed image to uint8
    img_svd_compressed = np.clip(img_svd_compressed, 0, 255).astype(np.uint8)

    # Save the SVD-compressed image
    cv2.imwrite(output_path, img_svd_compressed)
    print(f"Saved SVD-compressed image: {output_path}")

# Process train images
for folder_name in os.listdir(pca_train_folder):
    folder_path = os.path.join(pca_train_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing train folder: {folder_name}")
        
        # Create subfolder in SVD train folder
        svd_train_subfolder = os.path.join(svd_train_folder, folder_name)
        os.makedirs(svd_train_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(svd_train_subfolder, file_name)
                apply_svd(file_path, output_path, num_singular_values=100)

# Process test images
for folder_name in os.listdir(pca_test_folder):
    folder_path = os.path.join(pca_test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing test folder: {folder_name}")
        
        # Create subfolder in SVD test folder
        svd_test_subfolder = os.path.join(svd_test_folder, folder_name)
        os.makedirs(svd_test_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(svd_test_subfolder, file_name)
                apply_svd(file_path, output_path, num_singular_values=100)

print("SVD applied to PCA-compressed train and test images!")

