import cv2
import os
import numpy as np

# Paths to IDWT-processed images
idwt_train_folder = r'D:\face recoginization\archive (4)\final_data\train'
idwt_test_folder = r'D:\face recoginization\archive (4)\final_data\test'

# Output directories for mean-centered images
mean_centered_train_folder = r'D:\face recoginization\archive (4)\mean_centered_data\train'
mean_centered_test_folder = r'D:\face recoginization\archive (4)\mean_centered_data\test'

# Create directories if not exist
os.makedirs(mean_centered_train_folder, exist_ok=True)
os.makedirs(mean_centered_test_folder, exist_ok=True)

# Function to apply mean centering
def apply_mean_centering(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Compute the mean intensity value of the image
    mean_intensity = np.mean(image)

    # Subtract the mean from the image
    mean_centered_image = image - mean_intensity

    # Normalize to ensure values remain in valid range
    mean_centered_image = np.clip(mean_centered_image, 0, 255)

    # Save the mean-centered image
    cv2.imwrite(output_path, np.uint8(mean_centered_image))
    print(f"Saved mean-centered image: {output_path}")

# Process train images
for folder_name in os.listdir(idwt_train_folder):
    folder_path = os.path.join(idwt_train_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing train folder: {folder_name}")
        
        # Create subfolder in mean-centered train folder
        mean_centered_train_subfolder = os.path.join(mean_centered_train_folder, folder_name)
        os.makedirs(mean_centered_train_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(mean_centered_train_subfolder, file_name)
                apply_mean_centering(file_path, output_path)

# Process test images
for folder_name in os.listdir(idwt_test_folder):
    folder_path = os.path.join(idwt_test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing test folder: {folder_name}")
        
        # Create subfolder in mean-centered test folder
        mean_centered_test_subfolder = os.path.join(mean_centered_test_folder, folder_name)
        os.makedirs(mean_centered_test_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                output_path = os.path.join(mean_centered_test_subfolder, file_name)
                apply_mean_centering(file_path, output_path)

print("Mean Centering applied to train and test images!")
