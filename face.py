import cv2
import os
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

# Define the path to the test folder and the output directory for processed images
test_folder = r'D:\face recoginization\archive (4)\split_data\test'
processed_test_folder = r'D:\face recoginization\archive (4)\processed_data\test'

# Create the processed test directory if it doesn't exist
os.makedirs(processed_test_folder, exist_ok=True)

# Function to apply MICE for imputing missing pixels
def apply_mice(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale images
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Normalize the image to [0, 1] for MICE
    scaler = MinMaxScaler(feature_range=(0, 1))
    image_norm = scaler.fit_transform(image)

    # Identify missing pixels (e.g., assuming 0 indicates missing pixels)
    missing_mask = image_norm == 0
    image_norm[missing_mask] = np.nan  # Replace 0s with NaN for MICE

    # Apply MICE imputation
    imputer = IterativeImputer(max_iter=10, random_state=42)
    image_imputed = imputer.fit_transform(image_norm)

    # Rescale back to the original pixel range [0, 255]
    image_restored = scaler.inverse_transform(image_imputed)
    return np.uint8(image_restored)

# Loop through the test folder and apply MICE
for folder_name in os.listdir(test_folder):
    folder_path = os.path.join(test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Create corresponding subfolder in processed test directory
        processed_subfolder = os.path.join(processed_test_folder, folder_name)
        os.makedirs(processed_subfolder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                print(f"Applying MICE to: {file_path}")
                restored_image = apply_mice(file_path)
                
                if restored_image is not None:
                    # Save the processed image in the processed directory
                    output_path = os.path.join(processed_subfolder, file_name)
                    cv2.imwrite(output_path, restored_image)
                else:
                    print(f"Skipping: {file_path}")

print("MICE imputation complete. Processed images saved in the directory.")


