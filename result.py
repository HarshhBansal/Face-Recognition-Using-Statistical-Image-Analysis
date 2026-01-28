import os
import cv2
import numpy as np
from scipy.linalg import svd
import pickle  # For saving and loading the knowledge base
import matplotlib.pyplot as plt  # For displaying images
from sklearn.preprocessing import normalize  # For feature normalization

# Paths to SVD-compressed images
svd_train_folder = r'D:\face recoginization\archive (4)\svd_data\train'
svd_test_folder = r'D:\face recoginization\archive (4)\svd_data\test'

# Path to store knowledge base
knowledge_base_path = r'D:\face recoginization\archive (4)\knowledge_base.pkl'

# Predefined threshold for classification
distance_threshold = None  # Will be dynamically calculated

# Function to normalize a feature vector
def normalize_feature_vector(feature_vector):
    """Normalize the feature vector to have unit norm."""
    return normalize([feature_vector], norm='l2')[0]

# Function to extract and normalize features using SVD
def extract_features(image_path, num_singular_values=100):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Perform SVD
    U, S, Vt = svd(image, full_matrices=False)

    # Retain only the top num_singular_values singular values
    S[num_singular_values:] = 0

    # Create a feature vector using the reduced SVD components
    feature_vector = np.dot(U, np.diag(S)).flatten()  # Combine U and S

    # Normalize the feature vector
    return normalize_feature_vector(feature_vector)

# Step 5: Build Knowledge Base from Train Features
knowledge_base = {}
for folder_name in os.listdir(svd_train_folder):
    folder_path = os.path.join(svd_train_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                feature_vector = extract_features(file_path, num_singular_values=100)
                if feature_vector is not None:
                    # Use file name (or a unique identifier) as the key
                    knowledge_base[file_name] = (feature_vector, file_path)

# Save the knowledge base for future use
with open(knowledge_base_path, 'wb') as f:
    pickle.dump(knowledge_base, f)
    print(f"Knowledge base saved to {knowledge_base_path}")

# Step 6: Classification using City Block Distance (L1 Norm)
def calculate_city_block_distance(vec1, vec2):
    """Calculate City Block Distance (L1 Norm) between two normalized vectors."""
    return np.sum(np.abs(vec1 - vec2))

# Load the knowledge base
with open(knowledge_base_path, 'rb') as f:
    knowledge_base = pickle.load(f)
    print("Knowledge base loaded.")

results = []
distances = []
for folder_name in os.listdir(svd_test_folder):
    folder_path = os.path.join(svd_test_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.tiff'):  # Process only .tiff images
                test_feature = extract_features(file_path, num_singular_values=100)
                if test_feature is not None:
                    # Compare with each entry in the knowledge base
                    best_match = None
                    best_distance = float('inf')
                    best_match_path = None
                    for train_file, (train_feature, train_file_path) in knowledge_base.items():
                        distance = calculate_city_block_distance(test_feature, train_feature)
                        if distance < best_distance:
                            best_distance = distance
                            best_match = train_file
                            best_match_path = train_file_path

                    distances.append(best_distance)  # Collect distances for dynamic threshold

                    # Store the result
                    results.append({
                        "test_image": file_name,
                        "test_image_path": file_path,
                        "best_match": best_match,
                        "best_match_path": best_match_path,
                        "distance": best_distance
                    })

# Calculate dynamic threshold
distance_threshold = np.percentile(distances, 90)  # Top 10% as threshold
print(f"Dynamic Threshold: {distance_threshold:.2f}")

# Update results with classification status
for result in results:
    result["status"] = "Correct Match" if result["distance"] < distance_threshold else "Wrong Match"


# Summary table
print("\nSummary Results:")
for result in results:
    print(f"Test Image: {result['test_image']} -> Best Match: {result['best_match']} "
          f"(Distance: {result['distance']:.2f}, Status: {result['status']})")

print("\nClassification and visualization complete!")

# Calculate dynamic threshold
distance_threshold = np.percentile(distances, 90)  # Top 10% as threshold
print(f"Dynamic Threshold: {distance_threshold:.2f}")

# Update results with classification status
correct_matches = 0  # Count of correct matches
for result in results:
    result["status"] = "Correct Match" if result["distance"] < distance_threshold else "Wrong Match"
    if result["status"] == "Correct Match":
        correct_matches += 1

# Calculate recognition rate
total_test_images = len(results)
recognition_rate = (correct_matches / total_test_images) * 100
print(f"\nRecognition Rate: {recognition_rate:.2f}%")

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import metrics

# Define paths
original_test_folder = r'D:\face recoginization\archive (4)\split_data\test'  # Original test images
processed_test_folder = r'D:\face recoginization\archive (4)\processed_data\test'  # MICE-processed images
output_csv_path = r'D:\face recoginization\metrics_results.csv'  # CSV output path

# Function to calculate metrics for the jth image
def calculate_metrics_jth(original_image, processed_image):
    """
    Calculate metrics (Entropy, AMBE, PSNR, Contrast) for the jth image.
    """
    original_image = np.float32(original_image)
    processed_image = np.float32(processed_image)

    # Normalize histograms for probability distribution
    def calculate_histogram(image):
        histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
        histogram += 1e-10  # Avoid log(0)
        return histogram

    # 1. Entropy
    def entropy(image):
        """Calculate entropy for the jth image."""
        Pj = calculate_histogram(image)  # Probability density
        return -np.sum(Pj * np.log2(Pj))

    entropy_value = entropy(processed_image)

    # 2. Absolute Mean Brightness Error (AMBE)
    def ambe(original, processed):
        """Calculate AMBE for the jth image."""
        return np.abs(np.mean(original) - np.mean(processed))

    ambe_value = ambe(original_image, processed_image)

    # 3. Peak Signal-to-Noise Ratio (PSNR)
    def psnr(original, processed):
        """Calculate PSNR for the jth image."""
        mse = np.mean((original - processed) ** 2)  # Mean Squared Error
        max_pixel_value = 255  # Assuming 8-bit images
        return 10 * np.log10((max_pixel_value ** 2) / mse) if mse > 0 else float('inf')

    psnr_value = psnr(original_image, processed_image)

    # 4. Contrast
    def contrast(image):
        """Calculate contrast for the jth image."""
        Pj = calculate_histogram(image)
        mean_intensity = np.mean(image)  # Mean intensity
        return np.sqrt(np.sum(((np.arange(256) - mean_intensity) ** 2) * Pj))

    contrast_value = contrast(processed_image)

    return {
        "Entropy": entropy_value,
        "AMBE": ambe_value,
        "PSNR": psnr_value,
        "Contrast": contrast_value,
    }

# Process all images and calculate metrics
metrics_results = []

for folder_name in os.listdir(original_test_folder):
    original_subfolder = os.path.join(original_test_folder, folder_name)
    processed_subfolder = os.path.join(processed_test_folder, folder_name)

    if os.path.isdir(original_subfolder) and os.path.isdir(processed_subfolder):
        for file_name in os.listdir(original_subfolder):
            original_image_path = os.path.join(original_subfolder, file_name)
            processed_image_path = os.path.join(processed_subfolder, file_name)

            if os.path.isfile(original_image_path) and os.path.isfile(processed_image_path):
                original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
                processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)

                if original_image is not None and processed_image is not None:
                    metrics_result = calculate_metrics_jth(original_image, processed_image)
                    metrics_result["Image"] = file_name
                    metrics_results.append(metrics_result)

# Save results to a CSV file
df = pd.DataFrame(metrics_results)
df.to_csv(output_csv_path, index=False)
print(f"Metrics saved to CSV at: {output_csv_path}")

# Plot bar graphs for the first 10 images
first_10 = df.head(10)

metrics_to_plot = ["Entropy", "AMBE", "PSNR", "Contrast"]

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    plt.bar(first_10["Image"], first_10[metric])
    plt.title(f" {metric} ")
    plt.xlabel("Image")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("Bar graphs for the first 10 images plotted.")


# Print metrics for all processed images
print("\nMetrics for Processed Images:")
for result in metrics_results:
    print(f"Image: {result['Image']}")
    print(f"  Entropy: {result['Entropy']:.4f}")
    print(f"  AMBE: {result['AMBE']:.4f}")
    print(f"  PSNR: {result['PSNR']:.4f}")
    print(f"  Contrast: {result['Contrast']:.4f}\n")

print("Metrics calculation complete.")

import cv2
import os

def display_first_image_from_folders(folder_paths):
    """
    Displays the first image from each specified folder using OpenCV.
    
    Parameters:
    folder_paths (list): List of tuples containing folder paths and their names.
    """
    for folder_path, folder_name in folder_paths:
        print(f"Searching for images in {folder_name}...")
        image_displayed = False
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    if file_name.endswith('.tiff'):  # Look for .tiff images
                        # Read and display the image
                        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            print(f"Displaying image from {folder_name}: {file_path}")
                            cv2.imshow(f"{folder_name}", image)
                            cv2.waitKey(0)  # Wait for a key press to close the window
                            cv2.destroyAllWindows()
                            image_displayed = True
                            break
                        else:
                            print(f"Could not read image: {file_path}")
                if image_displayed:
                    break
        if not image_displayed:
            print(f"No images found in {folder_name}.")

if __name__ == "__main__":
    # Define the folders to process
    test_folder = r'D:\face recoginization\archive (4)\split_data\test'
    processed_test_folder = r'D:\face recoginization\archive (4)\processed_data\test'
    final_test_folder = r'D:\face recoginization\archive (4)\final_data\test'
    mean_centered_test_folder = r'D:\face recoginization\archive (4)\mean_centered_data\test'
    pca_test_folder = r'D:\face recoginization\archive (4)\pca_data\test'
    svd_test_folder = r'D:\face recoginization\archive (4)\svd_data\test'
    # List of folders and their descriptive names
    folder_paths = [
        (test_folder, "Input Image"),
        (processed_test_folder, "MICE Image"),
        (final_test_folder, "DWT Image"),
        (mean_centered_test_folder,"Mean centering Image"),
        (pca_test_folder,"PCA Image"),
        (svd_test_folder,"SVD Image")
    ]

    # Display the first image from each folder
    display_first_image_from_folders(folder_paths)

# Visualize and compare only the first five results
for i, result in enumerate(results[:5]):  # Limit to the first 5 results
    test_image = cv2.imread(result['test_image_path'], cv2.IMREAD_GRAYSCALE)
    best_match_image = cv2.imread(result['best_match_path'], cv2.IMREAD_GRAYSCALE)

    if test_image is None or best_match_image is None:
        print(f"Error loading images for result {i + 1}")
        continue

    # Show test and best match side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Test Image: {result['test_image']}")
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Best Match: {result['best_match']}\nStatus: {result['status']}\nDistance: {result['distance']:.2f}")
    plt.imshow(best_match_image, cmap='gray')
    plt.axis('off')

    plt.show()
