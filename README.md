# Face Recognition Pipeline Using MICE, DWT, PCA and SVD

## 📌 Project Overview
This project presents an **end-to-end face recognition system** built using **classical image processing and machine learning techniques**.  
Instead of deep learning, the system focuses on **statistical feature extraction, dimensionality reduction, and distance-based classification**, making it computationally efficient and interpretable.

The pipeline handles **noisy and incomplete face images**, enhances image quality, extracts discriminative features, and performs **face recognition with performance evaluation**.

## 🎯 Key Objectives
- Restore corrupted or incomplete face images
- Enhance facial features using signal processing techniques
- Reduce dimensionality while preserving important information
- Perform reliable face recognition
- Evaluate both **recognition accuracy** and **image quality metrics**

## 🧠 Methodology / Pipeline

Input Face Images
↓
MICE Imputation (Missing Pixel Restoration)
↓
DWT + Gaussian Filter + IDWT (Denoising & Enhancement)
↓
Mean Centering (Normalization)
↓
PCA (Dimensionality Reduction on RGB Images)
↓
SVD (Feature Compression on Grayscale Images)
↓
Feature Extraction & Normalization
↓
Distance-Based Classification (City Block Distance)
↓
Performance Evaluation & Visualization

## 🛠️ Techniques Used

### 🔹 MICE (Multiple Imputation by Chained Equations)
- Restores missing or corrupted pixels
- Improves robustness of the recognition system

### 🔹 Discrete Wavelet Transform (DWT)
- Separates low- and high-frequency components
- Gaussian filtering applied to reduce noise
- Image reconstructed using IDWT

### 🔹 Mean Centering
- Normalizes images by removing mean intensity
- Improves effectiveness of PCA and SVD

### 🔹 Principal Component Analysis (PCA)
- Applied on RGB channels
- Reduces dimensionality while preserving visual information

### 🔹 Singular Value Decomposition (SVD)
- Extracts compact and discriminative features
- Reduces redundancy in facial data

### 🔹 Classification
- Feature vectors normalized using L2 norm
- Face matching performed using **City Block (L1) distance**
  
## 📁 Project Structure

```text
Face-Recognition-Using-Statistical-Image-Analysis/
│
├── face.py              # MICE-based image restoration
├── DWT.py               # DWT + Gaussian filter + IDWT
├── mean_centering.py    # Mean centering of images
├── PCA.py               # PCA-based dimensionality reduction
├── SVD.py               # SVD-based feature compression
├── result.py            # Feature extraction, classification & evaluation
├── metrics_results.csv  # Evaluation metrics
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
└── results/             # Output results
```


## 📊 Performance Evaluation

### 🔸 Recognition Metrics
- Recognition Rate (%)
- Distance-based match validation
- Dynamic thresholding (90th percentile)

### 🔸 Image Quality Metrics
- **Entropy**
- **Peak Signal-to-Noise Ratio (PSNR)**
- **Absolute Mean Brightness Error (AMBE)**
- **Contrast**

### 🔸 Visual Analysis
- Side-by-side comparison of test images and matched results
- Bar plots for quality metrics

## ⚙️ Installation & Setup

1. Clone the repository:
Install dependencies:

pip install -r requirements.txt
Update dataset paths in the scripts as per your local directory structure.

▶️ How to Run
Run scripts in the following order:

python face.py
python DWT.py
python mean_centring.py
python pCA.py
python SVD.py
python result.py
Each stage generates intermediate outputs required for the next step.

📈 Results
The system demonstrates effective recognition using classical techniques
Image enhancement improves feature quality
SVD-based feature extraction provides compact and discriminative representations
The project validates that classical ML methods can still perform well for face recognition tasks
!![Output 1](results/Output_1.png)

🚀 Future Improvements
Dataset-level PCA fitting instead of per-image PCA
Replace distance-based classifier with SVM or KNN
Compare performance with deep learning models
Optimize runtime and memory usage
Add cross-dataset evaluation

🧑‍💻 Author
Harsh Bansal
AI / ML Developer
Interested in Computer Vision, Image Processing, and Classical ML

📄 License
This project is for educational and research purposes.
