# Lab 4 - Optical Character Recognition (OCR) Analysis

This repository contains various scripts and utilities for analyzing Optical Character Recognition (OCR) performance. The focus of the analysis is to evaluate the accuracy of OCR systems under different image transformations, noise conditions, blurring, resizing, and preprocessing techniques.

## Files Included

- [common_utils.py](common_utils.py) - Utility functions for OCR analysis
- [lab_4_aff_transf.py](lab_4_aff_transf.py) - Affine transformations for OCR evaluation
- [lab_4_noise.py](lab_4_noise.py) - Noise addition for OCR evaluation
- [lab_4_blur.py](lab_4_blur.py) - Blur effects for OCR evaluation
- [lab_4_resize.py](lab_4_resize.py) - Image resizing for OCR evaluation
- [lab_pre.py](lab_pre.py) - Preprocessing techniques for OCR analysis

---

## 1. `common_utils.py`

This file contains utility functions used across the OCR analysis scripts. These utilities help with calculating character accuracy, reading ground truth data, processing OCR results, and visualizing the outcomes.

### Key Features:
- **Character Accuracy Calculation**: Uses Levenshtein distance to compute accuracy at the character level.
- **Ground Truth Reading**: Reads and processes ground truth text from files.
- **Results Saving**: Saves OCR accuracy results to a CSV file for later analysis.
- **Word Highlighting**: Highlights correct and incorrect words on images for visual feedback.
- **OCR Data Processing**: Extracts text from images using Tesseract OCR.
- **Accuracy Plotting**: Generates plots to visualize accuracy trends from CSV data.

### Usage:
The functions in this file are used by other scripts to process images, calculate accuracy, and visualize OCR performance.

---

## 2. `lab_4_aff_transf.py`

This script applies affine transformations (rotation and shear) to images and evaluates OCR performance on the transformed images.

### Key Features:
- **Affine Transformations**: Rotates and shears images to simulate real-world distortions.
- **OCR Processing**: Extracts text from the transformed images using Tesseract OCR.
- **Accuracy Calculation**: Measures accuracy by comparing OCR results with ground truth.
- **Results Saving**: Saves OCR accuracy results to a CSV file and generates a plot for analysis.

### Usage:
1. Load images from the specified directory.
2. Apply rotation and shear transformations.
3. Process OCR results from the transformed images.
4. Compute character-level accuracy.
5. Highlight correct and incorrect words on images.
6. Save results and generate accuracy plots.

---

## 3. `lab_4_noise.py`

This script adds noise (Gaussian and salt-and-pepper) to images and evaluates the effect of noise on OCR performance.

### Key Features:
- **Noise Addition**: Adds Gaussian and salt-and-pepper noise to images.
- **OCR Processing**: Uses Tesseract OCR to extract text from noisy images.
- **Accuracy Calculation**: Measures accuracy by comparing OCR results with ground truth.
- **Results Saving**: Saves OCR accuracy results to a CSV file and generates a plot for analysis.

### Usage:
1. Load images from the specified directory.
2. Add noise (Gaussian or salt-and-pepper) to the images.
3. Process OCR results from the noisy images.
4. Compute accuracy and compare with ground truth.
5. Save results and generate accuracy plots.

---

## 4. `lab_4_blur.py`

This script applies blur effects (average and Gaussian) to images and evaluates the impact of blurring on OCR accuracy.

### Key Features:
- **Blur Effects**: Applies average and Gaussian blur to simulate image degradation.
- **OCR Processing**: Extracts text from the blurred images using Tesseract OCR.
- **Accuracy Calculation**: Computes character-level accuracy.
- **Results Saving**: Saves OCR accuracy results to a CSV file and generates a plot for analysis.

### Usage:
1. Load images from the specified directory.
2. Apply average and Gaussian blur to the images.
3. Process OCR results from the blurred images.
4. Measure accuracy and save the results.
5. Generate and save accuracy plots.

---

## 5. `lab_4_resize.py`

This script resizes images to various scales and evaluates the effect of resizing on OCR accuracy.

### Key Features:
- **Image Resizing**: Resizes images while maintaining or altering their aspect ratio.
- **OCR Processing**: Uses Tesseract OCR to extract text from resized images.
- **Accuracy Calculation**: Measures accuracy by comparing OCR results with ground truth.
- **Results Saving**: Saves OCR accuracy results to a CSV file and generates a plot for analysis.

### Usage:
1. Load images from the specified directory.
2. Resize images based on the specified parameters.
3. Extract text using Tesseract OCR.
4. Measure accuracy and save results.
5. Generate and visualize accuracy plots.

---

## 6. `lab_pre.py`

This script applies various preprocessing techniques (sharpening, thresholding, and morphological operations) to improve OCR accuracy.

### Key Features:
- **Sharpening**: Enhances image details by applying a sharpening filter.
- **Thresholding**: Converts images to binary format using thresholding.
- **Morphological Operations**: Applies erosion and dilation to refine image features.
- **OCR Processing**: Uses Tesseract OCR to extract text from preprocessed images.
- **Accuracy Calculation**: Computes character-level accuracy.
- **Results Saving**: Saves OCR accuracy results to a CSV file and generates a plot for analysis.

### Usage:
1. Load images from the specified directory.
2. Apply preprocessing techniques such as sharpening, thresholding, and morphological operations.
3. Process OCR results from the preprocessed images.
4. Measure accuracy and save results.
5. Generate and visualize accuracy plots.

---

## Running the Scripts

Each script in this repository is designed to be run in a Python environment. Before running, ensure that:
- The required directories for input images and output results are correctly set in your environment variables.
- Tesseract OCR is installed and accessible in your environment.

### Example:
To run one of the scripts, execute it in your Python environment:

```bash
python lab_4_aff_transf.py
```