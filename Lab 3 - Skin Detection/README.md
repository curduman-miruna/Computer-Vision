# LAB 3 - Skin and Face Detection with OpenCV

## Overview
This repository includes Python scripts that demonstrate skin detection and face detection techniques using OpenCV. The scripts showcase how to detect skin regions in images using different color models (RGB, HSV, YCbCr), evaluate the accuracy of these methods, and visualize the results through confusion matrices and performance plots. The face detection script uses skin detection results to locate faces within images.

## Files Included

1. [**main.py**](#main-py)
2. [**plot_results.py**](#plot_results-py)
3. [**face_detection.py**](#face_detection-py)

---

## 1. main.py

### Description:
This script contains functions for detecting skin regions in images using different color models: RGB, HSV, and YCbCr. It also evaluates the accuracy of skin detection methods by comparing the detected skin regions against ground truth data.

### Features:
- **RGB Skin Detection**: Detects skin regions using the RGB color model.
- **HSV Skin Detection**: Detects skin regions using the HSV color model.
- **YCbCr Skin Detection**: Detects skin regions using the YCbCr color model.
- **Accuracy Evaluation**: Compares the detected skin regions against ground truth and calculates accuracy.
- **Image Processing**: Processes multiple images in a directory and visualizes the original and detected skin regions.
- **Visualization**: Displays the original image along with the skin detection results for each method (RGB, HSV, and YCbCr).

### Dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`

---

## 2. plot_results.py

### Description:
This script generates confusion matrices and performance plots based on skin detection results. It evaluates the performance of skin detection methods (RGB, HSV, YCbCr) across multiple images and generates plots to visualize the results.

### Features:
- **Confusion Matrix Visualization**: Plots confusion matrices as tables for each skin detection method.
- **Performance Evaluation**: Compares the accuracy of skin detection methods and visualizes the results.
- **Line Plots for Performance**: Plots performance metrics (e.g., TP, FN, FP, TN, Accuracy) for each method across multiple images.
- **Ground Truth Comparison**: Evaluates the detected skin regions against ground truth data.
- **Real-time Plotting**: Uses `matplotlib` to visualize confusion matrices and performance over images.

### Dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`

---

## 3. face_detection.py

### Description:
This script uses skin detection methods (RGB, HSV, YCbCr) to detect faces in images. It identifies skin regions, draws bounding boxes around the largest skin areas, and visualizes the detected faces.

### Features:
- **Face Detection using Skin**: Detects faces by locating the largest skin regions in the image.
- **Multiple Detection Methods**: Uses three color models (RGB, HSV, YCbCr) for skin detection.
- **Bounding Boxes**: Draws red bounding boxes around detected faces for visualization.
- **Real-time Visualization**: Displays the image with bounding boxes for detected faces.
- **Error Handling**: Handles missing or incorrect image paths gracefully.

### Dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`

---