# LAB 1 - Image Processing and Emoji Generation with OpenCV

## Overview
This repository includes Python scripts that demonstrate basic image processing techniques and custom emoji creation using OpenCV. The scripts showcase how to apply filters to images, perform image manipulations such as rotation, cropping, and sharpening, and create custom emoji-like images with gradients and sparkle effects.

## Files Included

1. [**image_filters.py**](#image_filters-py)
2. [**emoji.py**](#emoji-py)

---

## 1. image_filters.py

### Description:
This script allows you to apply various image processing techniques on an input image, such as blurring, sharpening, rotating, and cropping. The script uses OpenCV's functionality for image manipulations and provides real-time previews of the applied effects.

### Features:
- **Image Blurring**: Applies Gaussian blur to the image with multiple kernel sizes.
- **Image Sharpening**: Uses a sharpening filter to enhance image details.
- **Custom Filter**: Applies a custom filter to the image (Laplacian filter).
- **Image Rotation**: Rotates the image by 90, 180, and arbitrary angles.
- **Image Cropping**: Crops a portion of the image and displays it.
- **Real-time Preview**: Uses OpenCV's GUI to show the image before and after processing.
- **File Saving**: Saves the processed images in various formats (e.g., JPG).

### Dependencies:
- `opencv-python`
- `numpy`

## 2. emoji.py

### Description
This script creates a custom emoji-like image using OpenCV by drawing geometric shapes (rectangles and circles), and applying gradient effects. The generated emoji includes a heart-shaped contour, rotation effects, and a sparkle effect with a custom gradient. The emoji is processed and displayed with these transformations in real-time.

### Features
- **Geometric Shape Drawing**: Creates an emoji-like image using a combination of rectangles and circles.
- **Gradient Effects**: Applies gradient coloring to the shapes, including the square and circles.
- **Heart Shape Contour**: Draws a heart-shaped contour inside the emoji and highlights it with contouring techniques.
- **Image Rotation**: Rotates the emoji by a specified angle (e.g., 315°).
- **Sparkle Effect**: Adds a sparkle effect using a custom gradient around the emoji.
- **File Saving**: Saves the generated emoji image as a `.jpg` file.

### Dependencies
- `opencv-python`
- `numpy`