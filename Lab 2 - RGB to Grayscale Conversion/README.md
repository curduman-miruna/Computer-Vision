# Lab 2 - Image Processing with OpenCV

## Description

This script performs various image processing techniques using OpenCV and NumPy on the input image (`lena.tif`). The user can interact with the program through keyboard inputs to apply different image transformations and visual effects in real-time. The results are displayed in separate windows, and the processed images are saved as `.jpg` files.

## Features

- **Simple Averaging:** Converts the image to grayscale by averaging the red, green, and blue channels equally.
- **Weighted Average:** Converts the image to grayscale using the luminance formula, which gives different weights to the color channels.
- **Desaturation:** Creates a grayscale image by averaging the maximum and minimum channel values.
- **Decomposition:** Extracts the maximum and minimum values of the red, green, and blue channels to create two new images.
- **Single Channel Extraction:** Displays each color channel (blue, green, and red) separately.
- **Custom Gray Shades:** Converts the image to grayscale with a custom number of shades. Users can specify the number of shades between 1 and 255.
- **Floyd-Steinberg Dithering:** Applies error diffusion dithering to the grayscale image, producing a halftone-like effect.
- **Stucki Dithering:** Another dithering technique that uses a different error diffusion matrix, resulting in a different halftone effect.
- **Interactive Controls:** Press keys to apply different transformations to the image.

## Key Press Commands

- **'1'** - Apply **Simple Averaging** to the image and display/save the result as `simple_averaging.jpg`.
- **'2'** - Apply **Weighted Average** (luminance) to the image and display/save the result as `weighted_average.jpg`.
- **'3'** - Apply **Desaturation** to the image and display/save the result as `desaturation.jpg`.
- **'4'** - Apply **Decomposition Max** and **Decomposition Min** to the image and save as `decomposition_max.jpg` and `decomposition_min.jpg`.
- **'5'** - Display each of the **Single Channels** (Blue, Green, Red).
- **'6'** - Apply **Custom Gray Shades** to the image with 4 shades and display the result.
- **'7'** - Apply **Floyd-Steinberg Dithering** and **Stucki Dithering** to the image, displaying and saving as `floyd_steinberg_dithered.jpg` and `stucki_dithered.jpg`.
- **'q'** - Quit the program.
- **'d'** - Close all open windows and return to the original image.

## Dependencies

- `opencv-python` – for image processing and GUI display.
- `numpy` – for array manipulations and mathematical operations.