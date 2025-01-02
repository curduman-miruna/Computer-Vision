import os
import cv2
import numpy as np
from common_utils import calculate_character_accuracy, read_ground_truth, save_accuracy_to_csv, process_ocr_data

# ----- Preprocessing Functions -----
def sharpen_image(image, intensity=1):
    """
    Apply a sharpening filter to the image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5 + intensity, -1], [0, -1, 0]])  # Adjusted sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def apply_thresholding(image, threshold_value):
    """
    Apply binary thresholding to the image.
    """
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

def apply_morphological_operations(image, kernel_size=(3, 3)):
    """
    Apply morphological operations (erosion and dilation) to the image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return eroded_image, dilated_image

# ----- OCR Function -----
def ocr_image(image, filename, ground_truth_path, results, preprocessing_type, preprocessing_param):
    """
    Perform OCR on the image, calculate accuracy, and store results.
    """
    if preprocessing_type == 'sharpen':
        image = sharpen_image(image, intensity=preprocessing_param)
    elif preprocessing_type == 'threshold':
        image = apply_thresholding(image, threshold_value=preprocessing_param)
    elif preprocessing_type == 'morphology':
        eroded, _ = apply_morphological_operations(image, kernel_size=preprocessing_param)
        image = eroded

    ocr_data = process_ocr_data(image)
    ocr_text = " ".join(ocr_data['text']).strip()

    ground_truth = read_ground_truth(ground_truth_path)
    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)

    results.append([filename, preprocessing_type, preprocessing_param, char_accuracy])
    return ocr_text

def process_images_in_directory(image_dir, ground_truth_dir, output_csv):
    """
    Process all images in a directory with multiple preprocessing methods.
    """
    results = []
    preprocessing_methods = {
        'sharpen': [1, 2, 3],
        'threshold': [100, 150, 200],
        'morphology': [(3, 3), (5, 5), (7, 7)]
    }

    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            ground_truth_filename = filename.rsplit('.', 1)[0] + '.txt'
            ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)

            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                continue

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Unable to load image {image_path}. Skipping.")
                continue

            for method, params in preprocessing_methods.items():
                for param in params:
                    ocr_image(image, filename, ground_truth_path, results, method, param)

    save_accuracy_to_csv(results, output_csv, ['Image Name', 'Preprocessing Type', 'Preprocessing Parameter', 'Character Accuracy'])

image_dir = os.getenv('IMAGE_DIR_SAMPLE')
ground_truth_dir = os.getenv('GROUND_TRUTH_DIR')
output_csv = os.getenv('OUTPUT_CSV_PATH')

process_images_in_directory(image_dir, ground_truth_dir, output_csv)
