import os
import numpy as np
from PIL import Image
from skimage.util import random_noise
from common_utils import calculate_character_accuracy, read_ground_truth, save_accuracy_to_csv, process_ocr_data, plot_csv_as_line_graphs

image_dir = 'D:/COMPUTER-VISION/Lab4R/sample/'
ground_truth_dir = 'D:/COMPUTER-VISION/Lab4R/ground_truth/'
output_dir = '/Lab4R/output/noise'
output_dir_all = 'D:/COMPUTER-VISION/Lab4R/output/all'
accuracy_csv_path = '/Lab4R/output/noise/accuracy_results.csv'

def add_gaussian_noise(image, sigma=25):
    """
    Add Gaussian noise to an image.
    """
    np_image = np.array(image)
    noise = np.random.normal(0, sigma, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))

def add_salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt-and-pepper noise to an image.
    """
    np_image = np.array(image)
    noisy_image = random_noise(np_image, mode='s&p', amount=amount)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

def ocr_image_with_noise(image, filename, ground_truth_path, results, noise_type, noise_param):
    """
    Perform OCR on an image with noise, calculate accuracy, and save results.
    """
    ocr_data = process_ocr_data(image)
    ocr_text = " ".join(ocr_data['text']).strip()
    ground_truth = read_ground_truth(ground_truth_path)
    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)
    results.append([filename, noise_type, noise_param, char_accuracy])

def process_images_with_noise():
    """
    Apply noise to images and process with OCR.
    """
    noise_params = {
        'gaussian': [10, 25, 50, 70],
        'salt_and_pepper': [0.01, 0.05, 0.1]
    }

    results = []
    for filename in os.listdir(image_dir):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename.replace('.jpg', '.txt'))

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {filename} not found. Skipping.")
            continue

        image = Image.open(image_path).convert("RGBA")
        for noise_type, params in noise_params.items():
            for param in params:
                if noise_type == 'gaussian':
                    noisy_image = add_gaussian_noise(image, sigma=param)
                elif noise_type == 'salt_and_pepper':
                    noisy_image = add_salt_and_pepper_noise(image, amount=param)

                noisy_image_path = os.path.join(output_dir_all, f"{filename}_{noise_type}_{param}.png")
                noisy_image.save(noisy_image_path)

                ocr_image_with_noise(noisy_image, filename, ground_truth_path, results, noise_type, param)

    save_accuracy_to_csv(results, accuracy_csv_path, ['Image Name', 'Noise Type', 'Noise Level', 'Character Accuracy'])

# Run the processing function and plot results
process_images_with_noise()
plot_csv_as_line_graphs(output_dir)
