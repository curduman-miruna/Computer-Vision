import os
from PIL import Image, ImageFilter
from common_utils import calculate_character_accuracy, read_ground_truth, save_accuracy_to_csv, process_ocr_data, highlight_words, plot_accuracy_from_csv

image_dir = 'D:/COMPUTER-VISION/Lab4R/sample/'
output_dir_all = 'D:/COMPUTER-VISION/Lab4R/output/all'
output_dir = 'D:/COMPUTER-VISION/Lab4R/output/blur'
accuracy_csv_path = 'D:/COMPUTER-VISION/Lab4R/output/blur/accuracy_results.csv'
ground_truth_dir = 'D:/COMPUTER-VISION/Lab4R/ground_truth/'

image_results = []

average_blur_sizes = [1, 2, 3, 5, 7]
gaussian_blur_sigmas = [1, 2, 3, 5, 7]

def apply_average_blur(image, size):
    return image.filter(ImageFilter.BoxBlur(size))

def apply_gaussian_blur(image, sigma):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def ocr_image(image, filename, ground_truth_path, blur_info):
    ocr_data = process_ocr_data(image)
    ocr_text = " ".join(ocr_data['text']).strip()

    if not ocr_text:
        ocr_text = ""

    ground_truth = read_ground_truth(ground_truth_path)
    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)

    image_results.append([filename, blur_info, char_accuracy])

    highlighted_image = highlight_words(image, ocr_data, ground_truth)
    highlighted_image_path = os.path.join(output_dir, f"{filename}_{blur_info}_highlighted.png")
    highlighted_image.save(highlighted_image_path)

def process_images_with_blur():
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            ground_truth_path = os.path.join(ground_truth_dir, filename.replace('.jpg', '.txt'))

            if not os.path.exists(ground_truth_path):
                continue

            image = Image.open(image_path).convert("RGBA")

            for size in average_blur_sizes:
                blurred_image = apply_average_blur(image, size)
                blur_info = f"average_{size}"

                blurred_image_path = os.path.join(output_dir_all, f"{filename}_{blur_info}.png")
                blurred_image.save(blurred_image_path)
                ocr_image(blurred_image, filename, ground_truth_path, blur_info)


            for sigma in gaussian_blur_sigmas:
                blurred_image = apply_gaussian_blur(image, sigma)
                blur_info = f"gaussian_{sigma}"

                blurred_image_path = os.path.join(output_dir_all, f"{filename}_{blur_info}.png")
                blurred_image.save(blurred_image_path)
                ocr_image(blurred_image, filename, ground_truth_path, blur_info)

    save_accuracy_to_csv(image_results, accuracy_csv_path, ['Image Name', 'Blur Type', 'Character Accuracy'])

process_images_with_blur()
