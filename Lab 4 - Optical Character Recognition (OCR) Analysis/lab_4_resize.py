import os
from PIL import Image
from common_utils import calculate_character_accuracy, read_ground_truth, save_accuracy_to_csv, process_ocr_data, highlight_words

image_dir = 'D:/COMPUTER-VISION/Lab4R/sample/'
output_dir_all = 'D:/COMPUTER-VISION/Lab4R/output/all'
output_dir = 'D:/COMPUTER-VISION/Lab4R/output/resize'
accuracy_csv_path = 'D:/COMPUTER-VISION/Lab4R/output/resize/accuracy_results.csv'
ground_truth_dir = 'D:/COMPUTER-VISION/Lab4R/ground_truth/'

image_results = []

resize_params = [
    {'new_width': 100, 'new_height': None, 'maintain_aspect_ratio': True},
    {'new_width': 200, 'new_height': None, 'maintain_aspect_ratio': True},
    {'new_width': None, 'new_height': 600, 'maintain_aspect_ratio': True},
    {'new_width': 500, 'new_height': 500, 'maintain_aspect_ratio': False},
    {'new_width': 150, 'new_height': 150, 'maintain_aspect_ratio': False},
]

def resize_image(image, new_width=None, new_height=None, maintain_aspect_ratio=True):
    width, height = image.size
    if maintain_aspect_ratio:
        if new_width and not new_height:
            new_height = int(new_width * height / width)
        elif new_height and not new_width:
            new_width = int(new_height * width / height)
        elif not new_width and not new_height:
            raise ValueError("Either new_width or new_height must be provided if maintaining aspect ratio.")

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def ocr_image(image, filename, ground_truth_path, resize_info):
    ocr_data = process_ocr_data(image)
    ocr_text = " ".join(ocr_data['text']).strip()

    if not ocr_text:
        ocr_text = ""

    ground_truth = read_ground_truth(ground_truth_path)
    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)

    image_results.append([filename, resize_info, char_accuracy])

    highlighted_image = highlight_words(image, ocr_data, ground_truth)
    highlighted_image_path = os.path.join(output_dir, f"{filename}_{resize_info}_highlighted.png")
    highlighted_image.save(highlighted_image_path)

def process_images_with_resizing():
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            ground_truth_path = os.path.join(ground_truth_dir, filename.replace('.jpg', '.txt'))

            if not os.path.exists(ground_truth_path):
                continue

            image = Image.open(image_path).convert("RGBA")

            for params in resize_params:
                resized_image = resize_image(image, **params)
                resize_info = f"{params['new_width']}x{params['new_height']}".replace("None", "auto")

                resized_image_path = os.path.join(output_dir_all, f"{filename}_{resize_info}.png")
                resized_image.save(resized_image_path)
                ocr_image(resized_image, filename, ground_truth_path, resize_info)

    save_accuracy_to_csv(image_results, accuracy_csv_path, ['Image Name', 'Resize Dimension', 'Character Accuracy'])

process_images_with_resizing()
