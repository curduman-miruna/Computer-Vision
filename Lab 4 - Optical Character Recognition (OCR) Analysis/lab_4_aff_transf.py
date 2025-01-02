import os
from PIL import Image
import numpy as np
from skimage.transform import AffineTransform, warp
from common_utils import calculate_character_accuracy, read_ground_truth, save_accuracy_to_csv, process_ocr_data, \
    highlight_words, plot_accuracy_from_csv
from dotenv import load_dotenv

load_dotenv()

image_dir = os.getenv('image_dir')
output_dir_all = os.getenv('output_dir_all')
output_dir = os.getenv('output_dir')
accuracy_csv_path = os.getenv('accuracy_csv_path')
ground_truth_dir = os.getenv('ground_truth_dir')

image_results = []

affine_transformations = [
    {'type': 'rotation', 'params': [15, 60, 90, 180, 270]},
    {'type': 'shear', 'params': [0.2, 0.5, -0.2, -0.5]},
]

def apply_affine_transformations(image):
    transformations = []

    for transformation in affine_transformations:
        if transformation['type'] == 'rotation':
            for angle in transformation['params']:
                rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
                transformations.append((f'Rotation {angle}°', rotated_image))
        elif transformation['type'] == 'shear':
            for shear_value in transformation['params']:
                shear_transform = AffineTransform(shear=shear_value)
                sheared_image = warp(np.array(image), shear_transform)
                sheared_image_pil = Image.fromarray((sheared_image * 255).astype(np.uint8))
                transformations.append((f'Shear {shear_value}', sheared_image_pil))
    return transformations

def ocr_image(image, filename, ground_truth_path, transformation_name):
    ocr_data = process_ocr_data(image)
    ocr_text = " ".join(ocr_data['text']).strip()

    if not ocr_text:
        ocr_text = ""

    ground_truth = read_ground_truth(ground_truth_path)
    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)

    image_results.append([filename, transformation_name, char_accuracy])

    highlighted_image = highlight_words(image, ocr_data, ground_truth)
    highlighted_image_path = os.path.join(output_dir, f"{filename}_{transformation_name}_highlighted.png")
    highlighted_image.save(highlighted_image_path)

def process_images_with_transformations():
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            ground_truth_path = os.path.join(ground_truth_dir, filename.replace('.jpg', '.txt'))

            if not os.path.exists(ground_truth_path):
                continue

            image = Image.open(image_path).convert("RGBA")
            transformations = apply_affine_transformations(image)

            for transformation_name, transformed_image in transformations:
                transformed_image_path = os.path.join(output_dir_all, f"{filename}_{transformation_name}.png")
                transformed_image.save(transformed_image_path)
                ocr_image(transformed_image, filename, ground_truth_path, transformation_name)

    save_accuracy_to_csv(image_results, accuracy_csv_path, ['Image Name', 'Transformation', 'Character Accuracy'])

process_images_with_transformations()
plot_accuracy_from_csv(
    csv_path=accuracy_csv_path,
    x_col='Transformation',
    y_col='Character Accuracy',
    group_col='Image Name',
    output_dir=output_dir,
    title='Accuracy vs. Affine Transformations'
)

