import csv
import os
import pytesseract
import Levenshtein
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

def calculate_character_accuracy(ocr_text, ground_truth):
    """
    Calculate character-level accuracy using Levenshtein distance.

    """
    lev_distance = Levenshtein.distance(ocr_text, ground_truth)
    max_len = max(len(ocr_text), len(ground_truth))
    return (max_len - lev_distance) / max_len if max_len > 0 else 0

def read_ground_truth(ground_truth_path):
    """
    Read the ground truth text from a file.
    """
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Ground truth file not found at {ground_truth_path}")
        return ""

def save_accuracy_to_csv(results, csv_filename, header):
    """
    Save OCR accuracy results to a CSV file.
    """
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Accuracy results saved to {csv_filename}")

def highlight_words(image, ocr_data, ground_truth):
    """
    Highlight the correct and incorrect words on the image.
    """
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    correct_words = [ocr_data['text'][i] for i in range(len(ocr_data['text'])) if
                     ocr_data['text'][i].strip() and ocr_data['text'][i] in ground_truth.split()]

    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            if ocr_data['text'][i] in correct_words:
                draw.rectangle([x, y, x + w, y + h], fill=(0, 255, 0, 50))  # Green for correct words
            else:
                draw.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 50))  # Red for incorrect words

    # Return the image with highlights
    return Image.alpha_composite(image.convert("RGBA"), overlay)

def process_ocr_data(image):
    """
    Process the image using pytesseract to extract OCR data.
    """
    return pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DICT)

def plot_accuracy_from_csv(csv_path, x_col, y_col, group_col=None, output_dir=None, title=None):
    """
    Generate a line plot from a CSV file for accuracy analysis.

    :param csv_path: Path to the CSV file.
    :param x_col: Column name to use as X-axis.
    :param y_col: Column name to use as Y-axis.
    :param group_col: Column name for grouping lines (optional).
    :param output_dir: Directory to save the plot.
    :param title: Title of the plot.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    plt.figure(figsize=(10, 6))

    if group_col:
        for group in df[group_col].unique():
            group_df = df[df[group_col] == group]
            plt.plot(group_df[x_col], group_df[y_col], marker='o', label=str(group))
    else:
        plt.plot(df[x_col], df[y_col], marker='o')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or 'Accuracy Analysis')
    plt.legend(title=group_col or "")
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'accuracy_plot.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    plt.show()
