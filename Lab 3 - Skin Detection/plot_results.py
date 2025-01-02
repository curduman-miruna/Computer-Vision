import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from main import detect_skin_rgb, detect_skin_hsv, detect_skin_ycrcb, evaluate_skin_detection

def plot_confusion_matrices(confusion_matrices, accuracies, title, ax):
    ax.axis('tight')
    ax.axis('off')
    table_data = [['Image', 'TP', 'FN', 'FP', 'TN', 'Accuracy']]

    sorted_data = []
    for j, matrix in enumerate(confusion_matrices):
        TP, FN = matrix[0]
        FP, TN = matrix[1]
        accuracy = accuracies[j]
        sorted_data.append((accuracy, f'Image {j + 1}', TP, FN, FP, TN))

    sorted_data.sort(key=lambda x: x[0], reverse=True)

    for entry in sorted_data:
        accuracy, img_name, TP, FN, FP, TN = entry
        table_data.append([img_name, TP, FN, FP, TN, f'{accuracy:.2f}'])

    pastel_colors = [
        '#FFCCCC',  # Color for TP
        '#FFCCFF',  # Color for FN
        '#CCFFFF',  # Color for FP
        '#CCFFCC',  # Color for TN
        '#FFFFCC'  # Color for Accuracy
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')

    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#D9EAD3')  # Light green for header
            cell.set_text_props(weight='bold')
        else:
            color_index = j - 1  # Skip header column
            cell.set_facecolor(pastel_colors[color_index])

    ax.set_title(title, fontsize=16, weight='bold', pad=30)


def plot_performance_over_methods(confusion_matrices, accuracies):
    metrics_values = {
        'TP': [],
        'FN': [],
        'FP': [],
        'TN': [],
        'Accuracy': []
    }

    # Prepare x values for each method
    image_indices = range(len(next(iter(confusion_matrices.values()))))  # Number of images processed

    for method, matrices in confusion_matrices.items():
        for j, matrix in enumerate(matrices):
            TP, FN = matrix[0]
            FP, TN = matrix[1]
            accuracy = accuracies[method][j]

            metrics_values['TP'].append((method, TP))
            metrics_values['FN'].append((method, FN))
            metrics_values['FP'].append((method, FP))
            metrics_values['TN'].append((method, TN))
            metrics_values['Accuracy'].append((method, accuracy))

    # Create subplots
    num_metrics = len(metrics_values)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))  # Adjust height as needed

    # Plot each metric for each method
    for idx, (metric, values) in enumerate(metrics_values.items()):
        for method in set([v[0] for v in values]):
            method_values = [v[1] for v in values if v[0] == method]
            axes[idx].plot(image_indices, method_values, marker='o', label=method)

        axes[idx].set_title(f'{metric} Over Images', pad=20)
        axes[idx].set_xlabel('Image Index')
        axes[idx].set_ylabel(metric)
        axes[idx].grid()
        axes[idx].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def process_skin_detection(detected_folder, ground_truth_folder):
    image_files = os.listdir(detected_folder)  # Get list of detected image files
    confusion_matrices = {'RGB': [], 'HSV': [], 'YCbCr': []}
    accuracies = {'RGB': [], 'HSV': [], 'YCbCr': []}

    # Process each image
    for idx, file in enumerate(image_files):
        detected_image = cv2.imread(os.path.join(detected_folder, file))
        if detected_image is None:
            print(f"Detected image {file} not found. Skipping this image.")
            continue

        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

        ground_truth_image_path = os.path.join(ground_truth_folder, file.replace('.jpg', '.png'))
        if not os.path.exists(ground_truth_image_path):
            ground_truth_image_path = os.path.join(ground_truth_folder, file.replace('.jpg', '.jpeg'))

        ground_truth_image = cv2.imread(ground_truth_image_path)
        if ground_truth_image is None:
            print(f"Ground truth for {file} not found. Skipping this image.")
            continue

        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # Skin detection
        skin_mask_rgb = detect_skin_rgb(detected_image)
        skin_mask_hsv = detect_skin_hsv(detected_image)
        skin_mask_ycrcb = detect_skin_ycrcb(detected_image)

        # Evaluation
        accuracy_rgb, confusion_matrix_rgb = evaluate_skin_detection(skin_mask_rgb, ground_truth_image)
        accuracy_hsv, confusion_matrix_hsv = evaluate_skin_detection(skin_mask_hsv, ground_truth_image)
        accuracy_ycrcb, confusion_matrix_ycrcb = evaluate_skin_detection(skin_mask_ycrcb, ground_truth_image)

        # Append results to respective metrics
        confusion_matrices['RGB'].append(confusion_matrix_rgb)
        confusion_matrices['HSV'].append(confusion_matrix_hsv)
        confusion_matrices['YCbCr'].append(confusion_matrix_ycrcb)

        accuracies['RGB'].append(accuracy_rgb)
        accuracies['HSV'].append(accuracy_hsv)
        accuracies['YCbCr'].append(accuracy_ycrcb)

    # Create a plot for confusion matrices as tables
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))  # Increased height for the tables
    fig.suptitle('Confusion Matrices for Skin Detection Methods', fontsize=16)

    plot_confusion_matrices(confusion_matrices['RGB'], accuracies['RGB'], 'RGB', axes[0])
    plot_confusion_matrices(confusion_matrices['HSV'], accuracies['HSV'], 'HSV', axes[1])
    plot_confusion_matrices(confusion_matrices['YCbCr'], accuracies['YCbCr'], 'YCbCr', axes[2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Plot performance over images with line plots
    plot_performance_over_methods(confusion_matrices, accuracies)


# Replace with your actual folder paths
detected_folder = 'D:\\COMPUTER-VISION\\Lab3\\Pratheepan_Dataset\\FacePhoto'
ground_truth_folder = 'D:\\COMPUTER-VISION\\Lab3\\Ground_Truth\\GroundT_FacePhoto'
process_skin_detection(detected_folder, ground_truth_folder)

detected_folder2 = 'D:\\COMPUTER-VISION\\Lab3\\Pratheepan_Dataset\\FamilyPhoto'
ground_truth_folder2 = 'D:\\COMPUTER-VISION\\Lab3\\Ground_Truth\\GroundT_FamilyPhoto'
process_skin_detection(detected_folder2, ground_truth_folder2)
