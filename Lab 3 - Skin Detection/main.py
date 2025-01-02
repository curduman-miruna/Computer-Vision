import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_skin_rgb(image):
    B, G, R = cv2.split(image)
    skin_mask = np.logical_and.reduce([
        R > 95,
        G > 40,
        B > 20,
        np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B)) > 15,
        np.abs(R - G) > 15,
        R > G,
        R > B
    ])
    return skin_mask.astype(np.uint8) * 255

def detect_skin_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    skin_mask = np.logical_and.reduce([
        (H >= 0) & (H <= 50),
        (S >= int(0.23 * 255)) & (S <= int(0.68 * 255)),
        (V >= int(0.35 * 255)) & (V <= 255)
    ])
    return skin_mask.astype(np.uint8) * 255

def detect_skin_ycrcb(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin_mask = np.logical_and.reduce([
        Cb >= 80,
        Cb <= 135,
        Cr >= 135,
        Cr <= 180,
        Y > 80
    ])
    return skin_mask.astype(np.uint8) * 255

def evaluate_skin_detection(prediction, ground_truth):
    TP = np.sum(np.logical_and(prediction == 255, ground_truth == 255))
    TN = np.sum(np.logical_and(prediction == 0, ground_truth == 0))
    FP = np.sum(np.logical_and(prediction == 255, ground_truth == 0))
    FN = np.sum(np.logical_and(prediction == 0, ground_truth == 255))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    confusion_matrix = np.array([[TP, FN], [FP, TN]])

    return accuracy, confusion_matrix

# Directory containing images
image_directory = 'D:\\COMPUTER-VISION\\Lab3\\test'  # Replace with your actual image directory path
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]  # List all jpg and png files

titles = ['Original', 'RGB Skin Detection', 'HSV Skin Detection', 'YCbCr Skin Detection']

# Process and display each image in a grid format
fig, axes = plt.subplots(len(image_files), 4, figsize=(20, 5 * len(image_files)))
fig.suptitle("Skin Detection Results for Multiple Images", fontsize=16)

for idx, file in enumerate(image_files):
    # Load the image
    image_path = os.path.join(image_directory, file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for displaying with plt

    # Apply skin detection methods
    skin_mask_rgb = detect_skin_rgb(image)
    skin_mask_hsv = detect_skin_hsv(image)
    skin_mask_ycrcb = detect_skin_ycrcb(image)

    # List of results, including the original image
    images = [image_rgb, skin_mask_rgb, skin_mask_hsv, skin_mask_ycrcb]

    # Plot the original and skin detection results
    for i in range(4):
        axes[idx, i].imshow(images[i], cmap='gray' if i > 0 else None)  # Gray for masks
        axes[idx, i].set_title(titles[i])
        axes[idx, i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
