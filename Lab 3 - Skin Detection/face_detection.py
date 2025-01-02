import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import detect_skin_rgb, detect_skin_hsv, detect_skin_ycrcb

def detect_face_using_skin(image):
    # Create an empty dictionary to store results for each method
    results = {}

    # Method 1: RGB
    skin_mask_rgb = detect_skin_rgb(image)
    contours_rgb, _ = cv2.findContours(skin_mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_rgb:
        largest_contour_rgb = max(contours_rgb, key=cv2.contourArea)
        x_rgb, y_rgb, w_rgb, h_rgb = cv2.boundingRect(largest_contour_rgb)
        rgb_image = image.copy()
        cv2.rectangle(rgb_image, (x_rgb, y_rgb), (x_rgb + w_rgb, y_rgb + h_rgb), (0, 0, 255), 2)  # Red square for RGB
        results['RGB'] = rgb_image

    # Method 2: HSV
    skin_mask_hsv = detect_skin_hsv(image)
    contours_hsv, _ = cv2.findContours(skin_mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_hsv:
        largest_contour_hsv = max(contours_hsv, key=cv2.contourArea)
        x_hsv, y_hsv, w_hsv, h_hsv = cv2.boundingRect(largest_contour_hsv)
        hsv_image = image.copy()
        cv2.rectangle(hsv_image, (x_hsv, y_hsv), (x_hsv + w_hsv, y_hsv + h_hsv), (0, 0, 255), 2)  # Red square for HSV
        results['HSV'] = hsv_image

    # Method 3: YCbCr
    skin_mask_ycrcb = detect_skin_ycrcb(image)
    contours_ycrcb, _ = cv2.findContours(skin_mask_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_ycrcb:
        largest_contour_ycrcb = max(contours_ycrcb, key=cv2.contourArea)
        x_ycrcb, y_ycrcb, w_ycrcb, h_ycrcb = cv2.boundingRect(largest_contour_ycrcb)
        ycrcb_image = image.copy()
        cv2.rectangle(ycrcb_image, (x_ycrcb, y_ycrcb), (x_ycrcb + w_ycrcb, y_ycrcb + h_ycrcb), (0, 0, 255), 2)  # Red square for YCbCr
        results['YCbCr'] = ycrcb_image

    return results

# Example usage
image = cv2.imread('D:\\COMPUTER-VISION\\Lab3\\Pratheepan_Dataset\\FacePhoto\\sara_badr.jpg')  # Replace with your image path
results = detect_face_using_skin(image)

# Display results for each method
plt.figure(figsize=(15, 5))

for idx, (method, result_image) in enumerate(results.items()):
    plt.subplot(1, len(results), idx + 1)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detectare față folosind metoda {method}')
    plt.axis('off')

plt.tight_layout()
plt.show()

if not results:
    print("Nu s-a detectat nicio față în imagine.")
