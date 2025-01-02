from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import os

load_dotenv()
model = YOLO("yolov8m.pt")

input_image_path = os.getenv("PHOTO_ONLY_PEOPLE_PATH")
output_dir = os.getenv("ONLY_PEOPLE_RESULT_PATH")
os.makedirs(output_dir, exist_ok=True)

results = model(input_image_path)
image = cv2.imread(input_image_path)

box_color = (0, 255, 0)
text_color = (255, 255, 255)
text_background_color = (0, 128, 0)

for result in results:
    for box in result.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            print(f"Detected person with confidence: {confidence:.2f}")
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            label = f"Person: {confidence:.2f}"
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x1 = x1
            text_y1 = y1 - text_height - baseline - 5
            text_x2 = x1 + text_width
            text_y2 = y1
            cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2), text_background_color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

output_path = os.path.join(output_dir, "people_detected_Taken_7.jpg")
cv2.imwrite(output_path, image)

print(f"Image with people detected saved at: {output_path}")
