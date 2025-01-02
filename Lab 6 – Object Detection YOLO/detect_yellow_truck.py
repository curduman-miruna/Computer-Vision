import cv2
import os
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
load_dotenv()
input_frames_dir = os.getenv("DATASET_IMAGE_PATH")
output_video_path = os.getenv("YELLOW_TRUCK_OUTPUT_PATH")
os.makedirs(input_frames_dir, exist_ok=True)

frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith('.PNG') or f.endswith('.jpeg')])
if not frame_files:
    print("No frames found in the directory.")
    exit()

frame_sample = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
frame_height, frame_width, _ = frame_sample.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

for frame_number, frame_file in enumerate(frame_files):
    frame_path = os.path.join(input_frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    results = model(frame)
    yellow_truck_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id == 7:
                truck_roi = frame[y1:y2, x1:x2]
                hsv_truck = cv2.cvtColor(truck_roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_truck, lower_yellow, upper_yellow)

                if np.sum(mask) > 5000:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Yellow Truck", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    yellow_truck_detected = True
                    print(f"Yellow truck detected at frame {frame_number}")

    out.write(frame)

out.release()

print(f"Processed video saved at: {output_video_path}")
