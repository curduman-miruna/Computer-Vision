from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import os

load_dotenv()

model = YOLO("yolov8m.pt")
input_frames_dir = os.getenv("DATASET_IMAGE_PATH")
output_video_path = os.getenv("KAGLE_RESULT_PATH")
os.makedirs(input_frames_dir, exist_ok=True)

frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith('.jpg') or f.endswith('.PNG')])
if not frame_files:
    print("No frames found in the directory.")
    exit()

frame_sample = cv2.imread(os.path.join(input_frames_dir, frame_files[0]))
frame_height, frame_width, _ = frame_sample.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

# Vehicle count variables
car_count = 0
truck_count = 0

# Process each frame
for frame_file in frame_files:
    frame_path = os.path.join(input_frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    results = model(frame)

    # Count cars and trucks, draw bounding boxes
    current_car_count = 0
    current_truck_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding boxes
            if class_id == 2:  # Car
                current_car_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for car
                label_car = f"Car: {confidence:.2f}"
                cv2.putText(frame, label_car, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif class_id == 7:  # Truck
                current_truck_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for truck
                label_truck = f"Truck: {confidence:.2f}"
                cv2.putText(frame, label_truck, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Update total counts
    car_count += current_car_count
    truck_count += current_truck_count

    # Display counts on the frame
    label_car = f"Cars: {current_car_count}"
    label_truck = f"Trucks: {current_truck_count}"
    cv2.putText(frame, label_car, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, label_truck, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release the video writer
out.release()

# Output final counts
print(f"Total cars detected: {car_count}")
print(f"Total trucks detected: {truck_count}")
print(f"Processed video saved at: {output_video_path}")
