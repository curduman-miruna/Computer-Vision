import os
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2

load_dotenv()
model = YOLO("yolov8n.pt")

input_video_path = os.getenv("TRAFFIC_INPUT_PATH")
output_video_path = os.getenv("TRAFFIC_OUTPUT_PATH")

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    # Sort the detections by confidence score (descending)
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                'cls': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'xyxy': box.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
            }
            detections.append(detection)

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    top_detections = detections[:4]

    # Draw bounding boxes for the top detections
    for detection in top_detections:
        x1, y1, x2, y2 = map(int, detection['xyxy'])
        confidence = detection['confidence']
        label = f"Conf: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label with confidence score
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              font_thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 128, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved at: {output_video_path}")
