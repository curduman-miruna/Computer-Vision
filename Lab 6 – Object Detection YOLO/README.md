# LAB 6 - YOLO Object Detection 
## Overview
This repository includes several Python scripts that use YOLO (You Only Look Once) models to perform object detection on different types of data. The models are built using the `ultralytics` YOLO implementation, allowing you to detect and count vehicles, people, traffic objects, and specifically yellow trucks in images or videos. The scripts use OpenCV for image/video processing and support output in video formats or image annotations.

## Files Included

1. [**detect_cars_trucks.py**](cars_trucks.py)
2. [**detect_only_people.py**](detect_only_people.py)
3. [**detect_traffic.py**](detect_traffic.py)
4. [**detect_yellow_truck.py**](detect_yellow_truck.py)

---

## 1. detect_cars_trucks.py

### Description:
This script detects and counts cars (class ID 2) and trucks (class ID 7) in a sequence of images, processes each frame, and generates an annotated video with bounding boxes around the vehicles. The vehicles are color-coded in green for cars and blue for trucks. It also outputs the total number of cars and trucks detected throughout the video.

### Features:
- **Car and Truck Detection**: Identifies and counts cars and trucks.
- **Bounding Boxes**: Draws bounding boxes around detected vehicles.
- **Color Coding**: Cars are marked with green, trucks with blue.
- **Output**: Generates an annotated video with bounding boxes and labels for cars and trucks.

### Dependencies:
- `opencv-python`
- `ultralytics`
- `python-dotenv`

### Setup:
1. Create a `.env` file and set the following environment variables:
   - `DATASET_IMAGE_PATH`: Path to the folder containing the images or video frames.
   - `KAGLE_RESULT_PATH`: Path where the output video will be saved.
2. Run the script to process the dataset and save the annotated video.

## 2. detect_only_people.py

### Description:
This script detects people (class ID 0) in a single image, draws bounding boxes around them, and saves the image with annotations. It is designed for detecting people in an image and labeling them with their confidence score.

### Features:
1. **People Detection**: Identifies and marks people in the input image.
2. **Bounding Boxes**: Draws bounding boxes around detected people with confidence labels.
3. **Output**: Saves the annotated image with the bounding boxes around detected people.

### Dependencies:
* `opencv-python`
* `ultralytics`
* `python-dotenv`

### Setup:
1. Set up the following environment variables in your .env file:
    - `PHOTO_ONLY_PEOPLE_PATH`: Path to the input image.
    - `ONLY_PEOPLE_RESULT_PATH`: Directory to save the output image.
2. Run the script to detect and annotate people in the image.

## 3. detect_traffic.py

### Description:
This script processes an input video, detects various objects, and annotates the top 4 most confident detections in each frame. It saves the processed video with bounding boxes and confidence scores displayed.
Features:
1. **Traffic Detection**: Detects objects in traffic scenes (using YOLO).
2. **Top 4 Detections**: Displays the top 4 most confident detections.
3. **Bounding Boxes & Confidence**: Annotates video frames with bounding boxes and confidence labels.
4. **Output**: Saves the annotated video with object detections.

### Dependencies:
- `opencv-python`
- `ultralytics`
- `python-dotenv`

### Setup:
1. Set up the following environment variables:
- `TRAFFIC_INPUT_PATH`: Path to the input video.
- `TRAFFIC_OUTPUT_PATH`: Path to save the output annotated video.
2. Run the script to process the video and generate the annotated output.

## 4. detect_yellow_truck.py
### Description:
This script detects yellow trucks in a sequence of images by first detecting trucks and then applying color filtering to specifically identify yellow trucks. It processes the frames and saves the annotated video with yellow trucks highlighted.
###Features:
1. Yellow Truck Detection: Detects yellow trucks using a combination of YOLO object detection and HSV color filtering.
2. Bounding Boxes & Labels: Annotates yellow trucks with bounding boxes and labels.
3. Output: Saves the annotated video with yellow trucks marked.

### Dependencies:
* `opencv-python`
* `numpy`
* `ultralytics`
* `python-dotenv`

### Setup:
1. Set the following environment variables:
    `DATASET_IMAGE_PATH`: Path to the folder containing input images or video frames.
    `YELLOW_TRUCK_OUTPUT_PATH`: Path to save the annotated output video.
2. Run the script to process the frames and generate the output.