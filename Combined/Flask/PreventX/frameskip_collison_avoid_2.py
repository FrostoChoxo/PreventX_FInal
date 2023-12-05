import cv2
from ultralytics import YOLO
import math
import time
import pygame.mixer

# Initialize the YOLO model
model = YOLO("yolov8s.pt")

# Define classes and exclusion list
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


exclude_classes=['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Initialize frame skip variable

# Distance estimation function
def estimate_distance(box_area, frame_area):
    return frame_area / box_area


# Function to process each frame
def process_frame(img, frame_width, frame_height, THRESHOLD_DISTANCE):
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_width * frame_height
            distance = estimate_distance(box_area, frame_area)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in exclude_classes:
                continue  # Skip drawing bounding boxes for excluded classes

            label = f"{class_name} - {distance:.2f}"  # display distance metric for debugging
            color = (0, 255, 0)  # default to green

            if distance < THRESHOLD_DISTANCE:
                color = (0, 0, 255)  # Red color for alert
                label += " TOO CLOSE!"
                if not pygame.mixer.music.get_busy():  # play the alert only if it's not already playing
                    pygame.mixer.music.play()
                last_alert_time = time.time()  # update the timestamp of the last alert
                alert_triggered = True

            # Draw bounding box and label inside the loop for every detected box
            # img = cv2.resize(img, (frame_width // 2, frame_height // 2))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
              # (Your existing logic here)

    return img


# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alert_forklift.mp3")

# Setup two cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

desired_width = 640
desired_height = 480
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Threshold distance
THRESHOLD_DISTANCE = 2
# Initialize frame skip variable
FRAME_SKIP = 5  # Skip every 5 frames

# ... (rest of your code until the main loop)

frame_count1 = 0
frame_count2 = 0

while True:
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    if not success1 or not success2:
        break

    # Increment frame counters
    frame_count1 += 1
    frame_count2 += 1

    # Process frames based on the frame skipping logic
    if frame_count1 % FRAME_SKIP == 0:
        processed_img1 = process_frame(img1, desired_width, desired_height, THRESHOLD_DISTANCE)
        cv2.imshow('Camera 1', processed_img1)
    if frame_count2 % FRAME_SKIP == 0:
        processed_img2 = process_frame(img2, desired_width, desired_height, THRESHOLD_DISTANCE)
        cv2.imshow('Camera 2', processed_img2)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):
        THRESHOLD_DISTANCE += 0.01
    elif key == ord('-'):
        THRESHOLD_DISTANCE -= 0.01

cap1.release()
cap2.release()
cv2.destroyAllWindows()