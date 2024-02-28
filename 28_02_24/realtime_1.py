import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import joblib

# Load the trained model
model_path = 'random_forest_model.joblib'
trained_model = joblib.load(model_path)

# Initialize tracking history
track_history = defaultdict(lambda: [])

# Load the YOLO model
yolo_model = YOLO("yolov8x-seg.pt")
names = yolo_model.model.names

# Define the region of interest
roi = np.array([[546, 683], [572, 729], [1098, 708], [1014, 667]], np.int32)
roi_binary = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Adjust to your video resolution if necessary.
cv2.fillPoly(roi_binary, [roi], color=(255, 255, 255))
area_of_roi = cv2.countNonZero(cv2.cvtColor(roi_binary, cv2.COLOR_BGR2GRAY))  # Calculate area of ROI

# Load your video file instead of webcam
video_file_path = '4.mp4'  # Replace this with the path to your video file
cap = cv2.VideoCapture(video_file_path)

frame_counter = 0  # Initialize a frame counter

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Failed to grab frame or end of video reached.")
        break

    results = yolo_model.track(im0, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    confidences = results[0].boxes.conf.float().cpu().tolist()
    cls_nums = results[0].boxes.cls.int().cpu().tolist()

    # Initialize lists for features
    features = []

    for mask, track_id, confidence, cls_num in zip(masks, track_ids, confidences, cls_nums):
        if mask.size == 0:
            continue  # Skip if mask is empty

        center = np.mean(mask, axis=0)
        mask_binary = np.zeros_like(im0)
        cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], color=(255, 255, 255))
        intersection_binary = cv2.bitwise_and(mask_binary, roi_binary)
        intersection_area = cv2.countNonZero(cv2.cvtColor(intersection_binary, cv2.COLOR_BGR2GRAY))
        mask_area = cv2.countNonZero(cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY))
        iou = intersection_area / (mask_area + area_of_roi - intersection_area) if mask_area + area_of_roi - intersection_area > 0 else 0

        if track_id in track_history:
            velocity = np.linalg.norm(center - track_history[track_id][-1]) if track_history[track_id] else 0
        else:
            velocity = 0
        track_history[track_id].append(center)

        # Collect features based on the object type
        if names[cls_num] in ['person', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']:
            features.append([center[0], center[1], velocity, iou * 100])  # Adjust feature list as needed

    # Perform inference if features are extracted
    if features:
        # Convert features to a suitable format for the model if needed
        prediction = trained_model.predict([features[0]])  # Example: using first feature set for prediction
        print("Prediction:", prediction)

        # If prediction is 1, save the frame
        if prediction == 1:
            frame_save_path = f'saved_frames/frame_{frame_counter}.jpg'  # Naming each saved frame uniquely
            cv2.imwrite(frame_save_path, im0)
            print(f"Frame saved as {frame_save_path}")
            frame_counter += 1  # Increment frame counter

    cv2.imshow("Real-Time Inference", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
