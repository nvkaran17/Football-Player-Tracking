import cv2
import os
import json
from ultralytics import YOLO
from utils.helpers import draw_boxes
from utils.tracker import PlayerTracker

# Paths
model_path = 'yolov11_model/best.pt'
video_path = 'inputs/15sec_input_720p.mp4'
output_video_path = 'outputs/annotated_video.mp4'
json_output_path = 'outputs/player_data.json'

# Load model
model = YOLO(model_path)

# Initialize video I/O
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Init tracker
tracker = PlayerTracker()

frame_num = 0
results_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Detect players
    detections = model(frame)[0].boxes.data.cpu().numpy()
    player_dets = [det[:4] for det in detections if int(det[5]) == 2]  # class 2 = player

    # Track
    tracked = tracker.update(player_dets)

    # Draw
    draw_boxes(frame, tracked)

    # Display player count
    cv2.putText(frame, f'Players in frame: {len(tracked)}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Log
    for t in tracked:
        results_log.append({
            'frame': int(frame_num),
            'id': int(t['id']),
            'bbox': [int(coord) for coord in t['bbox']]
        })

    out.write(frame)

cap.release()
out.release()

# Save JSON
with open(json_output_path, 'w') as f:
    json.dump(results_log, f, indent=2)

print("Processing complete. Output saved.")
