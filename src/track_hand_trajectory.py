import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from glob import glob
from pathlib import Path
import csv

model_path = "src/models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),running_mode=VisionRunningMode.VIDEO, num_hands = 1)


frames = sorted(glob("frames/raw/*.jpg"),key=lambda p: int(Path(p).stem.split("_")[-1])) # retain only the number to sort by number

trajectory = []

with HandLandmarker.create_from_options(options) as landmarker:

    for frame in frames:
        mp_image = mp.Image.create_from_file(frame)
        image = cv2.imread(frame)
        h, w = image.shape[:2] # image width and height 


        fps = 50
        frame_number = int(Path(frame).stem.split("_")[-1])
        timestamp_ms = round(frame_number * 1000 / fps)


        #perform hand landmark detection on the single image
        hand_landmarker_result = landmarker.detect_for_video(mp_image,timestamp_ms)

       
        # Count successful hand detections
        if hand_landmarker_result.hand_landmarks:
            hand_landmarks = hand_landmarker_result.hand_landmarks[0] #first detected hand (there is only 1 hand in the footage)

            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            # grasp midpoint 
            grasp_x = (thumb.x + index.x) / 2
            grasp_y = (thumb.y + index.y) / 2


            # save trajectory 
            trajectory.append([timestamp_ms, grasp_x, grasp_y])

with open("grasp_trajectory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms", "x", "y"])
    writer.writerows(trajectory)
            
