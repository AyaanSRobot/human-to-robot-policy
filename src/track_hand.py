import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from glob import glob
from pathlib import Path

model_path = "src/models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),running_mode=VisionRunningMode.IMAGE, num_hands = 1)


frames = sorted(glob("frames/raw/*.jpg"),key=lambda p: int(Path(p).stem.split("_")[-1])) # retain only the number to sort by number

detected = 0 

with HandLandmarker.create_from_options(options) as landmarker:

    for frame in frames:
        mp_image = mp.Image.create_from_file(frame)

        #perform hand landmark detection on the single image
        hand_landmarker_result = landmarker.detect(mp_image)

       
        # Count successful hand detections
        if hand_landmarker_result.hand_landmarks:
            detected += 1

print(f"Hands detected in {detected}/{len(frames)} frames")
