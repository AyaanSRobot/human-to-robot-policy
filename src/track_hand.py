import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from glob import glob
from pathlib import Path

model_path = "src/models/hand_landmarker.task" #add model path here 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),running_mode=VisionRunningMode.VIDEO, num_hands = 1)


frames = sorted(glob("frames/raw/*.jpg"),key=lambda p: int(Path(p).stem.split("_")[-1])) # retain only the number to sort by number

detected = 0 

fps = 50 # video FPS



with HandLandmarker.create_from_options(options) as landmarker:

    for frame in frames:
        mp_image = mp.Image.create_from_file(frame)
        image = cv2.imread(frame)
        h, w = image.shape[:2] # image width and height 

        frame_number = int(Path(frame).stem.split("_")[-1])
        timestamp_ms = round(frame_number * 1000 / fps)


        #perform hand landmark detection on the single image
        hand_landmarker_result = landmarker.detect_for_video(mp_image,timestamp_ms)

       
        # Count successful hand detections
        if hand_landmarker_result.hand_landmarks:
            landmarks = hand_landmarker_result.hand_landmarks[0] #first detected hand (there is only 1 hand in the footage)
            detected += 1


            # annotate a few frames to show detection 
            if detected % 112 == 0: # every 112th successful detection is saved
                for idx in [0, 4, 8]: # wrist, index finger and thumb
                    landmark = landmarks[idx]

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
                cv2.imwrite(f"frames/annotated/frame_{frame_number}.jpg",image)

print(f"Hands detected in {detected}/{len(frames)} frames")
# hands detected with 53.6% accuracy