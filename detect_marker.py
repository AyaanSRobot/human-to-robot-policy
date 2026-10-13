from glob import glob
import cv2 
from pathlib import Path

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

detector = cv2.aruco.ArucoDetector(dictionary)

frames = sorted(glob("frames/raw/*.jpg"),key=lambda p: int(Path(p).stem.split("_")[-1])) # retain only the number to sort by number



detected = 0 # markers detected in imgs

for frame in frames:
    image = cv2.imread(str(frame))
    if image is None:
        continue


    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None and 0 in ids.flatten():
        detected += 1

print(f"Marker detected in {detected}/{len(frames)} frames")
print(f"Detection rate: {100 * detected / len(frames):.1f}%")
