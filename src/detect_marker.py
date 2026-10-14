from glob import glob
import cv2 
from pathlib import Path
import csv

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

detector = cv2.aruco.ArucoDetector(dictionary)

frames = sorted(glob("frames/raw/*.jpg"),key=lambda p: int(Path(p).stem.split("_")[-1])) # retain only the number to sort by number

marker_data = []


detected = 0 # markers detected in imgs

for frame in frames:
    image = cv2.imread(str(frame))
    if image is None:
        continue

    frame_number = int(Path(frame).stem.split("_")[-1]) # collect frame number

    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None and 0 in ids.flatten():
        detected += 1

        idx = list(ids.flatten()).index(0) # position of marker ID 0
        pts = corners[idx].reshape(4,2)

        marker_data.append([frame_number, *pts.flatten()])

print(f"Marker detected in {detected}/{len(frames)} frames")
print(f"Detection rate: {100 * detected / len(frames):.1f}%")

# Markers detected in 1312/1393 frames as in 94.2% detection rate

with open("marker_corners.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame_number",
        "x0", "y0",
        "x1", "y1",
        "x2", "y2",
        "x3", "y3"
    ])
    writer.writerows(marker_data)