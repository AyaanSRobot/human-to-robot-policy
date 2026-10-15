import cv2
import pandas as pd

import numpy as np
import csv

# add frame number to grask trajectory csv

hand_df = pd.read_csv("grasp_trajectory.csv")
marker_df = pd.read_csv("marker_corners.csv")

fps = 50

hand_df["frame_number"] = (
    hand_df["timestamp_ms"] * fps / 1000
).round().astype(int)

# merge csv files 
df = hand_df.merge(
    marker_df,
    on="frame_number",
    how="left" # preserve hand observations even when there is no marker detected 
)

print(df.head())

# using the marker for homography
marker_world = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=np.float32)

def get_homography(row):
    corners = np.array([
        [row["x0"], row["y0"]],
        [row["x1"], row["y1"]],
        [row["x2"], row["y2"]],
        [row["x3"], row["y3"]]
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(corners, marker_world)


image = cv2.imread("frames/raw/frame_0.jpg")

height, width = image.shape[:2] # frame width and height in pixels

data_new=[]

for index, row in df.iterrows():

    # check if x and y are nan
    if row[["x", "y", "x0", "y0", "x1", "y1",
            "x2", "y2", "x3", "y3"]].isna().any():

        data_new.append([row["timestamp_ms"], np.nan, np.nan, row["frame_number"]])
        continue


    point = np.array(
        [[[row["x"] * width, row["y"] * height]]],
        dtype=np.float32
    )

    H = get_homography(row)

    transformed = cv2.perspectiveTransform(point, H)

    x_marker, y_marker = transformed[0, 0]

    data_new.append([row["timestamp_ms"], x_marker, y_marker, row["frame_number"]])


with open("grasp_trajectory_after_homography.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_ms",
        "x",
        "y",
        "frame_number"
    ])
    writer.writerows(data_new)