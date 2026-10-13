import cv2 

img = cv2.imread("frames/raw/frame_5.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


detector_params = cv2.aruco.DetectorParameters()
dictionaries = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_250",
    "DICT_7X7_50",
    "DICT_ARUCO_ORIGINAL",
]

for name in dictionaries:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,name))

    detector = cv2.aruco.ArucoDetector(dictionary,detector_params)

    corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None: #if the correct dictionary is found, annotate image
        print(f"Detected using {name}: IDs {ids.flatten()}")

        annotated = img.copy()
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        cv2.imwrite(f"detected_{name}.jpg", annotated)
