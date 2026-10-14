import cv2

video = cv2.VideoCapture("/zhome/27/3/234239/Desktop/SAM_5562.MP4")

fps = video.get(cv2.CAP_PROP_FPS)
print(f"Original video FPS: {fps}")

video.release()