import cv2


#from: https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

video_path= "/zhome/27/3/234239/Desktop/SAM_5562.MP4"
video = cv2.VideoCapture(video_path)

count, saved, success = 0, 0, True


fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract approximately 10 frames per second
target_fps = 10
frame_interval = max(1, round(fps / target_fps))



while success: # read a frame at a time
    success, image = video.read() # when video ends, success = False, could add error checks
    if success:
        if count % frame_interval == 0:
            cv2.imwrite(f"frames/raw/frame_{count}.jpg",image)
            saved += 1

    count += 1 #frame number, original

video.release()

#  AI generated print statements:
print(f"Original FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Extracted frames: {saved}")
