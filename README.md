# Human to robot policy 

The goal of this project wwas to translate human hand motion into manipulator motion using a simple egocentric setup that can be captured from home. 

## Setup
The experiment was conducted at home using a Samsung camera with a 18-55 mm OIS zoom lens set to 18 mm throughout recording, recoding at 50 fps. 

The experiment involved moving around an object in the workspace, with an ArUco marker placed nearby with an ID=0 for later homography.

Human manipulation demonstrations were personally recorded using a stationary camera positioned to approximate an egocentric, over-the-shoulder perspective. The fixed viewpoint simplified trajectory extraction while preserving a first-person-like view of hand-object interaction.

To get all the necessary imports, uv sync is enough after cloning the repository. 

## Video processing

The full video was placed outside the cloned repository, but is provided in the repository here. To run the extract_frames script in src, it is recommended that the video is downloaded and saved accordingly or that the "video_path" is modified accordingly. 

Instead of picking every frame from the video for analysis and increasing computational load, every 5th frame was selected, extracting 10 frames per second. 

The frames were saved in folder frames/raw within the human-to-robot-policy repository, but were not tracked by git. The frame number corresponded to the original 50 fps video, e.g. frame 5 from the original video was still named frame 5 despite being frame 1 in the 10 fps extraction process. 

## ArUco detection

After extracting the frames, the goal was to detect the ArUco marker. Since the marker's dictionary had to be confirmed, every dictionary present in cv2 was attempted and the id was checked to see if a valid id was generated. This was done in the detect_dictionary_used script within the src. 
![Dict_4X4_50](documentation/detected_DICT_4X4_50.jpg)
<em>DICT_4X4_50 image</em>

![Dict_4X4_100](documentation/detected_DICT_4X4_100.jpg)
<em>DICT_4X4_100 image</em>

Both DICT 4X4 50 and 100 generated the correct id with proper corner detection, thus 50 was chosen due to its smaller dictionary size.

Then, using this dictionary, the detect_marker script checked for a matching ArUco marker in every frame captured. The marker was detected in 1312/1393 frames with a 94.2% detection rate. The frames where a marker was detected were saved in a csv file with the corner positions and frame number. 

## Track hand 

Hand tracking is handled using MediaPipe's Hand landmarker's model bundle which has both a palm detection model and a hand landmarks detection model build using Convolutional Neural Networks (CNNs) (source: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models). 

The accuracy of the task was checked in track_hand.py by checking if it could produce sucessful hand landmarks within the all detected frames [could it detect a wrong hand?], with the resulting accuracy being 53.6% accuracy. 6 images were also taken across the dataset to see if the wrist, index and thumb were all being detected correctly through visual inspection. 

Running mode of Video was used as the goal was to map out a hand trajectory. The track_hand_trajectory script found the TCP as the midpoint between the thumb and index finger. The motion within the dataset was a pick and place operation with some manipulation as well using primarily the thumb and index finger. 
![hand_trajectory](results/hand_trajectory.png)
<em>Hand trajectory generated<\em>

In order to make sure that the motion scaled well with the video footage, a perspective homography was computed to express the detected hand positions in a tabletop-referenced coordinate system. As seen later, the resulting tranformed trajectories produced more pronounced movements. Since the hand moved above the marker, the resulting hand trajectories are approximate planar projections rather than accurate 3D reconstructions.

![hand_trajectory](results/hand_trajectory_after_homography.png)
<em>Hand trajectory generated after homography<\em>



## MuJoCo



Menagerie dependency in a sibling directory
## Results
![Interval_1](documentation/image-1.png)

![Interval_2](documentation/image.png)

![Interval_3](documentation/image-2.png)