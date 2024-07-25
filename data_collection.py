import mediapipe as mp
import pandas as pd
import glob
import os
import cv2
import numpy as np
from datetime import datetime
import threading

"""
This script uses MediaPipe to detect points on your body and collects them as
numerical values and stores them in a XLSX file that the XGBoost program will read.

Make sure to rename the filename before running this script so you do not overwrite data.

Yuvraj Dhadwal 2/24/2024

'''
modelType = "Ducking"
folder_path = "./" + modelType + "./Features./"
filename = folder_path + "training_values_001.xlsx"  # Important note: please change
=======
"""

# use glob to find the latest file, then increment the number by 1
files = glob.glob("")
directory = ""
while True:
    # user_input = input("Please enter 'p' for pointing or 'd' for ducking \t")
    user_input = "p"

    if user_input == "p":
        files = glob.glob("data/pointing/*.csv")
        directory = "data/pointing"
        break
    elif user_input == "d":
        files = glob.glob("data/ducking/*.csv")
        directory = "data/ducking"
        break
    else:
        continue

if len(files) == 0 or len(files) == 1:

    filename = "training_values_001.csv"
else:
    files.sort()
    filename = files[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    filename = filename[-1]
    filename = int(filename) + 1
    filename = "training_values_" + str(filename).zfill(3) + ".csv"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# calculates time for preferred format
def find_time():
    current_time = datetime.now()
    microseconds_rounded = round(current_time.microsecond / 100000) * 100000

    if microseconds_rounded >= 1000000:
        microseconds_rounded -= 1000000

        seconds = current_time.second + 1
        if seconds > 59:
            seconds -= 60

        current_time = current_time.replace(second=(seconds))

    rounded_time = current_time.replace(microsecond=microseconds_rounded)

    return rounded_time.strftime("%H:%M:%S.%f")[:-1]

def update_position(field, coordinate, visibility, mp_values, data_collection):
    if coordinate < 0 or coordinate > 1 or visibility < 0.8:
        coordinate = np.nan
    if len(mp_values) > 2:
        last_coordinate = float(list(list(mp_values)[-1].values())[field])
        second_last_coordinate = float(list(list(mp_values)[-2].values())[field])
        coordinate = 0.6 * coordinate + 0.2 * last_coordinate + 0.2 * second_last_coordinate
    data_collection.update({fields[field]: coordinate})


def capture_video(src, i, mp_values, data_collection, stop=False):
    limit = 0
    cap = cv2.VideoCapture(src)
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                continue
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            ## get height and width of the image
            height, width, _ = image.shape

            try:
                landmarks = results.pose_landmarks.landmark
            except AttributeError:
                continue


            rotation_matrix = np.array([[0, 1],
                                        [1, 0]])
            field = 0
            if field == 0:
                data_collection.update({fields[field]: find_time()})
            field = field + 1
            landmark = 0
            while field < 67:
                while landmark < 33:
                    x = landmarks[landmark].x
                    y = landmarks[landmark].y

                    point = np.array([[x],
                                      [y]])
                    rotated_point = np.dot(rotation_matrix, point)
                    x = rotated_point[0, 0]
                    y = rotated_point[1, 0]

                    update_position(field, x, landmarks[landmark].visibility, mp_values, data_collection)

                    field = field + 1

                    update_position(field, y, landmarks[landmark].visibility, mp_values, data_collection)

                    field = field + 1
                    landmark = landmark + 1

            mp_values.append(data_collection.copy())

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                # This is used for the circles/dots
                mp_drawing.DrawingSpec(color=(43, 80, 200), thickness=5, circle_radius=2),
                # This is used for the lines
                mp_drawing.DrawingSpec(color=(96, 25, 93), thickness=5),
            )

            limit += 1

            cv2.imshow("Mediapipe Feed" + str(i), cv2.flip(image, 1))
            cv2.moveWindow("Mediapipe Feed" + str(i), i*10, 0)

            # Press "x" or "q" to Exit Program/Camera
            if cv2.waitKey(10) & 0xFF == ord("q") or cv2.waitKey(10) & 0xFF == ord("x"):
                break
        
            if stop and limit > 3:
                break
    cap.release()
    cv2.destroyWindow("Mediapipe Feed" + str(i))


fields = [
    "Time",
    "x_nose",
    "y_nose",
    "x_left_eye_(inner)",
    "y_left_eye_(inner)",
    "x_left_eye",
    "y_left_eye",
    "x_left_eye_(outer)",
    "y_left_eye_(outer)",
    "x_right_eye_(inner)",
    "y_right_eye_(inner)",
    "x_right_eye",
    "y_right_eye",
    "x_right_eye_(outer)",
    "y_right_eye_(outer)",
    "x_left_ear",
    "y_left_ear",
    "x_right_ear",
    "y_right_ear",
    "x_mouth_(left)",
    "y_mouth_(left)",
    "x_mouth_(right)",
    "y_mouth_(right)",
    "x_left_shoulder",
    "y_left_shoulder",
    "x_right_shoulder",
    "y_right_shoulder",
    "x_left_elbow",
    "y_left_elbow",
    "x_right_elbow",
    "y_right_elbow",
    "x_left_wrist",
    "y_left_wrist",
    "x_right_wrist",
    "y_right_wrist",
    "x_left_pinky",
    "y_left_pinky",
    "x_right_pinky",
    "y_right_pinky",
    "x_left_index",
    "y_left_index",
    "x_right_index",
    "y_right_index",
    "x_left_thumb",
    "y_left_thumb",
    "x_right_thumb",
    "y_right_thumb",
    "x_left_hip",
    "y_left_hip",
    "x_right_hip",
    "y_right_hip",
    "x_left_knee",
    "y_left_knee",
    "x_right_knee",
    "y_right_knee",
    "x_left_ankle",
    "y_left_ankle",
    "x_right_ankle",
    "y_right_ankle",
    "x_left_heel",
    "y_left_heel",
    "x_right_heel",
    "y_right_heel",
    "x_left_foot_index",
    "y_left_foot_index",
    "x_right_foot_index",
    "y_right_foot_index"
]

values_camera1 = []  # default value for values,
values_camera2 = []
data_collection1 = {}
data_collection2 = {}
# 0 for now because I only have one camera on my computer
video_sources = [0, 1]
values = [values_camera1, values_camera2]
data_collections = [data_collection1, data_collection2]
threads = []
j = 0
i = 0
for source in video_sources:
    thread = threading.Thread(target=capture_video, args=(source, i, values[j], data_collections[j], False))
    threads.append(thread)
    i = i + 65
    j = j + 1

# Start the threads!
for thread in threads:
    thread.start()

# close the threads
for thread in threads:
    thread.join()


df1 = pd.DataFrame(values_camera1)
df2 = pd.DataFrame(values_camera2)


def add_suffix(df, suffix):
    return {col: f"{col}{suffix}" for col in df.columns}


# Rename columns
df1.rename(columns=add_suffix(df1, '1'), inplace=True)
df2.rename(columns=add_suffix(df2, '2'), inplace=True)
concatenated_df = pd.concat([df1, df2], axis=1)

concatenated_df.to_csv(directory + "/" + filename)