import xgboost as xgb
import pandas as pd
import mediapipe as mp
import numpy as np
import threading
import cv2
from datetime import datetime

"""
This script is simply used to test the accuracy of the model we have created in
model_training.py. This script will not be used for the exhibit. Nor does it have any pupose
except to let us know that our model is working well.

This file is not complete

Yuvraj Dhadwal 2/25/24
"""


model = xgb.XGBRegressor({"nthread": 4})
directory = ""
while True:
    user_input = input("Please enter 'p' for pointing or 'd' for ducking \t")

    if user_input == "p":
        directory = "data/pointing/"
        break
    elif user_input == "d":
        directory = "data/ducking/"
        break
    else:
        continue

model.load_model(directory + "pose_recognition_model.bin")

width, height = int(640*1.6875), 480*4
point_radius = 30
model_color = (255, 0, 0)
background_color = (255, 255, 255)  # White in BGR
model_speed = 60

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def move_point(current, target, speed):
    current = np.array(current, dtype=np.float32)
    target = np.array(target, dtype=np.float32)

    direction = target - current
    distance = np.linalg.norm(direction)

    # Normalize the direction vector
    if distance != 0:
        direction = direction / distance

    # Move by the speed along the direction, but do not overshoot the target
    step = direction * min(speed, distance)

    # Update the current position
    new_position = current + step

    # Round to the nearest integer to ensure that the point moves in both x and y directions
    new_position = np.round(new_position).astype(int)

    return new_position

def find_time():
    current_time = datetime.now()
    microseconds_rounded = round(current_time.microsecond / 100000) * 100000

    if microseconds_rounded >= 1000000:
        microseconds_rounded -= 1000000
        #current_time = current_time.replace(second=(current_time.second + 1))

    rounded_time = current_time.replace(microsecond=microseconds_rounded)

    return rounded_time.strftime("%H:%M")

def update_position(field, coordinate, visibility, mp_values, data_collection):
    if coordinate < 0 or coordinate > 1 or visibility < 0.8:
        coordinate = np.nan
    if len(mp_values) > 2:
        last_coordinate = float(list(list(mp_values)[-1].values())[field])
        second_last_coordinate = float(list(list(mp_values)[-2].values())[field])
        coordinate = 0.6 * coordinate + 0.2 * last_coordinate + 0.2 * second_last_coordinate
    data_collection.update({fields[field]: coordinate})

def capture_video(src, i, mp_values, data_collection):
    cap = src
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=0) as pose:
        while cap.isOpened():
            t2 = datetime.now()
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
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
            t3 = datetime.now()
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


            print(f"while {datetime.now() - t3}")

            #

            # prints current time

            mp_values.append(data_collection.copy())

            # Render detections
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     # This is used for the circles/dots
            #     mp_drawing.DrawingSpec(color=(43, 80, 200), thickness=5, circle_radius=2),
            #     # This is used for the lines
            #     mp_drawing.DrawingSpec(color=(96, 25, 93), thickness=5),
            # )

            # cv2.imshow("Mediapipe Feed" + str(i), cv2.flip(image, 1))
            # cv2.moveWindow("Mediapipe Feed" + str(i), i*10, 0)

            print(f"mediapipe {datetime.now()-t2}")

            # Press "x" or "q" to Exit Program/Camera
            if cv2.waitKey(10) & 0xFF == ord("q") or cv2.waitKey(10) & 0xFF == ord("x"):
                break
            return

        cap.release()
        cv2.destroyWindow("Mediapipe Feed" + str(i))

        return


current_prediction = np.array((1409//2, 2740//2))
#z = 0
video_sources = [cv2.VideoCapture(0), cv2.VideoCapture(1)]

while True:
    values_camera1 = []  # default value for values,
    values_camera2 = []
    data_collection1 = {}
    data_collection2 = {}
    # 0 for now because I only have one camera on my computer

    values = [values_camera1, values_camera2]
    data_collections = [data_collection1, data_collection2]
    threads = []
    j = 0
    i = 0
    for source in video_sources:
        thread = threading.Thread(target=capture_video, args=(source, i, values[j], data_collections[j]))
        threads.append(thread)
        i = i + 67
        j = j + 1

    # Start the threads!
    for thread in threads:
        thread.start()

    # close the threads
    for thread in threads:
        thread.join()

    df1 = pd.DataFrame(values_camera1)
    df2 = pd.DataFrame(values_camera2)

    # df1.to_csv(str(z) + "testeee.csv")
    #df2.to_csv(str(z) + "tessss.csv")
    #z = z+1

    def add_suffix(df, suffix):
        return {col: f"{col}{suffix}" for col in df.columns}

    t1 = datetime.now()
    # Rename columns
    df1.rename(columns=add_suffix(df1, '1'), inplace=True)
    df2.rename(columns=add_suffix(df2, '2'), inplace=True)

    # df1 = df1.iloc[:, 1:]
    # df2 = df2.iloc[:, 1:]

    camera1_new_name = {'Time1': 'Time'}
    camera2_new_name = {'Time2': 'Time'}

    df1.rename(columns=camera1_new_name, inplace=True)
    df2.rename(columns=camera2_new_name, inplace=True)

    # merged_cameras = pd.concat([df1, df2], axis=1)
    #merged_cameras.to_csv("qqqqqqqqqqqqqq.csv")

    merged_cameras = pd.merge(df1, df2, on="Time")
    merged_cameras.reset_index(drop=True, inplace=True)

    merged_cameras["Hip Width"] = abs(merged_cameras["x_right_hip1"] - merged_cameras["x_left_hip1"])
    merged_cameras["Torso Height"] = abs(merged_cameras["y_left_eye1"] - merged_cameras["y_left_hip1"])
    merged_cameras["Shoulder Width"] = abs(
        merged_cameras["x_left_shoulder1"] - merged_cameras["x_right_shoulder1"]
    )
    merged_cameras["Leg Length"] = abs(merged_cameras["y_left_ankle1"] - merged_cameras["y_left_hip1"])
    merged_cameras["Height"] = abs(merged_cameras["y_left_eye1"] - merged_cameras["y_left_ankle1"])
    merged_cameras["Height to Hips Ratio"] = abs(merged_cameras["Height"] / merged_cameras["Hip Width"])

    merged_cameras["left_arm_x_camera1"] = abs(merged_cameras["x_left_shoulder1"] - merged_cameras["x_left_wrist1"])
    merged_cameras["left_arm_y_camera1"] = abs(merged_cameras["y_left_shoulder1"] - merged_cameras["y_left_wrist1"])
    merged_cameras["right_arm_x_camera1"] = abs(merged_cameras["x_right_shoulder1"] - merged_cameras["x_right_wrist1"])
    merged_cameras["right_arm_y_camera1"] = abs(merged_cameras["y_right_shoulder1"] - merged_cameras["y_right_wrist1"])

    merged_cameras["left_arm_x_camera2"] = abs(merged_cameras["x_left_shoulder2"] - merged_cameras["x_left_wrist2"])
    merged_cameras["left_arm_y_camera2"] = abs(merged_cameras["y_left_shoulder2"] - merged_cameras["y_left_wrist2"])
    merged_cameras["right_arm_x_camera2"] = abs(merged_cameras["x_right_shoulder2"] - merged_cameras["x_right_wrist2"])
    merged_cameras["right_arm_y_camera2"] = abs(merged_cameras["y_right_shoulder2"] - merged_cameras["y_right_wrist2"])

    merged_features = merged_cameras.iloc[:, list(range(27,47)) + list(range(93,113))]
    print(merged_features)
    print(f"pandas: {datetime.now() - t1}")

    #merged_features = merged_features.drop("Unnamed: 0", axis=1)
    #merged_features.to_csv("test.csv")

    t = datetime.now()
    prediction = model.predict(merged_features)
    print(datetime.now() - t)
    print(prediction)
    prediction_points = pd.DataFrame(prediction, columns=["X", "Y"])

    #prediction_points.to_csv(directory + "predictionpoints.csv")
    pred_x = prediction_points.iloc[0]['X']
    pred_y = prediction_points.iloc[0]['Y']
    pred_point = pred_x, pred_y
    current_prediction = move_point(current_prediction, pred_point, model_speed)

    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)

    cv2.circle(canvas, tuple(current_prediction), point_radius, model_color, -1)

    cv2.imshow("Moving Point", canvas)
    cv2.moveWindow("Moving Point", 3440, 0)
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord("q"):  # 30 FPS display and 'q' to quit
        break

cv2.destroyAllWindows()
