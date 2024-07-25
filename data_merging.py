import pandas as pd
import glob

"""
This script reads data from csv files created from video.py and data_collection.py.
It then merges it into one dataframe and then saves it into another dataframe that
is a collection of all the data into one single excel file. This excel file will then
be exported for use by XGBoost.
Yuvraj Dhadwal 2/25/2024
"""

excel_number = 0
all_data = pd.DataFrame()

# excel_number should be given by glob to find the latest file, then increment the number by 1

files = glob.glob("")
directory = ""

while True:
    user_input = input(
        "Please enter 'p' for pointing or 'd' for ducking or 't' for testing \t"
    )

    if user_input == "p":
        files = glob.glob("data/pointing/*.*")
        directory = "data/pointing/"
        break
    elif user_input == "d":
        files = glob.glob("data/ducking/*.*")
        directory = "data/ducking/"
        break
    elif user_input == "t":
        files = glob.glob("data/testing/*.*")
        directory = "data/testing/"
        break

if len(files) == 0:
    raise Exception("No files found")
else:
    num = len(files) / 2
print(len(files))

while excel_number < num:
    excel_number = excel_number + 1

    features = pd.read_csv(
        directory + "training_values_{0:03}.csv".format(excel_number)
    )
    labels = pd.read_csv(
        directory + "point_coordinates_{0:03}.csv".format(excel_number)
    )

    camera1 = features.iloc[:, :68]
    camera2 = features.iloc[:, 68:]

    camera1_new_name = {"Time1": "Time"}
    camera2_new_name = {"Time2": "Time"}

    camera1.rename(columns=camera1_new_name, inplace=True)
    camera2.rename(columns=camera2_new_name, inplace=True)

    merged_cameras = pd.merge(camera1, camera2, on="Time")
    merged_cameras.reset_index(drop=True, inplace=True)

    merged_cameras["Hip Width"] = abs(
        merged_cameras["x_right_hip1"] - merged_cameras["x_left_hip1"]
    )
    merged_cameras["Torso Height"] = abs(
        merged_cameras["y_left_eye1"] - merged_cameras["y_left_hip1"]
    )
    merged_cameras["Shoulder Width"] = abs(
        merged_cameras["x_left_shoulder1"] - merged_cameras["x_right_shoulder1"]
    )
    merged_cameras["Leg Length"] = abs(
        merged_cameras["y_left_ankle1"] - merged_cameras["y_left_hip1"]
    )
    merged_cameras["Height"] = abs(
        merged_cameras["y_left_eye1"] - merged_cameras["y_left_ankle1"]
    )
    merged_cameras["Height to Hips Ratio"] = abs(
        merged_cameras["Height"] / merged_cameras["Hip Width"]
    )

    merged_cameras["left_arm_x_camera1"] = abs(
        merged_cameras["x_left_shoulder1"] - merged_cameras["x_left_wrist1"]
    )
    merged_cameras["left_arm_y_camera1"] = abs(
        merged_cameras["y_left_shoulder1"] - merged_cameras["y_left_wrist1"]
    )
    merged_cameras["right_arm_x_camera1"] = abs(
        merged_cameras["x_right_shoulder1"] - merged_cameras["x_right_wrist1"]
    )
    merged_cameras["right_arm_y_camera1"] = abs(
        merged_cameras["y_right_shoulder1"] - merged_cameras["y_right_wrist1"]
    )

    merged_cameras["left_arm_x_camera2"] = abs(
        merged_cameras["x_left_shoulder2"] - merged_cameras["x_left_wrist2"]
    )
    merged_cameras["left_arm_y_camera2"] = abs(
        merged_cameras["y_left_shoulder2"] - merged_cameras["y_left_wrist2"]
    )
    merged_cameras["right_arm_x_camera2"] = abs(
        merged_cameras["x_right_shoulder2"] - merged_cameras["x_right_wrist2"]
    )
    merged_cameras["right_arm_y_camera2"] = abs(
        merged_cameras["y_right_shoulder2"] - merged_cameras["y_right_wrist2"]
    )

    merged_features = pd.merge(labels, merged_cameras, on="Time")
    merged_features.reset_index(drop=True, inplace=True)

    merged_features = merged_features.drop("Time", axis=1)
    merged_features = merged_features.drop("Unnamed: 0", axis=1)

    all_data = pd.concat([all_data, merged_features])

all_data.to_csv(directory + "merged_data.csv")