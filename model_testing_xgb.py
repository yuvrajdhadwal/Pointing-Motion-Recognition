import pandas as pd
import mediapipe as mp
import numpy as np
import threading
import cv2
from datetime import datetime
import xgboost as xgb

"""
This script is simply used to test the accuracy of the model we have created in
model_training.py. This script will not be used for the exhibit. Nor does it have any pupose
except to let us know that our model is working well.
This file is not complete
Yuvraj Dhadwal 2/25/24
"""


model = xgb.XGBRegressor({"nthread": 4})
model.load_model("data/pointing/" + "pose_recognition_model.bin")

predict = model.predict(pd.read_csv("data/testing/" + "merged_data.csv").iloc[:, list(range(29,49)) + list(range(95,115))])
print(predict)
print(
    "Accuracy: "
    + str(
        model.score(
            pd.read_csv("data/testing/" + "merged_data.csv").iloc[:, list(range(29,49)) + list(range(95,115))],
            pd.read_csv("data/testing/" + "merged_data.csv").iloc[:, 1:3],
        )
    )
)