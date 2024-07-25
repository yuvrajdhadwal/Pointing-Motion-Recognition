import pandas as pd
import numpy as np
import xgboost as xgb

"""
This script uses XGBoost to train model based on training data and pre-tuned parameters
Then saves this model for use in actual exhibit.

Yuvraj Dhadwal 2/24/2024
"""

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

dataframe = pd.read_csv(directory + "merged_data.csv")
labels = dataframe.iloc[:, 1:3]
features = dataframe.iloc[:, list(range(29,49)) + list(range(95,115))]
print(features)
#features.to_csv(directory + "features.csv")
#labels.to_csv(directory + "labels.csv")

train = xgb.DMatrix(features, label=labels)

# currently random numbers needs to be tuned based on data we get
parameters = {
    "learning_rate": 0.3,
    "max_depth": 2,
    "colsample_bytree": 1,
    "subsample": 1,
    "min_child_weight": 1,
    "gamma": 0,
    "random_state": 1508,
    "eval_metric": "map@3",
    "missing": np.nan
}

model = xgb.XGBRegressor(**parameters, tree_method="hist", booster="dart")

model.fit(features, labels)

model.save_model(directory + "pose_recognition_model.bin")
