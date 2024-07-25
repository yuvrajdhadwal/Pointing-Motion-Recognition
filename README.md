### Overview

For this project, I created a Machine Learning Pipeline that made it very easy to collect massive amounts of data to train our models. In the semesters prior to my joining, the research group was only able to collect 1200 data points. Utilizing this pipeline, we were able to collect over 1 million data points in a week. This process utilizes multithreading, OpenCV, Pandas, Numpy, and Google MediaPipe to collect different parts of the body in rapid succession from multiple camera angles. I also utilized Bidirectional Elimination Stepwise Regression and parameter tuning to increase XGBoost Machine Learning model accuracy for predicting pointing location on a TV screen from 6 feet away from 2% to 97.84%.

The final product and model are run on the file `model_testing.py` however if you are more interested in the pipeline, it is as follows:

### Step 1:

Run `run.py` to start both files. Press `q` to quit the windows once data collection for a single session is complete.

### Step 2:

Redo Step 1 until you have collected sufficient data.

### Step 3:

Run `data_merging.py` to automatically match the data collected from `data_collection.py` and `video.py` for each set of data you collected.
Then, it will concatenate all the data you collected into a single excel file.

### Step 4:

Run `model_training.py`

### Step 5:

Use the model exported from this last file in whatever file you want! You can also run `model_testing_xgb.py` to see how good your model is at prediction.
