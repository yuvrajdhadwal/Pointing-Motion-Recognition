This folder's main purpose is for collecting a large dataset to train the machine learning model.

### Step 1:

Run `run.py` to start both files. Press `q` to quit the windows (this may be buggy). 

### Step 2:

Redo Step 1 until you have collected sufficient data.

### Step 3:

Run `data_merging.py` to automatically match the data collected from `data_collection.py` and `video.py` for each set of data you collected.
Then, it will concatenate all the data you collected into a single excel file.

### Step 4:

Tune data

**NOTE: THIS CODE IS NOT WRITTEN RIGHT NOW**

### Step 5:

Update the hyperparameters in `model_training.py` to create the predictive XGBoost model based on your results from Step 4 for your dataset.

Run `model_training.py`

### Step 6:

Use the model exported from this last file in whatever file you want! You can also run `model_testing.py` for to see how good your model is at prediction.

You can create a new dataset by repeating steps 1-3 and then test to see how accurate the model is at prediction.

**NOTE: MODEL_TESTING.PY IS NOT COMPLETED YET**
