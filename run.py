# this scripts runs both data_collection and video.py to collect data

import os
import subprocess
import time

# run data_collection.py
subprocess.Popen(["python", "data_collection.py"])

time.sleep(10)

# run video.py
files = ""

"""while True:
    user_input = input("Please enter 'p' for pointing or 'd' for ducking \t")

    if user_input == "p":
        files = "video_pointing.py"
        break
    elif user_input == "d":
        files = "video.py"
        break
    else:
        continue
"""


#Run Duck and Tilt Training
subprocess.Popen(["python", "video_pointing.py"])

# upon pressing q, the program will stop
# the data will be collected and saved to the excel file
# the video will be saved to the data folder
# the program will terminate

# terminate
print("Program terminated")
