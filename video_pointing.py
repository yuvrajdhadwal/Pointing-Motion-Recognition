import cv2
import csv
from datetime import datetime
import numpy as np
import glob
import os

"""
This script uses OpenCV library to create a video file that shows a dot that moves.
Intention is for this script to run alongside data_collection.py. This program is meant
to provide the labels for the features collected in the other file.

Keane Zhang 2/25/2024

'''
modelType = 'Ducking'  # rename based on the folder you want to save to
folder = './' + modelType + './Labels'
csv_filename = folder + './point_coordinates_001.csv'  # Rename this for every new time collecting data
=======
"""

# use glob to find the latest .csv file, then increment the number by 1

files = glob.glob("data/pointing/*.csv")
if len(files) == 0:
    csv_filename = "point_coordinates_001.csv"
else:
    files.sort()
    filename = files[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    filename = filename[-1]
    filename = int(filename) + 1
    csv_filename = "point_coordinates_" + str(filename).zfill(3) + ".csv"

directory = "data/pointing"

# if the directory does not exist, then the program will create the directory

try:
    os.mkdir(directory)
except FileExistsError:
    pass


# Parameters
width, height = (2160//2, 3840//2)
point_radius = 30
color = (0, 0, 255)  # Red in BGR
background_color = (255, 255, 255)  # White in BGR
speed = 8  # Speed of the point's movement

# Define the route points based on the image provided
route_points = [
    (91//2, 2740//2),
    (40//2, 3600//2),  # low point
    (91//2, 1979//2),
    (300//2, 3600//2),  # low point
    (750//2, 3121//2),
    (600//2, 3500//2),  # low point
    (750//2, 2360//2),
    (900//2, 3500//2),  # low point
    (750//2, 1599//2),
    (1300//2, 3500//2),  # low point
    (1409//2, 2740//2),
    (1800//2, 3600//2),  # low point
    (1409//2, 1979//2),
    (2100//2, 3600//2),  # low point
    (2068//2, 3121//2),
    (2068//2, 2360//2),
    (2068//2, 15//2),
]

# Prepare CSV file
with open(directory + "/" + csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "X", "Y"])  # Header

def find_time():
    current_time = datetime.now()
    microseconds_rounded = round(current_time.microsecond / 100000) * 100000

    if microseconds_rounded >= 1000000:
        microseconds_rounded -= 1000000
        #current_time = current_time.replace(second=(current_time.second + 1))

    rounded_time = current_time.replace(microsecond=microseconds_rounded)

    return rounded_time.strftime("%H:%M:%S.%f")[:-1]

# Function to move the point from current to target position
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

cv2.namedWindow("Moving Point", cv2.WINDOW_NORMAL)  # Make the window resizable
cv2.resizeWindow("Moving Point", width, height)  # Resize window to fill the screen

# Main loop
current_position = np.array(route_points[0])
target_index = 1
# Parameters for initial stop
stop_duration = 1000  # Duration in milliseconds to stop at each point

# Initial display and stop at the first point
canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
current_position = np.array(route_points[0])
cv2.circle(canvas, tuple(current_position), point_radius, color, -1)

# Display the frame
cv2.imshow("Moving Point", canvas)
cv2.moveWindow("Moving Point", 0, 0)  # Optionally move the window to the top-left corner


# Write system time and coordinates to CSV for the initial point
with open(directory + "/" + csv_filename, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([find_time(), *current_position])

# Wait for stop_duration milliseconds to simulate the stop at the first point
cv2.waitKey(stop_duration)


# Real-time display and writing to CSV
while True:
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Move the point
    current_position = move_point(current_position, route_points[target_index], speed)

    # Check if the point has reached the next target
    if np.array_equal(current_position, route_points[target_index]):
        # Draw the point at the target position
        cv2.circle(canvas, tuple(current_position), point_radius, color, -1)

        # Display the frame with the point stopped at the target
        cv2.imshow("Moving Point", canvas)

        # Write system time and coordinates to CSV for the stopped point
        with open(directory + "/" + csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([find_time(), *current_position])

        # Wait for stop_duration milliseconds to simulate the stop
        cv2.waitKey(stop_duration)

        target_index += 1
        if target_index == len(route_points):
            break  # End if last point is reached
        continue  # Skip the rest of the loop to immediately move to the next point

    # Draw the point
    cv2.circle(canvas, tuple(current_position), point_radius, color, -1)

    # Write system time and coordinates to CSV
    with open(directory + "/" + csv_filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([find_time(), *current_position])

    # Display the frame
    cv2.imshow("Moving Point", canvas)
    cv2.moveWindow("Moving Point", 3440, 0)
    if cv2.waitKey(int(1000 / 60)) & 0xFF == ord("q"):  # 60 FPS display and 'q' to quit
        break

cv2.destroyAllWindows()
print("video finished")
