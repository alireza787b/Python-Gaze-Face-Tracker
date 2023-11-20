"""
Face Landmark and Eye Tracker System
GitHub: https://github.com/alireza787b/Python-Gaze-Face-Tracker
Email: p30planets@gmail.com
LinkedIn: https://www.linkedin.com/in/alireza787b
Date: November 2023

Description:
This Python-based application is designed for advanced eye tracking, utilizing OpenCV and MediaPipe. 
It provides real-time iris tracking, logging of eye position data, and features socket communication for data transmission. 
A recent addition to the application includes blink detection, further enhancing its capabilities. 
The application is highly customizable, allowing users to control various parameters and logging options.

Features:
- Real-Time Eye Tracking: Utilizes webcam input for tracking and visualizing iris and eye corner positions.
- Data Logging: Records tracking data in CSV format, including timestamps, positional data, and blink counts, with an option to log all 468 facial landmarks.
- Socket Communication: Transmits eye tracking data via UDP sockets, configurable through user-defined IP and port.
- Blink Detection: Counts and logs the number of blinks detected during the tracking session.

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Inspiration:
Initially inspired by Asadullah Dal's iris segmentation project (https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python). 
The blink detection feature is also contributed by Asadullah Dal (GitHub: Asadullah-Dal17).

Parameters:
- SERVER_IP & SERVER_PORT: Configurable IP address and port for UDP socket communication. 
  These parameters define where the eye tracking data is sent via the network.
- DEFAULT_WEBCAM: Specifies the default webcam source number. If a webcam number is provided as a command-line argument, 
  that number is used; otherwise, the default value here is used.
- PRINT_DATA: When set to True, the program prints eye tracking data to the console. Set to False to disable printing.
- SHOW_ALL_FEATURES: If True, the program shows all facial landmarks. Set to False to display only the eye positions.
- LOG_DATA: Enables logging of eye tracking data to a CSV file when set to True.
- LOG_ALL_FEATURES: When True, all 468 facial landmarks are logged in the CSV file.
- BLINK_THRESHOLD: Threshold for the eye aspect ratio to trigger a blink.
- EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold to confirm a blink.

Usage:
Run the script in a Python environment with the necessary dependencies installed. The script accepts command-line arguments for camera source configuration. The application displays a real-time video feed with eye tracking visualization. Press 'q' to quit the application and save the log data.

Note:
This project is intended for educational and research purposes in fields like aviation, human-computer interaction, and more.
"""



import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os



# User-specific measurements (in mm)
# USER_FACE_WIDTH: Measure the horizontal distance between the outer edges of the cheekbones in mm.
USER_FACE_WIDTH = 140  # Example: 140mm, adjust based on actual measurement


#not used now
# NOSE_TO_CAMERA_DISTANCE: Measure the distance from the tip of the nose to the camera lens in mm.
NOSE_TO_CAMERA_DISTANCE = 600  # Example: 600mm, adjust based on actual distance


# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 0  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True

ENABLE_HEAD_POSE = True #Enable Head position and orientation estimator


LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070  # Set the server port

# eyes blinking variables
SHOW_ON_SCREEN_DATA = True  # Toggle to show the blink count on the video feed
TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
EYES_BLINK_FRAME_COUNTER = 0  # Counts the number of consecutive frames with a potential blink
BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames below the threshold to confirm a blink

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices for iris detection and eye aspect ratio calculations
LEFT_EYE_IRIS = [474, 475, 476, 477]  # Indices for the left eye iris
RIGHT_EYE_IRIS = [469, 470, 471, 472]  # Indices for the right eye iris
LEFT_EYE_OUTER_CORNER = [33]  # Left eye outer corner, for roll calculation
LEFT_EYE_INNER_CORNER = [133]  # Left eye inner corner, for yaw calculation
RIGHT_EYE_OUTER_CORNER = [362]  # Right eye outer corner, for roll calculation
RIGHT_EYE_INNER_CORNER = [263]  # Right eye inner corner, for yaw calculation

# Blinking Detection landmark's indices for eye blink detection
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]  # Right eye landmarks for blink detection
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]  # Left eye landmarks for blink detection

# Yaw detection landmarks indices for assessing head rotation around the vertical axis
LEFT_EYE_YAW_DETECTION = [23, 27, 130, 243]  # Left side landmarks around the eye for yaw detection
RIGHT_EYE_YAW_DETECTION = [253, 257, 359, 463]  # Right side landmarks around the eye for yaw detection

# Roll detection landmarks indices for assessing head tilt from side to side
EYE_CORNERS_ROLL_DETECTION = [130, 359]  # Indices for the outer corners of the eyes for roll detection

# Pitch detection landmarks indices for assessing head movement up and down
NOSE_TIP_PITCH_DETECTION = [4]  # Nose tip landmark for pitch detection reference
UPPER_LIP_PITCH_DETECTION = [13, 14]  # Upper lip landmarks for pitch detection
LOWER_LIP_PITCH_DETECTION = [0, 17]  # Lower lip landmarks for pitch detection


# Initial calibration values for head pose
initial_pitch = None
initial_yaw = None
initial_roll = None

# Server address for UDP socket communication
SERVER_ADDRESS = (SERVER_IP, 7070)


# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1


def euclidean_distance_3D(points):
    """Calculates the Euclidean distance between two points in 3D space.

    Args:
        points: A list of 3D points.

    Returns:
        The Euclidean distance between the two points.

        # Comment: This function calculates the Euclidean distance between two points in 3D space.
    """

    # Get the three points.
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    # Calculate the numerator.
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    # Calculate the denominator.
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    # Calculate the distance.
    distance = numerator / denominator

    return distance

def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = USER_FACE_WIDTH / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])

    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # 2D image points from landmarks
    image_points = np.array([
        landmarks[NOSE_TIP_PITCH_DETECTION[0]],     # Nose tip
        landmarks[LOWER_LIP_PITCH_DETECTION[0]],    # Chin
        landmarks[LEFT_EYE_OUTER_CORNER[0]],        # Left eye left corner
        landmarks[RIGHT_EYE_OUTER_CORNER[0]],       # Right eye right corner
        landmarks[UPPER_LIP_PITCH_DETECTION[0]],    # Left Mouth corner
        landmarks[LOWER_LIP_PITCH_DETECTION[1]],    # Right mouth corner
    ], dtype="double")

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()


    return pitch, yaw, roll


# This function calculates the blinking ratio of a person.
def blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person.

    Args:
        landmarks: A facial landmarks in 3D normalized.

    Returns:
        The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

    """

    # Get the right eye ratio.
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

    # Get the left eye ratio.
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

    # Calculate the blinking ratio.
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio


# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    if PRINT_DATA:
        head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
        print(f"Head pose estimation is {head_pose_status}.")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Preparing for CSV logging
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Column names for CSV file
column_names = [
    "Timestamp (ms)",
    "Left Eye Center X",
    "Left Eye Center Y",
    "Right Eye Center X",
    "Right Eye Center Y",
    "Left Iris Relative Pos Dx",
    "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx",
    "Right Iris Relative Pos Dy",
    "Total Blink Count"
]
# Add head pose columns if head pose estimation is enabled
if ENABLE_HEAD_POSE:
    column_names.extend(["Pitch", "Yaw", "Roll"])
    
if LOG_ALL_FEATURES:
    column_names.extend(
        [f"Landmark_{i}_X" for i in range(468)]
        + [f"Landmark_{i}_Y" for i in range(468)]
    )

# Main loop for video capture and processing
try:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping the frame for a mirror effect
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            # Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
            # just normalize values
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            # print(mesh_points_3D)

            # getting the blinking ratio
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            # print(f"Blinking ratio : {ratio}")
            # checking if ear less then or equal to required threshold if yes then
            # count the number of frame frame while eyes are closed.
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            # else check if eyes are closed is greater EYE_AR_CONSEC_FRAMES frame then
            # count the this as a blink
            # make frame counter equal to zero

            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            

            # Display all facial landmarks if enabled
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            cv.circle(
                frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Left eye right corner
            cv.circle(
                frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Left eye left corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Right eye right corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Total Blinks: {TOTAL_BLINKS}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
                # Check if head pose estimation is enabled
                if ENABLE_HEAD_POSE:
                    pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                    # Adjust with initial calibration
                    #check if it is the first frame eg pitch
                    if initial_pitch  == None:
                        initial_pitch = pitch
                        initial_yaw = yaw
                        initial_roll = roll
                    pitch -= initial_pitch
                    yaw -= initial_yaw
                    roll -= initial_roll
                    if PRINT_DATA:
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            # Logging data
            if LOG_DATA:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS]  # Include blink count in CSV
                
                # Append head pose data if enabled
                if ENABLE_HEAD_POSE:
                    log_entry.extend([pitch, yaw, roll])

                csv_data.append(log_entry)
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

            # Sending data through socket
            packet = np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32)
            iris_socket.sendto(bytes(packet), SERVER_ADDRESS)

        # Writing the on screen data on the frame
            if SHOW_ON_SCREEN_DATA:
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 50), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)


        
        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Calibrate on 'c' key press
        if key == ord('c'):
            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            if PRINT_DATA:
                print("Head pose recalibrated.")

        # Exit on 'q' key press
        if key == ord('q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
        
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()
    if PRINT_DATA:
        print("Program exited successfully.")

    # Writing data to CSV file
    if LOG_DATA:
        if PRINT_DATA:
            print("Writing data to CSV...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(
            LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv"
        )
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Writing column names
            writer.writerows(csv_data)  # Writing data rows
        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")
