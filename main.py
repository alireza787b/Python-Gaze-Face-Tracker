
"""
Eye Tracking and Head Pose Estimation

This script is designed to perform real-time eye tracking and head pose estimation using a webcam feed. 
It utilizes the MediaPipe library for facial landmark detection, which informs both eye tracking and 
head pose calculations. The purpose is to track the user's eye movements and head orientation, 
which can be applied in various domains such as HCI (Human-Computer Interaction), gaming, and accessibility tools.

Features:
- Real-time eye tracking to count blinks and calculate the eye aspect ratio for each frame.
- Head pose estimation to determine the orientation of the user's head in terms of pitch, yaw, and roll angles.
- Calibration feature to set the initial head pose as the reference zero position.
- Data logging for further analysis and debugging.

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Methodology:
- The script uses the 468 facial landmarks provided by MediaPipe's FaceMesh model.
- Eye tracking is achieved by calculating the Eye Aspect Ratio (EAR) for each eye and detecting blinks based on EAR thresholds.
- Head pose is estimated using the solvePnP algorithm with a predefined 3D facial model and corresponding 2D landmarks detected from the camera feed.
- Angles are normalized to intuitive ranges (pitch: [-90, 90], yaw and roll: [-180, 180]).

Theory:
- EAR is used as a simple yet effective metric for eye closure detection.
- Head pose angles are derived using a perspective-n-point approach, which estimates an object's pose from its 2D image points and 3D model points.

Parameters:
You can change parameters as in face width, moving average window, webcam ID, terminal outputs, on screen data, loggin detail , etc. from the code 

Author: Alireza Bagheri
GitHub: https://github.com/alireza787b/Python-Gaze-Face-Tracker
Email: p30planets@gmail.com
LinkedIn: https://www.linkedin.com/in/alireza787b
Date: November 2023

Inspiration:
Initially inspired by Asadullah Dal's iris segmentation project (https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python). 
The blink detection feature is also contributed by Asadullah Dal (GitHub: Asadullah-Dal17).

Usage:
-Run the script in a Python environment with the necessary dependencies installed. The script accepts command-line arguments for camera source configuration.
- Press 'c' to recalibrate the head pose estimation to the current orientation.
- Press 'r' to start/stop logging.
- Press 'q' to exit the program.
- Output is displayed in a window with live feed and annotations, and logged to a CSV file for further analysis.

Ensure that all dependencies, especially MediaPipe, OpenCV, and NumPy, are installed before running the script.

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
from AngleBuffer import AngleBuffer


#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

# Parameters Documentation

## User-Specific Measurements
# USER_FACE_WIDTH: The horizontal distance between the outer edges of the user's cheekbones in millimeters. 
# This measurement is used to scale the 3D model points for head pose estimation.
# Measure your face width and adjust the value accordingly.
USER_FACE_WIDTH = 140  # [mm]

## Camera Parameters (not currently used in calculations)
# NOSE_TO_CAMERA_DISTANCE: The distance from the tip of the nose to the camera lens in millimeters.
# Intended for future use where accurate physical distance measurements may be necessary.
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]

## Configuration Parameters
# PRINT_DATA: Enable or disable the printing of data to the console for debugging.
PRINT_DATA = True

# DEFAULT_WEBCAM: Default camera source index. '0' usually refers to the built-in webcam.
DEFAULT_WEBCAM = 0

# SHOW_ALL_FEATURES: If True, display all facial landmarks on the video feed.
SHOW_ALL_FEATURES = True

# LOG_DATA: Enable or disable logging of data to a CSV file.
LOG_DATA = True

# LOG_ALL_FEATURES: If True, log all facial landmarks to the CSV file.
LOG_ALL_FEATURES = False

# ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
ENABLE_HEAD_POSE = True

## Logging Configuration
# LOG_FOLDER: Directory where log files will be stored.
LOG_FOLDER = "logs"

## Server Configuration
# SERVER_IP: IP address of the server for sending data via UDP (default is localhost).
SERVER_IP = "127.0.0.1"

# SERVER_PORT: Port number for the server to listen on.
SERVER_PORT = 7070

## Blink Detection Parameters
# SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.
SHOW_ON_SCREEN_DATA = True

# TOTAL_BLINKS: Counter for the total number of blinks detected.
TOTAL_BLINKS = 0

# EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
EYES_BLINK_FRAME_COUNTER = 0

# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51

# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 0  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True
LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070  # Set the server port

# eyes blinking variables
SHOW_BLINK_COUNT_ON_SCREEN = True  # Toggle to show the blink count on the video feed
TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
EYES_BLINK_FRAME_COUNTER = (
    0  # Counts the number of consecutive frames with a potential blink
)
BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
EYE_AR_CONSEC_FRAMES = (
    2  # Number of consecutive frames below the threshold to confirm a blink
)
# SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)


#If set to false it will wait for your command (hittig 'r') to start logging.
IS_RECORDING = False  # Controls whether data is being logged

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]  # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

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

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")


        # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]


     # Normalize the pitch angle
    pitch = normalize_pitch(pitch)

    return pitch, yaw, roll

def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch


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
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
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
    "Total Blink Count",
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
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping the frame for a mirror effect
        # I think we better not flip to correspond with real world... need to make sure later...
        #frame = cv.flip(frame, 1)
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
            # getting the head pose estimation 3d points
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

            # collect nose three dimension and two dimension points
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            # Get rotational matrix
            rotation_matrix, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            # Get the y rotation degree
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            # if angle cross the values then
            threshold_angle = 10
            # See where the user's head tilting
            if angle_y < -threshold_angle:
                face_looks = "Left"
            elif angle_y > threshold_angle:
                face_looks = "Right"
            elif angle_x < -threshold_angle:
                face_looks = "Down"
            elif angle_x > threshold_angle:
                face_looks = "Up"
            else:
                face_looks = "Forward"
            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            # Display the nose direction
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
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
                    angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = angle_buffer.get_average()

                    # Set initial angles on first successful estimation or recalibrate
                    if initial_pitch is None or (key == ord('c') and calibrated):
                        initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                        calibrated = True
                        if PRINT_DATA:
                            print("Head pose recalibrated.")

                    # Adjust angles based on initial calibration
                    if calibrated:
                        pitch -= initial_pitch
                        yaw -= initial_yaw
                        roll -= initial_roll
                    
                    
                    if PRINT_DATA:
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            # Logging data
            if LOG_DATA:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                log_entry = [
                    timestamp,
                    l_cx,
                    l_cy,
                    r_cx,
                    r_cy,
                    l_dx,
                    l_dy,
                    r_dx,
                    r_dy,
                    TOTAL_BLINKS,
                ]  # Include blink count in CSV
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
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)


        
        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Calibrate on 'c' key press
        if key == ord('c'):
            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            if PRINT_DATA:
                print("Head pose recalibrated.")
                
        # Inside the main loop, handle the 'r' key press
        if key == ord('r'):
            
            IS_RECORDING = not IS_RECORDING
            if IS_RECORDING:
                print("Recording started.")
            else:
                print("Recording paused.")


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
    if LOG_DATA and IS_RECORDING:
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
