
# Python-Gaze-Face-Tracker

### Advanced Real-Time Eye and Facial Landmark Tracking System

---

## Description
**Python-Gaze-Face-Tracker**  is a Python-based application designed for advanced real-time eye tracking, facial landmark detection, and head tracking, utilizing OpenCV and MediaPipe technology. Specializing in uncalibrated gaze tracking, this tool is an easy to use Python eye and facial landmark tracker. It excels in visualizing iris positions and offers robust logging capabilities for both eye and facial landmark data. Equipped with the ability to transmit this iris and gaze information over UDP sockets, Python-Gaze-Face-Tracker stands out for various applications, including aviation, human-computer interaction (HCI), and augmented reality (AR). Its unmatched efficiency in iris tracking and face tracking positions it as a go-to solution for researchers and developers seeking a comprehensive Python facial landmark and gaze tracking system. This tool not only facilitates detailed eye movement analysis but also supports head tracking, making it a comprehensive package for advanced gaze tracking and facial feature analysis in any interactive technology application.

---

## Features
- **Real-Time Eye Tracking**: Tracks and visualizes iris and eye corner positions in real-time using webcam input.
- **Facial Landmark Detection**: Detects and displays up to 468 facial landmarks.
- **Data Logging**: Records tracking data to CSV files, including timestamps, eye positions, and optional facial landmark data. *Note: Enabling logging of all 468 facial landmarks can result in large log files.*
- **Socket Communication**: Supports transmitting only iris tracking data via UDP sockets for integration with other systems or applications.

---

## Requirements
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Python standard libraries: `math`, `socket`, `argparse`, `time`, `csv`, `datetime`, `os`

---

## Installation & Usage

1. **Clone the Repository:**
   ```
   git clone https://github.com/alireza787b/Python-Gaze-Face-Tracker.git
   ```

2. **Navigate to the Repository Directory:**
   ```
   cd Python-Gaze-Face-Tracker
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```
   python eye_face_tracker.py
   ```

   Optionally, specify the camera source:
   ```
   python eye_face_tracker.py -c <camera_source_number>
   ```

---

## Parameters
- **SERVER_IP & SERVER_PORT**: Defines the IP address and port for UDP socket communication.
- **DEFAULT_WEBCAM**: Default webcam source number. Can be overridden by command-line argument `-c`.
- **PRINT_DATA**: Toggles printing of eye tracking data in the console.
- **SHOW_ALL_FEATURES**: Controls whether all facial landmarks are displayed.
- **LOG_DATA**: Enables logging of tracking data to a CSV file.
- **LOG_ALL_FEATURES**: When enabled, logs all detected facial landmarks to the CSV file.

---

## Data Logging & Telemetry
- **CSV Logging**: The application generates CSV files with tracking data including timestamps, eye positions, and optional facial landmarks. These files are stored in the `logs` folder.
- **UDP Telemetry**: The application sends only iris position data through UDP sockets as defined by `SERVER_IP` and `SERVER_PORT`. The data is sent in the following order: [Left Eye Center X, Left Eye Center Y, Left Iris Relative Pos Dx, Left Iris Relative Pos Dy].

---

## Acknowledgements
This project was initially inspired by [Asadullah Dal's iris segmentation project](https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python).

---

## Note
The **Python-Gaze-Face-Tracker** is intended for educational and research purposes and is particularly suited for applications in aviation, HCI, AR, and similar fields.

---
