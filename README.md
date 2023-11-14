
# Python-Gaze-Face-Tracker

### Advanced Real-Time Eye and Facial Landmark Tracking System

---
<img src="https://github.com/alireza787b/Python-Gaze-Face-Tracker/assets/30341941/0e4b8068-9d80-4573-b5e7-2a2a6061c594" style="text-align:center">

![image](https://github.com/alireza787b/Python-Gaze-Face-Tracker/assets/30341941/ce20ac3a-6785-448e-85df-4d2dd5f22040)

## Description
**Python-Gaze-Face-Tracker**  is a Python-based application designed for advanced real-time eye tracking, facial landmark detection, and head tracking, utilizing OpenCV and MediaPipe technology. Specializing in uncalibrated gaze tracking, this tool is an easy to use Python eye and facial landmark tracker. It excels in visualizing iris positions and offers robust logging capabilities for both eye and facial landmark data. Equipped with the ability to transmit this iris and gaze information over UDP sockets, Python-Gaze-Face-Tracker stands out for various applications, including aviation, human-computer interaction (HCI), and augmented reality (AR). The tool also includes a blink detection feature, contributing to detailed eye movement analysis and supporting head tracking. This makes it a comprehensive package for advanced gaze tracking and facial feature analysis in interactive technology applications.





---

## Features
- **Real-Time Eye Tracking**: Tracks and visualizes iris and eye corner positions in real-time using webcam input.
- **Facial Landmark Detection**: Detects and displays up to 468 facial landmarks.
- **Data Logging**: Records tracking data to CSV files, including timestamps, eye positions, and optional facial landmark data. *Note: Enabling logging of all 468 facial landmarks can result in large log files.*
- **Socket Communication**: Supports transmitting only iris tracking data via UDP sockets for integration with other systems or applications.
- **Blink Detection**: Monitors and records blink frequency, enhancing eye movement analysis.
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
   python main.py
   ```

   Optionally, specify the camera source:
   ```
   python main.py -c <camera_source_number>
   ```

---

## Parameters
- **SERVER_IP & SERVER_PORT**: Configurable IP address and port for UDP socket communication. These parameters define where the eye-tracking data is sent via the network.
- **DEFAULT_WEBCAM**: Specifies the default webcam source number. If a webcam number is provided as a command-line argument, that number is used; otherwise, the default value here is used.
- **PRINT_DATA**: When set to True, the program prints eye-tracking data to the console. Set to False to disable printing.
- **SHOW_ALL_FEATURES**: If True, the program shows all facial landmarks. Set to False to display only the eye positions.
- **LOG_DATA**: Enables logging of eye tracking data to a CSV file when set to True.
- **LOG_ALL_FEATURES**: When True, all 468 facial landmarks are logged in the CSV file.
- **BLINK_THRESHOLD**: Threshold for the eye aspect ratio to trigger a blink.
- **EYE_AR_CONSEC_FRAMES**: Number of consecutive frames below the threshold to confirm a blink.

---

## Data Logging & Telemetry
- **CSV Logging**: The application generates CSV files with tracking data including timestamps, eye positions, and optional facial landmarks. These files are stored in the `logs` folder.
- **UDP Telemetry**: The application sends only iris position data through UDP sockets as defined by `SERVER_IP` and `SERVER_PORT`. The data is sent in the following order: [Left Eye Center X, Left Eye Center Y, Left Iris Relative Pos Dx, Left Iris Relative Pos Dy].

---

## Acknowledgements
This project was initially inspired by [Asadullah Dal's iris segmentation project](https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python).
The blink detection feature is also contributed by Asadullah Dal.

---

## Note
The **Python-Gaze-Face-Tracker** is intended for educational and research purposes and is particularly suited for applications in aviation, HCI, AR, and similar fields.

---
