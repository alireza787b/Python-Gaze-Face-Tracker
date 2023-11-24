import cv2
import mediapipe as mp
import threading
import queue

# Parameters (easy to change)
WEBCAM_NUMBER = 0  # Change this to use a different webcam
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_LANDMARKS = 467  # Max landmark number in MediaPipe (0-467)

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# Function to mark landmarks on the image.
def mark_landmarks(image, landmarks, landmark_ids):
    img_height, img_width, _ = image.shape
    for landmark_id in landmark_ids:
        if 0 <= landmark_id <= MAX_LANDMARKS:
            landmark = landmarks.landmark[landmark_id]
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Increased dot size
            cv2.putText(image, str(landmark_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # Larger text
    return image

def validate_input(user_input):
    try:
        landmark_ids = [int(id.strip()) for id in user_input.split(',') if id.strip().isdigit()]
        if all(0 <= id <= MAX_LANDMARKS for id in landmark_ids):
            return landmark_ids
        else:
            raise ValueError
    except ValueError:
        print(f"Invalid input. Please enter numbers between 0 and {MAX_LANDMARKS}, comma-separated.")
        return None

# Function to handle user input in a separate thread.
def input_thread(input_queue):
    while True:
        user_input = input()
        input_queue.put(user_input)

def main():
    print("MediaPipe Landmark Visualizer")
    print("Instructions:")
    print("1. Enter landmark IDs in the console (comma-separated, e.g., 1,5,30,150).")
    print("2. Press 'q' to quit the application.")
    print("3. You can enter new landmark IDs anytime to update the visualization.")

    # Open webcam.
    cap = cv2.VideoCapture(WEBCAM_NUMBER)
    if not cap.isOpened():
        print(f"Could not open webcam #{WEBCAM_NUMBER}.")
        return

    landmark_ids = []
    input_queue = queue.Queue()

    # Start the thread for handling user input.
    threading.Thread(target=input_thread, args=(input_queue,), daemon=True).start()

    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Convert back to BGR for OpenCV rendering.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image = mark_landmarks(image, face_landmarks, landmark_ids)

            cv2.imshow('MediaPipe Landmarks', image)

            # Check for 'q' key to quit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Check for input from the input thread
            try:
                user_input = input_queue.get_nowait()
                validated_ids = validate_input(user_input)
                if validated_ids is not None:
                    landmark_ids = validated_ids
                    print("Selected Landmarks: ", ", ".join(map(str, landmark_ids)))
                    print("To see new landmarks, type their IDs again (comma-separated) and press enter.")
            except queue.Empty:
                pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
