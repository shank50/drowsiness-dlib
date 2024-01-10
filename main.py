import cv2
import numpy as np
import dlib
from imutils import face_utils

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for status tracking
STATUS_AWAKE = "Awake"
STATUS_DROWSY = "Drowsy"
STATUS_VERY_DROWSY = "Very Drowsy"
COLOR_AWAKE = (0, 255, 0)
COLOR_DROWSY = (0, 0, 255)
COLOR_VERY_DROWSY = (255, 0, 0)

# State counters
sleep_counter = 0
drowsy_counter = 0
active_counter = 0
current_status = ""
status_color = (0, 0, 0)

def calculate_distance(point_a, point_b):
    #Calculate the Euclidean distance between two points
    return np.linalg.norm(point_a - point_b)

def detect_blink(eye_points):
    #Detect if an eye is blinked based on landmark points
    upper_lid_distance = calculate_distance(eye_points[1], eye_points[3]) + calculate_distance(eye_points[2], eye_points[4])
    lower_lid_distance = calculate_distance(eye_points[0], eye_points[5])
    ratio = upper_lid_distance / (2.0 * lower_lid_distance)

    if ratio > 0.20:
        return 2  # Fully open
    elif 0.15 < ratio <= 0.20:
        return 1  # Partially closed
    else:
        return 0  # Fully closed

def update_status(left_eye_state, right_eye_state):
    #Update the drowsiness status based on eye states
    global sleep_counter, drowsy_counter, active_counter, current_status, status_color

    if left_eye_state == 0 or right_eye_state == 0:
        sleep_counter += 1
        drowsy_counter = 0
        active_counter = 0
        if sleep_counter > 6:
            current_status = STATUS_VERY_DROWSY
            status_color = COLOR_VERY_DROWSY

    elif left_eye_state == 1 or right_eye_state == 1:
        sleep_counter = 0
        active_counter = 0
        drowsy_counter += 1
        if drowsy_counter > 6:
            current_status = STATUS_DROWSY
            status_color = COLOR_DROWSY

    else:
        drowsy_counter = 0
        sleep_counter = 0
        active_counter += 1
        if active_counter > 6:
            current_status = STATUS_AWAKE
            status_color = COLOR_AWAKE

def draw_landmarks(frame, landmarks):
    #Draw facial landmarks on the frame
    for point in landmarks[36:48]:  
        cv2.circle(frame, tuple(point), 1, (255, 255, 255), -1)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray_frame)

    for face in detected_faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = landmark_predictor(gray_frame, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_state = detect_blink(landmarks[36:42])
        right_eye_state = detect_blink(landmarks[42:48])

        update_status(left_eye_state, right_eye_state)

        cv2.putText(frame, current_status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        draw_landmarks(frame, landmarks)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()