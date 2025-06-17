import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

# Load model and class names
model = load_model("asl_model.h5")
actions = np.load("classes.npy")

# Setup mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.85  # Increased threshold for more confidence
CONFIDENCE_WINDOW = 10  # Frames to average predictions over
COOLDOWN_FRAMES = 15  # Minimum frames before changing prediction
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Prediction smoothing buffers
prediction_queue = deque(maxlen=CONFIDENCE_WINDOW)
last_prediction_time = 0
current_action = ""
cooldown_counter = 0


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # Optimized drawing using styles
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sequence = deque(maxlen=SEQUENCE_LENGTH)
fps_counter = 0
start_time = time.time()
show_landmarks = True  # Toggle landmark visibility

with mp_holistic.Holistic(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    model_complexity=1,  # Balanced performance
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)

        # Process frame
        image, results = mediapipe_detection(frame, holistic)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Only predict when we have a full sequence
        if len(sequence) == SEQUENCE_LENGTH:
            # Convert to numpy array and add batch dimension
            input_data = np.expand_dims(np.array(sequence), axis=0)

            # Get prediction
            res = model.predict(input_data, verbose=0)[0]
            prediction_idx = np.argmax(res)
            confidence = res[prediction_idx]

            # Store prediction for smoothing
            prediction_queue.append(prediction_idx)

            # Get most frequent prediction in window
            if prediction_queue:
                most_common = max(set(prediction_queue), key=prediction_queue.count)
                confidence = res[most_common]

                # Apply threshold and cooldown logic
                if confidence > PREDICTION_THRESHOLD:
                    predicted_action = actions[most_common]

                    # Cooldown logic prevents rapid prediction changes
                    if predicted_action != current_action:
                        cooldown_counter += 1
                        if cooldown_counter >= COOLDOWN_FRAMES:
                            current_action = predicted_action
                            cooldown_counter = 0
                    else:
                        cooldown_counter = 0
                else:
                    current_action = ""

        # Calculate FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()

        # Visualization
        if show_landmarks:
            draw_landmarks(image, results)

        # Display prediction
        if current_action:
            cv2.putText(
                image,
                f"Sign: {current_action}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (10, 200, 10),
                3,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_width = 300
            bar_height = 30
            fill_width = int(confidence * bar_width)
            cv2.rectangle(
                image, (20, 80), (20 + bar_width, 80 + bar_height), (50, 50, 50), -1
            )
            cv2.rectangle(
                image, (20, 80), (20 + fill_width, 80 + bar_height), (10, 200, 10), -1
            )
            cv2.putText(
                image,
                f"Confidence: {confidence:.2f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Display FPS

        # Display sequence status
        seq_status = f"Sequence: {len(sequence)}/{SEQUENCE_LENGTH}"
        cv2.putText(
            image,
            seq_status,
            (image.shape[1] - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        # Progress bar for sequence collection
        progress_width = int(len(sequence) * (image.shape[1] / SEQUENCE_LENGTH))
        cv2.rectangle(
            image,
            (0, image.shape[0] - 10),
            (progress_width, image.shape[0]),
            (0, 255, 0),
            -1,
        )

        # Show image
        cv2.imshow("Sign Language Recognition", image)

        # Key controls
        key = cv2.waitKey(10)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("l"):  # Toggle landmarks
            show_landmarks = not show_landmarks
        elif key & 0xFF == ord("r"):  # Reset sequence
            sequence.clear()
            prediction_queue.clear()
            current_action = ""
            cooldown_counter = 0

cap.release()
cv2.destroyAllWindows()
