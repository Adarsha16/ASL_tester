import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

# Load model and class names
model = load_model("asl_model.h5", compile=False)
actions = np.load("classes.npy")
training_mean = np.load("training_mean.npy")
training_std = np.load("training_std.npy")

# Setup mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.7  # High confidence threshold
CONFIDENCE_DELTA = 0.15  # Min difference between top predictions
CONFIDENCE_WINDOW = 15  # Frames to average predictions over
COOLDOWN_FRAMES = 20  # Minimum frames before changing prediction
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Prediction smoothing buffers
prediction_queue = deque(maxlen=CONFIDENCE_WINDOW)
confidence_history = deque(maxlen=CONFIDENCE_WINDOW)
last_prediction_time = 0
current_action = ""
cooldown_counter = 0


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def draw_landmarks(image, results):
    # Optimized drawing

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
    keypoint_list = []

    # Pose
    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            keypoint_list.extend([res.x, res.y, res.z, res.visibility])
    else:
        keypoint_list.extend([0] * 33 * 4)

    # Left Hand
    if results.left_hand_landmarks:
        for res in results.left_hand_landmarks.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    # Right Hand
    if results.right_hand_landmarks:
        for res in results.right_hand_landmarks.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    return np.array(keypoint_list)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sequence = deque(maxlen=SEQUENCE_LENGTH)
fps_counter = 0
start_time = time.time()
show_landmarks = True
hand_detected = False
hand_detection_time = 0

with mp_holistic.Holistic(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    model_complexity=2,  # Higher accuracy
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Process frame
        image, results = mediapipe_detection(frame, holistic)

        # Check hand presence
        current_time = time.time()
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hand_detected = True
            hand_detection_time = current_time
        elif current_time - hand_detection_time > 2.0:  # 2 seconds cooldown
            hand_detected = False

        # Extract keypoints only if hands detected
        if hand_detected:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
        else:
            sequence.append(np.zeros_like(extract_keypoints(results)))

        # Only predict when we have a full sequence
        if len(sequence) == SEQUENCE_LENGTH and hand_detected:
            # Normalize
            input_data = (np.array(sequence) - training_mean) / (training_std + 1e-8)
            input_data = np.expand_dims(input_data, axis=0)

            # Get prediction
            res = model.predict(input_data, verbose=0)[0]
            top_idx = np.argsort(res)[-2:]  # Get top 2 predictions
            top_actions = actions[top_idx]
            top_confidences = res[top_idx]

            # Confidence difference check
            confidence_delta = top_confidences[-1] - top_confidences[-2]

            if (
                top_confidences[-1] > PREDICTION_THRESHOLD
                and confidence_delta > CONFIDENCE_DELTA
            ):
                prediction_queue.append(top_idx[-1])
                confidence_history.append(top_confidences[-1])

                # Get most frequent prediction in window
                if prediction_queue:
                    most_common = max(set(prediction_queue), key=prediction_queue.count)
                    avg_confidence = np.mean(
                        [
                            c
                            for i, c in zip(prediction_queue, confidence_history)
                            if i == most_common
                        ]
                    )

                    if avg_confidence > PREDICTION_THRESHOLD:
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
                # Reset if confidence is low
                current_action = ""
                cooldown_counter = 0
        else:
            # Reset if hands not detected
            current_action = ""
            cooldown_counter = 0

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

        # Display hand detection status
        hand_status = "Hands: " + ("✅" if hand_detected else "❌")
        cv2.putText(
            image,
            hand_status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255) if hand_detected else (50, 50, 200),
            2,
        )

        # Display prediction
        if current_action:
            cv2.putText(
                image,
                f"Sign: {current_action}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (10, 200, 10),
                3,
                cv2.LINE_AA,
            )

            # Confidence bar
            bar_width = 300
            bar_height = 30
            fill_width = int(avg_confidence * bar_width)
            cv2.rectangle(
                image, (20, 100), (20 + bar_width, 100 + bar_height), (50, 50, 50), -1
            )
            cv2.rectangle(
                image, (20, 100), (20 + fill_width, 100 + bar_height), (10, 200, 10), -1
            )
            cv2.putText(
                image,
                f"Confidence: {avg_confidence:.2f}",
                (20, 150),
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
            (image.shape[1] - 250, 70),
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
            confidence_history.clear()
            current_action = ""
            cooldown_counter = 0

cap.release()
cv2.destroyAllWindows()
