import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

# Load model and class names
model = load_model("asl_model_reduced.h5", compile=False)
actions = np.load("classes_reduced.npy")
train_median = np.load("train_median_reduced.npy")
train_iqr = np.load("train_iqr_reduced.npy")

# Setup mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Important pose landmarks (same as training)
IMPORTANT_POSE_LANDMARKS = {
    0: "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
}

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


# Optimized keypoint extraction (170 features)
def extract_keypoints(results):
    keypoint_list = []

    # Pose features (only important landmarks)
    for idx in IMPORTANT_POSE_LANDMARKS.keys():
        if results.pose_landmarks:
            res = results.pose_landmarks.landmark[idx]
            keypoint_list.extend([res.x, res.y, res.z, res.visibility])
        else:
            keypoint_list.extend([0] * 4)

    # Left Hand (21 landmarks)
    if results.left_hand_landmarks:
        for res in results.left_hand_landmarks.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    # Right Hand (21 landmarks)
    if results.right_hand_landmarks:
        for res in results.right_hand_landmarks.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    return np.array(keypoint_list)


# Optimized drawing function
def draw_landmarks(image, results):
    # Draw only important pose landmarks
    if results.pose_landmarks:
        for idx in IMPORTANT_POSE_LANDMARKS.keys():
            landmark = results.pose_landmarks.landmark[idx]
            if landmark.visibility > 0.5:  # Only draw if visible
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    # Draw hands
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


# Initialize camera
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
last_prediction = ""
prediction_history = deque(maxlen=5)  # For smoothing display

with mp_holistic.Holistic(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    model_complexity=1,  # Balanced accuracy/performance
) as holistic:
    # Warm-up camera
    for _ in range(5):
        cap.read()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        fps_counter += 1

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
            sequence.append(np.zeros(170))  # Match reduced feature size

        # Only predict when we have a full sequence
        prediction_made = False
        if len(sequence) == SEQUENCE_LENGTH and hand_detected:
            # Robust scaling
            input_data = (np.array(sequence) - train_median) / (train_iqr + 1e-8)
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
                prediction_made = True

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

        # Update display prediction with smoothing
        if current_action:
            prediction_history.append(current_action)
            display_action = max(set(prediction_history), key=prediction_history.count)
        else:
            display_action = ""

        # Visualization
        if show_landmarks:
            draw_landmarks(image, results)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
            fps_text = f"FPS: {fps:.1f}"

        # Display info panel
        info_height = 180
        info_panel = np.zeros((info_height, image.shape[1], 3), dtype=np.uint8)

        # Hand detection status
        hand_status = "Hands: " + ("✅" if hand_detected else "❌")
        cv2.putText(
            info_panel,
            hand_status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if hand_detected else (0, 0, 255),
            2,
        )

        # Sequence status
        seq_status = f"Sequence: {len(sequence)}/{SEQUENCE_LENGTH}"
        cv2.putText(
            info_panel,
            seq_status,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        # FPS display
        cv2.putText(
            info_panel,
            fps_text,
            (image.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 0),
            2,
        )

        # Display prediction
        if display_action:
            cv2.putText(
                info_panel,
                f"Sign: {display_action}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (10, 200, 10),
                2,
            )

            # Confidence bar
            if prediction_made:
                bar_width = 300
                bar_height = 20
                fill_width = int(avg_confidence * bar_width)
                cv2.rectangle(
                    info_panel,
                    (20, 140),
                    (20 + bar_width, 140 + bar_height),
                    (50, 50, 50),
                    -1,
                )
                cv2.rectangle(
                    info_panel,
                    (20, 140),
                    (20 + fill_width, 140 + bar_height),
                    (10, 200, 10),
                    -1,
                )
                cv2.putText(
                    info_panel,
                    f"Confidence: {avg_confidence:.2f}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        # Combine with main image
        combined = np.vstack((image, info_panel))

        # Show image
        cv2.imshow("Sign Language Recognition", combined)

        # Key controls
        key = cv2.waitKey(1)
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
            prediction_history.clear()

cap.release()
cv2.destroyAllWindows()
