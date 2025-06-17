import cv2
import os
import numpy as np
import mediapipe as mp
import time
from tqdm import tqdm


DATA_PATH = os.path.join("MP_Data")
actions = np.array(["Hello", "Thanks", "Help", "Father", "Mother"])
no_sequences = 30
sequence_length = 30
COLLECTION_WINDOW = 2.0


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
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


print("Creating dataset directories...")
for action in tqdm(actions, desc="Actions"):
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as holistic:
    for action_idx, action in enumerate(actions):
        for sequence in range(no_sequences):
            start_time = time.time()
            while time.time() - start_time < COLLECTION_WINDOW:
                ret, frame = cap.read()
                if not ret:
                    continue

                countdown = int(COLLECTION_WINDOW - (time.time() - start_time) + 1)
                cv2.putText(
                    frame,
                    f"Starting '{action}' sequence {sequence + 1}/{no_sequences} in {countdown}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Get ready to perform: {action}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                cv2.putText(
                    image,
                    f"Collecting: {action}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Sequence: {sequence + 1}/{no_sequences}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"Frame: {frame_num + 1}/{sequence_length}",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                progress_width = int(frame_num * (image.shape[1] / sequence_length))
                cv2.rectangle(
                    image,
                    (0, image.shape[0] - 30),
                    (progress_width, image.shape[0]),
                    (0, 255, 0),
                    -1,
                )

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                )
                np.save(npy_path, keypoints)

                cv2.imshow("Data Collection", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
print("Data collection completed successfully!")
