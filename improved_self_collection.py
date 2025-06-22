import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy import interpolate
import shutil

# ===== Configuration =====
VIDEO_PATH = "videos"  # Still needed if you plan to process existing videos too, but not for webcam capture
ANNOTATION_FILE = (
    "WLASL_v0.3.json"  # Not directly used for webcam capture, but kept for context
)
DATA_PATH = "MP_Data"

target_actions = [
    "drink",
    "book",
    "computer",
    "go",
    "before",
    "who",
    "walk",
    "what",
    "thin",
    "year",
    "yes",
    "all",
    "black",
    "cool",
    "finish",
    "hot",
    "like",
    "many",
    "mother",
    "now",
    "orange",
    "table",
    "thanksgiving",
    "woman",
    "bed",
    "blue",
    "bowling",
    "can",
    "dog",
    "family",
    "fish",
    "graduate",
    "hat",
    "help",
    "no",
    "clothes",
    "pizza",
    "play",
    "school",
    "shirt",
    "study",
    "tall",
    "white",
    "wrong",
    "accident",
    "apple",
    "bird",
    "change",
    "color",
    "corn",
]
no_sequences = 30  # Total desired sequences per action (webcam + augmented)
no_sequences_webcam = 15  # Number of sequences to capture from webcam
sequence_length = 30

# ===== MediaPipe Setup =====
mp_holistic = mp.solutions.holistic

# Important pose landmarks (reduced from 33 to 11 key points)
# These include shoulders, elbows, wrists, hips, and the nose for orientation
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


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


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


def temporal_augmentation(sequence):
    """Enhanced time warping with multiple interpolation methods"""
    # Randomly select warping parameters
    warp_type = np.random.choice(["speed", "reverse", "jitter"])

    if warp_type == "speed":
        # Speed variation
        factor = np.random.uniform(0.5, 1.5)
        new_length = int(len(sequence) * factor)

        # Handle edge cases
        if new_length < 5 or new_length > 100:  # Ensure reasonable lengths
            return sequence

        # Select interpolation method
        methods = ["linear", "nearest", "slinear"]
        if len(sequence) > 4:  # Cubic requires at least 4 points
            methods.append("cubic")
        kind = np.random.choice(methods)

        # Time interpolation
        x_old = np.linspace(0, 1, len(sequence))
        x_new = np.linspace(0, 1, new_length)

        augmented = []
        for dim in range(sequence.shape[1]):
            f = interpolate.interp1d(
                x_old,
                sequence[:, dim],
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            augmented.append(f(x_new))
        sequence = np.array(augmented).T

    elif warp_type == "reverse":
        # Reverse sequence
        sequence = sequence[::-1]

    elif warp_type == "jitter":
        # Frame jittering
        if len(sequence) > 0:  # Ensure sequence is not empty
            jitter_amount = np.random.randint(1, 3)
            # Add random frames from sequence to create jitter
            indices = np.arange(len(sequence))
            np.random.shuffle(indices)
            jittered_indices = np.sort(
                indices[: min(jitter_amount, len(sequence))]
            )  # Pick few frames to duplicate

            jittered_sequence = []
            current_idx = 0
            for idx in range(len(sequence)):
                jittered_sequence.append(sequence[idx])
                if idx in jittered_indices:
                    jittered_sequence.append(sequence[idx])  # Duplicate frame
            sequence = np.array(jittered_sequence)

        # Original simple jitter (can be combined or replaced)
        # jitter_amount = np.random.randint(1, 3)
        # sequence = np.concatenate(
        #     [sequence, sequence[-1:].repeat(jitter_amount, axis=0)]
        # )

    return sequence


def spatial_augmentation(frame):
    """Enhanced spatial transformations with coherent transformations"""
    # Global transformations
    if np.random.rand() < 0.7:  # 70% chance
        # Uniform scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        frame = frame * scale_factor

    if np.random.rand() < 0.7:  # 70% chance
        # Translation
        # Ensure translation is applied to each coordinate for each point
        translation = np.random.uniform(-0.1, 0.1, size=(3,))
        num_coords = frame.shape[0] // 3  # Assuming (x,y,z) for each point
        for i in range(num_coords):
            frame[i * 3] += translation[0]  # x-coordinates
            frame[i * 3 + 1] += translation[1]  # y-coordinates
            frame[i * 3 + 2] += translation[2]  # z-coordinates

    if np.random.rand() < 0.5:  # 50% chance
        # Rotation (2D plane)
        angle = np.random.uniform(-15, 15)
        rad = np.deg2rad(angle)
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

        # Center normalization
        # Calculate center based on actual detected points, ignore (0,0,0) if no detection
        valid_x = frame[0::3][frame[0::3] != 0]
        valid_y = frame[1::3][frame[1::3] != 0]

        center_x = np.mean(valid_x) if np.any(valid_x) else 0.5
        center_y = np.mean(valid_y) if np.any(valid_y) else 0.5

        # Apply rotation to x,y coordinates
        # Iterate over keypoints, assuming each keypoint has (x,y,z) or (x,y,z,visibility)
        # Our extract_keypoints gives (x,y,z,visibility) for pose, and (x,y,z) for hands
        # The frame will be a flattened array.
        # Pose landmarks are 4 values, hand landmarks are 3 values.
        # Re-evaluate the frame structure here. Currently, extract_keypoints flattens to a single list.

        # Let's assume for spatial augmentation, we're working with the flattened keypoint_list structure:
        # Pose (11 * 4) + Left Hand (21 * 3) + Right Hand (21 * 3) = 44 + 63 + 63 = 170 elements

        # Pose part (indices 0 to 43, in groups of 4: x,y,z,v)
        for i in range(0, 11 * 4, 4):  # For each pose landmark
            x_idx, y_idx = i, i + 1
            if x_idx + 1 < len(frame):  # Ensure indices are within bounds
                x = frame[x_idx] - center_x
                y = frame[y_idx] - center_y
                rotated = rot_matrix @ np.array([x, y])
                frame[x_idx] = rotated[0] + center_x
                frame[y_idx] = rotated[1] + center_y

        # Left Hand part (indices 44 to 44 + 21*3 - 1, in groups of 3: x,y,z)
        for i in range(44, 44 + 21 * 3, 3):  # For each left hand landmark
            x_idx, y_idx = i, i + 1
            if x_idx + 1 < len(frame):
                x = frame[x_idx] - center_x
                y = frame[y_idx] - center_y
                rotated = rot_matrix @ np.array([x, y])
                frame[x_idx] = rotated[0] + center_x
                frame[y_idx] = rotated[1] + center_y

        # Right Hand part (indices 44 + 21*3 to end, in groups of 3: x,y,z)
        for i in range(44 + 21 * 3, len(frame), 3):  # For each right hand landmark
            x_idx, y_idx = i, i + 1
            if x_idx + 1 < len(frame):
                x = frame[x_idx] - center_x
                y = frame[y_idx] - center_y
                rotated = rot_matrix @ np.array([x, y])
                frame[x_idx] = rotated[0] + center_x
                frame[y_idx] = rotated[1] + center_y

    # Per-joint transformations
    if np.random.rand() < 0.6:  # 60% chance
        # Random noise
        noise = np.random.normal(0, 0.03, size=frame.shape)
        frame = frame + noise

    if np.random.rand() < 0.3:  # 30% chance
        # Axis dropout
        # Create a mask to zero out some x,y,z coordinates
        # For a coordinate, it's either all kept or all dropped.
        dropout_mask_per_coord = np.random.choice(
            [0, 1], size=(frame.shape[0]), p=[0.1, 0.9]
        )
        frame = frame * dropout_mask_per_coord

    return frame


def augment_sequence(original_path, target_dir):
    original_seq = [
        np.load(os.path.join(original_path, f"{i}.npy")) for i in range(sequence_length)
    ]
    original_seq = np.array(original_seq)

    # Apply temporal augmentation
    augmented_seq = temporal_augmentation(original_seq)

    # Pad/truncate to fixed length
    if len(augmented_seq) > sequence_length:
        augmented_seq = augmented_seq[:sequence_length]
    elif len(augmented_seq) < sequence_length:
        pad_length = sequence_length - len(augmented_seq)
        padding = np.zeros((pad_length, augmented_seq.shape[1]))
        augmented_seq = np.vstack((augmented_seq, padding))

    # Apply spatial augmentations
    for i in range(len(augmented_seq)):
        augmented_seq[i] = spatial_augmentation(augmented_seq[i])

    # Save augmented sequence
    os.makedirs(target_dir, exist_ok=True)
    for i, frame in enumerate(augmented_seq):
        np.save(os.path.join(target_dir, f"{i}.npy"), frame)


# ===== Create Dataset Directories =====
print("🔧 Creating dataset folders...")
for action in tqdm(target_actions):
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# ===== Start Processing =====
with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2,  # Higher accuracy
) as holistic:
    for action in target_actions:
        print(f"\n🚀 Processing action: {action}")
        sequence_count = 0

        # --- Webcam Capture ---
        print(
            f"Starting webcam capture for action: {action}. Capture {no_sequences_webcam} sequences."
        )

        cap = cv2.VideoCapture(0)  # Open default webcam
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Exiting.")
            break  # Exit the function if webcam not available

        for seq_num in range(no_sequences_webcam):
            if sequence_count >= no_sequences_webcam:
                break

            keypoints_buffer = []
            recording = False
            frame_counter = 0

            print(
                f"\n--- Sequence {seq_num + 1}/{no_sequences_webcam} for '{action}' ---"
            )
            print("Press 'S' to START recording.")
            print("Press 'S' again to STOP recording this sequence and save.")
            print(
                "Press 'Q' to QUIT collection for this action (and proceed to augmentation if needed)."
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, trying again...")
                    continue

                # Flip frame horizontally for mirror effect (common for webcam)
                frame = cv2.flip(frame, 1)

                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks for visualization
                # For pose: mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # For hands: mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # You might need to import drawing_utils if you want to visualize:
                # import mediapipe.python.solutions.drawing_utils as mp_drawing
                # Example for drawing:
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                #                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                #                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                #                            )
                # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                #                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                #                            )
                # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                #                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                #                            )

                display_text = "PRESS 'S' TO START RECORDING"
                if recording:
                    display_text = f"RECORDING... ({len(keypoints_buffer)}/{sequence_length} frames)"
                elif len(keypoints_buffer) > 0 and not recording:
                    display_text = (
                        "RECORDING PAUSED. Press 'S' to continue or 'Q' to quit."
                    )

                cv2.putText(
                    image,
                    display_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("OpenCV Feed - Data Collection", image)

                key = cv2.waitKey(10) & 0xFF

                if key == ord("s"):  # Start/Stop recording
                    recording = not recording
                    if recording:
                        print(
                            f"Starting recording sequence {seq_num + 1} for '{action}'..."
                        )
                    else:
                        print(
                            f"Pausing recording sequence {seq_num + 1} for '{action}'. Current frames: {len(keypoints_buffer)}"
                        )

                if recording:
                    keypoints = extract_keypoints(results)
                    # Only add frames with *some* detection (pose or hands)
                    if np.any(keypoints != 0):  # Check if any keypoint is non-zero
                        keypoints_buffer.append(keypoints)

                    if len(keypoints_buffer) >= sequence_length:
                        print(
                            f"Sequence {seq_num + 1} for '{action}' captured {len(keypoints_buffer)} frames."
                        )
                        recording = False  # Automatically stop recording after sequence_length frames
                        break  # Exit inner while loop to save sequence

                if key == ord("q"):  # Quit for current action
                    print("Quitting webcam collection for this action.")
                    recording = False
                    break  # Exit inner while loop and proceed to augmentation

            if len(keypoints_buffer) < sequence_length:
                print(
                    f"Warning: Sequence {seq_num + 1} for '{action}' has only {len(keypoints_buffer)} frames. Padding with zeros."
                )
                pad_length = sequence_length - len(keypoints_buffer)
                if len(keypoints_buffer) > 0:
                    padding = [np.zeros_like(keypoints_buffer[0])] * pad_length
                else:
                    # If buffer is empty, create a zero array of the expected keypoint size
                    padding = [
                        np.zeros(170)
                    ] * pad_length  # 170 is the expected size from extract_keypoints
                keypoints_buffer.extend(padding)
            else:
                keypoints_buffer = keypoints_buffer[
                    :sequence_length
                ]  # Truncate if too long

            if len(keypoints_buffer) == sequence_length:
                seq_dir = os.path.join(DATA_PATH, action, str(sequence_count))
                os.makedirs(seq_dir, exist_ok=True)
                for frame_num, keypoints in enumerate(keypoints_buffer):
                    np.save(os.path.join(seq_dir, f"{frame_num}.npy"), keypoints)
                sequence_count += 1
                print(
                    f"✅ Sequence {sequence_count}/{no_sequences_webcam} collected for '{action}'."
                )
            else:
                print(
                    f"Skipping sequence {seq_num + 1} for '{action}' due to insufficient frames after padding."
                )

            if key == ord("q"):  # If user pressed 'q', break outer loop as well
                break

        cap.release()
        cv2.destroyAllWindows()  # Close the webcam window

        # === Augmentation Phase ===
        if sequence_count == 0:
            # Remove empty action directory if no real sequences were collected
            action_dir = os.path.join(DATA_PATH, action)
            if os.path.exists(action_dir):
                shutil.rmtree(action_dir)
            print(f"⚠️ No sequences collected for '{action}'. Directory removed.")
            continue

        if sequence_count < no_sequences:
            num_augmentations = no_sequences - sequence_count
            print(
                f"⚠️ Not enough real sequences for '{action}' ({sequence_count}), generating {num_augmentations} augmentations..."
            )
            existing_dirs = [
                os.path.join(DATA_PATH, action, str(i)) for i in range(sequence_count)
            ]

            for aug_idx in range(num_augmentations):
                # Cycle through the collected sequences for augmentation
                base_dir = existing_dirs[aug_idx % len(existing_dirs)]
                target_dir = os.path.join(
                    DATA_PATH, action, str(sequence_count + aug_idx)
                )
                augment_sequence(base_dir, target_dir)
                print(
                    f"✅ Augmented sequence {sequence_count + aug_idx + 1}/{no_sequences} created"
                )

print("\n✅ All done! Dataset created under `MP_Data/`")
total_sequences_collected = 0
for act in target_actions:
    action_path = os.path.join(DATA_PATH, act)
    if os.path.exists(action_path):
        total_sequences_collected += len(os.listdir(action_path))
print(f"Total sequences collected: {total_sequences_collected}")
