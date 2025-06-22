import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy import interpolate
import shutil
import json

# ===== Configuration =====
VIDEO_PATH = "videos"
ANNOTATION_FILE = "WLASL_v0.3.json"
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
no_sequences = 30
no_sequences_webcam = 15
sequence_length = 30

# ===== MediaPipe Setup =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Important pose landmarks including facial features
IMPORTANT_POSE_LANDMARKS = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
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

# Ask for handedness once at start
is_left_handed = input("Are you left-handed? (y/n): ").lower().strip() == "y"


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

    # Hand extraction with handedness swapping
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    # Swap hands if user is left-handed
    if is_left_handed:
        left_hand, right_hand = right_hand, left_hand

    # Left Hand (21 landmarks)
    if left_hand:
        for res in left_hand.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    # Right Hand (21 landmarks)
    if right_hand:
        for res in right_hand.landmark:
            keypoint_list.extend([res.x, res.y, res.z])
    else:
        keypoint_list.extend([0] * 21 * 3)

    return np.array(keypoint_list)


def draw_landmarks(image, results):
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return image


def temporal_augmentation(sequence):
    """Enhanced time warping with safer interpolation"""
    if len(sequence) < 2:  # Need at least 2 frames to interpolate
        return sequence

    # Randomly select warping parameters
    warp_type = np.random.choice(["speed", "reverse", "jitter"])

    if warp_type == "speed":
        # Safer speed variation
        factor = np.random.uniform(0.8, 1.2)  # Reduced range for safety
        new_length = max(5, min(100, int(len(sequence) * factor)))  # Constrained length

        # Select interpolation method based on available points
        methods = ["linear", "nearest", "slinear"]
        if len(sequence) > 4:
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
        # Frame jittering with bounds check
        if len(sequence) > 1:
            jitter_amount = np.random.randint(1, min(3, len(sequence) // 2))
            jittered_indices = np.random.choice(
                len(sequence), jitter_amount, replace=False
            )
            sequence = np.insert(
                sequence, jittered_indices, sequence[jittered_indices], axis=0
            )

    return sequence


def spatial_augmentation(frame):
    """Enhanced spatial transformations with bounds checking"""
    # Create copy to avoid modifying original
    frame = frame.copy()
    num_coords = len(frame)

    # Global transformations
    if np.random.rand() < 0.7:
        scale_factor = np.random.uniform(0.8, 1.2)
        frame = frame * scale_factor

    if np.random.rand() < 0.7:
        translation = np.random.uniform(-0.1, 0.1, size=3)
        for i in range(0, num_coords, 3):
            if i < num_coords:
                frame[i] += translation[0]
            if i + 1 < num_coords:
                frame[i + 1] += translation[1]
            if i + 2 < num_coords:
                frame[i + 2] += translation[2]

    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        rad = np.deg2rad(angle)
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

        # Calculate center from non-zero points
        valid_x = frame[0::3][(frame[0::3] != 0) & (frame[1::3] != 0)]
        valid_y = frame[1::3][(frame[0::3] != 0) & (frame[1::3] != 0)]
        center_x = np.mean(valid_x) if len(valid_x) > 0 else 0.5
        center_y = np.mean(valid_y) if len(valid_y) > 0 else 0.5

        # Apply rotation to all (x,y) pairs
        for i in range(0, num_coords, 3):
            if i + 1 >= num_coords:
                continue

            x = frame[i] - center_x
            y = frame[i + 1] - center_y
            rotated = rot_matrix @ np.array([x, y])
            frame[i] = rotated[0] + center_x
            frame[i + 1] = rotated[1] + center_y

    if np.random.rand() < 0.6:
        noise = np.random.normal(0, 0.03, size=frame.shape)
        frame = frame + noise

    if np.random.rand() < 0.3:
        dropout_mask = np.random.choice([0, 1], size=frame.shape, p=[0.1, 0.9])
        frame = frame * dropout_mask

    return frame


def augment_sequence(original_path, target_dir):
    """Augment sequence stored in single file"""
    try:
        # Load entire sequence from single file
        original_seq = np.load(os.path.join(original_path, "sequence.npy"))

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

        # Save augmented sequence as single file
        os.makedirs(target_dir, exist_ok=True)
        np.save(os.path.join(target_dir, "sequence.npy"), augmented_seq)
        return True

    except Exception as e:
        print(f"Error augmenting sequence: {e}")
        return False


# ===== Create Dataset Directories =====
print("🔧 Creating dataset folders...")
for action in tqdm(target_actions):
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# ===== Start Processing =====
with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2,
) as holistic:
    for action in target_actions:
        print(f"\n🚀 Processing action: {action}")
        sequence_count = 0

        # --- Webcam Capture ---
        print(f"Starting webcam capture for: {action}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Skipping action.")
            continue

        for seq_num in range(no_sequences_webcam):
            keypoints_buffer = []
            recording = False
            discarded = False

            print(
                f"\n--- Sequence {seq_num + 1}/{no_sequences_webcam} for '{action}' ---"
            )
            print("S - Start/Stop recording | D - Discard | Q - Quit action")
            print("Press 'S' to begin recording...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame capture error, skipping...")
                    continue

                # Mirror display
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                image = draw_landmarks(image, results)

                # Display instructions
                status = "RECORDING" if recording else "READY"
                cv2.putText(
                    image,
                    f"Status: {status} | Frames: {len(keypoints_buffer)}/{sequence_length}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"Action: {action} | Seq: {sequence_count + 1}/{no_sequences_webcam}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.imshow(f"Sign Collection: {action}", image)

                key = cv2.waitKey(10) & 0xFF

                # Toggle recording
                if key == ord("s"):
                    recording = not recording
                    if recording:
                        print("Recording started...")
                    else:
                        print("Recording paused.")

                # Discard current sequence
                elif key == ord("d"):
                    if recording:
                        recording = False
                    print("Sequence discarded.")
                    keypoints_buffer = []
                    discarded = True
                    break

                # Quit action
                elif key == ord("q"):
                    print("Quitting action...")
                    if recording:
                        recording = False
                    break

                # Capture frame if recording
                if recording:
                    keypoints = extract_keypoints(results)

                    # Only save frames with valid data
                    if np.any(keypoints != 0):
                        keypoints_buffer.append(keypoints)

                    # Stop when sequence length reached
                    if len(keypoints_buffer) >= sequence_length:
                        recording = False
                        break

            # Handle quit command
            if key == ord("q"):
                break

            # Skip saving if discarded
            if discarded:
                continue

            # Final sequence validation
            if len(keypoints_buffer) == 0:
                print("No valid frames captured. Sequence skipped.")
                continue

            # Pad sequence if needed
            if len(keypoints_buffer) < sequence_length:
                pad_length = sequence_length - len(keypoints_buffer)
                padding = [np.zeros_like(keypoints_buffer[0])] * pad_length
                keypoints_buffer.extend(padding)
                print(f"Padded with {pad_length} empty frames")
            else:
                keypoints_buffer = keypoints_buffer[:sequence_length]

            # Convert to numpy array
            sequence_array = np.array(keypoints_buffer)

            # Skip if all frames are empty
            if np.all(sequence_array == 0):
                print("All frames empty. Sequence skipped.")
                continue

            # Create sequence directory
            seq_dir = os.path.join(DATA_PATH, action, str(sequence_count))
            os.makedirs(seq_dir, exist_ok=True)

            # Save as single file
            np.save(os.path.join(seq_dir, "sequence.npy"), sequence_array)
            sequence_count += 1
            print(f"✅ Sequence saved: {seq_dir}")

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Skip augmentation if no sequences captured
        if sequence_count == 0:
            print(f"⚠️ No sequences collected for '{action}'. Skipping.")
            action_dir = os.path.join(DATA_PATH, action)
            if os.path.exists(action_dir) and not os.listdir(action_dir):
                shutil.rmtree(action_dir)
            continue

        # --- Augmentation Phase ---
        if sequence_count < no_sequences:
            num_augmentations = no_sequences - sequence_count
            print(f"Generating {num_augmentations} augmentations...")

            # Get existing sequence directories
            existing_dirs = [
                d
                for d in os.listdir(os.path.join(DATA_PATH, action))
                if os.path.isdir(os.path.join(DATA_PATH, action, d))
            ]

            for aug_idx in range(num_augmentations):
                # Randomly select base sequence
                base_dir = os.path.join(
                    DATA_PATH, action, np.random.choice(existing_dirs)
                )

                target_dir = os.path.join(
                    DATA_PATH, action, str(sequence_count + aug_idx)
                )

                if augment_sequence(base_dir, target_dir):
                    print(f"✅ Augmented sequence {sequence_count + aug_idx} created")
                else:
                    print(f"⚠️ Failed to augment sequence {sequence_count + aug_idx}")

print("\n✅ All done! Dataset created under `MP_Data/`")

# Final summary
total_sequences = 0
for action in target_actions:
    action_path = os.path.join(DATA_PATH, action)
    if os.path.exists(action_path):
        count = len(
            [
                d
                for d in os.listdir(action_path)
                if os.path.isdir(os.path.join(action_path, d))
            ]
        )
        total_sequences += count
        print(f"{action}: {count} sequences")

print(f"Total sequences collected: {total_sequences}")
print(f"Handedness: {'Left' if is_left_handed else 'Right'}-handed")
