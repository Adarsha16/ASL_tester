import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy import interpolate
import random
import shutil  # Added for directory removal

# ===== Configuration =====
VIDEO_PATH = "videos"
ANNOTATION_FILE = "WLASL_v0.3.json"
DATA_PATH = "MP_Data"
target_actions = [
    "book",
    "help",
    "ljaf",
    "yes",
    "no",
    "want",
    "eat",
    "drink",
    "bathroom",
]  # Expanded vocabulary
no_sequences = 30  # Increased data volume
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
        if new_length < 5 or new_length > 100:
            return sequence

        # Select interpolation method
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
        # Frame jittering
        jitter_amount = np.random.randint(1, 3)
        sequence = np.concatenate(
            [sequence, sequence[-1:].repeat(jitter_amount, axis=0)]
        )

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
        translation = np.random.uniform(-0.1, 0.1, size=(3,))
        frame[0::3] += translation[0]  # x-coordinates
        frame[1::3] += translation[1]  # y-coordinates
        frame[2::3] += translation[2]  # z-coordinates

    if np.random.rand() < 0.5:  # 50% chance
        # Rotation (2D plane)
        angle = np.random.uniform(-15, 15)
        rad = np.deg2rad(angle)
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

        # Center normalization
        center_x = np.mean(frame[0::3]) if np.any(frame[0::3]) else 0.5
        center_y = np.mean(frame[1::3]) if np.any(frame[1::3]) else 0.5

        # Apply rotation to x,y coordinates
        for i in range(0, len(frame), 3):
            x = frame[i] - center_x
            y = frame[i + 1] - center_y
            rotated = rot_matrix @ np.array([x, y])
            frame[i] = rotated[0] + center_x
            frame[i + 1] = rotated[1] + center_y

    # Per-joint transformations
    if np.random.rand() < 0.6:  # 60% chance
        # Random noise
        noise = np.random.normal(0, 0.03, size=frame.shape)
        frame = frame + noise

    if np.random.rand() < 0.3:  # 30% chance
        # Axis dropout
        dropout_mask = np.random.choice([0, 1], size=frame.shape, p=[0.1, 0.9])
        frame = frame * dropout_mask

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


with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

action_video_map = {action: [] for action in target_actions}
for entry in annotations:
    gloss = entry["gloss"].lower()
    if gloss in target_actions:
        for inst in entry["instances"]:
            action_video_map[gloss].append(
                {
                    "video_id": inst["video_id"] + ".mp4",
                    "start": inst.get("frame_start", 0),
                    "end": inst.get("frame_end", None),
                    "signer_id": inst.get("signer_id", 0),
                }
            )

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
        video_entries = action_video_map[action]

        if not video_entries:
            print(f"⚠️ No videos found for action: {action}")
            continue

        # Sort by signer for better diversity
        video_entries.sort(key=lambda x: x["signer_id"])

        # Process real videos
        for video_info in tqdm(video_entries, desc=f"{action} videos"):
            if sequence_count >= no_sequences:
                break

            file_path = os.path.join(VIDEO_PATH, video_info["video_id"])
            if not os.path.isfile(file_path):
                tqdm.write(f"⚠️ File not found: {file_path}")
                continue

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                tqdm.write(f"⚠️ Could not open {file_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start = video_info["start"] or 0
            end = video_info["end"]
            if end is None or end <= 0 or end > total_frames:
                end = total_frames

            if start >= end:
                tqdm.write(
                    f"⚠️ Invalid frame range ({start}-{end}) in {video_info['video_id']}"
                )
                cap.release()
                continue

            # Calculate optimal frame step
            frame_step = max(1, (end - start) // sequence_length)

            seq_dir = os.path.join(DATA_PATH, action, str(sequence_count))
            keypoints_list = []
            valid = True

            for frame_idx in range(start, end, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    valid = False
                    break

                # Process every nth frame
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                # Only add frames with detected hands
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints_list.append(keypoints)

                if len(keypoints_list) >= sequence_length:
                    break

            cap.release()

            if not valid or not keypoints_list:
                tqdm.write(
                    f"⚠️ Skipping {video_info['video_id']} due to frame read error."
                )
                continue

            # Pad sequence if needed
            if len(keypoints_list) < sequence_length:
                pad_length = sequence_length - len(keypoints_list)
                padding = [np.zeros_like(keypoints_list[0])] * pad_length
                keypoints_list.extend(padding)
            else:
                keypoints_list = keypoints_list[:sequence_length]

            os.makedirs(seq_dir, exist_ok=True)
            for frame_num, keypoints in enumerate(keypoints_list):
                np.save(os.path.join(seq_dir, f"{frame_num}.npy"), keypoints)

            sequence_count += 1
            tqdm.write(
                f"✅ Sequence {sequence_count}/{no_sequences} collected from {video_info['video_id']}"
            )

        # === FIX: Handle case with no real sequences ===
        if sequence_count == 0:
            # Remove empty action directory
            action_dir = os.path.join(DATA_PATH, action)
            if os.path.exists(action_dir):
                shutil.rmtree(action_dir)
            print(f"⚠️ No sequences collected for '{action}'. Directory removed.")
            continue

        # Generate augmentations if needed
        if sequence_count < no_sequences:
            num_augmentations = no_sequences - sequence_count
            print(
                f"⚠️ Not enough real sequences for '{action}' ({sequence_count}), generating {num_augmentations} augmentations..."
            )
            existing_dirs = [
                os.path.join(DATA_PATH, action, str(i)) for i in range(sequence_count)
            ]

            for aug_idx in range(num_augmentations):
                base_dir = existing_dirs[aug_idx % sequence_count]
                target_dir = os.path.join(
                    DATA_PATH, action, str(sequence_count + aug_idx)
                )
                augment_sequence(base_dir, target_dir)
                tqdm.write(
                    f"✅ Augmented sequence {sequence_count + aug_idx + 1}/{no_sequences} created"
                )

print("\n✅ All done! Dataset created under `MP_Data/`")
total_sequences = sum(
    len(os.listdir(os.path.join(DATA_PATH, act)))
    for act in target_actions
    if os.path.exists(os.path.join(DATA_PATH, act))  # Handle removed actions
)
print(f"Total sequences collected: {total_sequences}")
