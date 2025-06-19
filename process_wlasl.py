import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ===== Configuration =====
VIDEO_PATH = "videos"
ANNOTATION_FILE = "WLASL_v0.3.json"
DATA_PATH = "MP_Data"
target_actions = ["book", "help"]
no_sequences = 30
sequence_length = 30

# ===== MediaPipe Setup =====
mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


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


def augment_sequence(original_path, target_dir):
    original_seq = [
        np.load(os.path.join(original_path, f"{i}.npy")) for i in range(sequence_length)
    ]
    augmented_seq = []

    for frame in original_seq:
        jitter = np.random.normal(loc=0.0, scale=0.02, size=frame.shape)
        scale = np.random.uniform(0.9, 1.1)
        frame_aug = (frame + jitter) * scale
        augmented_seq.append(frame_aug)

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
                }
            )

# ===== Create Dataset Directories =====
print("🔧 Creating dataset folders...")
for action in tqdm(target_actions):
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# ===== Start Processing =====
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
) as holistic:
    for action in target_actions:
        print(f"\n🚀 Processing action: {action}")
        sequence_count = 0
        video_entries = action_video_map[action]

        if not video_entries:
            print(f"⚠️ No videos found for action: {action}")
            continue

        # Process real videos first
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

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if actual_start != start:
                tqdm.write(
                    f"⚠️ Seek failed ({actual_start} vs {start}) in {video_info['video_id']}"
                )
                cap.release()
                continue

            seq_dir = os.path.join(DATA_PATH, action, str(sequence_count))
            keypoints_list = []
            valid = True

            for _ in range(min(sequence_length, end - start)):
                ret, frame = cap.read()
                if not ret:
                    valid = False
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                keypoints_list.append(keypoints)

            cap.release()

            if not valid or not keypoints_list:
                tqdm.write(
                    f"⚠️ Skipping {video_info['video_id']} due to frame read error."
                )
                continue

            if len(keypoints_list) < sequence_length:
                pad_length = sequence_length - len(keypoints_list)
                padding = [np.zeros_like(keypoints_list[0])] * pad_length
                keypoints_list.extend(padding)

            os.makedirs(seq_dir, exist_ok=True)
            for frame_num, keypoints in enumerate(keypoints_list):
                np.save(os.path.join(seq_dir, f"{frame_num}.npy"), keypoints)

            sequence_count += 1
            tqdm.write(
                f"✅ Sequence {sequence_count}/{no_sequences} collected from {video_info['video_id']}"
            )

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

            sequence_count += num_augmentations  # Update total count

print("\n✅ All done! Dataset created under `MP_Data/`")
total_sequences = sum(
    len(os.listdir(os.path.join(DATA_PATH, act))) for act in target_actions
)
print(f"Total sequences collected: {total_sequences}")
