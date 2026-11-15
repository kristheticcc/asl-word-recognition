# Imports
import csv
from pathlib import Path
from urllib.request import urlretrieve

import mediapipe as mp
import numpy as np

from data.video_loader import load_video
from data.landmark_extractor import extract_landmarks

# Setting up paths
project_root = Path(__file__).resolve().parent.parent

video_dir = project_root / 'data' / 'raw' / 'videos'    # Original wlasl videos
manifest_path = project_root / 'data' / 'metadata' / 'selected_videos.csv'  # 345 selected videos and their labels

model_dir = project_root / "models"
model_path = model_dir / "hand_landmarker.task"     # mediapipe's pretrained hand detection model

output_dir = project_root / 'data' / 'preprocessed' / 'landmarks'    # Individual landmark arrays
metadata_path = project_root / 'data' / 'preprocessed' / 'metadata.csv'    # Info about successfully processed videos

# Download the model if needed
model_url = (
"https://storage.googleapis.com/mediapipe-models/"
"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Download the model only if it does not exist
if not model_path.exists():
    print("Downloading MediaPipe hand model...")
    urlretrieve(model_url, model_path)

# Read the manifest
with open(manifest_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    selected_videos = list(reader)

print("Videos in manifest:", len(selected_videos))

if len(selected_videos) != 345:
    raise ValueError("Expected exactly 345 selected videos.")

# Configure the MediaPipe
# Initialize MediaPipe:
# Use the downloaded model
# process frames sequentially in video mode
# Detect up to 2 hands per frama
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path = str(model_path),
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)


# The processing loop
metadata_rows = []      # List for adding rows to metadata csv

for index, video in enumerate(selected_videos, start=1):
    video_id = video["video_id"]
    video_path = video_dir / f"{video_id}.mp4"

    print(f"[{index}/{len(selected_videos)}] Processing {video_id}")

    try:
        frames, fps = load_video(video_path)    # Load the video and fetch frames and fps

        with HandLandmarker.create_from_options(options) as landmarker:     # Extract the landmarks
            landmarks = extract_landmarks(frames, fps, landmarker)

        # Save the landmarks to .npy file
        output_path = output_dir / f"{video_id}.npy"
        np.save(output_path, landmarks)

        # Count number of frames with hands
        frames_with_hands = int(
            np.any(landmarks!=0, axis=(1, 2, 3)).sum()
        )

        # Append to metadata rows
        metadata_rows.append({
            "video_id": video_id,
            "gloss": video["gloss"],
            "signer_id": video["signer_id"],
            "split": video["split"],
            "fps": fps,
            "frame_count": len(landmarks),
            "frames_with_hands": frames_with_hands,
        })

    except Exception as e:
        print(f"FAILED {video_id}: {e}")


print("Successfully processed:", len(metadata_rows))


# Adding metadata rows to metadata dataset
# Columns names
fieldnames = [
    "video_id", "gloss", "signer_id", "split", "fps", "frame_count", "frames_with_hands",
]

# Write fieldnames (column names) and rows to metadata (overwrites if already exists)
with open(metadata_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metadata_rows)

print("Metadata saved to:", metadata_path)