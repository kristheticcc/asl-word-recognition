# Imports
import csv
from pathlib import Path
from collections import Counter
import numpy as np

# Setting up paths
project_root = Path(__file__).resolve().parent.parent
landmark_dir = project_root / 'data' / 'preprocessed' / 'landmarks'
metadata_path = project_root / 'data' / 'preprocessed' / 'metadata.csv'

# Read the metadata
with open(metadata_path, "r", newline="") as f:
    metadata = list(csv.DictReader(f))

# Basic counts
video_ids = [row["video_id"] for row in metadata]
landmark_files = list(landmark_dir.glob("*.npy"))

print(f"Metadata rows: {len(metadata)}")
print(f"Unique video ids: {len(set(video_ids))}")
print(f"Landmark files: {len(landmark_files)}")

# Check missing and extra files
expected_ids = set(video_ids)
actual_ids = {path.stem for path in landmark_files}

print(f"Missing landmark files: {len(expected_ids - actual_ids)}")
print(f"Unexpected landmark files: {len(actual_ids - expected_ids)}")

# Validate every landmark array
invalid_shapes = []
non_finite = []
empty_sequences = []
no_hand_sequences = []
frame_count_mismatches = []

for row in metadata:
    video_id = row["video_id"]
    path = landmark_dir / f"{video_id}.npy"

    if not path.exists():
        continue

    landmarks = np.load(path)

    if landmarks.ndim != 4 or landmarks.shape[1:] != (2, 21, 3):
        invalid_shapes.append(video_id)

    if not np.isfinite(landmarks).all():
        non_finite.append(video_id)

    if len(landmarks) == 0:
        empty_sequences.append(video_id)

    if len(landmarks) != int(row["frame_count"]):
        frame_count_mismatches.append(video_id)

    if len(landmarks) > 0 and not np.any(landmarks != 0):
        no_hand_sequences.append(video_id)

# Check labels and splits
gloss_counts = Counter(row["gloss"] for row in metadata)
split_counts = Counter(row["split"] for row in metadata)

print(f"\nVocabulary size: {len(gloss_counts)}")
print(f"Split size: {len(split_counts)}")

print(f"\nInvalid shapes: {invalid_shapes}")
print(f"Non-finite arrays: {non_finite}")
print(f"Empty sequences: {empty_sequences}")
print(f"Sequences with no detected hands: {no_hand_sequences}")
print(f"Frame count mismatches: {frame_count_mismatches}")