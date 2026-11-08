from pathlib import Path
import json
from collections import Counter, defaultdict

# Find the project root directory.
# This script is inside the src/ folder, so parents[1] is the project root.
project_root = Path(__file__).resolve().parents[1]

# Location of the WLASL metadata file.
raw_dir = project_root / "data" / "raw"

# Load the metadata JSON file.
with (raw_dir / "WLASL_v0.3.json").open(
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

# Store the IDs of videos that are available locally.
# For example, 52291.mp4 becomes the ID "52291".
local_video_ids = {
    video.stem
    for video in (raw_dir / "videos").glob("*.mp4")
}

# Count the number of available videos in each split.
# Example: train -> 8313 videos.
split_counts = Counter()

# Store unique signer IDs for each split.
# Example: train -> {1, 2, 3, ...}.
split_signers = defaultdict(set)

# Loop through every class in the metadata.
for record in metadata:

    # Loop through every video instance for that class.
    for instance in record.get("instances", []):

        # Convert the video ID to a string so it matches the MP4 filename.
        video_id = str(instance["video_id"])

        # Ignore videos that are listed in the metadata
        # but are not available in the local videos folder.
        if video_id not in local_video_ids:
            continue

        # Get the dataset split.
        # Use "unknown" if the metadata does not contain a split.
        split = instance.get("split", "unknown")

        # Get the signer ID.
        # Use "unknown" if signer information is missing.
        signer = str(instance.get("signer_id", "unknown"))

        # Add one video to the appropriate split count.
        split_counts[split] += 1

        # Add the signer to the split's set.
        # Sets automatically remove duplicate signer IDs.
        split_signers[split].add(signer)

# Display the number of available videos in each split.
print("Available videos by split:")

for split, count in split_counts.items():
    print(f"{split}: {count} videos")

# Display the number of unique signers in each split.
print("\nUnique signers by split:")

for split, signers in split_signers.items():
    print(f"{split}: {len(signers)} signers")

# Get the names of the splits that were found.
splits = list(split_signers)

print("\nSigner overlap between splits:")

# Compare every pair of splits.
# The index ranges prevent comparing a split with itself
# or repeating the same comparison twice.
for first_index in range(len(splits)):
    for second_index in range(first_index + 1, len(splits)):

        first = splits[first_index]
        second = splits[second_index]

        # Find signers that appear in both splits.
        overlap = split_signers[first] & split_signers[second]

        print(
            f"{first} and {second}: "
            f"{len(overlap)} shared signers"
        )