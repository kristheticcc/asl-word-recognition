from pathlib import Path
import json
from collections import Counter, defaultdict

# Restrict analysis to 26 observed classes
SELECTED_GLOSSES = {
    "thin", "before", "cool", "drink", "go", "who", "help",
    "computer", "cousin", "tall", "short", "candy", "thanksgiving",
    "accident", "bowling", "trade", "what", "basketball", "cold",
    "yes", "change", "pizza", "shirt", "dark", "last", "call"
}

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

# For counting missing signer ids
missing_signer_counts = Counter()

# For unique signer ids per class and split
class_split_signers = defaultdict(lambda: defaultdict(set))

# For counting available videos per class and split
class_split_counts = defaultdict(Counter)

# For counting videos with missing signer ids per class and split
class_missing_signers = defaultdict(Counter)

# Loop through every class in the metadata.
for record in metadata:

    # Skip if gloss not a part of 26 selected classes
    if record["gloss"] not in SELECTED_GLOSSES:
        continue

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
        signer_id = instance.get("signer_id")

        gloss = record["gloss"]

        # Add one video to the appropriate split count.
        split_counts[split] += 1
        class_split_counts[gloss][split] += 1

        if signer_id is None or str(signer_id).strip() in {"", "unknown"}:
            missing_signer_counts[split] += 1
            class_missing_signers[gloss][split] += 1
            continue

        signer_id = str(signer_id).strip()  # Normalize signer id

        # Add the signer to the split's set.
        # Sets automatically remove duplicate signer IDs.
        split_signers[split].add(signer_id)
        class_split_signers[gloss][split].add(signer_id)

# Display the number of available videos in each split.
print("Available videos by split:")
for split, count in split_counts.items():
    print(f"{split}: {count} videos")

print("\nVideos with missing signer IDs:")
for split, count in missing_signer_counts.items():
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

# Signers per class and split analysis
print("\nPer-class signer and video distribution:")

for gloss in sorted(SELECTED_GLOSSES):
    print(f"\n{gloss}:")

    for split in ["train", "val", "test"]:
        videos = class_split_counts[gloss][split]
        signers = len(class_split_signers[gloss][split])
        missing = class_missing_signers[gloss][split]

        print(
            f"    {split:5} "
            f"videos={videos:2} "
            f"signers={signers:2} "
            f"missing_signer_ids={missing:2}"
        )
