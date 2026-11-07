from pathlib import Path
import json
from collections import Counter

# Minimum requirements for a class to be considered valid.
MIN_VIDEOS = 12
MIN_SIGNERS = 7
REQUIRED_SPLITS = {"train", "val", "test"}

# Locate the project and dataset folders.
project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"

# Load the WLASL metadata.
with (raw_dir / "WLASL_v0.3.json").open(
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

# Store the video IDs that exist locally.
local_video_ids = {
    video.stem
    for video in (raw_dir / "videos").glob("*.mp4")
}

valid_classes = []

# Examine every gloss/class in the metadata.
for record in metadata:
    gloss = record["gloss"]

    # Keep only video instances whose MP4 files exist locally.
    available_instances = [
        instance
        for instance in record.get("instances", [])
        if str(instance["video_id"]) in local_video_ids
    ]

    # Count unique available videos.
    available_video_ids = {
        str(instance["video_id"])
        for instance in available_instances
    }

    # Count unique signers.
    signer_ids = {
        str(instance.get("signer_id", "unknown"))
        for instance in available_instances
    }

    # Count videos in train, validation, and test.
    split_counts = Counter(
        instance.get("split", "unknown")
        for instance in available_instances
    )

    # Keep the class only if it satisfies all requirements.
    has_required_splits = REQUIRED_SPLITS.issubset(
        set(split_counts)
    )

    if (
        len(available_video_ids) >= MIN_VIDEOS
        and len(signer_ids) >= MIN_SIGNERS
        and has_required_splits
    ):
        valid_classes.append(
            {
                "gloss": gloss,
                "videos": len(available_video_ids),
                "signers": len(signer_ids),
                "splits": split_counts,
            }
        )

# Rank valid classes by video count and then signer count.
valid_classes.sort(
    key=lambda item: (
        item["videos"],
        item["signers"],
    ),
    reverse=True,
)

print("Number of valid classes:", len(valid_classes))
print("\nValid classes:\n")

for rank, item in enumerate(valid_classes, start=1):
    print(
        f"{rank:2}. {item['gloss']:<15} "
        f"videos={item['videos']:<2} "
        f"signers={item['signers']:<2} "
        f"splits={dict(item['splits'])}"
    )

print("\nVocabulary decision:")

if len(valid_classes) < 20:
    print("Fewer than 20 classes passed the requirements.")
elif len(valid_classes) <= 30:
    print(
        f"Use all {len(valid_classes)} valid classes."
    )
else:
    print(
        "More than 30 classes passed. "
        "Use the top 30 classes."
    )