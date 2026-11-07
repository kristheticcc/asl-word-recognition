from pathlib import Path
import json
from collections import Counter

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"

with (raw_dir / "WLASL_v0.3.json").open(
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

local_video_ids = {
    video.stem
    for video in (raw_dir / "videos").glob("*.mp4")
}

class_results = []

for record in metadata:
    gloss = record["gloss"]

    available_instances = [
        instance
        for instance in record.get("instances", [])
        if str(instance["video_id"]) in local_video_ids
    ]

    available_video_ids = {
        str(instance["video_id"])
        for instance in available_instances
    }

    signer_ids = {
        str(instance.get("signer_id", "unknown"))
        for instance in available_instances
    }

    split_counts = Counter(
        instance.get("split", "unknown")
        for instance in available_instances
    )

    class_results.append(
        {
            "gloss": gloss,
            "videos": len(available_video_ids),
            "signers": len(signer_ids),
            "splits": split_counts,
        }
    )

# Rank classes by available videos, then by number of signers.
ranked_classes = sorted(
    class_results,
    key=lambda item: (
        item["videos"],
        item["signers"],
    ),
    reverse=True,
)

print("Top 30 candidate classes:\n")

for rank, item in enumerate(ranked_classes[:30], start=1):
    print(
        f"{rank:2}. {item['gloss']:<15} "
        f"videos={item['videos']:<2} "
        f"signers={item['signers']:<2} "
        f"splits={dict(item['splits'])}"
    )

print("\nSummary:")

for vocabulary_size in (20, 30):
    selected = ranked_classes[:vocabulary_size]

    print(
        f"Top {vocabulary_size}: "
        f"{sum(item['videos'] for item in selected)} total videos, "
        f"minimum class size={min(item['videos'] for item in selected)}, "
        f"minimum signer count={min(item['signers'] for item in selected)}"
    )