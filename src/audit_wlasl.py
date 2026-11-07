from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
metadata_path = raw_dir / "WLASL_v0.3.json"
video_dir = raw_dir / "videos"

with metadata_path.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

local_video_ids = {
    video.stem for video in video_dir.glob("*.mp4")
}

class_results = []

for record in metadata:
    gloss = record["gloss"]

    metadata_ids = {
        str(instance["video_id"])
        for instance in record.get("instances", [])
    }

    available_ids = metadata_ids.intersection(local_video_ids)
    missing_ids = metadata_ids.difference(local_video_ids)

    class_results.append(
        (
            gloss,
            len(metadata_ids),
            len(available_ids),
            len(missing_ids),
        )
    )

print("Classes in JSON:", len(class_results))
print("Local MP4 files:", len(local_video_ids))
print("Metadata video instances:", sum(row[1] for row in class_results))
print("Available video instances:", sum(row[2] for row in class_results))
print("Missing video instances:", sum(row[3] for row in class_results))

print("\nTop 20 classes by available videos:")

for gloss, total, available, missing in sorted(
    class_results,
    key=lambda row: row[2],
    reverse=True,
)[:20]:
    print(
        f"{gloss}: total={total}, "
        f"available={available}, missing={missing}"
    )