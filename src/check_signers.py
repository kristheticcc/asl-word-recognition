from pathlib import Path
import json
from collections import Counter, defaultdict

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"

with (raw_dir / "WLASL_v0.3.json").open("r", encoding="utf-8") as file:
    metadata = json.load(file)

local_video_ids = {
    video.stem for video in (raw_dir / "videos").glob("*.mp4")
}

split_counts = Counter()
split_signers = defaultdict(set)

for record in metadata:
    for instance in record.get("instances", []):
        video_id = str(instance["video_id"])

        # Only analyze videos that actually exist locally
        if video_id not in local_video_ids:
            continue

        split = instance.get("split", "unknown")
        signer = str(instance.get("signer_id", "unknown"))

        split_counts[split] += 1
        split_signers[split].add(signer)

print("Available videos by split:")

for split, count in split_counts.items():
    print(f"{split}: {count} videos")

print("\nUnique signers by split:")

for split, signers in split_signers.items():
    print(f"{split}: {len(signers)} signers")

splits = list(split_signers)

print("\nSigner overlap between splits:")

for first_index in range(len(splits)):
    for second_index in range(first_index + 1, len(splits)):
        first = splits[first_index]
        second = splits[second_index]
        overlap = split_signers[first] & split_signers[second]

        print(f"{first} and {second}: {len(overlap)} shared signers")