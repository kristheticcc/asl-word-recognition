# Imports
from pathlib import Path
import mediapipe as mp
import numpy as np

from data.video_loader import load_video
from data.landmark_extractor import extract_landmarks
from urllib.request import urlretrieve

# Set up paths
project_root = Path(__file__).resolve().parents[1]
model_dir = project_root / "models"
model_path = model_dir / "hand_landmarker.task"
video_path = project_root / "data" / "raw" / "videos" / "69302.mp4"

# Download the model
model_url = (
"https://storage.googleapis.com/mediapipe-models/"
"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

model_dir.mkdir(parents=True, exist_ok=True)

# Download the model only if it does not exist
if not model_path.exists():
    print("Downloading MediaPipe hand model...")
    urlretrieve(model_url, model_path)

# Load the videos
frames, fps = load_video(video_path)

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

# Run the extraction. With block initializes model and releases resources later
with HandLandmarker.create_from_options(options) as landmarker:
    landmarks = extract_landmarks(frames, fps, landmarker)

# Inspect the output
print(f"Video: {video_path.name}")
print(f"Landmark shape: {landmarks.shape}")
print(f"Data Type: {landmarks.dtype}")
print(f"All values finite: {np.isfinite(landmarks).all()}")