from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

project_root = Path(__file__).resolve().parents[1]
video_dir = project_root / "data" / "raw" / "videos"
model_dir = project_root / "models"
output_dir = project_root / "data" / "processed"

model_path = model_dir / "hand_landmarker.task"
output_path = output_dir / "sample_landmarks.npy"

model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

if not model_path.exists():
    print("Downloading MediaPipe hand model...")
    urlretrieve(model_url, model_path)

video_path = next(video_dir.glob("*.mp4"))
capture = cv2.VideoCapture(str(video_path))

if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)

landmark_frames = []
detected_frames = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for frame_number in range(30):
        success, frame = capture.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(frame_number * 1000 / fps)
        result = landmarker.detect_for_video(image, timestamp_ms)

        coordinates = np.zeros((2, 21, 3), dtype=np.float32)

        for hand_number, hand in enumerate(result.hand_landmarks[:2]):
            coordinates[hand_number] = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in hand
            ]

        if result.hand_landmarks:
            detected_frames += 1

        landmark_frames.append(coordinates)

capture.release()

landmarks = np.stack(landmark_frames)
np.save(output_path, landmarks)

print("Video:", video_path.name)
print("Frames processed:", len(landmarks))
print("Frames with detected hands:", detected_frames)
print("Landmark array shape:", landmarks.shape)
print("Saved to:", output_path)