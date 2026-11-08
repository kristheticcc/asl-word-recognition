from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


# Find the project root directory.
project_root = Path(__file__).resolve().parents[1]

# Define important project folders.
video_dir = project_root / "data" / "raw" / "videos"
model_dir = project_root / "models"
output_dir = project_root / "data" / "processed"

# Define the model and output file paths.
model_path = model_dir / "hand_landmarker.task"
output_path = output_dir / "sample_landmarks.npy"

# URL for the MediaPipe hand-landmark model.
model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Create the folders if they do not already exist.
model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Download the MediaPipe model only if it is not already available.
if not model_path.exists():
    print("Downloading MediaPipe hand model...")
    urlretrieve(model_url, model_path)

# Select the first MP4 file found in the videos directory.
# This video is used only as a small test sample.
video_path = next(video_dir.glob("*.mp4"))

# Open the video using OpenCV.
capture = cv2.VideoCapture(str(video_path))

# Stop the program if OpenCV cannot open the video.
if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

# Read the video's frames-per-second value.
fps = capture.get(cv2.CAP_PROP_FPS)

# Use 30 FPS if the video does not provide a valid FPS value.
if fps <= 0:
    fps = 30


# Create shorter names for the MediaPipe classes.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure the MediaPipe hand detector.
options = HandLandmarkerOptions(
    # Tell MediaPipe where the downloaded model is located.
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),

    # VIDEO mode requires timestamps for each frame.
    running_mode=RunningMode.VIDEO,

    # Detect a maximum of two hands per frame.
    num_hands=2,
)

# Store landmark arrays for every processed frame.
landmark_frames = []

# Count how many frames contain at least one detected hand.
detected_frames = 0


# Create the hand landmarker.
# The "with" statement automatically releases MediaPipe resources.
with HandLandmarker.create_from_options(options) as landmarker:

    # Process only the first 30 frames.
    for frame_number in range(30):

        # Read one frame from the video.
        success, frame = capture.read()

        # Stop if there are no more frames.
        if not success:
            break

        # OpenCV reads images in BGR format.
        # MediaPipe expects RGB format, so convert the frame.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # Convert the NumPy image into a MediaPipe image object.
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # Convert the frame number into milliseconds.
        # MediaPipe VIDEO mode requires increasing timestamps.
        timestamp_ms = int(frame_number * 1000 / fps)

        # Detect hands in the current frame.
        result = landmarker.detect_for_video(
            image,
            timestamp_ms,
        )

        # Create an empty array for up to two hands.
        #
        # Shape:
        #   2  = maximum number of hands
        #   21 = landmarks per hand
        #   3  = x, y, and z coordinates
        #
        # If a hand is not detected, its values remain zero.
        coordinates = np.zeros(
            (2, 21, 3),
            dtype=np.float32,
        )

        # Copy the detected hand landmarks into the array.
        for hand_number, hand in enumerate(
            result.hand_landmarks[:2]
        ):
            coordinates[hand_number] = [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
                for landmark in hand
            ]

        # Count the frame if at least one hand was detected.
        if result.hand_landmarks:
            detected_frames += 1

        # Save this frame's coordinates.
        landmark_frames.append(coordinates)

# Release the OpenCV video resource.
capture.release()

# Combine all frame arrays into one NumPy array.
#
# Expected shape:
#   (number of frames, 2 hands, 21 landmarks, 3 coordinates)
landmarks = np.stack(landmark_frames)

# Save the landmark data as a NumPy file.
np.save(output_path, landmarks)

# Print a summary of the extraction process.
print("Video:", video_path.name)
print("Frames processed:", len(landmarks))
print("Frames with detected hands:", detected_frames)
print("Landmark array shape:", landmarks.shape)
print("Saved to:", output_path)