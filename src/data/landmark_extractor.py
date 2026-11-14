# Imports
import cv2
import mediapipe as mp
import numpy as np

# Function for extracting hand landmarks from frames
def extract_landmarks(frames, fps, landmarker):

    landmark_frames = []

    # Safety checks
    if not frames:
        raise ValueError("No frames provided.")
    if fps <= 0:
        fps = 30

    # Extract hand landmarks from each frame
    for frame_number, frame in enumerate(frames):

        # convert BGR (OpenCV) to RGB(Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame into Mediapipe image
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data = rgb_frame,
        )

        # For temporal order of frames
        timestamp_ms = int(frame_number * 1000 / fps)

        # Result contains detected hand landmarks if any (also contains left or right hand classification)
        result = landmarker.detect_for_video(image, timestamp_ms)

        coordinates = np.zeros(
            (2, 21, 3),     # 2 hands, 21 landmarks per hand, 3 coordinates (x, y, z)
            dtype=np.float32,
        )

        # Fill slots with actual coordinates
        for hand_number, hand in enumerate(result.hand_landmarks[:2]):
            coordinates[hand_number] = [
                [landmark.x, landmark.y, landmark.z] for landmark in hand
            ]

        # Add formatted landmarks through coordinates into landmark frames
        landmark_frames.append(coordinates)

    landmarks = np.stack(landmark_frames)   # Convert to numpy array

    return landmarks