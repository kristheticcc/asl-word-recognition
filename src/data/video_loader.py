# Imports
from pathlib import Path
import cv2
import numpy as np

# Function for opening a video
def load_video(video_path: Path):
    # Getting the capture object from the video
    capture = cv2.VideoCapture(video_path)

    # If capture is not opened, raise an error
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # List for all frames of the video
    frames = []
    try:
        while True:
            # success: frame opened or not, frame: the current frame
            success, frame = capture.read()

            # If no frames anymore, break out of the loop
            if not success:
                break

            # Add the current frame to the list
            frames.append(frame)
    finally:
        # Release the capture object
        capture.release()

    # If video has no frames
    if not frames:
        raise RuntimeError(f"No frames can be read from: {video_path}")

    return frames   # return the list of frames

# Testing function on a video
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "data" / "raw" / "videos" / "69302.mp4"

    frames = load_video(video_path)

    print(f"Video: {video_path}")
    print(f"Frames loaded: {len(frames)}")
    print(f"Frame shape: {frames[0].shape}")    # first frame's height, width, color channels