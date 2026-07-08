from pathlib import Path
import subprocess
import yaml

import cv2
from decord import VideoReader
import numpy as np


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_video(video_path, end_frame=None):
    """Load video using decord."""
    video_path = Path(video_path)

    if video_path.is_file():
        vr = VideoReader(str(video_path))

        n = len(vr) if end_frame is None else min(end_frame, len(vr))
        frames = [vr[i].asnumpy() for i in range(n)]
    elif video_path.is_dir():
        frames = []
        for frame_path in sorted(video_path.glob("*.*")):
            frame = cv2.imread(str(frame_path))[..., [2, 1, 0]]
            frames.append(frame)
        if end_frame is not None:
            frames = frames[:end_frame]
    else:
        raise ValueError(f"Invalid video path: {video_path}")

    return frames


def save_video(frames, output_path: Path, mute=True, fps=30):
    """Render frames to video file using ffmpeg."""
    height, width = frames[0].shape[:2]

    # Write frames using ffmpeg
    stdout = subprocess.DEVNULL if mute else None
    stderr = subprocess.DEVNULL if mute else None
    process = subprocess.Popen(
        [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=stderr
    )

    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()


def downsample_video(video, target_size=(512, 512)):
    """Downsample video to target resolution"""
    downsampled_video = []

    for frame in video:
        downsampled_video.append(cv2.resize(frame, dsize=[target_size[1], target_size[0]]))

    return downsampled_video