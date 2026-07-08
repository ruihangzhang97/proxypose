import argparse
import glob
import os
import sys

import numpy as np
import torch

from inference.utils import load_video


def process_video(model, video_path, device, n_frames=None):
    images = load_video(video_path, end_frame=n_frames)
    prediction = model.inference(images)

    out_path = os.path.splitext(video_path)[0] + ".da3.npz"
    save_kwargs = dict(depth=prediction.depth)
    if prediction.extrinsics is not None:
        save_kwargs["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_kwargs["intrinsics"] = prediction.intrinsics
    np.savez_compressed(out_path, **save_kwargs)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on a video or directory of videos."
    )
    parser.add_argument(
        "input", help="Path to a .mp4 file or a directory containing .mp4 files."
    )
    parser.add_argument(
        "--n_frames", type=int, default=49,
        help="Number of frames to process per video (default: 49)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)."
    )
    parser.add_argument(
        "--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="HuggingFace model ID for Depth Anything 3."
    )
    args = parser.parse_args()

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print(
            "depth_anything_3 is not installed.\n"
            "Install it from: https://github.com/DepthAnything/Depth-Anything-V3"
        )
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Loading Depth Anything 3 model: {args.model} ...")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)

    if os.path.isfile(args.input):
        video_paths = [args.input]
    elif os.path.isdir(args.input):
        video_paths = sorted(glob.glob(os.path.join(args.input, "*.mp4")))
        if not video_paths:
            print(f"No .mp4 files found in {args.input}")
            sys.exit(1)
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)

    for video_path in video_paths:
        out_path = os.path.splitext(video_path)[0] + ".da3.npz"
        if os.path.exists(out_path):
            print(f"Skipping {video_path} (output already exists: {out_path})")
            continue
        print(f"Processing {video_path} ...")
        process_video(model, video_path, device, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
