import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inference.generate_proxy import load_model
from inference.evaluate_all import (
    crop_video_and_prompt,
    full_inference,
    get_square_crop_prompt,
    adjust_intrinsics_crop,
)
from inference.utils import load_video, save_video, load_config


def load_intrinsics_from_da3(da3_path, video_shape):
    """Load and rescale camera intrinsics from a Depth Anything 3 .da3.npz file.

    Args:
        da3_path:    Path to the .da3.npz file.
        video_shape: (H, W) of the full input video frames.

    Returns:
        intrinsics: (3, 3) numpy array, or None if the file is missing or has no intrinsics.
    """
    da3_path = Path(da3_path)
    if not da3_path.exists():
        print(f"Warning: Depth Anything 3 file not found at {da3_path}. Using default FOV.")
        return None

    data = np.load(da3_path)
    if "intrinsics" not in data:
        print(f"Warning: no intrinsics in {da3_path}. Using default FOV.")
        return None

    depth_map = data["depth"][0]
    intrinsics = data["intrinsics"][0].copy()
    # Rescale focal length and principal point from DA3 resolution to video resolution
    intrinsics[0] = intrinsics[0] / depth_map.shape[1] * video_shape[1]
    intrinsics[1] = intrinsics[1] / depth_map.shape[0] * video_shape[0]
    print(f"Loaded intrinsics from Depth Anything 3: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
    return intrinsics


def _parse_prompt(args_prompt: list[str], parser: argparse.ArgumentParser) -> dict:
    """Parse --prompt as either a JSON file path or two integer pixel coordinates.

    JSON form  : --prompt path/to/query.json
    Legacy form: --prompt 320 240
    """
    if len(args_prompt) == 1:
        json_path = Path(args_prompt[0]).expanduser().resolve()
        if not json_path.exists():
            parser.error(f"--prompt: file not found: {json_path}")
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            parser.error(f"--prompt: failed to parse {json_path}: {e}")
        try:
            first_group = next(iter(data["groups"].values()))
            u, v = int(first_group[0][0]), int(first_group[0][1])
        except (KeyError, IndexError, TypeError) as e:
            parser.error(f"--prompt: unexpected JSON structure in {json_path}: {e}")
        print(f"Loaded prompt from {json_path}")
        return {"type": "point", "coordinates": [u, v]}

    if len(args_prompt) == 2:
        try:
            u, v = int(args_prompt[0]), int(args_prompt[1])
        except ValueError:
            parser.error("--prompt: expected two integers 'U V' or a path to a .json file")
        return {"type": "point", "coordinates": [u, v]}

    parser.error("--prompt: expected either a .json file path or two integers 'U V'")


def main():
    parser = argparse.ArgumentParser(description="ProxyPose in-the-wild inference")
    parser.add_argument("--gen_config",   type=str, default="configs/generation/default.yaml",
                        help="Path to generation config YAML")
    parser.add_argument("--track_config", type=str, default="configs/tracking/default.yaml",
                        help="Path to tracking config YAML")
    parser.add_argument("--video_path",   type=str, required=True,  help="Path to input video")
    parser.add_argument("--output_path",  type=str, required=True,  help="Output video path")
    parser.add_argument("--device",       type=str, default="cuda", help="Torch device")
    parser.add_argument("--prompt", nargs="+", required=True,
                        metavar="ARG",
                        help="Either a path to a .json query file produced by proxypose-annotate, "
                             "or two integers 'U V' (pixel coordinate of the target in the first frame)")
    parser.add_argument("--depth_anything_path", type=str, default=None,
                        help="Path to a Depth Anything 3 .da3.npz file for focal length estimation. "
                             "If not set, falls back to the default FOV in the tracking config.")
    parser.add_argument("--fps",        type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=49,  help="Number of frames to process")
    parser.add_argument("--seed",       type=int, default=123)
    parser.add_argument("--debug",      type=int, default=0,   help="Skip generation (for testing)")
    args = parser.parse_args()

    gen_config   = load_config(args.gen_config)
    track_config = load_config(args.track_config)
    video = load_video(args.video_path)[:args.num_frames]

    print(f"Video: {len(video)} frames at {video[0].shape[1]}x{video[0].shape[0]}")

    prompt = _parse_prompt(args.prompt, parser)
    print(f"Prompt: {prompt}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve intrinsics: Depth Anything 3 > default FOV
    h, w = video[0].shape[:2]
    intrinsics = None
    if args.depth_anything_path:
        intrinsics = load_intrinsics_from_da3(args.depth_anything_path, video_shape=(h, w))

    pipe = None if args.debug else load_model(gen_config, device=args.device)

    crop_box = get_square_crop_prompt(prompt["coordinates"], resolution=(h, w))
    cropped_video, new_prompt, cropped_intrinsics = crop_video_and_prompt(
        video, prompt, crop_box, intrinsics=intrinsics
    )

    output_frames, _, _, _, _ = full_inference(
        pipe, cropped_video, new_prompt, gen_config, track_config,
        intrinsics=cropped_intrinsics, device=args.device,
        seed=args.seed,
        debug=bool(args.debug),
    )

    save_video(output_frames, output_path, fps=args.fps)
    print(f"Saved {len(output_frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
