import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inference.generate_proxy import load_model, prompt_to_proxy, generate_proxy_video
from inference.track_proxy import track_video, visualize_transform
from inference.utils import load_video, save_video, load_config


def mask_to_prompt(mask):
    """Convert a binary segmentation mask to a point prompt at the mask centroid.

    Args:
        mask: (H, W) uint8 binary mask (non-zero = object).

    Returns:
        Dict with 'type'='point' and 'coordinates'=(u, v) in (col, row) order.
    """
    dist_inside = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_dist_idx = np.unravel_index(np.argmax(dist_inside), dist_inside.shape)
    return {"type": "point", "coordinates": (max_dist_idx[1], max_dist_idx[0])}


def get_square_crop_prompt(prompt_pixel, resolution):
    """Compute the largest centered square crop that keeps `prompt_pixel` inside.

    Args:
        prompt_pixel: (u, v) prompt pixel in (col, row) order.
        resolution:   (H, W) of the full image.

    Returns:
        (x1, y1, x2, y2) crop box in pixel coordinates.
    """
    h, w = resolution
    u, v = prompt_pixel
    side = min(h, w)
    x1 = int(np.clip(u - side // 2, 0, w - side))
    y1 = int(np.clip(v - side // 2, 0, h - side))
    return x1, y1, x1 + side, y1 + side


def adjust_image_crop(image, crop_box, target_resolution=(512, 512)):
    """Crop and resize an image."""
    x1, y1, x2, y2 = crop_box
    return cv2.resize(image[y1:y2, x1:x2], target_resolution, interpolation=cv2.INTER_LINEAR)


def adjust_pixel_crop(prompt_pixel, crop_box, target_resolution=(512, 512)):
    """Map a pixel coordinate from the original image into the cropped/resized image."""
    x1, y1, x2, y2 = crop_box
    u, v = prompt_pixel
    scale_x = target_resolution[0] / (x2 - x1)
    scale_y = target_resolution[1] / (y2 - y1)
    return int((u - x1) * scale_x), int((v - y1) * scale_y)


def adjust_intrinsics_crop(intrinsics, crop_box, target_resolution=(512, 512)):
    """Adjust a camera intrinsic matrix for a crop and resize operation."""
    x1, y1, x2, y2 = crop_box
    K = intrinsics.copy()
    K[0, 2] -= x1
    K[1, 2] -= y1
    scale_x = target_resolution[0] / (x2 - x1)
    scale_y = target_resolution[1] / (y2 - y1)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def make_prompt_panel(vis_img, prompt):
    """Draw the prompt point on a copy of `vis_img` for visualization."""
    panel = vis_img.copy()
    u, v = prompt["coordinates"]
    cv2.circle(panel, (u, v), radius=5, color=(255, 0, 0), thickness=-1)
    return panel


def crop_video_and_prompt(video, prompt, crop_box, intrinsics=None, target_resolution=(512, 512)):
    """Crop and resize a video and adjust the prompt and intrinsics accordingly."""
    cropped_video = [adjust_image_crop(f, crop_box, target_resolution) for f in video]
    new_prompt = {
        "type": prompt["type"],
        "coordinates": adjust_pixel_crop(prompt["coordinates"], crop_box, target_resolution),
    }
    if intrinsics is not None:
        intrinsics = adjust_intrinsics_crop(intrinsics, crop_box, target_resolution)
    return cropped_video, new_prompt, intrinsics


def convert_tracking_results(input_depth, input_transform, rvecs, tvecs):
    """Convert proxy-space tracking results to metric object poses.

    Args:
        input_depth:     Depth of the object at the first frame (same units as GT).
        input_transform: (4, 4) GT camera-to-object transform at the first frame.
        rvecs:           List of tracked rotation vectors.
        tvecs:           List of tracked translation vectors.

    Returns:
        T_camera_object: (N, 4, 4) array of estimated camera-to-object transforms.
    """
    T_camera_proxys = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.squeeze()
        T_camera_proxys.append(T)

    T_camera_proxys = np.stack(T_camera_proxys, axis=0)

    z_ratio = input_depth / T_camera_proxys[0][2, 3]
    T_camera_proxys[:, :3, 3] *= z_ratio

    T_proxy_object = np.linalg.inv(T_camera_proxys[0]) @ input_transform
    T_camera_object = T_camera_proxys @ T_proxy_object[None]

    return T_camera_object


def full_inference(pipe, video, prompt, gen_config, track_config, intrinsics=None, device="cuda", seed=None, debug=False):
    """ProxyPose inference pipeline.

    Args:
        pipe:         Loaded WanVideoPipeline (or None in debug mode).
        video:        List of RGB frames (H, W, 3) uint8.
        prompt:       Dict with 'type' and 'coordinates'.
        gen_config:   Generation config dict.
        track_config: Tracking config dict.
        intrinsics:   Optional (3, 3) numpy intrinsic matrix.
        device:       Torch device string.
        seed:         Optional random seed.
        debug:        If True, skip generation and use the proxy image as a static video.

    Returns:
        output_frames: List of BGR visualization frames (5× width).
        rvecs:         Tracked rotation vectors (one per frame).
        tvecs:         Tracked translation vectors (one per frame).
    """
    first_frame_rgb = video[0]
    h, w = first_frame_rgb.shape[:2]

    prompt_panel = make_prompt_panel(first_frame_rgb, prompt)[..., [2, 1, 0]]

    if intrinsics is None:
        focal_deg = track_config["default_focal_deg"]
        focal_length_px = 0.5 * w / np.tan(np.radians(focal_deg / 2))
        intrinsics = np.array([
            [focal_length_px, 0, w / 2],
            [0, focal_length_px, h / 2],
            [0, 0, 1],
        ])

    proxy_img = prompt_to_proxy(
        prompt, intrinsics, image_size=(h, w),
        proxy_scale=gen_config["proxy_scale"], device=device
    )

    if debug:
        proxy_video = [proxy_img] * len(video)
    else:
        proxy_video = generate_proxy_video(pipe, video, proxy_img, gen_config, device=device, seed=seed)

    proxy_resolution = proxy_video[0].shape[:2]
    proxy_video_intrinsics = adjust_intrinsics_crop(
        intrinsics, crop_box=(0, 0, proxy_resolution[1], proxy_resolution[0]),
        target_resolution=proxy_resolution
    )
    proxy_video_bgr = [frame[..., [2, 1, 0]] for frame in proxy_video]
    rvecs, tvecs, tracking_vis, points_2d_list, points_3d_list = track_video(
        proxy_video_bgr, track_config, intrinsics=proxy_video_intrinsics
    )

    output_frames = []
    for i in range(len(video)):
        panel1 = video[i][..., [2, 1, 0]].copy()
        panel2 = prompt_panel.copy()
        panel3 = proxy_video[i][..., [2, 1, 0]].copy()
        panel4 = video[i][..., [2, 1, 0]].copy()
        if i < len(rvecs) and rvecs[i] is not None:
            panel4 = visualize_transform(panel4, rvecs[i], tvecs[i], intrinsics)
        row = np.concatenate([panel1, panel3, tracking_vis[i], panel4, panel2], axis=1)
        output_frames.append(row)

    return output_frames, rvecs, tvecs, points_2d_list, points_3d_list

TARGET_RESOLUTION = (512, 512)

def main():
    parser = argparse.ArgumentParser(description="Batch benchmark evaluation for ProxyPose")
    parser.add_argument("--gen_config",      type=str, default="configs/generation/default.yaml")
    parser.add_argument("--track_config",    type=str, default="configs/tracking/default.yaml")
    parser.add_argument("--benchmark_path",  type=str, required=True, help="Root path of the benchmark dataset")
    parser.add_argument("--output_path",     type=str, required=True, help="Output directory for results")
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--fps",             type=int, default=24)
    parser.add_argument("--seed",            type=int, default=None)
    parser.add_argument("--debug",           type=int, default=0, help="Skip generation (use proxy image as video)")
    parser.add_argument("--num_attempts",    type=int, default=1, help="Repeated inference attempts per sequence")
    args = parser.parse_args()

    gen_config   = load_config(args.gen_config)
    track_config = load_config(args.track_config)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    input_sequences = sorted(
        seq for seq in Path(args.benchmark_path).glob("*")
        if seq.is_dir() and seq.stem != "meshes"
    )

    pipe = None if args.debug else load_model(gen_config, device=args.device)

    for seq in input_sequences:
        frame_meta_path = seq / "frame_meta.json"
        if not frame_meta_path.is_file():
            print(f"Warning: frame_meta.json not found for {seq.name}, skipping.")
            continue

        frame_meta = json.loads(frame_meta_path.read_text())
        start_id   = frame_meta["top_windows"][0]["anchor_frame"]
        end_id     = frame_meta["top_windows"][0]["end_frame"]

        print(f"\nProcessing sequence: {seq.name}")

        mask_files = sorted((seq / "mask").glob("*_mask.png"))
        mask       = cv2.imread(str(mask_files[start_id]), cv2.IMREAD_UNCHANGED)
        prompt     = mask_to_prompt(mask)
        crop_box   = get_square_crop_prompt(prompt["coordinates"], resolution=mask.shape[:2])

        scene_meta           = json.loads((seq / "scene_meta.json").read_text())
        scene_object_meta    = scene_meta["objects"][str(frame_meta["reference_obj_id"])]
        original_intrinsics  = np.array(scene_meta["K"])

        depth_files  = sorted((seq / "depth").glob("*.png"))
        depth        = cv2.imread(str(depth_files[start_id]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth       *= scene_meta["depth_scale_to_meters"]
        prompt_depth = depth[prompt["coordinates"][1], prompt["coordinates"][0]]

        object_meta_files    = sorted((seq / "meta").glob("*.json"))
        object_transform_meta = json.loads(object_meta_files[start_id].read_text())["objects"][str(frame_meta["reference_obj_id"])]
        T_camera_object_gt   = np.eye(4)
        T_camera_object_gt[:3, :3] = np.array(object_transform_meta["R"])
        T_camera_object_gt[:3, 3]  = np.array(object_transform_meta["t"])

        original_video = load_video(seq / "color")[start_id:end_id + 1]
        cropped_video, new_prompt, new_intrinsics = crop_video_and_prompt(
            original_video, prompt, crop_box,
            intrinsics=original_intrinsics, target_resolution=TARGET_RESOLUTION
        )

        print(f"  Frames: {len(cropped_video)}, prompt: {new_prompt}")

        seq_output_path = output_path / seq.name
        seq_output_path.mkdir(exist_ok=True)

        for attempt_id in range(args.num_attempts):
            print(f"\n  === Attempt {attempt_id + 1}/{args.num_attempts} ===")

            output_frames, rvecs, tvecs, _, _ = full_inference(
                pipe, cropped_video, new_prompt, gen_config, track_config,
                intrinsics=new_intrinsics, device=args.device,
                seed=(args.seed + attempt_id) if args.seed is not None else None,
                debug=bool(args.debug),
            )

            pose_result = convert_tracking_results(
                input_depth=prompt_depth,
                input_transform=T_camera_object_gt,
                rvecs=rvecs,
                tvecs=tvecs,
            )

            attempt_json_path = output_path / f"ours_results_attempt_{attempt_id + 1:02d}.json"
            existing = json.loads(attempt_json_path.read_text()) if attempt_json_path.exists() else []
            for frame_id, T in enumerate(pose_result):
                existing.append({
                    "scene_name":    scene_meta["scene_name"],
                    "dataset":       scene_meta["dataset"],
                    "obj_id":        frame_meta["reference_obj_id"],
                    "object_class":  scene_object_meta["class_name"],
                    "frame_idx":     start_id + frame_id,
                    "pose_4x4":      T.tolist(),
                    "time_sec":      0.0,
                })
            attempt_json_path.write_text(json.dumps(existing, indent=4))

            attempt_video_path = seq_output_path / f"tracking_{attempt_id + 1:02d}.mp4"
            save_video(output_frames, attempt_video_path, fps=args.fps)
            print(f"  Saved {len(output_frames)} frames to {attempt_video_path}")


if __name__ == "__main__":
    main()
