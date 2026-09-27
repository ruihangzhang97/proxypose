
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mask_to_prompt(mask, top_percentile=0.5, border_ratio=0.15):
    # mask is binary mask 0 and 1

    # get border pixel height and width in pixels
    border_height = int(mask.shape[0] * border_ratio)
    border_width = int(mask.shape[1] * border_ratio)

    # crop mask to remove border pixels
    cropped_mask = mask[border_height:-border_height, border_width:-border_width]

    # make signed distance field
    dist_inside = cv2.distanceTransform((cropped_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # find maximum distance pixel coordinate
    valid_pixel_ids = np.where(dist_inside > 0)

    # sort valid pixels by distance and take top percentile
    sorted_indices = np.argsort(dist_inside[valid_pixel_ids])[::-1]
    num_top_pixels = int(len(sorted_indices) * top_percentile)
    top_pixel_indices = sorted_indices[:num_top_pixels]

    # randomly select one of the top pixels
    selected_pixel_id = np.random.choice(top_pixel_indices)

    selected_pixel = valid_pixel_ids[0][selected_pixel_id], valid_pixel_ids[1][selected_pixel_id]  # (v, u) in (y, x) order

    # adjust for cropping
    selected_pixel = (selected_pixel[0] + border_height, selected_pixel[1] + border_width)

    return int(selected_pixel[1]), int(selected_pixel[0])


def prompt_pixel_to_3d_coordinate(prompt_pixel, depth, intrinsics):
    u, v = prompt_pixel
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z_cam = depth[v, u]
    x_ndc = (u - cx) / fx
    y_ndc = (v - cy) / fy
    x_cam = x_ndc * z_cam
    y_cam = y_ndc * z_cam
    return np.array([x_cam, y_cam, z_cam])


def main():
    parser = argparse.ArgumentParser(
        description="Sample prompt pixels on the anchor frame of each scene and write <scene>/prompt_meta.json."
    )
    parser.add_argument("--benchmark_path", type=str, required=True, help="Root of the benchmark in uniform format")
    parser.add_argument("--distance_percentile", type=float, default=0.25, help="Ratio of pixels to select based on distance to border")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode")

    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path)

    for seq_id, seq_dir in enumerate(sorted(list(benchmark_path.glob("*")))):
        if not seq_dir.is_dir():
            continue

        frame_meta_path = seq_dir / "frame_meta.json"

        if not frame_meta_path.exists():
            continue

        print(f"Processing {seq_dir.name}...")

        frame_meta = json.load(open(frame_meta_path, "r"))
        start_id = frame_meta["top_windows"][0]["anchor_frame"]
        
        mask_files = sorted((seq_dir / "mask").glob("*_mask.png"))
        mask = cv2.imread(str(mask_files[start_id]), cv2.IMREAD_UNCHANGED)

        # select random pixels
        np.random.seed(args.seed + seq_id)
        try:
            prompt_1 = mask_to_prompt(mask, top_percentile=args.distance_percentile)
            prompt_2 = mask_to_prompt(mask, top_percentile=args.distance_percentile)
            prompt_3 = mask_to_prompt(mask, top_percentile=args.distance_percentile)
        except Exception as e:
            print(f"Error processing {seq_dir.name}: {e}")
            continue

        prompts = [prompt_1, prompt_2, prompt_3]
        depths = []
        t_prompts = []
        t_object_prompts = []

        ### Extract intrinsics from scene_meta.json and adjust for cropping and target resolution
        scene_meta_path = seq_dir / "scene_meta.json"
        scene_meta = json.load(open(scene_meta_path, "r"))
        scene_object_meta = scene_meta["objects"][str(frame_meta["reference_obj_id"])]
        intrinsics = np.array(scene_meta["K"])

        ### Load depth at prompt pixel
        depth_files = sorted((seq_dir / "depth").glob("*.png"))
        depth = cv2.imread(str(depth_files[start_id]), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) * scene_meta["depth_scale_to_meters"]  # convert depth

        # Load GT object transform
        object_meta_files = sorted((seq_dir / "meta").glob("*.json"))
        object_meta = json.load(open(object_meta_files[start_id], "r"))
        object_transform_meta = object_meta["objects"][str(frame_meta["reference_obj_id"])]
        T_camera_object_gt = np.eye(4)
        T_camera_object_gt[:3, :3] = np.array(object_transform_meta["R"])
        T_camera_object_gt[:3, 3] = np.array(object_transform_meta["t"])

        for prompt in prompts:
            prompt_depth = depth[prompt[1], prompt[0]]  # depth at the prompt pixel
            
            # get world coordinate from pixel coordinate using intrinsics and depth
            t_prompt = prompt_pixel_to_3d_coordinate(prompt, depth, intrinsics)

            T_object_camera = np.linalg.inv(T_camera_object_gt)
            t_object_prompt = T_object_camera[:3, :3] @ t_prompt + T_object_camera[:3, 3]

            depths.append(prompt_depth.tolist())
            t_prompts.append(t_prompt.tolist())
            t_object_prompts.append(t_object_prompt.tolist())

        output_json = {
            "pixel_coordinates": prompts,
            "depths": depths,
            "3d_coordinates": t_prompts,
            "object_coordinates": t_object_prompts,
        }

        with open(seq_dir / "prompt_meta.json", "w") as f:
            json.dump(output_json, f, indent=4)

        if args.debug:
            color_files = sorted((seq_dir / "color").glob("*.png"))
            vis_img = cv2.imread(str(color_files[start_id]))
            cv2.circle(vis_img, prompt_1, radius=10, color=(0, 0, 255), thickness=-1)  # red dot in BGR
            cv2.imwrite(str(f"prompt_pixel_{seq_dir.name}.png"), vis_img)


if __name__ == "__main__":
    main()

