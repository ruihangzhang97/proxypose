#!/usr/bin/env python3
"""Render the ground-truth colored-cube "proxy" video for a data_generation clip.

Given the output of one data_generation render (``0000.meta.json`` with per-frame
camera/object 6-DoF poses, ``depth/*.exr``, and ``instance_maps/*.png``), this picks a
random on-object pixel to seed the proxy's position/scale, then renders the full-trajectory
colored cube so it tracks the object exactly (using the ground-truth poses, not a trained
model). The result is the paired proxy video used to train ProxyPose's video-to-video model,
alongside the corresponding ``0000.rgb.mp4``.

Requires the ``pytorch3d``/``torch``/``opencv-python`` environment described in this
directory's README, and a system ``ffmpeg`` binary on PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesAtlas,
    AmbientLights,
    HardPhongShader,
    BlendParams,
)


def _open_ffmpeg_rgb_pipe(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    codec: str = "libx264",
) -> subprocess.Popen:
    """Stream raw RGB24 frames to H.264/HEVC via ffmpeg (yuv420p output)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


class ColoredCube:
    """A unit cube mesh with a distinct solid color on each face."""

    def __init__(self, size=1.0, device="cuda"):
        self.size = float(size)
        self.device = device
        self.face_colors = {
            "front": torch.tensor((255, 255, 255), device=device) / 255.,  # white
            "back": torch.tensor((0, 255, 255), device=device) / 255.,  # yellow
            "left": torch.tensor((230, 230, 0), device=device) / 255.,  # cyan
            "right": torch.tensor((0, 240, 75), device=device) / 255.,  # green
            "bottom": torch.tensor((255, 30, 30), device=device) / 255.,  # blue
            "top": torch.tensor((10, 10, 255), device=device) / 255.,  # red
        }
        self.mesh = self._create_colored_cube_mesh()

    def _create_colored_cube_mesh(self) -> Meshes:
        s = self.size / 2.0
        vertices = torch.tensor([
                [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
                [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
            ], dtype=torch.float32, device=self.device)
        faces = torch.tensor([
                [0, 1, 2], [0, 2, 3], [5, 4, 7], [5, 7, 6],
                [4, 0, 3], [4, 3, 7], [1, 5, 6], [1, 6, 2],
                [4, 5, 1], [4, 1, 0], [3, 2, 6], [3, 6, 7],
            ], dtype=torch.int64, device=self.device)
        atlas_size = 1
        atlas = torch.zeros((1, 12, atlas_size, atlas_size, 3), device=self.device)
        face_colors_list = [self.face_colors["front"]] * 2 + [self.face_colors["back"]] * 2 + \
                          [self.face_colors["left"]] * 2 + [self.face_colors["right"]] * 2 + \
                          [self.face_colors["bottom"]] * 2 + [self.face_colors["top"]] * 2
        for tri_idx in range(12):
            atlas[0, tri_idx, :, :, :] = face_colors_list[tri_idx]
        textures = TexturesAtlas(atlas=atlas)
        return Meshes(verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures)


def decompose_with_qr(matrix):
    """Use QR decomposition for robust scale/rotation extraction from a 4x4 transform."""
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.float64)

    RS = matrix[:3, :3].astype(np.float64)
    t = matrix[:3, 3].astype(np.float64)

    Q, R_upper = np.linalg.qr(RS)
    scale = np.abs(np.diag(R_upper))
    signs = np.sign(np.diag(R_upper))
    rotation = Q @ np.diag(signs)

    return scale, rotation, t


def load_frames(dir_path, pattern="*.png"):
    frame_paths = sorted(list(dir_path.glob(pattern)))
    frames = [cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED) for frame_path in frame_paths]
    return frames


def check_num_objects(json_path):
    with open(json_path) as f:
        data = json.load(f)
    num_objects = len(data["frames"][0]["object_transforms"])
    print(f"{json_path} contains {num_objects} objects.")
    return num_objects


def load_trajectory(json_path, object_id=1):
    """Load per-frame camera + object transforms from a data_generation meta.json.

    Converts Blender's (X, Y, Z) world convention to PyTorch3D's (X, Z, -Y), and negates
    the camera's local X/Z axes to go from Blender's camera convention (-Z forward) to
    PyTorch3D's (+Z forward, row-vector) convention.
    """
    print(f"Loading from {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    frames = data['frames']
    num_frames = len(frames)
    fov_deg = np.degrees(data.get('camera_angle_x', 0.785398))

    print(f"  Frames: {num_frames}, FOV: {fov_deg:.2f} deg")

    # Blender (X, Y, Z) -> PyTorch3D (X, Z, -Y)
    R_coords = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    T_coords = np.eye(4, dtype=np.float64)
    T_coords[:3, :3] = R_coords

    rotation_matrices_list = []
    translations_list = []
    scales_list = []
    camera_transforms_list = []

    for frame in frames:
        obj_transform = None
        for obj in frame.get('object_transforms', []):
            if obj['object_id'] == object_id:
                obj_transform = obj['transform_matrix']
                break

        if obj_transform is None:
            raise ValueError(f"No transform for object_id={object_id}")

        cam_transform = np.array(frame['transform_matrix'], dtype=np.float64)
        cam_transform = T_coords @ cam_transform
        cam_transform[:, 0] = -cam_transform[:, 0]  # negate X axis to fix handedness
        cam_transform[:, 2] = -cam_transform[:, 2]  # negate Z axis to fix handedness
        camera_transforms_list.append(cam_transform.astype(np.float32))

        scale, R_blender, t_blender = decompose_with_qr(obj_transform)

        R_pytorch3d = (R_coords @ R_blender @ R_coords.T).astype(np.float32)
        t_pytorch3d = (R_coords @ t_blender).astype(np.float32)

        rotation_matrices_list.append(R_pytorch3d)
        translations_list.append(t_pytorch3d)
        scales_list.append(scale.astype(np.float32))

    cam_transforms = np.array(camera_transforms_list)
    rotation_matrices = np.array(rotation_matrices_list)
    translations = np.array(translations_list)
    scales = np.array(scales_list)

    print(
        "  Position range: "
        f"X=[{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}], "
        f"Y=[{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}], "
        f"Z=[{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]"
    )

    return rotation_matrices, translations, scales, cam_transforms, fov_deg, num_frames


def get_first_instance_map_pixel(instance_map_dir, object_id, n_border_px, target_num_frames, all_object_ids=None):
    """Find the first frame where the object is at least partially within the border, and pick a pixel on it."""
    if object_id == -1:
        # Background: invert the union of all instance maps to get valid (non-object) pixels.
        assert all_object_ids is not None
        instance_map_list = []
        for oid in all_object_ids:
            instance_maps = load_frames(instance_map_dir, pattern=f"{oid:04d}.*.png")
            instance_map_list.append(np.stack(instance_maps))
        instance_maps = np.stack(instance_map_list)
        instance_maps = instance_maps.astype(np.int64).sum(axis=0).clip(0, 255).astype(np.uint8)
        instance_maps = 255 - instance_maps
    else:
        instance_maps = load_frames(instance_map_dir, pattern=f"{object_id:04d}.*.png")

    min_start_id = None
    for frame_id in range(len(instance_maps)):
        instance_map = instance_maps[frame_id] / 255.
        h, w = instance_map.shape
        valid_frame = instance_map[n_border_px:h - n_border_px, n_border_px:w - n_border_px]
        if (valid_frame > 0.5).any():
            min_start_id = frame_id
            break

    assert min_start_id is not None, (
        f"No valid frame found for object_id={object_id} with given proxy scale and image size "
        "(proxy cube too large or object too close to border)."
    )

    max_start_id = len(instance_maps) - target_num_frames
    start_id = np.random.randint(min_start_id, max(max_start_id + 1, min_start_id + 1))

    instance_map = instance_maps[start_id] / 255.
    h, w = instance_map.shape
    valid_frame = instance_map[n_border_px:h - n_border_px, n_border_px:w - n_border_px]
    valid_pixels = np.argwhere(valid_frame > 0.5)
    start_pixel = valid_pixels[np.random.randint(len(valid_pixels))] + n_border_px

    return start_id, start_pixel  # (y, x)


def camera_space_point_to_proxy(fov_deg, image_size, pixel_xy, depth, proxy_scale=0.1):
    """Convert a 2D pixel + depth to a 3D proxy cube transform in camera space.

    Places the cube at the 3D point corresponding to the given pixel, oriented so a
    corner points toward the camera (for easy color visibility), and scaled so its
    on-screen size stays consistent across depths.
    """
    fov_rad = np.radians(fov_deg)
    focal_length = (image_size / 2) / np.tan(fov_rad / 2)

    # Convert pixel (x=col, y=row) to camera-space NDC; y is negated because image rows
    # increase downward while camera y increases upward.
    x_ndc = -(pixel_xy[0] - image_size / 2) / focal_length
    y_ndc = -(pixel_xy[1] - image_size / 2) / focal_length

    # Camera-space point (PyTorch3D convention: +z points into the scene).
    z_cam = depth
    x_cam = x_ndc * z_cam
    y_cam = y_ndc * z_cam
    t_obj = np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    scale = z_cam * proxy_scale * (image_size / focal_length)

    # Orient the cube by aligning its z axis with the camera ray, y axis with world up.
    z_dir = t_obj / np.linalg.norm(t_obj)
    y_dir = np.array([0, 1, 0], dtype=np.float32)
    y_dir = y_dir - np.dot(y_dir, z_dir) * z_dir
    y_dir = y_dir / np.linalg.norm(y_dir)
    x_dir = np.cross(y_dir, z_dir)
    R_proxy = np.stack([x_dir, y_dir, z_dir], axis=1)

    # Rotate 45 deg around y then tilt around x so a cube corner faces the camera.
    angle_y = np.radians(45)
    rot_y = np.array([
        [ np.cos(angle_y), 0, np.sin(angle_y)],
        [             0, 1,             0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ], dtype=np.float32)
    angle_x = -np.arctan(1 / np.sqrt(2))
    rot_x = np.array([
        [1,              0,             0],
        [0,  np.cos(angle_x), -np.sin(angle_x)],
        [0,  np.sin(angle_x),  np.cos(angle_x)],
    ], dtype=np.float32)

    R_proxy = R_proxy @ rot_x @ rot_y

    T_proxy = np.eye(4, dtype=np.float32)
    T_proxy[:3, :3] = R_proxy
    T_proxy[:3, 3] = t_obj

    return T_proxy, scale


class CubeTrajectoryRenderer:
    """Renders and encodes the proxy cube for a full trajectory of camera/object poses."""

    def __init__(self, output_path, rotation_matrices, translations, scales, cam_rotations, cam_translations, fov,
                 image_size=512, fps=24, device="cuda", supersample=2, cube_size=1.0,
                 cube_scale_multiplier=1.0,
                 object_space_rotation=None,
                 object_space_translation=None,
                 video_crf=15,
                 video_codec="libx264"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_size, self.fps, self.fov, self.device = int(image_size), int(fps), float(fov), device
        self.video_crf = int(video_crf)
        self.video_codec = str(video_codec)
        self.num_frames = len(rotation_matrices)
        self.supersample = max(1, int(supersample))
        self.render_size = self.image_size * self.supersample

        self.rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)
        self.translations = torch.tensor(translations, dtype=torch.float32)

        cam_rotations_torch = torch.tensor(cam_rotations, dtype=torch.float32, device=device)
        cam_translations_torch = torch.tensor(cam_translations, dtype=torch.float32, device=device)

        self.R_cam = cam_rotations_torch.permute(0, 2, 1)  # PyTorch3D expects (R @ X) for point transformation
        self.t_cam = cam_translations_torch

        sm = float(scales.mean()) * float(cube_scale_multiplier)
        full_edge = float(cube_size) * sm
        self.cube = ColoredCube(size=full_edge, device=device)

        if object_space_rotation is not None:
            q = np.asarray(object_space_rotation, dtype=np.float32)
            self.R_obj = torch.tensor(q, dtype=torch.float32, device=device)
        else:
            self.R_obj = None

        if object_space_translation is not None:
            t = np.asarray(object_space_translation, dtype=np.float32).reshape(3)
            self.t_obj = torch.tensor(t, dtype=torch.float32, device=device)
        else:
            self.t_obj = None

        self.renderer = self._setup_renderer()

    def _setup_renderer(self):
        raster_settings = RasterizationSettings(
            image_size=self.render_size, blur_radius=0.0, faces_per_pixel=1,
            perspective_correct=True, cull_backfaces=False)
        lights = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=self.device, lights=lights, blend_params=blend_params))

    @torch.no_grad()
    def render_frame(self, rotation_matrix, translation, r_cam, t_cam):
        rotation_matrix = rotation_matrix.to(self.device)
        translation = translation.to(self.device)

        cameras = FoVPerspectiveCameras(device=self.device, R=r_cam[None], T=t_cam[None], fov=self.fov)

        original_verts = self.cube.mesh.verts_packed()
        verts = original_verts
        if self.R_obj is not None:
            verts = verts @ self.R_obj.T
        if self.t_obj is not None:
            verts = verts + self.t_obj

        transformed_verts = (verts @ rotation_matrix.T) + translation
        transformed_mesh = self.cube.mesh.update_padded(transformed_verts.unsqueeze(0))

        images = self.renderer(transformed_mesh, cameras=cameras)
        rgb = images[0, ..., :3].clamp(0, 1)

        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

        if self.supersample > 1:
            rgb_np = cv2.resize(rgb_np, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        return bgr_np

    def generate_video(self):
        print(f"\nGenerating {self.num_frames} frames (ffmpeg {self.video_codec}, CRF {self.video_crf})...\n")

        frames: list[np.ndarray] = []
        for frame_idx in tqdm(range(self.num_frames), desc="Rendering"):
            bgr = self.render_frame(
                self.rotation_matrices[frame_idx],
                self.translations[frame_idx],
                self.R_cam[frame_idx],
                self.t_cam[frame_idx]
            )
            frames.append(np.ascontiguousarray(bgr))

        return frames

    def save_video(self, frames):
        w, h = self.image_size, self.image_size

        proc = _open_ffmpeg_rgb_pipe(self.output_path, w, h, self.fps, self.video_crf, self.video_codec)
        try:
            for fr in frames:
                proc.stdin.write(fr.tobytes())
        finally:
            if proc.stdin:
                proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg failed (exit {ret}) encoding proxy video: {err.strip() or '(no stderr)'}")

        print(f"\n✓ Video: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render the ground-truth proxy cube video for a data_generation clip, "
        "given its 0000.meta.json, depth/, and instance_maps/."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Clip directory containing 0000.meta.json, 0000.rgb.mp4, depth/, and instance_maps/ "
        "(the output of one data_generation render).",
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=4,
        help="Maximum number of proxy videos rendered. Objects are selected at random.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the output proxy video(s) (default: same as --input-dir).",
    )
    parser.add_argument("--proxy-scale", type=float, default=0.3, help="Proxy scale as ratio of image-size")
    parser.add_argument("--proxy-scale-deviation", type=float, default=0.3, help="Random proxy scale deviation as ratio of original proxy")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--has-camera-motion", type=int, default=0, help="Flag indicating if the camera has motion")
    parser.add_argument("--target-num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' (default; recommended — rendering a 12-triangle cube "
                        "doesn't need a GPU) or 'cuda' if you'd rather use one.")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0, help="Annotate the prompt pixel on the debug side-by-side video")
    parser.add_argument("--cube-size", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for object/pixel selection (omit for nondeterministic).",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=15,
        help="ffmpeg/libx264 (or chosen --video-codec) CRF; lower = higher quality/larger files.",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="ffmpeg video codec (e.g. libx264, libx265).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    traj_path = input_dir / "0000.meta.json"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(exist_ok=True)

    num_objects = check_num_objects(traj_path)
    np.random.seed(args.seed)

    selected_object_ids = np.random.choice(
        list(range(num_objects)), size=min(num_objects, args.max_num_objects), replace=False)

    if args.has_camera_motion:
        selected_object_ids = np.append(selected_object_ids, -1)  # add a background video for camera-motion cases

    print("selected objects", selected_object_ids, "for proxy video generation")

    print("loading depth maps")
    depth_maps = load_frames(input_dir / "depth", pattern="*.exr")
    assert len(depth_maps) > 0

    pixel_proxy_scale = args.proxy_scale * (1.0 + args.proxy_scale_deviation * (2 * np.random.random() - 1))
    n_border_px = int(args.image_size * 0.5 * pixel_proxy_scale)

    for i, object_id in enumerate(selected_object_ids):
        if object_id == -1:
            rotation_matrices, translations, scales, cam_transforms, fov_deg, num_frames = load_trajectory(
                str(traj_path), 1)
            rotation_matrices = np.eye(3, dtype=np.float32)[None].repeat(num_frames, axis=0)
            translations = np.zeros((num_frames, 3), dtype=np.float32)
        else:
            rotation_matrices, translations, scales, cam_transforms, fov_deg, num_frames = load_trajectory(
                str(traj_path), object_id + 1)

        # 1. Pick a start frame + pixel for the proxy cube from the instance masks (simulates a user click).
        start_id, start_pixel = get_first_instance_map_pixel(
            input_dir / "instance_maps", object_id, n_border_px, args.target_num_frames,
            all_object_ids=list(range(num_objects)))

        print(f"\nObject {object_id}: start_id={start_id}, start_pixel={start_pixel.tolist()} "
              f"(border {n_border_px}px), proxy_scale={pixel_proxy_scale:.3f} (deviation {args.proxy_scale_deviation:.3f})")

        # 2. Use depth + FOV to place a proxy cube at that pixel, sized to roughly match the object
        #    at the start frame.
        depth = depth_maps[start_id][start_pixel[0], start_pixel[1]]
        T_cam_proxy, proxy_scale = camera_space_point_to_proxy(
            fov_deg, args.image_size, start_pixel[::-1], depth, pixel_proxy_scale)

        print(f"  Estimated object-space position for proxy cube at start frame: {T_cam_proxy[:3, 3].tolist()}")

        # 3. Express that offset in the object's local frame at the start frame, so it can be applied
        #    rigidly across the whole trajectory (the cube then "sticks" to the object as it moves).
        T_cams = torch.from_numpy(cam_transforms).float().inverse()
        R_cams = T_cams[:, :3, :3]
        t_cams = T_cams[:, :3, 3]

        T_cam_world = T_cams[start_id]

        T_world_object = torch.eye(4, dtype=torch.float32)
        T_world_object[:3, :3] = torch.from_numpy(rotation_matrices[start_id])
        T_world_object[:3, 3] = torch.from_numpy(translations[start_id])

        T_world_proxy = T_cam_world.inverse() @ torch.from_numpy(T_cam_proxy)
        T_object_proxy = T_world_object.inverse() @ T_world_proxy

        all_T_world_object = torch.eye(4, dtype=torch.float32)[None].repeat(num_frames, 1, 1)
        all_T_world_object[:, :3, :3] = torch.from_numpy(rotation_matrices)
        all_T_world_object[:, :3, 3] = torch.from_numpy(translations)
        all_T_cam_proxy = T_cams @ all_T_world_object @ T_world_proxy

        # TODO: set the cube scale from depth/FOV directly instead of this fixed multiplier.
        cube_scale = 0.5
        scales[:] = proxy_scale

        renderer = CubeTrajectoryRenderer(
            output_dir / f"0000.proxy.{object_id + 1:04d}.mp4",
            rotation_matrices,
            translations,
            scales,
            R_cams,
            t_cams,
            fov_deg,
            args.image_size,
            args.fps,
            args.device,
            args.supersample,
            args.cube_size,
            cube_scale_multiplier=cube_scale,
            object_space_rotation=T_object_proxy[:3, :3],
            object_space_translation=T_object_proxy[:3, 3],
            video_crf=args.video_crf,
            video_codec=args.video_codec,
        )
        frames = renderer.generate_video()

        if args.debug:
            for frame_id, frame in enumerate(frames):
                if frame_id == start_id:
                    cv2.circle(frame, tuple(start_pixel[::-1]), 5, (0, 0, 255), -1)

        renderer.save_video(frames)

        meta = {
            "prompt_frame_id": int(start_id),
            "prompt_pixel": start_pixel.tolist(),
            "proxy_scale": float(pixel_proxy_scale),
            "T_cam_proxy": all_T_cam_proxy.tolist(),
        }
        meta_path = output_dir / f"0000.proxy.{object_id + 1:04d}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"\n✓ Proxy video meta: {meta_path}")

        if i == 0 or args.debug:
            # Side-by-side debug video of the source RGB and the rendered proxy.
            debug_path = output_dir / f"0000.proxy_debug.{object_id + 1:04d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "quiet",
                "-i", str(input_dir / "0000.rgb.mp4"),
                "-i", str(output_dir / f"0000.proxy.{object_id + 1:04d}.mp4"),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2",
                "-crf", "35",  # this is just for debugging, so favor small/fast over quality
                str(debug_path),
            ]
            subprocess.run(cmd, check=True)
            print(f"\n✓ Debug video (source | proxy side by side): {debug_path}")


if __name__ == "__main__":
    main()
