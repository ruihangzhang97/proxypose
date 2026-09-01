#!/usr/bin/env python3

from re import I
import bpy
import bmesh
import math
import random
import json
import os
import shutil
import sys
import logging
import time
import argparse
import glob
import numpy as np
import copy
import torch
import imageio
import imageio.v3 as iio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from mathutils import Vector, Matrix, Quaternion
from pathlib import Path
from shapely.geometry import Polygon
from utils import blender_utils, render_utils, image_utils
from omegaconf import OmegaConf, DictConfig
from types import SimpleNamespace

DEPTH_MAX = 1000.0

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s - %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)

# Create a logger
logger = logging.getLogger(__name__)


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # If seed is None, set seed using current time and process id for randomness
        seed_val = (int(time.time() * 1000) + os.getpid()) % (2**32)
        random.seed(seed_val)
        np.random.seed(seed_val)

def check_msh_bbox(msh):
    vmin, vmax = msh.aabb
    bbox = (vmax - vmin) # [x, y, z]
    
    # Filter objects that are too large (before rescaling)
    # This catches buildings, large structures, etc.
    max_dimension = max(bbox)
    if max_dimension > 5.0:  # Reject objects larger than 5 units in any dimension
        return False
    
    hori_ratio = bbox[0] / bbox[2]
    if hori_ratio > 6 or hori_ratio < 1/6:
        return False
    else:
        vert_ratio = (bbox[1] / bbox[0], bbox[1] / bbox[2])
        if max(vert_ratio) < 0.15: # too flat
            return False
        if min(vert_ratio) > 8: # too tall
            return False
    return True

def post_process_rendering(output_dir, feature_fmt='jpg', dump_video=False, video_fps=24):
    blender_passes = ['rgb', 'normal', 'depth', 'albedo', 'orm']
    mask = 0
    for p in blender_passes:
        img_list = sorted(glob.glob(os.path.join(output_dir, f'{p}.*')))
        if p == 'normal' and len(img_list) > 0:
            meta_file = json.load(open(os.path.join(output_dir, f'0000.meta.json')))
            meta_frames = meta_file['frames']
        for img_path in img_list:
            img_basename = os.path.basename(img_path)
            img_name_part = img_basename.split('.')
            pidx, fidx, img_fmt = int(img_name_part[1]), int(img_name_part[2]) - 1, img_name_part[3]
            img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{img_fmt}'
            if p == 'normal':
                w_normal = image_utils.read_normal_exr(img_path)[..., :3] # [H, W, 3]
                mask = (w_normal == 0).all(axis=-1, keepdims=True) # [H, W, 1]
                bg_normal = np.array([0, 0, 1])
                c2w = np.array(meta_frames[fidx]['transform_matrix'])
                w2c_rot = np.linalg.inv(c2w[:3, :3])
                s_normal = w_normal @ w2c_rot.T
                s_normal = s_normal * (1-mask) + mask * bg_normal
                s_normal = (s_normal + 1) * 0.5
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                image_utils.save_image(os.path.join(output_dir, img_new_name), s_normal)
                # remove the original normal
                os.remove(img_path)
            elif p == 'depth':
                depth = image_utils.read_depth_exr(img_path)
                depth_dir = os.path.join(output_dir, 'depth')
                os.makedirs(depth_dir, exist_ok=True)
                depth_name = f'{fidx:04d}.depth.exr'
                iio.imwrite(os.path.join(depth_dir, depth_name), depth, plugin='opencv')
                os.remove(img_path)
            elif p == 'albedo':
                albedo = image_utils.read_img(img_path)
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                # NOTE: albedo is in sRGB!!!
                albedo = render_utils.rgb_to_srgb(albedo)
                image_utils.save_image(os.path.join(output_dir, img_new_name), albedo)
                os.remove(img_path)
            elif p == 'orm':
                orm = image_utils.read_img(img_path)
                roughness, metallic = orm[..., 1:2], orm[..., 2:3]
                for key, value, bg_color in zip(['roughness', 'metallic'], [roughness, metallic], [0.5, 0]):
                    img_new_name = f'{pidx:04d}.{fidx:04d}.{key}.{feature_fmt}'
                    value = value * (1-mask) + mask * bg_color
                    image_utils.save_image(os.path.join(output_dir, img_new_name), value[..., 0])
                os.remove(img_path)
            else:
                # os.rename(img_path, os.path.join(output_dir, img_new_name))
                shutil.move(img_path, os.path.join(output_dir, img_new_name))

    # Reorganize instance masks: instance_XXX.PPPP.FFFF.png -> instance_maps/XXXX.FFFF.instance.png
    instance_files = sorted(glob.glob(os.path.join(output_dir, 'instance_*.png')))
    if instance_files:
        instance_dir = os.path.join(output_dir, 'instance_maps')
        os.makedirs(instance_dir, exist_ok=True)
        for fp in instance_files:
            base = os.path.basename(fp)
            parts = base.split('.')
            obj_idx_str = parts[0].replace('instance_', '')
            obj_idx = int(obj_idx_str)
            frame_num = int(parts[2]) - 1
            new_name = f'{obj_idx:04d}.{frame_num:04d}.instance.png'
            shutil.move(fp, os.path.join(instance_dir, new_name))
        logger.info(f"Moved {len(instance_files)} instance masks to {instance_dir}")

    # Optionally dump videos from RGB frames grouped by lighting index
    if dump_video:
        # Accept any extension for RGB frames (png/jpg)
        rgb_list = sorted(glob.glob(os.path.join(output_dir, f'*.rgb.*')))
        # Group frames by lighting index (first token)
        group_dict = {}
        for fp in rgb_list:
            base = os.path.basename(fp)
            parts = base.split('.')
            if len(parts) < 4:
                continue
            pidx_str, fidx_str = parts[0], parts[1]
            try:
                pidx = int(pidx_str)
                fidx = int(fidx_str)
            except Exception:
                continue
            group = group_dict.setdefault(pidx, [])
            group.append((fidx, fp))

        for pidx, frames in group_dict.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) < 2:
                continue

            out_path = os.path.join(output_dir, f"{pidx:04d}.rgb.mp4")
            with imageio.get_writer(out_path, fps=float(video_fps)) as writer:
                for _, frame_path in frames:
                    frame = iio.imread(frame_path)
                    if frame is None:
                        continue
                    # Convert float images to uint8
                    if frame.dtype.kind == 'f':
                        frame = np.clip(frame, 0.0, 1.0)
                        frame = (frame * 255.0 + 0.5).astype(np.uint8)
                    elif frame.dtype != np.uint8:
                        # Best-effort cast
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    # Ensure color (H,W,3)
                    if frame.ndim == 2:
                        frame = np.repeat(frame[..., None], 3, axis=-1)
                    writer.append_data(frame)
            logger.info(f"Wrote video: {out_path}")
        
        # Generate masked videos if mask files exist
        logger.info("Checking for mask files to generate masked videos...")
        for pidx, frames in group_dict.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) < 2:
                continue
            
            # Check if mask files exist for this lighting index
            mask_pattern = os.path.join(output_dir, f'mask.{pidx:04d}.*.jpg')
            mask_list = sorted(glob.glob(mask_pattern))
            
            if mask_list and len(mask_list) == len(frames):
                logger.info(f"Generating masked video for lighting {pidx} from {len(mask_list)} masks...")
                masked_video_path = os.path.join(output_dir, f"{pidx:04d}.masked.mp4")
                
                with imageio.get_writer(masked_video_path, fps=float(video_fps)) as writer:
                    for (_, rgb_path), mask_path in zip(frames, mask_list):
                        try:
                            # Load RGB frame
                            rgb_frame = iio.imread(rgb_path)
                            
                            # Load mask (grayscale)
                            mask_frame = iio.imread(mask_path)
                            
                            # Normalize to 0-1 range
                            if rgb_frame.dtype != np.uint8:
                                rgb_frame = np.clip(rgb_frame, 0.0, 1.0)
                                rgb_frame = (rgb_frame * 255.0).astype(np.uint8)
                            
                            if mask_frame.dtype != np.uint8:
                                mask_frame = np.clip(mask_frame, 0.0, 1.0)
                                mask_frame = (mask_frame * 255.0).astype(np.uint8)
                            
                            # Ensure RGB is 3-channel
                            if rgb_frame.ndim == 2:
                                rgb_frame = np.repeat(rgb_frame[..., None], 3, axis=-1)
                            elif rgb_frame.shape[2] == 4:
                                rgb_frame = rgb_frame[:, :, :3]
                            
                            # Ensure mask is single channel
                            if mask_frame.ndim == 3:
                                mask_frame = mask_frame[:, :, 0]
                            
                            # Resize mask to match RGB if needed
                            if mask_frame.shape[:2] != rgb_frame.shape[:2]:
                                from PIL import Image
                                mask_pil = Image.fromarray(mask_frame)
                                mask_pil = mask_pil.resize((rgb_frame.shape[1], rgb_frame.shape[0]), Image.LANCZOS)
                                mask_frame = np.array(mask_pil)
                            
                            # Normalize mask to [0, 1]
                            # Cryptomatte mask: white (255) for object, black (0) for background
                            mask_normalized = mask_frame.astype(np.float32) / 255.0
                            
                            # Apply mask: multiply RGB by mask to keep object, remove background
                            # Where mask=1 (white), keep RGB. Where mask=0 (black), make black.
                            masked_frame = rgb_frame.astype(np.float32) * mask_normalized[:, :, np.newaxis]
                            masked_frame = masked_frame.astype(np.uint8)
                            
                            writer.append_data(masked_frame)
                        except Exception as e:
                            logger.warning(f"Error processing frame for masked video: {e}")
                            continue
                
                logger.info(f"Wrote masked video: {masked_video_path}")
            else:
                logger.info(f"No masks found for lighting {pidx}, skipping masked video generation")
        
        # Cleanup intermediate files, keeping only masked videos, RGB videos, and metadata
        logger.info("Cleaning up intermediate files...")
        
        keep_patterns = ['*.masked.mp4', '*.rgb.mp4', '*.meta.json']
        keep_files = set()
        for pattern in keep_patterns:
            keep_files.update(glob.glob(os.path.join(output_dir, pattern)))
        keep_dirs = {'depth', 'instance_maps', 'meshes'}
        
        # Remove all other files (preserve depth/ and instance_maps/ subdirs)
        all_files = glob.glob(os.path.join(output_dir, "*"))
        removed_count = 0
        for file_path in all_files:
            if os.path.isdir(file_path) and os.path.basename(file_path) in keep_dirs:
                continue
            if file_path not in keep_files and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Removed {removed_count} intermediate files")
        kept_names = [os.path.basename(f) for f in sorted(keep_files)]
        logger.info(f"Kept: {', '.join(kept_names)}")
        # # Remove all other files
        # all_files = glob.glob(os.path.join(output_dir, "*"))
        # removed_count = 0
        # for file_path in all_files:
        #     if file_path not in keep_files and os.path.isfile(file_path):
        #         try:
        #             os.remove(file_path)
        #             removed_count += 1
        #         except Exception as e:
        #             logger.warning(f"Failed to remove {file_path}: {e}")
        # 
        # logger.info(f"Removed {removed_count} intermediate files")
        kept_names = [os.path.basename(f) for f in keep_files]
        logger.info(f"Kept: {', '.join(kept_names)}")


def get_object_transforms_at_frame(mesh_list, frame_idx):
    """Get world transformation matrices for all objects at a specific frame"""
    scene = bpy.context.scene
    
    # Set frame and force a full scene update INCLUDING animations/rigid bodies
    scene.frame_set(frame_idx)
    
    # Force view layer update to apply animations and physics
    bpy.context.view_layer.update()
    
    # Re-fetch the depsgraph AFTER the view layer update
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    transforms = []
    for mesh_obj in mesh_list:
        if hasattr(mesh_obj, 'empty') and mesh_obj.empty is not None:
            # Get the EVALUATED world matrix (with animations/physics applied)
            empty_evaluated = mesh_obj.empty.evaluated_get(depsgraph)
            world_matrix = empty_evaluated.matrix_world.copy()
            transforms.append(np.array(world_matrix))
        else:
            # No transform data available
            transforms.append(None)
    return transforms

def render_scene(
    mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS
):
    cam_radius = FLAGS.radius_range[0] + np.random.uniform() * (FLAGS.radius_range[1] - FLAGS.radius_range[0])

    fovx = np.deg2rad(FLAGS.fov_range[0]+np.random.uniform()*(FLAGS.fov_range[1] - FLAGS.fov_range[0]))
    fovx_list = None
    azimuth = np.random.uniform(*FLAGS.cam_phi_range)
    elevation = np.random.uniform(*FLAGS.cam_theta_range)
    if FLAGS.cam_t_range is not None:
        t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
    else:
        t = np.zeros(3)
    
    cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)

    num_frames = FLAGS.num_frames
    if FLAGS.video_mode == 'orbit_cam':
        azimuth_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        elevation_offset = np.zeros(num_frames)
    elif FLAGS.video_mode == 'oscil_cam':
        phi_center = sum(FLAGS.cam_phi_range) / 2
        phi_ratio = 0.6
        phi_range = (FLAGS.cam_phi_range[1] - FLAGS.cam_phi_range[0]) / 2
        phi_range_clip = phi_range * phi_ratio
        azimuth = np.random.uniform(phi_center - phi_range_clip, phi_center + phi_range_clip)
        cone_angle = np.random.uniform(0.2 * (1-phi_ratio) * phi_range, phi_range - np.abs(phi_center - azimuth))
        # azimuth = np.random.uniform(*FLAGS.cam_phi_range)
        azimuth_offset = np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
        elevation_offset = np.cos(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
    elif FLAGS.video_mode == 'dolly_cam':
        # start_fov
        fovx_fix = np.deg2rad(45)
        radius_fix = 2.5
        fovx_perturb = np.random.uniform(-10, 10, size=2)
        fovx_motion = np.linspace(
            max(10, FLAGS.fov_range[0] + fovx_perturb[0]),
            FLAGS.fov_range[1] + fovx_perturb[1],
            num_frames, endpoint=True
        )
        if random.random() < 0.5:
            fovx_motion = fovx_motion[::-1]
        fovx_list = []
    elif FLAGS.video_mode == 'orbit_lgt':
        env_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    elif FLAGS.video_mode == 'rotat_obj':
        obj_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        mesh_id = [1]
        if mesh_meta is not None and random.random() > 0.5:
            if len(mesh_list) > 2 and 'metallic' not in mesh_meta[2]:
                mesh_id.append(2)
    elif FLAGS.video_mode == 'vtran_obj':
        num_obj = len(mesh_list) - 1
        drop_id = np.random.permutation(num_obj) + 1
        drop_id = drop_id.tolist()
        drop_id = drop_id[:1] if random.random() < 0.5 else drop_id[:2]
        if 1 not in drop_id and random.random() < 0.8:
            drop_id = drop_id + [1]
        num_drop = len(drop_id)
        drop_prev = [0] * num_drop
        drop_range = [0.5, 1.5]
        drop_list = np.random.uniform(*drop_range, size=[num_drop])
        drop_offset_list = []
        bounce = random.random() < 0.5 # always bounce 1/3 of the height
        if bounce:
            bounce_nframes = num_frames // 3 + random.randint(-num_frames//6, num_frames//6)
        else:
            bounce_nframes = 0
        for drop in drop_list:
            drop_offset = np.linspace(drop, 0, num_frames - bounce_nframes, endpoint=True) # the last is 0
            if bounce:
                bounce_factor = random.uniform(0.33, 0.8)
                bounce_offset = np.linspace(drop * bounce_factor, 0, bounce_nframes, endpoint=False)[::-1] # the last might not be 0
                drop_offset = np.concatenate([drop_offset, bounce_offset])
            drop_offset_list.append(drop_offset)

    cam_radius_list = None
    if FLAGS.varying_radius:
        if random.random() < 0.3:
            # sin wave 
            # random roll
            roll_step = np.random.uniform(-np.pi/2, np.pi/2)
            cam_radius_list = FLAGS.radius_range[0] + (FLAGS.radius_range[1] - FLAGS.radius_range[0]) * \
                (1 + np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False) + roll_step )) / 2

    azimuth_0, elevation_0 = azimuth, elevation
    cubemap, vec, vec_ref, latlong_img = None, None, None, None
    if FLAGS.dump_envmap: # TODO:
        vec = render_utils.latlong_vec(FLAGS.resolution)
        vec_ball, mask = render_utils.get_ideal_normal_ball(FLAGS.resolution[0], flip_x=False)
        vec_ref = render_utils.get_ref_vector(vec_ball, np.array([0,0, 1]))
        vec_ref = vec_ref.float() #.to('cuda')
    
    ori_shortname = shortname
    skip_features = False
    dump_format = FLAGS.dump_format # TODO:
    for lgt_i in range(FLAGS.num_lighting):
        prefix = f'{lgt_i:04d}.'
        if FLAGS.prefix_in_folder:
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)
            
        skip_features = lgt_i > 0
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if FLAGS.random_env_scale is not None else FLAGS.env_scale
        envmap_flip = False
        if FLAGS.random_env_flip:
            if random.random() > 0.5:
                envmap_flip = True

        if FLAGS.random_env_rotation:
            envmap_rotation_y = random.uniform(0, 2*np.pi)
        else:
            envmap_rotation_y = 0
        envmap_rotation_y_0 = envmap_rotation_y

        if FLAGS.dump_envmap:
            latlong_img = image_utils.read_img(envlight_path)
            latlong_img = latlong_img * envmap_strength
            latlong_img = np.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0) 
            latlong_img = np.clip(latlong_img, 0.0, 65504.0)
            latlong_img = torch.tensor(latlong_img, dtype=torch.float32)
            if envmap_flip:
                latlong_img = latlong_img.flip(1)

            # TODO: rotate
            cubemap = render_utils.latlong_to_cubemap_torch(latlong_img, [512, 512])
            env_proj = render_utils.cubemap_sample_torch(cubemap, -vec)
            env_proj = env_proj.flip(0).flip(1)

            env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
            env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_ldr.{dump_format}'), env_ev0)
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_log.{dump_format}'), env_log)

            if FLAGS.dump_env_bg:
                intrinsic = render_utils.cam_intrinsics(fovx, FLAGS.resolution[1], FLAGS.resolution[0])
                env_uv = render_utils.uv_mesh(FLAGS.resolution[1], FLAGS.resolution[0])
                pos_cam = env_uv @ np.linalg.inv(intrinsic).T

        blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)    
        logger.info(f"EnvProbe {lgt_i}/{FLAGS.num_lighting}: {envlight_path}")

        meta_dict = {
            # 'tone_mapping': FLAGS.tonemap_type,
            # 'envmap': os.path.basename(envlight_path),
            'camera_angle_x': fovx, # fov along width
            'cam_radius': cam_radius,
        }
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            meta_dict['envmap'] = os.path.basename(envlight_path)
        meta_frames = []

        # =============================================================================================
        # Start rendering
        # =============================================================================================
        cam_list = []
        for it in range(num_frames):
            if FLAGS.video_mode in ['orbit_cam', 'oscil_cam']:
                azimuth = azimuth_0 + azimuth_offset[it]
                elevation = elevation_0 + elevation_offset[it]
                # to avoid camera or object flipping
                elevation = np.clip(elevation, 5*np.pi/180, 85*np.pi/180)
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)
                
            elif FLAGS.video_mode == 'orbit_lgt':
                envmap_rotation_y = envmap_rotation_y_0 + env_rot_offset[it]
                # blender_utils.rotate_envmap(envmap_rotation_y) # TODO:
                        
            elif FLAGS.video_mode == 'vtran_obj':
                pass
            elif FLAGS.video_mode == 'dolly_cam':
                fovx_frame = np.deg2rad(fovx_motion[it])
                radius_frame = radius_fix * np.tan(fovx_fix/2) / np.tan(fovx_frame/2)
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, radius_frame)
                fovx_list.append(fovx_frame)

            if FLAGS.varying_radius and cam_radius_list is not None and FLAGS.video_mode != 'dolly_cam':
                cam_radius = cam_radius_list[it]
                cam_matrix = blender_utils.get_cam_matrix(azimuth, elevation, t, cam_radius)
            
            cam_list.append(cam_matrix)

            if FLAGS.dump_envmap:
                c2w = render_utils.convert_cam_mat_blender_to_dr(cam_matrix)
                c2w = torch.tensor(c2w, dtype=torch.float32, device=vec.device)
                vec_cam = vec.reshape(-1, 3) @ c2w[:3, :3].T
                y_rot = render_utils.rotate_y(envmap_rotation_y, device=vec.device)
                vec_query = (vec_cam @ y_rot[:3, :3].T).reshape(1, *FLAGS.resolution, 3)
                env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                env_proj = env_proj.flip(0).flip(1)
                env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_ldr.{dump_format}'), env_ev0)
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_log.{dump_format}'), env_log)

                if FLAGS.dump_ball_env:
                    vec_ball = -vec_ref.reshape(-1, 3) @ c2w[:3, :3].T
                    vec_query = (vec_ball @ y_rot[:3, :3].T).reshape(1, FLAGS.resolution[0], FLAGS.resolution[0], 3)
                    env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                    env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_ldr.{dump_format}'), env_ev0)
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_log.{dump_format}'), env_log)

                if FLAGS.dump_env_bg:
                    bg_dir = pos_cam @ c2w[:3, :3].T
                    bg_dir = (bg_dir @ y_rot[:3, :3].T)
                    bg_q_dir = -bg_dir.flip(1).contiguous().reshape(1, *FLAGS.resolution, 3)
                    bg_proj = render_utils.cubemap_sample_torch(cubemap, bg_q_dir)[0]
                    bg_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(bg_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_bg.{dump_format}'), bg_ev0)



            meta_frame = {
                # camera attributes
                'transform_matrix': cam_matrix.tolist(), # standard blender c2w
                'elevation': elevation,
                'azimuth': azimuth,
                # envmap
                'envmap_rot': envmap_rotation_y,
                'envmap_strength': envmap_strength,
                'envmap_flip': envmap_flip,
            }
            if FLAGS.video_mode == 'rotat_obj':
                meta_frame['obj_rot'] = obj_rot_offset[it]
                meta_frame['obj_rot_id'] = mesh_id
            if FLAGS.video_mode == 'vtran_obj':
                meta_frame['drop_offset'] = [drop_offset_list[i][it] for i in range(num_drop)]
                meta_frame['drop_id'] = drop_id
            if FLAGS.video_mode == 'dolly_cam':
                meta_frame['fov'] = fovx_frame

            meta_frames.append(meta_frame)

        if not skip_features or lgt_i == 0: # only setup the camera update for the first lighting setup
            blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX', fov_sequence=fovx_list)
            if FLAGS.video_mode == 'rotat_obj':
                for mi in mesh_id:
                    init_rot = mesh_list[mi].empty.rotation_euler[2]
                    mesh_list[mi].setup_realtime_update(num_frames, rotation=obj_rot_offset + init_rot)
            elif FLAGS.video_mode == 'vtran_obj':
                for di,mi in enumerate(drop_id):
                    init_loc = mesh_list[mi].empty.location
                    drop_offset_vec3 = np.zeros((num_frames, 3))
                    drop_offset_vec3[:, 2] = drop_offset_list[di]
                    drop_offset_vec3 += init_loc
                    mesh_list[mi].setup_realtime_update(num_frames, translation=drop_offset_vec3)
        
        if FLAGS.video_mode == 'orbit_lgt': # on orbit lgt needs reset.
            envmap_rotation_y_list = envmap_rotation_y_0 + env_rot_offset
            blender_utils.setup_realtime_envmap_update(envmap_rotation_y_list.tolist())

        # Capture object transforms for all frames (if enabled)
        object_transforms_per_frame = []
        if FLAGS.export_object_motion and num_frames > 1:
            logger.info("Capturing object transforms for all frames...")
            # Blender renders frames 1 to num_frames, so capture those frames
            for frame_idx in range(1, num_frames + 1):
                transforms = get_object_transforms_at_frame(mesh_list, frame_idx)
                object_transforms_per_frame.append(transforms)

            for it, transforms in enumerate(object_transforms_per_frame):
                if it >= len(meta_frames):
                    break
                obj_transforms = []
                for obj_idx, transform in enumerate(transforms):
                    if transform is None:
                        continue
                    if getattr(FLAGS, 'export_motion_skip_plane', True) and obj_idx == 0:
                        continue
                    obj_name = mesh_meta[obj_idx].get('name', f'object_{obj_idx}') if obj_idx < len(mesh_meta) else f'object_{obj_idx}'
                    if hasattr(FLAGS, 'export_motion_filter') and FLAGS.export_motion_filter:
                        if not any(pattern in obj_name for pattern in FLAGS.export_motion_filter):
                            continue
                    obj_transforms.append({
                        'object_id': obj_idx,
                        'object_name': obj_name,
                        'transform_matrix': transform.tolist()
                    })
                if len(obj_transforms) > 0:
                    meta_frames[it]['object_transforms'] = obj_transforms

        # set up the rendering
        save_folder = os.path.join(FLAGS.out_dir, shortname)
        blender_utils.setup_camera_settings(
            resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
        )
        blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

        blender_passes = ['rgb']
        if not skip_features:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, 
                passes=['normal', 'depth'], suffix=f'.{0:04d}') # NOTE: ['rgb', 'diffcol'] is removed here
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')
            blender_passes.extend(['normal', 'depth', 'albedo', 'orm'])
        # else:
        #     blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['rgb'], suffix=f'.{lgt_i:04d}')
        blender_utils.render_all_frames(output_dir=save_folder, num_frames=num_frames, suffix=f'rgb.{lgt_i:04d}')
        
        # dump the meta data
        meta_dict['frames'] = meta_frames
        meta_dict['file_path'] = shortname
        with open(os.path.join(FLAGS.out_dir, shortname, f'{prefix}meta.json'), 'w') as f:
            _meta_dict = copy.deepcopy(meta_dict)
            _meta_dict['mesh_list'] = mesh_meta
            json.dump(_meta_dict, f, indent=4)

class GLTFFileManger:
    def __init__(
        self, files, random_sample=True, rescale=True, 
        multi_sample_weight=None, check_bbox=False
    ):
        # Deep copy + sort each sublist so np.random.choice indices match across runs
        # (glob order is filesystem-dependent) and bbox-removal does not corrupt shared lists.
        self.files = [sorted(list(f)) for f in files]
        self.files_list = []
        for f in self.files:
            self.files_list.extend(f)
        self.num_file_lists = len(files)

        self.num_files = len(self.files_list)
        self.random_sample = random_sample
        self.rescale = rescale
        self.multi_sample_weight = multi_sample_weight
        self.check_bbox = check_bbox
        if self.multi_sample_weight is not None:
            assert len(self.multi_sample_weight) == self.num_file_lists
            
    def __len__(self):
        return self.num_files

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self): # inifi-gen
        self.idx += 1
        sample_success = False
        obj_mesh = None
        while not sample_success:
            if self.random_sample:
                idx = np.random.choice(self.num_file_lists, p=self.multi_sample_weight)
                file = np.random.choice(self.files[idx])
            else:
                file = self.files_list[(self.idx-1) % self.num_files]
                sample_success = True
            try:
                logger.info(f'Processing: {file}')
                # Load object WITHOUT rescaling first to check original size
                _obj_mesh = blender_utils.add_object_file(
                    file, with_empty=True, recenter=True, rescale=False
                )
                if _obj_mesh is None:
                    raise Exception('Failed to load object')
                
                # Check original size before rescaling
                if self.check_bbox:
                    vmin, vmax = _obj_mesh.aabb
                    bbox = vmax - vmin
                    max_dim = max(bbox)
                    min_dim = min(bbox)
                    
                    # Filter extremely large objects (buildings, scenes)
                    if max_dim > 5.0:
                        _obj_mesh.clear_objects()
                        del _obj_mesh
                        raise Exception(f'Object too large: max_dim={max_dim:.2f}')
                    
                    # Filter extremely small objects
                    if max_dim < 0.1:
                        _obj_mesh.clear_objects()
                        del _obj_mesh
                        raise Exception(f'Object too small: max_dim={max_dim:.2f}')
                    
                    # Check aspect ratio
                    if not check_msh_bbox(_obj_mesh):
                        _obj_mesh.clear_objects()
                        del _obj_mesh
                        raise Exception('Invalid bbox aspect ratio')
                
                # Now apply rescaling if needed
                if self.rescale:
                    vmin, vmax = _obj_mesh.aabb
                    scale_factor = 1.0 / max(vmax - vmin)
                    _obj_mesh.unit_rescale = float(scale_factor)
                    if _obj_mesh.empty:
                        _obj_mesh.empty.scale = _obj_mesh.empty.scale * scale_factor
                    else:
                        for obj in _obj_mesh.objs:
                            obj.scale = obj.scale * scale_factor
                    _obj_mesh.aabb = _obj_mesh.get_aabb()
                
                obj_mesh = _obj_mesh
                sample_success = True
            except Exception as e:
                logger.info(f'---> Error: {e}, skipping file {file}')
                self.files[idx].remove(file)
        
        return {'mesh': obj_mesh, 'name': file}
    
def main():
    # Parse --config and support legacy flags; additional overrides use OmegaConf dotlist (key=val)
    parser = argparse.ArgumentParser(description='composition_rendering')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    # Legacy convenience flags retained for compatibility
    parser.add_argument('-n', '--num_frames', type=int, default=None)
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('--base_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args, unknown = parser.parse_known_args()

    # Defaults
    default_cfg = {
        'seed': None,
        'out_dir': '.',
        'base_path': None,
        'cam_near_far': [0.1, 1000.0],
        'resolution': [256, 256],
        'baseshape_path': None,
        'envlight': 'data/envmap/aerodynamics_workshop_512.hdr',
        'env_scale': 1.0,
        'env_res': [256, 512],
        'probe_res': 256,
        'spp': 8,
        'use_denoise': 'OPTIX',
        'transparent_bg': True,
        'radius_range': [2.0, 4.0],
        'varying_radius': False,
        'num_files': None,
        'random_env_rotation': True,
        'random_env_flip': True,
        'random_env_scale': None,
        'bg_color': [1.0, 1.0, 1.0],
        'bg_features': [0.0, 0.0, 0.0],
        'dump_env_bg': False,
        'dump_alpha': False,
        'ray_depth': 1,
        'num_workers': 0,
        'timeout': -1,
        'fov_range': [45.0, 45.0],
        'dump_features': False,
        'num_rendering': 10,
        'num_lighting': 1,
        'cam_phi_range': [0, 360],
        'cam_theta_range': [0, 90],
        'cam_t_range': [0, 0],
        'dump_shading': False,
        'dump_splitsum': False,
        'dump_envmap': False,
        'dump_ball_env': False,
        'dump_irradiance': False,
        'dump_format': 'jpg',
        'dump_video': False,
        'dump_complete': False,
        'dump_blend': False,
        'dump_placement': False,
        'video_mode': 'orbit_cam',
        # Sampling config
        'glbs_check_bbox': False,
        'glbs_multi_sample_weight': None,
        'glbs_rescale': True,
        'glbs_random_sample': True,
        'glbs_per_scene': 2,
        'glbs_scale_range': [1.0, 1.5],
        'glbs_rotation_range': [-45, 45],
        'glbs_placement_bbox': [-1.0, -1.0, 1.0, 1.0],
        'shapes_per_scene': 3,
        'shapes_scale_range': [0.3, 0.8],
        'shapes_rotation_range': [-45, 45],
        'shapes_placement_bbox': [-1.5, -1.5, 1.5, 1.5],
        'placement_centered': False,
        'placement_bbox': [-1.5, -1.5, 1.5, 1.5],
        'placement_grid_res': [40, 40],
        'placement_bbox_scale': 1.1,
        'placement_plane_offset': [0, 0, -0.5],
        'placement_plane_scale': 10,
        'placement_plane': 'data/plane_basic/plane.glb',
        'plane_sample_weight': None,
        'envlight_sample_weight': None,
        'texture_sample_weight': None,
        'placement_plane_textures': None,
        'prefix_in_folder': False,
        # Unsupported/unused but kept for completeness
        'analytical_sky': False,
        'use_objaverse': False,
        'objaverse_selection': None,
        'no_physics': False,
        'num_frames': 8,
        'sample_shape_texture': False,
        'export_object_motion': True,
        'export_motion_skip_plane': True,
        'export_motion_filter': None,
    }

    cfg: DictConfig = OmegaConf.create(default_cfg)
    if args.config is not None:
        file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)
    # Dotlist CLI overrides (e.g., num_frames=8 out_dir=output/ path.with.dots=value)
    if len(unknown) > 0:
        dotlist = [tok for tok in unknown if '=' in tok]
        if len(dotlist) > 0:
            cli_cfg = OmegaConf.from_cli(dotlist)
            cfg = OmegaConf.merge(cfg, cli_cfg)

    # Apply legacy flags if provided
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.num_frames is not None:
        cfg.num_frames = int(args.num_frames)
    if args.base_path is not None:
        cfg.base_path = args.base_path
    if args.seed is not None:
        cfg.seed = int(args.seed)

    # Convert to plain python containers to safely mutate later
    _flags_container = OmegaConf.to_container(cfg, resolve=True)
    FLAGS = SimpleNamespace(**_flags_container)

    logger.info('Config / Flags (OmegaConf):')
    logger.info('---------')
    logger.info('\n' + OmegaConf.to_yaml(cfg))
    logger.info('---------')

    if FLAGS.seed is not None:
        sub_folder = f"s{FLAGS.seed:06d}"         
        set_seed(FLAGS.seed)
    else:
        set_seed(None)
        sub_folder = 's' + time.strftime("%m%d%H")

    sub_folder = f"{FLAGS.video_mode}_{sub_folder}"
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, sub_folder)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    try:
        bpy.context.preferences.filepaths.use_relative_paths = False
    except Exception:
        pass

    FLAGS.cam_phi_range = [float(np.deg2rad(float(x)) + np.pi) for x in list(FLAGS.cam_phi_range)]
    FLAGS.cam_theta_range = [float(np.deg2rad(float(x))) for x in list(FLAGS.cam_theta_range)]
    assert FLAGS.video_mode in ['orbit_cam', 'oscil_cam', 'orbit_lgt', 'rotat_obj', 'vtran_obj', 'dolly_cam', 'drop_phy', 'replay_trajectory', 'arbitrary_motion']

    # Composition Sampling Related
    glbs_placement_vmin, glbs_placement_vmax = np.array(FLAGS.glbs_placement_bbox[:2]), np.array(FLAGS.glbs_placement_bbox[2:])
    shapes_placement_vmin, shapes_placement_vmax = np.array(FLAGS.shapes_placement_bbox[:2]), np.array(FLAGS.shapes_placement_bbox[2:])
    bbox_scale = 0.5 * FLAGS.placement_bbox_scale

    placement_vmin, placement_vmax = np.array(FLAGS.placement_bbox[:2]), np.array(FLAGS.placement_bbox[2:])
    placement_range = placement_vmax - placement_vmin
    placement_grid = np.zeros(FLAGS.placement_grid_res, dtype=np.int32)
    placement_bbox2grid = np.array(FLAGS.placement_grid_res) / placement_range
    placement_plane_offset = np.array(FLAGS.placement_plane_offset, dtype=np.float32)
    placement_plane_scale = np.array(FLAGS.placement_plane_scale, dtype=np.float32)
    placement_plane_path = FLAGS.placement_plane

    if placement_plane_path is None:
        placement_plane_path = []
    elif os.path.isdir(placement_plane_path):
        placement_plane_path = sorted([os.path.abspath(p) for p in glob.glob(placement_plane_path + "/*.glb")])
    else:
        placement_plane_path = [os.path.abspath(placement_plane_path)]

    num_planes = len(placement_plane_path)
    plane_sample_weight = np.ones(num_planes)
    if FLAGS.plane_sample_weight is not None:
        for i, plane in enumerate(placement_plane_path):
            for k, w in FLAGS.plane_sample_weight.items():
                if k in plane:
                    plane_sample_weight[i] = w
    FLAGS.plane_sample_weight = plane_sample_weight / plane_sample_weight.sum()

    def sample_glb(data_iter, obj_idx=0):
        target = next(data_iter) # centered, scaled
        ref_mesh = target['mesh']
        mesh_name = target['name']

        if ref_mesh is None:
            logger.info(f'Mesh {mesh_name} is None')
            return None
        num_materials = ref_mesh.get_num_materials()
        if num_materials == 0:
            logger.info(f'Mesh {mesh_name} has invalid materials {num_materials}')
            return None

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.glbs_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.glbs_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        # After recentering, geometric center is always at [0,0,0] relative to empty
        # Do NOT recalculate from AABB after rotation (AABB changes shape, but center doesn't move)
        mesh_center = np.array([0.0, 0.0, 0.0])
        cz = 0  # Use geometric center, not bottom offset
        mesh_bounds = np.array(vmax - vmin)[:2]
        if FLAGS.video_mode == 'rotat_obj':
            # use square bbox
            mesh_bounds[:] = np.max(mesh_bounds)
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        
        mesh_placement_vmin =  glbs_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  glbs_placement_vmax - mesh_bounds * 0.5

        if not FLAGS.placement_centered:
            # Sample the object placement
            find_placement = False
            for t in range(5): # 8 tries
                # sample the center
                cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
                cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
                # convert to grid
                gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
                gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
                # check if the placement is valid
                x_lb, x_ub = max(gx-bx, 0), gx+bx
                y_lb, y_ub = max(gy-by, 0), gy+by
                if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                    find_placement = True
                    placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                    break
                
            if not find_placement:
                logger.info(f'Cannot find valid placement for {mesh_name}')
                ref_mesh.clear_objects()
                del ref_mesh
                return None
        else:
            cx, cy = 0, 0
            cz = 0 if FLAGS.glbs_rescale else mesh_center[2]

        # logger.info('cx, cy, cz, smpl_scale', cx, cy, cz, smpl_scale)
        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'unit_rescale': ref_mesh.unit_rescale,
            'parts_matrix_local': ref_mesh.get_parts_matrix_local(),
            'num_materials': num_materials
        }

        return ref_mesh, placement_meta

    def sample_shape(shapes_files, obj_idx=0):
        target_file = np.random.choice(shapes_files)
        ref_mesh = blender_utils.add_object_file(target_file, with_empty=True, recenter=True, rescale=True)
        mesh_name = target_file

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.shapes_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.shapes_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        # After recentering, geometric center is always at [0,0,0] relative to empty
        # Do NOT recalculate from AABB after rotation (AABB changes shape, but center doesn't move)
        mesh_center = np.array([0.0, 0.0, 0.0])
        cz = 0  # Use geometric center, not bottom offset
        mesh_bounds = np.array(vmax - vmin)[:2]
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        mesh_placement_vmin =  shapes_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  shapes_placement_vmax - mesh_bounds * 0.5

        # Sample the object placement
        find_placement = False
        for t in range(8): # 8 tries
            # sample the center
            cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
            cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
            # convert to grid
            gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
            gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
            # check if the placement is valid
            x_lb, x_ub = max(gx-bx, 0), gx+bx
            y_lb, y_ub = max(gy-by, 0), gy+by
            if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                find_placement = True
                placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                break

        if not find_placement:
            logger.info(f'Cannot find valid placement for {mesh_name}')
            ref_mesh.clear_objects()
            del ref_mesh
            return None

        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        # Material sampling
        roughness_range = [0, 0.8]
        roughness = np.random.uniform(roughness_range[0], roughness_range[1]) ** 2
        metallic = np.abs(np.random.normal(0, 0.25))
        # 50% metallic (close to 1)
        if np.random.uniform() < 0.5:
            metallic = 1 - metallic
        if metallic > 0.5:
            base_color = np.random.randint(170, 255, size=3) / 255
        else:
            base_color = np.random.randint(30, 240, size=3) / 255

        logger.info(f'shape material: {base_color}, {roughness}, {metallic}')
        ref_mesh.set_principled_material(base_color, roughness, metallic)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'unit_rescale': ref_mesh.unit_rescale,
            'parts_matrix_local': ref_mesh.get_parts_matrix_local(),
            'roughness': roughness,
            'metallic': metallic,
            'base_color': base_color.tolist()
        }

        return ref_mesh, placement_meta

    # plane textures
    plane_textures_path = []
    num_plane_textures = 0  # Initialize to 0
    if FLAGS.placement_plane_textures is not None:
        logger.info(f"Looking for plane textures in: {FLAGS.placement_plane_textures}")
        
        # Check if the path is a single texture directory or a parent containing multiple texture directories
        texture_path = os.path.abspath(FLAGS.placement_plane_textures)
        
        # Check if this directory directly contains texture files (diff*, rough*, nor*, etc.)
        has_texture_files = any(glob.glob(os.path.join(texture_path, pattern)) 
                               for pattern in ['diff*', 'col*', 'albedo*', 'basecol*'])
        
        if has_texture_files:
            # This is a single texture directory, use it directly
            plane_textures_path = [texture_path]
            logger.info(f"Using single texture directory: {texture_path}")
        else:
            # This is a parent directory, look for subdirectories
            plane_textures_path = sorted([
                os.path.abspath(p) for p in glob.glob(os.path.join(texture_path, '*'))
                if os.path.isdir(p)
            ])
            logger.info(f"Found {len(plane_textures_path)} texture subdirectories: {plane_textures_path}")
        
        num_plane_textures = len(plane_textures_path)
        texture_sample_weight = np.ones(num_plane_textures)
        if FLAGS.texture_sample_weight is not None:
            for i, texture in enumerate(plane_textures_path):
                for k, w in FLAGS.texture_sample_weight.items():
                    if k in texture:
                        texture_sample_weight[i] = w
        FLAGS.texture_sample_weight = texture_sample_weight / texture_sample_weight.sum()
        
    obj_files = []
    if not FLAGS.use_objaverse and FLAGS.base_path is not None:
        base_path_abs = os.path.abspath(FLAGS.base_path)
        if not os.path.isfile(base_path_abs):
            _paths = (
                glob.glob(os.path.join(base_path_abs, "*.glb"))
                + glob.glob(os.path.join(base_path_abs, "*.gltf"))
                + glob.glob(os.path.join(base_path_abs, "*.obj"))
                + glob.glob(os.path.join(base_path_abs, "*.ply"))
            )
            obj_files.append(sorted(os.path.abspath(p) for p in _paths))
        else:
            obj_files.append(sorted(os.path.abspath(p) for p in np.loadtxt(base_path_abs, dtype=str).tolist()))
    else:
        raise NotImplementedError("Objaverse is not supported yet")

   
    # obj dataloader
    obj_dataloader = GLTFFileManger(
        obj_files, 
        random_sample=FLAGS.glbs_random_sample, 
        rescale=FLAGS.glbs_rescale, 
        multi_sample_weight=FLAGS.glbs_multi_sample_weight, 
        check_bbox=FLAGS.glbs_check_bbox
    )
    logger.info(f"Use {len(obj_dataloader)} glbs")

    # envlight 
    env_src = os.path.abspath(FLAGS.envlight)
    if not os.path.isfile(env_src):
        envlight_path_list = sorted([os.path.abspath(p) for p in (glob.glob(os.path.join(env_src, "*.exr"))  + \
            glob.glob(os.path.join(env_src, "*.hdr")))])
    elif env_src.endswith('.txt'):
        envlight_path_list = sorted(os.path.abspath(p) for p in np.loadtxt(env_src, dtype=str).tolist())
    else:
        envlight_path_list = [env_src]

    num_envlights = len(envlight_path_list)
    envlight_sample_weight = np.ones(num_envlights)
    if FLAGS.envlight_sample_weight is not None:
        for i, envlight in enumerate(envlight_path_list):
            for k, w in FLAGS.envlight_sample_weight.items():
                if k in envlight:
                    envlight_sample_weight[i] = w
    FLAGS.envlight_sample_weight = envlight_sample_weight / envlight_sample_weight.sum()

    baseshape_files = []
    if FLAGS.baseshape_path is not None:
        bsp = os.path.abspath(FLAGS.baseshape_path)
        if os.path.isdir(bsp):
            baseshape_files = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(bsp, "*.glb")))
        else:
            baseshape_files = [bsp]
    
    start_idx = 0
    iter_start_time = time.time()
    obj_iter = iter(obj_dataloader)
    for i in range(start_idx, FLAGS.num_rendering):
        logger.info(f"Rendering iteration {i}/{FLAGS.num_rendering}")
        name = f"{i:06d}"
        if FLAGS.dump_complete:
            new_complete_file = os.path.join(FLAGS.out_dir, f"COMPLETE_{name}")
            if os.path.exists(new_complete_file):
                logger.info(f"COMPLETE_{name} already exists, skip")
                continue
        prefix = ''
        if FLAGS.num_frames > 1:
            prefix = f"{0:04d}."
        placement_grid = placement_grid * 0
        blender_utils.clear_scene()
        mesh_list = []
        mesh_meta = []

        # sample placement plane (skip for flying physics mode)
        skip_plane = getattr(FLAGS, 'skip_placement_plane', False)
        if skip_plane:
            mesh_meta.append({'name': 'none'})
            mesh_list.append(None)
            logger.info("Skipping placement plane (skip_placement_plane=true)")
        elif num_planes > 0:
            plane_idx = np.random.choice(num_planes, size=1, p=FLAGS.plane_sample_weight)[0]
            # insert plane 
            placement_plane = blender_utils.add_object_file(placement_plane_path[plane_idx], with_empty=True, recenter=True, rescale=True)
            placement_plane.apply_transform((0, 0, 0), scale=placement_plane_scale)
            plane_vmin, plane_vmax = placement_plane.aabb
            vz = plane_vmin[2]
            vplane = np.array(placement_plane_offset) - np.array([0, 0, vz])
            placement_plane.apply_transform(vplane)
            # 
            plane_meta = {'name': os.path.basename(placement_plane_path[plane_idx])}
            # sample the plane texture (randomly skip to leave plane black)
            skip_plane_texture = getattr(FLAGS, 'skip_plane_texture_prob', 0.0)
            if num_plane_textures > 0 and np.random.random() >= skip_plane_texture:
                plane_tex_idx = np.random.choice(num_plane_textures, size=1, p=FLAGS.texture_sample_weight)[0]
                texture_scale = np.random.uniform(1.5, 2.5)
                use_emissive = (FLAGS.env_scale == 0.0)
                selected_texture_dir = plane_textures_path[plane_tex_idx]
                logger.info(f"Applying plane texture directory: {selected_texture_dir} (scale={texture_scale:.2f}, emissive={use_emissive})")
                placement_plane.apply_texture(selected_texture_dir, texture_scale, emissive=use_emissive)
                plane_meta['texture'] = selected_texture_dir
                plane_meta['texture_scale'] = texture_scale
            else:
                logger.info("Skipping plane texture (black plane)")
            mesh_meta.append(plane_meta)
            mesh_list.append(placement_plane)

        # Re-seed before object selection so the same seed always picks the same
        # objects regardless of mode-specific random calls (e.g. plane sampling).
        set_seed(FLAGS.seed + i * 100003)

        # add glbs (retry up to 10 extra times per slot to hit the requested count)
        glb_retry_budget = FLAGS.glbs_per_scene * 10
        glb_loaded = 0
        glb_attempts = 0
        while glb_loaded < FLAGS.glbs_per_scene and glb_attempts < glb_retry_budget:
            glb_attempts += 1
            target = sample_glb(obj_iter, obj_idx=glb_loaded)
            if target is not None:
                glb_loaded += 1
                ref_mesh, meta = target
                
                # SKIP emissive conversion for ArUco cube - use regular materials with texture
                # Emissive causes bloom that destroys sharp ArUco patterns
                skip_emissive = True  # ArUco needs sharp edges, not emission glow
                if not skip_emissive:
                    for obj in ref_mesh.objs:
                        if obj.data and hasattr(obj.data, 'materials'):
                            for mat in obj.data.materials:
                                if mat and mat.use_nodes:
                                    nodes = mat.node_tree.nodes
                                    links = mat.node_tree.links
                                    
                                    # Find Principled BSDF and get its base color
                                    shader = None
                                    for node in nodes:
                                        if node.type == 'BSDF_PRINCIPLED':
                                            shader = node
                                            break
                                    
                                    if shader:
                                        # Check if Base Color has a connected texture
                                        base_color_input = shader.inputs['Base Color']
                                        texture_node = None
                                        if base_color_input.is_linked:
                                            # Get the connected node
                                            texture_node = base_color_input.links[0].from_node
                                        
                                        # Find or create Emission shader
                                        emission = None
                                        for node in nodes:
                                            if node.type == 'EMISSION':
                                                emission = node
                                                break
                                        
                                        if not emission:
                                            emission = nodes.new(type='ShaderNodeEmission')
                                            emission.location = (200, 0)
                                        
                                        # Connect texture or color to emission
                                        if texture_node:
                                            # Preserve texture connection
                                            links.new(texture_node.outputs[0], emission.inputs['Color'])
                                            # Use VERY LOW strength to avoid bloom (we have white background now!)
                                            emission.inputs['Strength'].default_value = 0.1
                                        else:
                                            # Use flat color
                                            base_color = base_color_input.default_value[:]
                                            emission.inputs['Color'].default_value = base_color
                                            # Higher strength for solid colors too (for consistency)
                                            emission.inputs['Strength'].default_value = 5.0
                                        
                                        # Connect to output
                                        output = None
                                        for node in nodes:
                                            if node.type == 'OUTPUT_MATERIAL':
                                                output = node
                                                break
                                        
                                        if output:
                                            # Clear existing connections to output
                                            for link in list(output.inputs['Surface'].links):
                                                links.remove(link)
                                            # Connect emission to output
                                            links.new(emission.outputs['Emission'], output.inputs['Surface'])
                
                mesh_list.append(ref_mesh)
                mesh_meta.append(meta)


        if glb_loaded < FLAGS.glbs_per_scene:
            logger.warning(f"Only loaded {glb_loaded}/{FLAGS.glbs_per_scene} GLBs after {glb_attempts} attempts")

        num_glbs = len(mesh_list) - 1
        shape_list = []
        shape_meta = []
        for j in range(FLAGS.shapes_per_scene):
            target = sample_shape(baseshape_files, obj_idx=j+num_glbs)
            if target is not None:
                ref_mesh, meta = target
                shape_list.append(ref_mesh)

                if FLAGS.sample_shape_texture and num_plane_textures > 0:
                    if random.random() < 0.25:
                        plane_tex_idx = np.random.choice(num_plane_textures, size=1, p=FLAGS.texture_sample_weight)[0]
                        texture_scale = 1.0
                        ref_mesh.apply_texture(plane_textures_path[plane_tex_idx], texture_scale)
                        meta['texture'] = plane_textures_path[plane_tex_idx]
                        meta['texture_scale'] = texture_scale
                shape_meta.append(meta)

        if len(shape_list) > 0:
            mesh_list.extend(shape_list)
            mesh_meta.extend(shape_meta)
        else:
            logger.info("No shapes added")

        # Render multiple views and dump results
        if not FLAGS.prefix_in_folder:
            os.makedirs(os.path.join(FLAGS.out_dir, name), exist_ok=True)

        if FLAGS.video_mode == 'drop_phy':
            if getattr(FLAGS, 'no_physics', False):
                from modes.arbitrary_motion import run as run_arbitrary_motion
                render_fn = run_arbitrary_motion
                logger.info("Using arbitrary motion (no physics)")
            else:
                from modes.drop_physics import run as run_drop_physics
                render_fn = run_drop_physics
                logger.warning("Drop physics is not fully tested")
        elif FLAGS.video_mode == 'arbitrary_motion':
            from modes.arbitrary_motion import run as run_arbitrary_motion
            render_fn = run_arbitrary_motion
            logger.info("Using arbitrary motion mode (flying trajectories)")
        elif FLAGS.video_mode == 'replay_trajectory':
            from modes.replay_trajectory import run as run_replay_trajectory
            render_fn = run_replay_trajectory
            logger.info("Using trajectory replay mode")
        else:
            render_fn = render_scene

        # Export canonical PLY meshes and copy raw GLBs
        meshes_out_dir = os.path.join(FLAGS.out_dir, name, 'meshes')
        os.makedirs(meshes_out_dir, exist_ok=True)
        for obj_idx, (mesh_obj, meta_entry) in enumerate(zip(mesh_list, mesh_meta)):
            if mesh_obj is None or not hasattr(mesh_obj, 'export_canonical_ply'):
                continue
            if 'scale' not in meta_entry:
                continue
            src_path = meta_entry.get('name', '')
            glb_name = os.path.splitext(os.path.basename(src_path))[0]
            ply_path = os.path.join(meshes_out_dir, f'{glb_name}.ply')
            try:
                mesh_obj.export_canonical_ply(ply_path)
                logger.info(f"Exported canonical PLY: {ply_path}")
            except Exception as e:
                logger.warning(f"Failed to export PLY for object {obj_idx}: {e}")
            if os.path.isfile(src_path):
                glb_dst = os.path.join(meshes_out_dir, os.path.basename(src_path))
                if not os.path.exists(glb_dst):
                    shutil.copy2(src_path, glb_dst)
                    logger.info(f"Copied raw GLB: {glb_dst}")

        render_fn(mesh_list, mesh_meta, envlight_path_list, name, prefix, FLAGS)

        post_process_rendering(
            os.path.join(FLAGS.out_dir, name),
            feature_fmt=FLAGS.dump_format,
            dump_video=FLAGS.dump_video
        )

        # Flatten: move contents of 000000/ up into the parent dir and remove 000000/
        sub_dir = os.path.join(FLAGS.out_dir, name)
        parent_dir = FLAGS.out_dir
        if os.path.isdir(sub_dir):
            for item in os.listdir(sub_dir):
                src = os.path.join(sub_dir, item)
                dst = os.path.join(parent_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(sub_dir)
            logger.info(f"Flattened {sub_dir} into {parent_dir}")

        if FLAGS.dump_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FLAGS.out_dir, f"scene.blend"))
        if FLAGS.dump_placement:
            # Optional debug plot — lazy import so NumPy/matplotlib ABI mismatches on cluster
            # (e.g. numpy 2.x + scipy-stack matplotlib) do not break normal rendering.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError as e:
                logger.warning("dump_placement: matplotlib unavailable (%s); skip placement.png", e)
            else:
                fig = plt.figure()
                plt.imshow(
                    placement_grid.T,
                    cmap="tab20",
                    vmin=0,
                    vmax=placement_grid.max().item() + 1,
                )
                plt.savefig(os.path.join(FLAGS.out_dir, "placement.png"))
                plt.close(fig)
        if FLAGS.dump_complete:
            new_complete_file = os.path.join(FLAGS.out_dir, f"COMPLETE_{name}")
            open(new_complete_file, 'w').close()

    # clean up and safely exit blender
    blender_utils.clear_scene()
    bpy.ops.wm.quit_blender()
    # sys.exit(0)

if __name__ == "__main__":
    main()
