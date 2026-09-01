import bpy
import numpy as np
import random
import os
import json
import copy
from mathutils import Vector, Matrix, Euler

from utils import blender_utils, render_utils, image_utils
from modes.drop_physics import get_object_transforms_at_frame
import torch
import logging

logger = logging.getLogger(__name__)


def _smooth_noise(n, num_harmonics=5):
    """Sum of random sinusoids for smooth, non-periodic motion."""
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for _ in range(num_harmonics):
        freq = np.random.uniform(0.5, 4.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.3, 1.0)
        out += amp * np.sin(2 * np.pi * freq * t + phase)
    out /= num_harmonics
    return out


def generate_arbitrary_trajectory(num_frames, seed=None):
    """Generate non-physical 3D trajectory where objects fly around.

    Returns positions (num_frames, 3) and rotations (num_frames, 3) in radians.
    Objects stay roughly within a bounding region suitable for camera view.
    """
    if seed is not None:
        np.random.seed(seed)

    traj_type = np.random.choice([
        'hover', 'fly_through', 'orbit', 'figure8',
        'zigzag', 'random_smooth',
    ])

    t = np.linspace(0, 1, num_frames)
    positions = np.zeros((num_frames, 3))

    height_base = np.random.uniform(0.1, 0.35)
    R = 0.3

    if traj_type == 'hover':
        cx, cy = np.random.uniform(-0.05, 0.05, 2)
        positions[:, 0] = cx + _smooth_noise(num_frames) * 0.06
        positions[:, 1] = cy + _smooth_noise(num_frames) * 0.06
        positions[:, 2] = height_base + _smooth_noise(num_frames) * 0.05

    elif traj_type == 'fly_through':
        start = np.random.uniform(-R, R, 3)
        end = np.random.uniform(-R, R, 3)
        start[2] = np.random.uniform(0.1, 0.4)
        end[2] = np.random.uniform(0.1, 0.4)
        for d in range(3):
            positions[:, d] = np.interp(t, [0, 1], [start[d], end[d]])
            positions[:, d] += _smooth_noise(num_frames) * 0.03

    elif traj_type == 'orbit':
        radius = np.random.uniform(0.1, R)
        speed = np.random.uniform(0.5, 2.0)
        positions[:, 0] = radius * np.cos(2 * np.pi * speed * t)
        positions[:, 1] = radius * np.sin(2 * np.pi * speed * t)
        positions[:, 2] = height_base + _smooth_noise(num_frames) * 0.05

    elif traj_type == 'figure8':
        scale = np.random.uniform(0.15, R)
        positions[:, 0] = scale * np.sin(2 * np.pi * t)
        positions[:, 1] = scale * 0.5 * np.sin(4 * np.pi * t)
        positions[:, 2] = height_base + _smooth_noise(num_frames) * 0.06

    elif traj_type == 'zigzag':
        num_zigs = np.random.randint(3, 7)
        kp_t = np.linspace(0, 1, num_zigs)
        kp_x = np.random.uniform(-R, R, num_zigs)
        kp_y = np.random.uniform(-R, R, num_zigs)
        kp_z = np.random.uniform(0.1, 0.4, num_zigs)
        for d, kp in enumerate([kp_x, kp_y, kp_z]):
            positions[:, d] = np.interp(t, kp_t, kp)

    else:  # random_smooth
        for d in range(3):
            positions[:, d] = _smooth_noise(num_frames, num_harmonics=6) * 0.2
        positions[:, 2] = np.abs(positions[:, 2]) + 0.1

    rotations = np.zeros((num_frames, 3))
    for axis in range(3):
        if np.random.random() > 0.3:
            rot_speed = np.random.uniform(-2 * np.pi, 2 * np.pi)
            rotations[:, axis] = t * rot_speed + np.random.uniform(0, 2 * np.pi)

    logger.info(f"Generated trajectory type='{traj_type}' height_base={height_base:.2f}")
    return positions, rotations


def simulate_flying_physics(
    num_objects, num_frames, cam_matrices, cam_radius, fovx, resolution,
    spawn_center, speed_range=(5.0, 12.0), ang_speed_range=(30.0, 120.0),
    restitution=0.8, obj_radius=0.3, fps=24, fixed_depth=None,
):
    """Simulate N objects flying with collisions, visible in ALL camera frames.

    Uses Euler integration with sphere-sphere collision and frustum-aware
    wall bounces. Checks visibility against the camera at each timestep.
    When fixed_depth is set, objects are constrained to move in the camera XY
    plane at that depth (no motion toward/away from camera).
    Returns (all_positions, all_rotations) arrays of shape (N, num_frames, 3).
    """
    dt = 1.0 / fps
    W, H = resolution
    focal = 0.5 * W / np.tan(fovx / 2.0)

    # Pre-compute inverse (world-to-camera) for every frame
    cam_exts = [np.linalg.inv(m) for m in cam_matrices]

    margin = 0.20

    def _project(world_pos, cam_ext):
        p = cam_ext @ np.append(world_pos, 1.0)
        if p[2] >= -1e-6:
            return None
        px = focal * p[0] / (-p[2]) + W / 2.0
        py = focal * (-p[1]) / (-p[2]) + H / 2.0
        return px, py

    def _is_visible(world_pos, cam_ext):
        proj = _project(world_pos, cam_ext)
        if proj is None:
            return False
        px, py = proj
        return (W * margin < px < W * (1 - margin) and
                H * margin < py < H * (1 - margin))

    def _is_visible_all(world_pos):
        """Check visibility in first, middle, and last camera frames."""
        for idx in [0, num_frames // 2, num_frames - 1]:
            if not _is_visible(world_pos, cam_exts[idx]):
                return False
        return True

    cam_matrix = cam_matrices[0]
    cam_right = np.array(cam_matrix[:3, 0])
    cam_up = np.array(cam_matrix[:3, 1])
    cam_fwd = np.array(cam_matrix[:3, 2])
    half_w = cam_radius * np.tan(fovx / 2.0)

    pos = np.zeros((num_objects, 3))
    vel = np.zeros((num_objects, 3))
    ang_vel = np.zeros((num_objects, 3))
    rot = np.zeros((num_objects, 3))

    spawn_min_depth = cam_radius * 0.35

    if fixed_depth is not None:
        cam_pos = np.array(cam_matrix[:3, 3])
        depth_center = cam_pos - cam_fwd * fixed_depth
        logger.info(f"Fixed-depth mode: depth={fixed_depth:.2f}, plane center={depth_center.round(2)}")

    # Spawn objects visible in ALL camera frames (first, mid, last)
    spread_frac = 0.35
    for i in range(num_objects):
        for attempt in range(100):
            rx = random.uniform(-spread_frac, spread_frac) * half_w
            ry = random.uniform(-spread_frac, spread_frac) * half_w
            if fixed_depth is not None:
                candidate = depth_center + cam_right * rx + cam_up * ry
            else:
                candidate = spawn_center + cam_right * rx + cam_up * ry
                p_cam_spawn = cam_exts[0] @ np.append(candidate, 1.0)
                if -p_cam_spawn[2] < spawn_min_depth:
                    continue
            if _is_visible_all(candidate):
                too_close = False
                for j in range(i):
                    if np.linalg.norm(candidate - pos[j]) < obj_radius * 3:
                        too_close = True
                        break
                if not too_close:
                    pos[i] = candidate
                    break
        else:
            pos[i] = depth_center if fixed_depth is not None else spawn_center

        vel_dir = np.random.randn(3)
        vel_dir /= np.linalg.norm(vel_dir) + 1e-8
        if fixed_depth is not None:
            vel_dir -= np.dot(vel_dir, cam_fwd) * cam_fwd
            vel_dir /= np.linalg.norm(vel_dir) + 1e-8
        speed = random.uniform(*speed_range)
        vel[i] = vel_dir * speed

        for ax in range(3):
            ang_vel[i, ax] = np.deg2rad(random.uniform(*ang_speed_range)) * random.choice([-1, 1])

        rot[i] = np.random.uniform(0, 2 * np.pi, 3)
        logger.info(f"Flying obj {i}: pos={pos[i].round(2)} vel={vel[i].round(2)} visible_all={_is_visible_all(pos[i])}")

    all_positions = np.zeros((num_objects, num_frames, 3))
    all_rotations = np.zeros((num_objects, num_frames, 3))

    for f in range(num_frames):
        all_positions[:, f] = pos.copy()
        all_rotations[:, f] = rot.copy()

        pos += vel * dt
        rot += ang_vel * dt

        if fixed_depth is not None:
            cam_pos_f = np.array(cam_matrices[f][:3, 3])
            cam_fwd_f = np.array(cam_matrices[f][:3, 2])
            for i in range(num_objects):
                offset = pos[i] - cam_pos_f
                actual_depth = -np.dot(offset, cam_fwd_f)
                drift = actual_depth - fixed_depth
                pos[i] += cam_fwd_f * drift
                vel[i] -= np.dot(vel[i], cam_fwd_f) * cam_fwd_f

        cam_ext = cam_exts[f]
        cam_rot = cam_ext[:3, :3]
        cam_rot_inv = np.linalg.inv(cam_rot)

        min_depth = cam_radius * 0.35

        for i in range(num_objects):
            if fixed_depth is None:
                p_cam = cam_ext @ np.append(pos[i], 1.0)
                depth = -p_cam[2]

                if depth < min_depth:
                    vel_cam = cam_rot @ vel[i]
                    vel_cam[2] = -abs(vel_cam[2]) * restitution
                    vel[i] = cam_rot_inv @ vel_cam
                    pos[i] += vel[i] * dt * 2
                    continue

            proj = _project(pos[i], cam_ext)
            if proj is None:
                vel[i] = -vel[i] * restitution
                pos[i] += vel[i] * dt * 2
                continue
            px, py = proj
            if px < W * margin:
                vel_cam = cam_rot @ vel[i]
                vel_cam[0] = abs(vel_cam[0]) * restitution
                vel[i] = cam_rot_inv @ vel_cam
            elif px > W * (1 - margin):
                vel_cam = cam_rot @ vel[i]
                vel_cam[0] = -abs(vel_cam[0]) * restitution
                vel[i] = cam_rot_inv @ vel_cam
            if py < H * margin:
                vel_cam = cam_rot @ vel[i]
                vel_cam[1] = -abs(vel_cam[1]) * restitution
                vel[i] = cam_rot_inv @ vel_cam
            elif py > H * (1 - margin):
                vel_cam = cam_rot @ vel[i]
                vel_cam[1] = abs(vel_cam[1]) * restitution
                vel[i] = cam_rot_inv @ vel_cam

        # Sphere-sphere collision
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                min_dist = obj_radius * 2
                if dist < min_dist and dist > 1e-6:
                    normal = diff / dist
                    overlap = min_dist - dist
                    pos[i] += normal * overlap * 0.5
                    pos[j] -= normal * overlap * 0.5

                    rel_vel = vel[i] - vel[j]
                    vn = np.dot(rel_vel, normal)
                    if vn < 0:
                        impulse = -(1 + restitution) * vn * 0.5
                        vel[i] += impulse * normal
                        vel[j] -= impulse * normal

    return all_positions, all_rotations


def run(mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS):
    """Run arbitrary motion rendering (non-physics, keyframed trajectories)."""

    fovx = np.deg2rad(FLAGS.fov_range[0] + np.random.uniform() * (FLAGS.fov_range[1] - FLAGS.fov_range[0]))
    if FLAGS.cam_t_range is not None:
        t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
    else:
        t = np.zeros(3)

    # Linear camera motion: A is random, B = A + small drift
    cam_motion_cfg = getattr(FLAGS, 'camera_motion', {})
    if not isinstance(cam_motion_cfg, dict):
        cam_motion_cfg = {}
    max_az_drift = cam_motion_cfg.get('max_azimuth_drift', 15.0)
    max_el_drift = cam_motion_cfg.get('max_elevation_drift', 8.0)
    max_rad_drift = cam_motion_cfg.get('max_radius_drift', 0.3)

    az_a = np.random.uniform(*FLAGS.cam_phi_range)
    az_b = az_a + np.random.uniform(-max_az_drift, max_az_drift)
    el_a = np.random.uniform(*FLAGS.cam_theta_range)
    el_b = el_a + np.random.uniform(-max_el_drift, max_el_drift)
    rad_a = np.random.uniform(*FLAGS.radius_range)
    rad_b = rad_a + np.random.uniform(-max_rad_drift, max_rad_drift)

    cam_radius = rad_a  # used for spawn calculations
    azimuth = az_a
    elevation = el_a

    cam_list = []
    for it in range(FLAGS.num_frames):
        frac = it / max(FLAGS.num_frames - 1, 1)
        az_t = az_a + (az_b - az_a) * frac
        el_t = np.clip(el_a + (el_b - el_a) * frac, 1.0, 85.0)
        rad_t = rad_a + (rad_b - rad_a) * frac
        cam_matrix = blender_utils.get_cam_matrix(az_t, el_t, t, rad_t)
        cam_list.append(cam_matrix)

    blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX')
    blender_utils.setup_camera_settings(
        resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
    )

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FLAGS.num_frames

    for idx in range(1, len(mesh_list)):
        mesh_list[idx].join_meshes()

    W, H = FLAGS.resolution[1], FLAGS.resolution[0]
    focal = 0.5 * W / np.tan(fovx / 2.0)
    cam_ext = np.linalg.inv(cam_list[0])
    margin = 0.15

    def _is_visible(world_pos):
        p = cam_ext @ np.append(world_pos, 1.0)
        if p[2] >= 0:
            return False
        px = focal * p[0] / (-p[2]) + W / 2.0
        py = focal * (-p[1]) / (-p[2]) + H / 2.0
        return (W * margin < px < W * (1 - margin) and
                H * margin < py < H * (1 - margin))

    spawn_cfg = getattr(FLAGS, 'spawn', {})
    use_flying = spawn_cfg.get('mode', 'region') == 'edges' if isinstance(spawn_cfg, dict) else False

    num_objects = len(mesh_list) - 1
    obj_drivers = []
    for idx in range(1, len(mesh_list)):
        mesh_obj = mesh_list[idx]
        driver = getattr(mesh_obj, 'empty', None)
        if driver is None and mesh_obj.objs:
            driver = mesh_obj.objs[0]
        if driver is not None:
            obj_drivers.append((driver, idx))

    if use_flying and len(obj_drivers) > 0:
        spawn_center = np.array(spawn_cfg.get('region', {}).get('center', [0.0, 0.0, 0.5]))
        motion_cfg = getattr(FLAGS, 'initial_motion', {})
        if isinstance(motion_cfg, dict):
            speed_range = tuple(motion_cfg.get('speed_range', [5.0, 12.0]))
            ang_range = tuple(motion_cfg.get('angular_speed_range', [30.0, 120.0]))
        else:
            speed_range = (5.0, 12.0)
            ang_range = (30.0, 120.0)

        physics_cfg = getattr(FLAGS, 'physics', {})
        if isinstance(physics_cfg, dict):
            restitution = np.mean(physics_cfg.get('restitution_range', [0.6, 0.95]))
        else:
            restitution = 0.8

        fixed_depth_range = None
        fd_cfg = getattr(FLAGS, 'fixed_depth_range', None)
        if fd_cfg is not None and isinstance(fd_cfg, (list, tuple)) and len(fd_cfg) == 2:
            fixed_depth_range = fd_cfg
        fixed_depth = None
        if fixed_depth_range is not None:
            fixed_depth = random.uniform(fixed_depth_range[0], fixed_depth_range[1])
            logger.info(f"Using fixed depth: {fixed_depth:.2f} (range {fixed_depth_range})")

        all_positions, all_rotations = simulate_flying_physics(
            num_objects=len(obj_drivers),
            num_frames=FLAGS.num_frames,
            cam_matrices=cam_list,
            cam_radius=cam_radius,
            fovx=fovx,
            resolution=(W, H),
            spawn_center=spawn_center,
            speed_range=speed_range,
            ang_speed_range=ang_range,
            restitution=restitution,
            obj_radius=0.3,
            fps=getattr(scene.render, 'fps', 24) or 24,
            fixed_depth=fixed_depth,
        )

        for oi, (driver, idx) in enumerate(obj_drivers):
            if driver.animation_data:
                driver.animation_data_clear()
            for fi in range(FLAGS.num_frames):
                driver.location = Vector(all_positions[oi, fi].tolist())
                driver.keyframe_insert(data_path="location", frame=fi + 1)
                driver.rotation_euler = Euler(all_rotations[oi, fi].tolist(), 'XYZ')
                driver.keyframe_insert(data_path="rotation_euler", frame=fi + 1)
            logger.info(f"Keyframed {FLAGS.num_frames} flying frames for object {idx}")

    else:
        obj_init_locs = []
        obj_trajectories = []

        for driver, idx in obj_drivers:
            seed_for_obj = (FLAGS.seed + idx * 997) if hasattr(FLAGS, 'seed') else None
            positions, rotations = generate_arbitrary_trajectory(FLAGS.num_frames, seed=seed_for_obj)

            init_loc = np.array(driver.location)

            if not _is_visible(init_loc):
                for shrink in np.linspace(0.9, 0.0, 20):
                    candidate = init_loc * shrink
                    if _is_visible(candidate):
                        logger.info(f"Object {idx}: initial pos shrunk by {shrink:.2f} to stay in frame")
                        init_loc = candidate
                        driver.location = Vector(init_loc.tolist())
                        break

            scale = 1.0
            for attempt in range(20):
                all_visible = True
                for fi in range(FLAGS.num_frames):
                    world_pos = init_loc + positions[fi] * scale
                    if not _is_visible(world_pos):
                        all_visible = False
                        break
                if all_visible:
                    break
                scale *= 0.7
            if scale < 1.0:
                logger.info(f"Object {idx}: trajectory scaled to {scale:.3f} to stay in frame")

            obj_init_locs.append(init_loc)
            obj_trajectories.append((positions * scale, rotations))

        for (driver, idx), init_loc, (positions, rotations) in zip(
            obj_drivers, obj_init_locs, obj_trajectories
        ):
            init_rot = np.array(driver.rotation_euler)

            if driver.animation_data:
                driver.animation_data_clear()

            for fi in range(FLAGS.num_frames):
                frame_num = fi + 1
                pos = init_loc + positions[fi]
                rot = init_rot + rotations[fi]
                driver.location = Vector(pos.tolist())
                driver.keyframe_insert(data_path="location", frame=frame_num)
                driver.rotation_euler = Euler(rot.tolist(), 'XYZ')
                driver.keyframe_insert(data_path="rotation_euler", frame=frame_num)

            logger.info(f"Keyframed {FLAGS.num_frames} frames for object {idx}")

    object_transforms_per_frame = []

    ori_shortname = shortname
    blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

    for lgt_i in range(FLAGS.num_lighting):
        prefix = f'{lgt_i:04d}.'
        shortname = ori_shortname
        if getattr(FLAGS, 'prefix_in_folder', False):
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)

        save_folder = f"{FLAGS.out_dir}/{shortname}"

        if getattr(FLAGS, 'envlight_sample_weight', None) is not None:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        else:
            envlight_path = random.choice(envlight_path_list)
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if getattr(FLAGS, 'random_env_scale', None) is not None else FLAGS.env_scale
        envmap_flip = getattr(FLAGS, 'random_env_flip', False) and (random.random() > 0.5)
        envmap_rotation_y = random.uniform(0, 2 * np.pi) if getattr(FLAGS, 'random_env_rotation', False) else 0.0

        if envmap_strength == 0.0 or FLAGS.env_scale == 0.0:
            bg_color = getattr(FLAGS, 'bg_color', [1.0, 1.0, 1.0])
            blender_utils.set_solid_color_background(color=tuple(bg_color), strength=1.0, add_uniform_light=False)
        else:
            blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)

        if lgt_i == 0:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['normal', 'depth'], suffix=f'.{0:04d}')
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')

            if FLAGS.dump_features:
                object_names_for_mask = []
                for idx in range(1, len(mesh_list)):
                    if hasattr(mesh_list[idx], 'objs') and len(mesh_list[idx].objs) > 0:
                        for obj in mesh_list[idx].objs:
                            if obj.type == 'MESH':
                                object_names_for_mask.append(obj.name)
                logger.info(f"Masking {len(object_names_for_mask)} objects")
                blender_utils.setup_object_mask_output(output_dir=save_folder, object_names=object_names_for_mask, suffix=f'.{lgt_i:04d}')
                blender_utils.setup_instance_mask_outputs(output_dir=save_folder, object_names=object_names_for_mask, suffix=f'.{lgt_i:04d}')

        if lgt_i == 0 and getattr(FLAGS, 'export_object_motion', True):
            def capture_callback(frame_number):
                return get_object_transforms_at_frame(mesh_list, frame_number)

            object_transforms_per_frame = blender_utils.render_all_frames(
                output_dir=save_folder, num_frames=FLAGS.num_frames,
                suffix=f'rgb.{lgt_i:04d}', capture_transforms_callback=capture_callback
            )
            logger.info(f"Captured {len(object_transforms_per_frame)} frames of object motion")
        else:
            blender_utils.render_all_frames(output_dir=save_folder, num_frames=FLAGS.num_frames, suffix=f'rgb.{lgt_i:04d}')

        fx = float(focal)
        fy = float(focal)
        cx = W / 2.0
        cy = H / 2.0

        meta_dict = {
            'camera_angle_x': fovx,
            'cam_radius': cam_radius,
            'resolution': [int(H), int(W)],
            'intrinsics': {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
            },
            'envmap': os.path.basename(envlight_path),
            'motion_type': 'flying' if use_flying else 'arbitrary',
        }
        meta_frames = []
        for it in range(FLAGS.num_frames):
            frac = it / max(FLAGS.num_frames - 1, 1)
            az_frame = az_a + (az_b - az_a) * frac
            el_frame = float(np.clip(el_a + (el_b - el_a) * frac, 1.0, 85.0))
            rad_frame = rad_a + (rad_b - rad_a) * frac

            meta_frame = {
                'transform_matrix': cam_list[it].tolist(),
                'elevation': float(el_frame),
                'azimuth': float(az_frame),
                'cam_radius': float(rad_frame),
                'envmap_rot': float(envmap_rotation_y),
                'envmap_strength': float(envmap_strength),
                'envmap_flip': bool(envmap_flip),
            }

            if getattr(FLAGS, 'export_object_motion', True) and len(object_transforms_per_frame) > it:
                obj_transforms = []
                for obj_idx, transform in enumerate(object_transforms_per_frame[it]):
                    if transform is not None:
                        if getattr(FLAGS, 'export_motion_skip_plane', True) and obj_idx == 0:
                            continue
                        obj_name = mesh_meta[obj_idx].get('name', f'object_{obj_idx}') if obj_idx < len(mesh_meta) else f'object_{obj_idx}'
                        entry = {
                            'object_id': obj_idx,
                            'object_name': obj_name,
                            'transform_matrix': transform['transform_matrix'].tolist(),
                        }
                        for key in ('center', 'bbox_extent', 'world_aabb_min', 'world_aabb_max', 'obb_corners', 'empty_matrix_world'):
                            if key in transform:
                                entry[key] = transform[key]
                        obj_transforms.append(entry)
                if len(obj_transforms) > 0:
                    meta_frame['object_transforms'] = obj_transforms

            meta_frames.append(meta_frame)
        meta_dict['frames'] = meta_frames
        meta_dict['file_path'] = shortname
        with open(os.path.join(FLAGS.out_dir, shortname, f'{prefix}meta.json'), 'w') as f:
            _meta_dict = copy.deepcopy(meta_dict)
            _meta_dict['mesh_list'] = mesh_meta
            json.dump(_meta_dict, f, indent=4)
