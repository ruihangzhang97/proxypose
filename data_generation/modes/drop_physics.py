import bpy
import numpy as np
import random
import os
import json
import copy
from mathutils import Vector, Matrix

from utils import blender_utils, render_utils, image_utils
import torch
from physics import rigid_body_utils as rb
import logging

logger = logging.getLogger(__name__)


def get_object_transforms_at_frame(mesh_list, frame_idx):
    """Get world transformation matrices for all objects at a specific frame.

    **Translation (exported center):** On ``scene.frame_start``, we take the world AABB
    midpoint of all vertices, map it to the primary mesh's **local** space with
    ``inv(matrix_world)``, and cache it on the ``ObjContainer``. On every frame we set
    ``center_world = matrix_world @ center_local`` so the point is **rigidly attached**
    to the object (same as marking a fixed point on the mesh), instead of recomputing
    the world AABB midpoint each frame (which can drift numerically and is redundant for
    rigid motion).

    ``world_aabb_min`` / ``world_aabb_max`` are still the true **world** axis-aligned
    bounds from all vertices each frame (for overlays / debugging).

    For rigid body physics, we need to ensure the cache is properly evaluated.
    """
    scene = bpy.context.scene
    
    # Set the frame - this updates the rigid body cache point
    scene.frame_set(frame_idx)
    
    # CRITICAL: Force a full scene update including physics evaluation
    # We need to update the scene itself, not just the view layer
    scene.frame_current = frame_idx  # Ensure frame is set
    bpy.context.view_layer.update()  # Update view layer first
    
    # Trigger a full depsgraph update - this should apply rigid body transforms
    for obj in bpy.data.objects:
        obj.update_tag()
    
    # Force depsgraph evaluation
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    
    transforms = []
    for obj_idx, mesh_obj in enumerate(mesh_list):
        if not (hasattr(mesh_obj, 'objs') and len(mesh_obj.objs) > 0):
            transforms.append(None)
            continue

        # World-space AABB over all vertices (for bbox center) + samples for SVD rotation.
        vert_count = 0
        sample_local = []
        sample_world = []
        ws_min = Vector((1e9, 1e9, 1e9))
        ws_max = Vector((-1e9, -1e9, -1e9))
        max_samples = 300

        for part in mesh_obj.objs:
            part_eval = part.evaluated_get(depsgraph)
            if part_eval.type != 'MESH' or part_eval.data is None:
                continue

            mw = part_eval.matrix_world
            for v in part_eval.data.vertices:
                ws = mw @ v.co
                vert_count += 1
                for i in range(3):
                    if ws[i] < ws_min[i]:
                        ws_min[i] = ws[i]
                    if ws[i] > ws_max[i]:
                        ws_max[i] = ws[i]
                if len(sample_local) < max_samples:
                    sample_local.append(np.array(v.co))
                    sample_world.append(np.array(ws))

        if vert_count == 0:
            transforms.append(None)
            continue

        bbox_center_from_verts = (ws_min + ws_max) * 0.5

        primary = None
        for part in mesh_obj.objs:
            pe = part.evaluated_get(depsgraph)
            if pe.type == "MESH" and pe.data is not None:
                primary = pe
                break

        ref_frame = int(scene.frame_start)
        if (
            primary is not None
            and frame_idx == ref_frame
            and not getattr(mesh_obj, "_bbox_center_local_cached", False)
        ):
            mw0 = primary.matrix_world.copy()
            inv0 = mw0.inverted()
            mesh_obj._bbox_center_local = inv0 @ bbox_center_from_verts

            x0, y0, z0 = ws_min.x, ws_min.y, ws_min.z
            x1, y1, z1 = ws_max.x, ws_max.y, ws_max.z
            aabb_corners_world = [
                Vector((x0, y0, z0)), Vector((x1, y0, z0)),
                Vector((x0, y1, z0)), Vector((x1, y1, z0)),
                Vector((x0, y0, z1)), Vector((x1, y0, z1)),
                Vector((x0, y1, z1)), Vector((x1, y1, z1)),
            ]
            mesh_obj._obb_corners_local = [inv0 @ c for c in aabb_corners_world]
            mesh_obj._bbox_center_local_cached = True

        if getattr(mesh_obj, "_bbox_center_local_cached", False) and primary is not None:
            mw_now = primary.matrix_world
            bbox_center_world = mw_now @ mesh_obj._bbox_center_local
            obb_corners_world = [mw_now @ c for c in mesh_obj._obb_corners_local]
        else:
            bbox_center_world = bbox_center_from_verts
            obb_corners_world = None

        # Compute rotation via SVD (Procrustes).
        # This is robust regardless of parenting or rigid body quirks.
        L = np.array(sample_local)
        W = np.array(sample_world)
        L_centered = L - L.mean(axis=0)
        W_centered = W - np.array(bbox_center_world)
        H = L_centered.T @ W_centered
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        rot_matrix = Matrix(R.tolist()).to_3x3()

        # Uniform scale: average AABB extent, frozen on first frame we compute it.
        if not hasattr(mesh_obj, "_uniform_scale"):
            extents = ws_max - ws_min
            mesh_obj._uniform_scale = (extents.x + extents.y + extents.z) / 3.0
        uni_s = mesh_obj._uniform_scale
        scale = Vector((uni_s, uni_s, uni_s))

        transform_matrix = Matrix.LocRotScale(
            bbox_center_world, rot_matrix.to_quaternion(), scale
        )
        bbox_ext = ws_max - ws_min
        entry = {
            "transform_matrix": np.array(transform_matrix),
            "center": [float(bbox_center_world.x), float(bbox_center_world.y), float(bbox_center_world.z)],
            "bbox_extent": [float(bbox_ext.x), float(bbox_ext.y), float(bbox_ext.z)],
            "world_aabb_min": [float(ws_min.x), float(ws_min.y), float(ws_min.z)],
            "world_aabb_max": [float(ws_max.x), float(ws_max.y), float(ws_max.z)],
        }
        if obb_corners_world is not None:
            entry["obb_corners"] = [[float(c.x), float(c.y), float(c.z)] for c in obb_corners_world]

        if hasattr(mesh_obj, 'empty') and mesh_obj.empty is not None:
            empty_eval = mesh_obj.empty.evaluated_get(depsgraph)
            entry["empty_matrix_world"] = np.array(empty_eval.matrix_world).tolist()

        transforms.append(entry)

        if frame_idx <= 3 and obj_idx == 1:
            logger.info(
                f"  Frame {frame_idx}: object[{obj_idx}] bbox_center_world="
                f"[{bbox_center_world.x:.4f}, {bbox_center_world.y:.4f}, {bbox_center_world.z:.4f}] "
                f"scale={uni_s:.4f} verts={vert_count}"
            )
    return transforms


def _sample_velocity(downward_bias=0.7, speed_range=(0.0, 3.0)):
    speed = random.uniform(*speed_range)
    dir_vec = np.random.randn(3)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    if downward_bias > 0:
        down = np.array([0.0, 0.0, -1.0])
        blended = downward_bias * down + (1.0 - downward_bias) * dir_vec
        blended = blended / (np.linalg.norm(blended) + 1e-8)
        return (blended * speed).tolist()
    return (dir_vec * speed).tolist()


def _sample_angular_speed(range_deg_s=(0.0, 30.0)):
    # Random axis with random speed mapped to radians
    deg_s = random.uniform(*range_deg_s)
    rad_s = np.deg2rad(deg_s)
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (axis * rad_s).tolist()


def run(mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS):
    # Camera setup (randomized similar to vtran_obj / rotat_obj)
    cam_radius = FLAGS.radius_range[0] + np.random.uniform() * (FLAGS.radius_range[1] - FLAGS.radius_range[0])
    fovx = np.deg2rad(FLAGS.fov_range[0] + np.random.uniform() * (FLAGS.fov_range[1] - FLAGS.fov_range[0]))
    azimuth = np.random.uniform(*FLAGS.cam_phi_range)
    elevation = np.random.uniform(*FLAGS.cam_theta_range)
    if FLAGS.cam_t_range is not None:
        t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
    else:
        t = np.zeros(3)

    # Optional varying radius like in render_scene
    cam_radius_list = None
    if getattr(FLAGS, 'varying_radius', False):
        if random.random() < 0.3:
            roll_step = np.random.uniform(-np.pi/2, np.pi/2)
            cam_radius_list = FLAGS.radius_range[0] + (FLAGS.radius_range[1] - FLAGS.radius_range[0]) * \
                (1 + np.sin(np.linspace(0, 2*np.pi, FLAGS.num_frames, endpoint=False) + roll_step)) / 2

    # Linear camera motion: A → B with gentle drift
    cam_motion_cfg = getattr(FLAGS, 'camera_motion', {})
    if not isinstance(cam_motion_cfg, dict):
        cam_motion_cfg = {}
    max_az_drift = cam_motion_cfg.get('max_azimuth_drift', 0.0)
    max_el_drift = cam_motion_cfg.get('max_elevation_drift', 0.0)
    max_rad_drift = cam_motion_cfg.get('max_radius_drift', 0.0)

    az_b = azimuth + np.random.uniform(-max_az_drift, max_az_drift)
    el_b = elevation + np.random.uniform(-max_el_drift, max_el_drift)
    rad_b = cam_radius + np.random.uniform(-max_rad_drift, max_rad_drift)

    cam_list = []
    for it in range(FLAGS.num_frames):
        frac = it / max(FLAGS.num_frames - 1, 1)
        az_t = azimuth + (az_b - azimuth) * frac
        el_t = np.clip(elevation + (el_b - elevation) * frac, 1.0, 85.0)
        rad_t = cam_radius + (rad_b - cam_radius) * frac
        if cam_radius_list is not None:
            rad_t = cam_radius_list[it]
        cam_matrix = blender_utils.get_cam_matrix(az_t, el_t, t, rad_t)
        cam_list.append(cam_matrix)

    blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX')
    blender_utils.setup_camera_settings(
        resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
    )

    # Physics world
    rb.ensure_rigidbody_world(
        gravity=tuple(FLAGS.physics.get('gravity', [0.0, 0.0, -9.81])),
        steps_per_second=int(FLAGS.physics.get('steps_per_second', 240)),
        substeps_per_frame=int(FLAGS.physics.get('substeps_per_frame', 5)),
        solver_iterations=int(FLAGS.physics.get('solver_iterations', 10)),
        split_impulse=bool(FLAGS.physics.get('split_impulse', True)),
        cache_frames=int(FLAGS.physics.get('cache_frames', 250)),
    )
    # Ground and optional walls
    ground_restitution = float(FLAGS.physics.get('ground_restitution', 0.5))
    ground_friction = float(FLAGS.physics.get('ground_friction', 0.8))
    skip_ground = FLAGS.environment.get('skip_ground', False)

    if not skip_ground:
        if len(mesh_list) > 0 and FLAGS.physics.get('set_rigidbody_plane', True):
            try:
                rb.add_passive_rigidbody(mesh_list[0], collision_shape='MESH', friction=ground_friction, restitution=ground_restitution)
            except Exception:
                ground_size = float(FLAGS.environment.get('ground_size', 8.0))
                rb.add_passive_ground(size=ground_size, location=FLAGS.placement_plane_offset, transparent=True, 
                                    restitution=ground_restitution, friction=ground_friction)
        else:
            ground_size = float(FLAGS.environment.get('ground_size', 8.0))
            rb.add_passive_ground(size=ground_size, location=FLAGS.placement_plane_offset, transparent=True,
                                restitution=ground_restitution, friction=ground_friction)

    if FLAGS.environment.get('walls', {}).get('enabled', False):
        size_xy = FLAGS.environment['walls'].get('size', [6.0, 6.0])
        height = float(FLAGS.environment['walls'].get('height', 2.0))
        x, y = size_xy
        wall_center_z = float(FLAGS.environment['walls'].get('center_z', height / 2.0))
        rb.add_wall(size_x=x, size_y=0.2, height=height, location=(0.0, -y/2.0, wall_center_z), name='WallSouth', 
                   restitution=ground_restitution, friction=ground_friction)
        rb.add_wall(size_x=x, size_y=0.2, height=height, location=(0.0,  y/2.0, wall_center_z), name='WallNorth',
                   restitution=ground_restitution, friction=ground_friction)
        rb.add_wall(size_x=0.2, size_y=y, height=height, location=(-x/2.0, 0.0, wall_center_z), name='WallWest',
                   restitution=ground_restitution, friction=ground_friction)
        rb.add_wall(size_x=0.2, size_y=y, height=height, location=( x/2.0, 0.0, wall_center_z), name='WallEast',
                   restitution=ground_restitution, friction=ground_friction)
        # Floor and ceiling walls for full enclosure
        if FLAGS.environment['walls'].get('enclose', False):
            rb.add_passive_ground(size=max(x, y), location=(0.0, 0.0, wall_center_z - height / 2.0),
                                transparent=True, restitution=ground_restitution, friction=ground_friction)
            rb.add_passive_ground(size=max(x, y), location=(0.0, 0.0, wall_center_z + height / 2.0),
                                transparent=True, restitution=ground_restitution, friction=ground_friction)

    # Spawn region
    spawn_mode = FLAGS.spawn.get('mode', 'region')
    region = FLAGS.spawn.get('region', {'center': [0.0, 0.0, 1.5], 'size': [1.0, 1.0, 0.4]})
    center = np.array(region.get('center', [0.0, 0.0, 1.5]), dtype=float)
    size = np.array(region.get('size', [1.0, 1.0, 0.4]), dtype=float)

    # Pre-compute camera view vectors for "edges" spawn mode.
    # cam_list[0] is a camera-to-world 4x4 matrix.
    cam_right = np.array(cam_list[0][:3, 0])
    cam_up = np.array(cam_list[0][:3, 1])
    cam_forward = -np.array(cam_list[0][:3, 2])
    cam_pos = np.array(cam_list[0][:3, 3])
    half_w = cam_radius * np.tan(fovx / 2.0)
    # Distance from scene center at which objects are just off-screen
    edge_offset = half_w * float(FLAGS.spawn.get('edge_overshoot', 1.3))

    # Join multi-part GLBs into one mesh per ObjContainer (skip plane at index 0).
    # Without this, add_active_rigidbody gives every mesh part its own body and the object "explodes".
    for idx in range(1, len(mesh_list)):
        mesh_list[idx].join_meshes()

    # Unparent meshes from empties so rigid body has full control over matrix_world.
    # Without this, Blender's depsgraph recomputes matrix_world from the parent chain,
    # overwriting the physics rotation.
    for idx in range(1, len(mesh_list)):
        mesh_obj = mesh_list[idx]
        if getattr(mesh_obj, 'empty', None) is not None:
            empty = mesh_obj.empty
            mesh_obj._empty_scale = Vector(empty.scale)
            for obj in mesh_obj.objs:
                world_mat = obj.matrix_world.copy()
                obj.parent = None
                obj.matrix_world = world_mat
            bpy.context.view_layer.update()
    
    # Create active bodies for objects (skip plane if present at index 0)
    physics_cfg = FLAGS.physics
    for idx in range(1, len(mesh_list)):
        rb.add_active_rigidbody(
            mesh_list[idx],
            mass=float(physics_cfg.get('mass', 1.0)),
            friction=float(random.uniform(*physics_cfg.get('friction_range', [0.3, 0.9]))),
            restitution=float(random.uniform(*physics_cfg.get('restitution_range', [0.2, 0.8]))),
            collision_shape=str(physics_cfg.get('collision_shape', 'CONVEX_HULL')),
            collision_margin=float(physics_cfg.get('collision_margin', 0.001)),
            use_deactivation=bool(physics_cfg.get('use_deactivation', True)),
        )

        if spawn_mode == 'edges':
            # Spawn just outside a random edge of the camera frame, aimed inward.
            edge = random.choice(['left', 'right', 'top', 'bottom'])
            slide = random.uniform(-0.8, 0.8)
            z_jitter = random.uniform(-0.3, 0.3)
            if edge == 'left':
                loc = center - cam_right * edge_offset + cam_up * slide * half_w
            elif edge == 'right':
                loc = center + cam_right * edge_offset + cam_up * slide * half_w
            elif edge == 'top':
                loc = center + cam_up * edge_offset + cam_right * slide * half_w
            else:
                loc = center - cam_up * edge_offset + cam_right * slide * half_w
            loc[2] += z_jitter

            # Velocity: toward scene center with some spread
            to_center = center - loc
            to_center = to_center / (np.linalg.norm(to_center) + 1e-8)
            spread = np.random.randn(3) * 0.2
            vel_dir = to_center + spread
            vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-8)
            speed = random.uniform(*FLAGS.initial_motion.get('speed_range', [4.0, 10.0]))
            lin_v = (vel_dir * speed).tolist()
            logger.info(f"Object {idx}: edge spawn '{edge}' at {loc.round(2)} vel={np.array(lin_v).round(2)}")
        else:
            loc = center + (np.random.rand(3) - 0.5) * size
            gravity = physics_cfg.get('gravity', [0.0, 0.0, -9.81])
            has_gravity = any(abs(g) > 0.01 for g in gravity)
            if has_gravity:
                loc[2] = max(loc[2], 0.6)
            lin_v = _sample_velocity(
                downward_bias=float(FLAGS.initial_motion.get('downward_bias', 0.7)),
                speed_range=tuple(FLAGS.initial_motion.get('speed_range', [0.0, 3.0]))
            )

        for obj in mesh_list[idx].objs:
            obj.location = Vector(loc.tolist())
        bpy.context.view_layer.update()

        ang_v = _sample_angular_speed(tuple(FLAGS.initial_motion.get('angular_speed_range', [0.0, 30.0])))
        rb.set_initial_velocity(mesh_list[idx], linear=lin_v, angular=ang_v)

    # After keyframing the initial conditions, ensure frame range
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FLAGS.num_frames

    logger.info(f"Baking physics simulation for frames 1-{FLAGS.num_frames}...")
    
    # Bake physics once before rendering under multiple lights
    try:
        rb.bake_rigidbody_cache(frame_start=1, frame_end=FLAGS.num_frames)
        logger.info("Physics bake completed successfully")
    except Exception as e:
        logger.warning(f"Physics bake failed: {e}, falling back to frame stepping")
        for f in range(1, FLAGS.num_frames + 1):
            scene.frame_set(f)

    # We'll capture transforms DURING rendering (not before) for rigid body physics
    # This will be populated by the render callback
    object_transforms_per_frame = []

    # Precompute envmap helper data for dumping
    dump_format = FLAGS.dump_format
    vec = None
    if FLAGS.dump_envmap:
        vec = render_utils.latlong_vec(FLAGS.resolution)

    ori_shortname = shortname
    blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

    for lgt_i in range(FLAGS.num_lighting):
        # Prefix and folder handling similar to render_scene
        prefix = f'{lgt_i:04d}.'
        shortname = ori_shortname
        if getattr(FLAGS, 'prefix_in_folder', False):
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)

        save_folder = f"{FLAGS.out_dir}/{shortname}"

        # Envmap augmentation
        if getattr(FLAGS, 'envlight_sample_weight', None) is not None:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        else:
            envlight_path = random.choice(envlight_path_list)
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if getattr(FLAGS, 'random_env_scale', None) is not None else FLAGS.env_scale
        envmap_flip = False
        if getattr(FLAGS, 'random_env_flip', False) and (random.random() > 0.5):
            envmap_flip = True
        if getattr(FLAGS, 'random_env_rotation', False):
            envmap_rotation_y = random.uniform(0, 2*np.pi)
        else:
            envmap_rotation_y = 0.0

        # Optional envmap dump (pass-level overview)
        cubemap = None
        if FLAGS.dump_envmap:
            latlong_img = image_utils.read_img(envlight_path)
            latlong_img = latlong_img * envmap_strength
            latlong_img = np.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0)
            latlong_img = np.clip(latlong_img, 0.0, 65504.0)
            latlong_img = torch.tensor(latlong_img, dtype=torch.float32)
            if envmap_flip:
                latlong_img = latlong_img.flip(1)
            cubemap = render_utils.latlong_to_cubemap_torch(latlong_img, [512, 512])
            env_proj = render_utils.cubemap_sample_torch(cubemap, -vec)
            env_proj = env_proj.flip(0).flip(1)
            env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
            env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_ldr.{dump_format}'), env_ev0)
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_log.{dump_format}'), env_log)

        # Set envmap in Blender for actual rendering
        # Check if we should use solid color background instead
        if envmap_strength == 0.0 or FLAGS.env_scale == 0.0:
            bg_color = getattr(FLAGS, 'bg_color', [1.0, 1.0, 1.0])
            logger.info(f"Using solid color background: {bg_color}")
            # Disable uniform light to prevent lighting on objects
            blender_utils.set_solid_color_background(color=tuple(bg_color), strength=1.0, add_uniform_light=False)
        else:
            blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)

        # Setup feature passes only for first lighting to save time
        if lgt_i == 0:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['normal', 'depth'], suffix=f'.{0:04d}')
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')
            
            # Setup object mask output if dump_features is enabled
            if FLAGS.dump_features:
                logger.info("Setting up object mask output...")
                # Extract object names from mesh_list (skip plane at index 0)
                object_names_for_mask = []
                for idx in range(1, len(mesh_list)):
                    if hasattr(mesh_list[idx], 'objs') and len(mesh_list[idx].objs) > 0:
                        for obj in mesh_list[idx].objs:
                            if obj.type == 'MESH':
                                object_names_for_mask.append(obj.name)
                                logger.info(f"Will mask object: {obj.name}")
                logger.info(f"Masking {len(object_names_for_mask)} objects explicitly")
                blender_utils.setup_object_mask_output(output_dir=save_folder, object_names=object_names_for_mask, suffix=f'.{lgt_i:04d}')
                blender_utils.setup_instance_mask_outputs(output_dir=save_folder, object_names=object_names_for_mask, suffix=f'.{lgt_i:04d}')

        # Per-frame env projections if requested
        if FLAGS.dump_envmap and cubemap is not None:
            y_rot = render_utils.rotate_y(envmap_rotation_y, device=latlong_img.device)
            for it in range(FLAGS.num_frames):
                c2w = render_utils.convert_cam_mat_blender_to_dr(cam_list[it])
                c2w = torch.tensor(c2w, dtype=torch.float32, device=latlong_img.device)
                vec_cam = vec.reshape(-1, 3) @ c2w[:3, :3].T
                vec_query = (vec_cam @ y_rot[:3, :3].T).reshape(1, *FLAGS.resolution, 3)
                env_frame = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                env_frame = env_frame.flip(0).flip(1)
                env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_frame, max_point=16).clip(0, 1)).cpu().numpy()
                env_log = render_utils.rgb_to_srgb(torch.log1p(env_frame) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_ldr.{dump_format}'), env_ev0)
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_log.{dump_format}'), env_log)

        # Render RGB frames for this lighting setup
        # For first lighting pass, capture object transforms during rendering
        if lgt_i == 0 and getattr(FLAGS, 'export_object_motion', True):
            # Define callback to capture transforms during rendering.
            # IMPORTANT: use geometric-center-aware export helper, not raw mesh matrix_world.
            def capture_callback(frame_number):
                """Capture transforms AFTER rendering using geometric centers."""
                transforms = get_object_transforms_at_frame(mesh_list, frame_number)
                # Keep a compact debug log for first frames.
                if frame_number <= 5 and len(transforms) > 1 and transforms[1] is not None:
                    pos = transforms[1]["transform_matrix"][:3, 3]
                    logger.info(
                        f"  Frame {frame_number}: object[1] bbox center "
                        f"[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]"
                    )
                return transforms
            
            # Render with transform capture
            object_transforms_per_frame = blender_utils.render_all_frames(
                output_dir=save_folder, num_frames=FLAGS.num_frames, 
                suffix=f'rgb.{lgt_i:04d}', capture_transforms_callback=capture_callback
            )
            logger.info(f"Captured {len(object_transforms_per_frame)} frames of object motion during rendering")
        else:
            # Just render without capture
            blender_utils.render_all_frames(output_dir=save_folder, num_frames=FLAGS.num_frames, suffix=f'rgb.{lgt_i:04d}')

        W, H = FLAGS.resolution[1], FLAGS.resolution[0]
        focal = 0.5 * W / np.tan(fovx / 2.0)
        fx = float(focal)
        fy = float(focal)
        cx = W / 2.0
        cy = H / 2.0

        meta_dict = {
            'camera_angle_x': fovx,
            'cam_radius': float(cam_radius),
            'resolution': [int(H), int(W)],
            'intrinsics': {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
            },
            'envmap': os.path.basename(envlight_path),
            'motion_type': 'drop_physics',
        }
        meta_frames = []
        for it in range(FLAGS.num_frames):
            frac = it / max(FLAGS.num_frames - 1, 1)
            az_frame = azimuth + (az_b - azimuth) * frac
            el_frame = float(np.clip(elevation + (el_b - elevation) * frac, 1.0, 85.0))
            rad_frame = cam_radius + (rad_b - cam_radius) * frac
            meta_frame = {
                'transform_matrix': cam_list[it].tolist(),
                'elevation': el_frame,
                'azimuth': float(az_frame),
                'cam_radius': float(rad_frame),
                'envmap_rot': float(envmap_rotation_y),
                'envmap_strength': float(envmap_strength),
                'envmap_flip': bool(envmap_flip),
            }
            
            # Add object transforms if available
            if getattr(FLAGS, 'export_object_motion', True) and len(object_transforms_per_frame) > it:
                obj_transforms = []
                for obj_idx, transform in enumerate(object_transforms_per_frame[it]):
                    if transform is not None:
                        # Skip objects based on config
                        if getattr(FLAGS, 'export_motion_skip_plane', True) and obj_idx == 0:
                            continue  # Skip ground plane (always static)
                        
                        obj_name = mesh_meta[obj_idx].get('name', f'object_{obj_idx}') if obj_idx < len(mesh_meta) else f'object_{obj_idx}'
                        
                        # Optional: filter by name pattern
                        if hasattr(FLAGS, 'export_motion_filter') and FLAGS.export_motion_filter:
                            if not any(pattern in obj_name for pattern in FLAGS.export_motion_filter):
                                continue
                        
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


