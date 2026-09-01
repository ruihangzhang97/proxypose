import bpy
from mathutils import Vector


def ensure_rigidbody_world(
    gravity=(0.0, 0.0, -9.81),
    steps_per_second=240,
    substeps_per_frame=5,
    solver_iterations=10,
    split_impulse=True,
    cache_frames=250,
):
    scene = bpy.context.scene
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    # Cache frames
    if hasattr(rbw, 'point_cache'):
        rbw.point_cache.frame_start = 1
        rbw.point_cache.frame_end = max(getattr(scene, 'frame_end', cache_frames), cache_frames)
    # Time scale
    if hasattr(rbw, 'time_scale'):
        rbw.time_scale = 1.0
    # Steps per second (version-safe)
    if hasattr(rbw, 'steps_per_second'):
        rbw.steps_per_second = int(steps_per_second)
    elif hasattr(rbw, 'time_steps'):
        rbw.time_steps = int(steps_per_second)
    # Solver iterations
    if hasattr(rbw, 'solver_iterations'):
        rbw.solver_iterations = int(solver_iterations)
    # Substeps per frame
    if hasattr(rbw, 'substeps_per_frame'):
        rbw.substeps_per_frame = int(substeps_per_frame)
    elif hasattr(rbw, 'substeps'):
        rbw.substeps = int(substeps_per_frame)
    # Split impulse
    if hasattr(rbw, 'use_split_impulse'):
        rbw.use_split_impulse = bool(split_impulse)
    # Gravity
    scene.gravity = Vector(gravity)
    return rbw


def add_passive_ground(size=10.0, location=(0.0, 0.0, 0.0), transparent=False, restitution=0.5, friction=0.8):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    ground = bpy.context.active_object
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.collision_shape = 'BOX'
    ground.rigid_body.restitution = restitution
    ground.rigid_body.friction = friction
    ground.name = 'GroundPlane'
    if transparent:
        # Assign transparent material for invisible ground in renders
        mat = bpy.data.materials.new(name="GroundPlane_Transparent")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent_node.location = (0, 0)
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)
        links.new(transparent_node.outputs['BSDF'], output.inputs['Surface'])
        mat.blend_method = 'BLEND'
        if hasattr(mat, 'shadow_method'):
            mat.shadow_method = 'NONE'
        ground.data.materials.clear()
        ground.data.materials.append(mat)
    return ground


def add_passive_rigidbody(obj, collision_shape='BOX', friction=0.5, restitution=0.0):
    """Convert an existing object or ObjContainer into a PASSIVE rigid body.

    This preserves existing geometry and materials (e.g., textured placement plane).
    """
    targets = []
    if hasattr(obj, 'objs'):
        targets = obj.objs
    elif isinstance(obj, bpy.types.Object):
        targets = [obj]
    else:
        return

    for o in targets:
        if o.type not in {'MESH', 'CURVE'}:
            continue
        bpy.context.view_layer.objects.active = o
        bpy.ops.rigidbody.object_add()
        o.rigid_body.type = 'PASSIVE'
        o.rigid_body.collision_shape = collision_shape
        o.rigid_body.friction = friction
        o.rigid_body.restitution = restitution

def add_wall(size_x=6.0, size_y=0.2, height=2.0, location=(0.0, 0.0, 0.0), name='Wall', restitution=0.5, friction=0.5):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    wall = bpy.context.active_object
    wall.scale = Vector((size_x / 2.0, size_y / 2.0, height / 2.0))
    bpy.context.view_layer.update()
    bpy.ops.rigidbody.object_add()
    wall.rigid_body.type = 'PASSIVE'
    wall.rigid_body.collision_shape = 'BOX'
    wall.rigid_body.restitution = restitution
    wall.rigid_body.friction = friction
    wall.name = name
    # Assign transparent material so walls are invisible in renders
    mat = bpy.data.materials.new(name=f"{name}_Transparent")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    transparent.location = (0, 0)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    links.new(transparent.outputs['BSDF'], output.inputs['Surface'])
    # Ensure full transparency behavior
    mat.blend_method = 'BLEND'
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'
    wall.data.materials.clear()
    wall.data.materials.append(mat)
    return wall


def add_active_rigidbody(obj, mass=1.0, friction=0.5, restitution=0.4, collision_shape='CONVEX_HULL', collision_margin=0.002, use_deactivation=True):
    # obj can be a Blender object or an ObjContainer with .objs
    targets = []
    if hasattr(obj, 'objs'):
        targets = obj.objs
    elif isinstance(obj, bpy.types.Object):
        targets = [obj]
    else:
        return

    for o in targets:
        if o.type not in {'MESH', 'CURVE'}:
            continue
        bpy.context.view_layer.objects.active = o
        bpy.ops.rigidbody.object_add()
        o.rigid_body.type = 'ACTIVE'
        o.rigid_body.mass = mass
        o.rigid_body.friction = friction
        o.rigid_body.restitution = restitution
        o.rigid_body.collision_shape = collision_shape
        o.rigid_body.collision_margin = collision_margin
        o.rigid_body.use_deactivation = use_deactivation


def set_initial_velocity(obj, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
    targets = []
    if hasattr(obj, 'objs'):
        targets = obj.objs
    elif isinstance(obj, bpy.types.Object):
        targets = [obj]
    else:
        return

    scene = bpy.context.scene
    fps = getattr(scene.render, 'fps', 24) or 24
    dt = 1.0 / float(fps)

    for o in targets:
        rb = getattr(o, 'rigid_body', None)
        if rb is None:
            continue

        # Save the spawn location/rotation BEFORE frame_set (which triggers
        # a depsgraph update that can overwrite obj.location).
        saved_loc = o.location.copy()
        saved_rot = tuple(o.rotation_euler)

        start_f = max(getattr(scene, 'frame_start', 1), 1)

        # Make object kinematic for frames 1-2 so we control its position
        if hasattr(rb, 'kinematic'):
            rb.kinematic = True
            rb.keyframe_insert(data_path="kinematic", frame=start_f)
            rb.keyframe_insert(data_path="kinematic", frame=start_f + 1)

        # Restore spawn transform and keyframe at frame start_f
        scene.frame_set(start_f)
        o.location = saved_loc
        o.rotation_euler = saved_rot
        o.keyframe_insert(data_path="location", frame=start_f)
        o.keyframe_insert(data_path="rotation_euler", frame=start_f)

        # Frame start_f+1: advance by velocity * dt
        new_loc = saved_loc + Vector(linear) * dt
        new_rot = (saved_rot[0] + angular[0] * dt,
                   saved_rot[1] + angular[1] * dt,
                   saved_rot[2] + angular[2] * dt)
        o.location = new_loc
        o.rotation_euler = new_rot
        o.keyframe_insert(data_path="location", frame=start_f + 1)
        o.keyframe_insert(data_path="rotation_euler", frame=start_f + 1)

        # Frame start_f+2: switch to dynamic so physics takes over
        if hasattr(rb, 'kinematic'):
            rb.kinematic = False
            rb.keyframe_insert(data_path="kinematic", frame=start_f + 2)


def bake_rigidbody_cache(frame_start=1, frame_end=None):
    scene = bpy.context.scene
    if frame_end is None:
        frame_end = scene.frame_end
    scene.frame_set(frame_start)
    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frame_start)


