import os
import bpy
import math
import random
import mathutils
from mathutils import Vector, Quaternion, Matrix
import numpy as np
import logging
import glob
logger = logging.getLogger(__name__)

PI_2 = 1.5707963267948966

class ObjContainer:
    def __init__(self, appended_objs, with_empty=True, recenter=True, rescale=True, file_name=None):
        self.objs = appended_objs
        self.empty = None
        self.recenter = recenter
        self.rescale = rescale
        self.file_name=file_name
        
        appended_obj_name = appended_objs[0].name
        if with_empty:
            empty_name = f"{appended_obj_name}_root"
            self.empty = bpy.data.objects.new(empty_name, None)
            bpy.context.collection.objects.link(self.empty)
            for obj in appended_objs:
                obj.parent = self.empty
        
        # Calculate AABB after setting up the hierarchy
        self.aabb = self.get_aabb()
        min_corner, max_corner = self.aabb
        center = (min_corner + max_corner) * 0.5
 
        if recenter:
            for obj in appended_objs:
                obj.location = obj.location - center
            # Recompute AABB after transforms to ensure it's up-to-date
            self.aabb = self.get_aabb()

        if rescale:
            vmin, vmax = self.aabb
            scale_factor = 1.0 / max(vmax - vmin)
            self.unit_rescale = float(scale_factor)
            if with_empty:
                self.empty.scale = self.empty.scale * scale_factor
            else:
                for obj in appended_objs:
                    obj.scale = obj.scale * scale_factor
            self.aabb = self.get_aabb()
        else:
            self.unit_rescale = 1.0

    def get_aabb(self):
        # Ensure scene/depsgraph updates are applied before reading transforms
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass

        min_corner = Vector((float('inf'), float('inf'), float('inf')))
        max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for obj in self.objs:
            # Skip objects without a bounding box (e.g., empties, lights)
            if not hasattr(obj, "bound_box") or obj.bound_box is None:
                continue
            # Get world-space bounding box corners
            for v in obj.bound_box:
                # Use the object's world matrix which already accounts for parent transforms
                world_v = obj.matrix_world @ Vector(v)
                min_corner = Vector((min(min_corner[i], world_v[i]) for i in range(3)))
                max_corner = Vector((max(max_corner[i], world_v[i]) for i in range(3)))

        return (min_corner, max_corner)

    def join_meshes(self):
        """Merge all mesh objects in this container into one mesh.

        Multi-part GLBs are imported as many mesh objects. Drop-physics assigns an
        active rigid body to *each* mesh unless they are joined first, which makes
        the object look like it explodes. Joining preserves materials and yields a
        single rigid solid.
        """
        mesh_objs = [
            o
            for o in self.objs
            if getattr(o, "type", None) == "MESH" and o.data is not None
        ]
        if len(mesh_objs) <= 1:
            return

        others = [o for o in self.objs if o not in mesh_objs]
        prev_active = bpy.context.view_layer.objects.active
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for o in mesh_objs:
                o.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objs[0]
            if mesh_objs[0].mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.join()
            joined = bpy.context.active_object
            if joined is None or joined.type != "MESH":
                logger.warning("join_meshes: join() did not yield a mesh; leaving objs unchanged")
                return
            self.objs = others + [joined]
            if self.empty is not None:
                mw = joined.matrix_world.copy()
                joined.parent = self.empty
                joined.matrix_world = mw
            bpy.context.view_layer.update()
            self.aabb = self.get_aabb()
            logger.info("join_meshes: merged %d mesh parts into one object", len(mesh_objs))
        except Exception as e:
            logger.warning("join_meshes failed (%s); physics may still use separate bodies per part", e)
        finally:
            try:
                bpy.ops.object.select_all(action="DESELECT")
            except Exception:
                pass
            if prev_active and prev_active.name in bpy.data.objects:
                bpy.context.view_layer.objects.active = prev_active

    def get_parts_matrix_local(self):
        """Return list of per-part matrix_local (relative to the empty) for MESH parts."""
        parts = []
        for obj in self.objs:
            if obj.type != 'MESH' or obj.data is None:
                continue
            parts.append({
                'name': obj.name,
                'matrix_local': [list(row) for row in obj.matrix_local],
            })
        return parts

    def export_canonical_ply(self, output_path):
        """Export all mesh parts as a single PLY in the empty's local space.

        Vertices have GLTF import + axis conversion + recenter + unit_rescale
        baked in (via each part's matrix_local).  The empty's own transforms
        (smpl_scale, rotation, translation, per-frame animation) are NOT
        included.

        To get world-space vertices at frame t::

            empty_mw = np.array(frame['object_transforms'][j]['empty_matrix_world'])
            v_world = (empty_mw @ np.c_[v_ply, np.ones(N)].T).T[:, :3]
        """
        import struct
        bpy.context.view_layer.update()

        all_verts = []
        all_faces = []
        vert_offset = 0
        for obj in self.objs:
            if obj.type != 'MESH' or obj.data is None:
                continue
            ml = obj.matrix_local
            mesh_data = obj.data
            mesh_data.calc_loop_triangles()
            for v in mesh_data.vertices:
                local_co = ml @ v.co
                all_verts.append((float(local_co.x), float(local_co.y), float(local_co.z)))
            for tri in mesh_data.loop_triangles:
                all_faces.append(tuple(idx + vert_offset for idx in tri.vertices))
            vert_offset += len(mesh_data.vertices)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            header = (
                f"ply\n"
                f"format binary_little_endian 1.0\n"
                f"element vertex {len(all_verts)}\n"
                f"property float x\n"
                f"property float y\n"
                f"property float z\n"
                f"element face {len(all_faces)}\n"
                f"property list uchar int vertex_indices\n"
                f"end_header\n"
            )
            f.write(header.encode('ascii'))
            for v in all_verts:
                f.write(struct.pack('<fff', *v))
            for face in all_faces:
                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<iii', *face))

    def apply_transform(self, translation, rotation=0, scale=1, move_empty=True):
        """Apply the given location and rotation to the object"""
        vec_location = Vector(translation)
        if self.empty is not None and move_empty:
            self.empty.scale = self.empty.scale * scale
            self.empty.location = self.empty.location + vec_location
            self.empty.rotation_euler[2] = self.empty.rotation_euler[2] + rotation
        else:
            for obj in self.objs:
                obj.scale = obj.scale * scale
                obj.location = obj.location + vec_location
                obj.rotation_euler[2] = obj.rotation_euler[2] + rotation

        self.aabb = self.get_aabb()

    def apply_texture(self, texture_dir, texture_scale=1.0, emissive=False):
        """
        Apply PBR textures from a directory to all objects in the container.
        
        Args:
            texture_dir: Directory path containing texture files with naming convention:
                - xxx_diff_*.* (diffuse/albedo)
                - xxx_disp_*.* (displacement/height)
                - xxx_nor_gl_*.* (normal map)
                - xxx_rough_*.* (roughness)
                - xxx_metal_*.* (metallic)
            emissive: If True, drive Principled emission from albedo (for planes when env_scale=0).
        """
        texture_dir = os.path.abspath(texture_dir)
        if not os.path.exists(texture_dir):
            logger.warning(f"Texture directory does not exist: {texture_dir}")
            return
        
        # Create material from texture directory
        material = self._create_material_from_textures(texture_dir, texture_scale, emissive=emissive)
        
        # Apply material to all objects in the container
        for obj in self.objs:
            if obj.data and hasattr(obj.data, 'materials'):
                # Clear existing materials
                obj.data.materials.clear()
                # Add the new material
                obj.data.materials.append(material)
    
    def _create_material_from_textures(self, texture_dir, texture_scale=1.0, emissive=False):
        """Create a Blender material from texture files in the directory"""
        texture_dir = os.path.abspath(texture_dir)
        texture_name = os.path.basename(texture_dir)
        material = bpy.data.materials.new(name=f"tex_{texture_name}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create texture coordinate node
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)
        
        # Create mapping node
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 0)
        mapping.inputs['Scale'].default_value = (texture_scale, texture_scale, 1.0)
        # Create principled BSDF node
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        
        # Create material output node
        material_output = nodes.new(type='ShaderNodeOutputMaterial')
        material_output.location = (300, 0)
        
        # Link texture coordinate to mapping
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        
        # Link principled to output
        links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])
        
        # Define texture file patterns
        texture_patterns = {
            'diffuse': ['col*', 'diff*', 'albedo*', 'basecol*'],
            'roughness': ['rough*'],
            'normal': ['nor*'],
            'metallic': ['metal*'],
            'displacement': ['disp*', 'height*']
        }
        
        # Load diffuse/albedo texture
        diffuse_tex_node = None
        diffuse_tex = self._find_texture_file(texture_dir, texture_patterns['diffuse'])
        if diffuse_tex:
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, 200)
            tex_node.image = bpy.data.images.load(diffuse_tex, check_existing=True)
            diffuse_tex_node = tex_node
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
        
        # Load roughness texture
        roughness_tex = self._find_texture_file(texture_dir, texture_patterns['roughness'])
        if roughness_tex:
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, 0)
            tex_node.image = bpy.data.images.load(roughness_tex, check_existing=True)
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
        
        # Load normal texture
        normal_tex = self._find_texture_file(texture_dir, texture_patterns['normal'])
        if normal_tex:
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, -200)
            tex_node.image = bpy.data.images.load(normal_tex, check_existing=True)
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            
            # Add normal map node
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-100, -200)
            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        
        # Load metallic texture
        metallic_tex = self._find_texture_file(texture_dir, texture_patterns['metallic'])
        if metallic_tex:
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, -400)
            tex_node.image = bpy.data.images.load(metallic_tex, check_existing=True)
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
        
        # Load displacement texture
        displacement_tex = self._find_texture_file(texture_dir, texture_patterns['displacement'])
        if displacement_tex:
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.location = (-300, -600)
            tex_node.image = bpy.data.images.load(displacement_tex, check_existing=True)
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            
            # Add displacement node
            displacement = nodes.new(type='ShaderNodeDisplacement')
            displacement.location = (-100, -600)
            displacement.inputs['Scale'].default_value = 0.1  # Adjust displacement strength
            
            links.new(tex_node.outputs['Color'], displacement.inputs['Height'])
            links.new(displacement.outputs['Displacement'], material_output.inputs['Displacement'])
        
        # Set default values
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        principled.inputs['Roughness'].default_value = random.uniform(0.5, 0.8)
        principled.inputs['Metallic'].default_value = 0.0

        if emissive:
            if diffuse_tex_node is not None:
                links.new(diffuse_tex_node.outputs['Color'], principled.inputs['Emission Color'])
            else:
                principled.inputs['Emission Color'].default_value = (0.9, 0.9, 0.9, 1.0)
            principled.inputs['Emission Strength'].default_value = 2.0
        
        return material
    
    def _find_texture_file(self, texture_dir, possible_names):
        """Find a texture file with one of the possible names in the directory"""
        for name in possible_names:
            if '*' in name:
                # Handle glob patterns
                matches = glob.glob(os.path.join(texture_dir, name))
                if matches:
                    return matches[0]  # Return first match
            else:
                # Handle exact filenames
                file_path = os.path.join(texture_dir, name)
                if os.path.isfile(file_path):
                    return file_path
        return None

    def set_principled_material(self, base_color, roughness, metallic):
        """Set the principled material for all objects in the container.
        If the object already has a principled material, then just set the values.
        If the object doesn't have a (principled) material, then clear or create a new one.
        """
        for obj in self.objs:
            # Get or create a material slot
            if not hasattr(obj.data, "materials"):
                continue
            # If there is at least one material, try to use the first one
            if len(obj.data.materials) > 0 and obj.data.materials[0] is not None:
                mat = obj.data.materials[0]
                if not mat.use_nodes:
                    mat.use_nodes = True
                nodes = mat.node_tree.nodes
                principled = None
                for node in nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled = node
                        break
                if principled is None:
                    # Add a Principled BSDF node if not found
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    # Connect to output
                    material_output = None
                    for node in nodes:
                        if node.type == 'OUTPUT_MATERIAL':
                            material_output = node
                            break
                    if material_output is not None:
                        mat.node_tree.links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])
                # Set values
                principled.inputs['Base Color'].default_value = (
                    float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0
                )
                principled.inputs['Roughness'].default_value = float(roughness)
                principled.inputs['Metallic'].default_value = float(metallic)
            else:
                # No material: create a new one
                mat = bpy.data.materials.new(name="PrincipledMaterial")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                # Create output and principled nodes
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (200, 0)
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.location = (0, 0)
                mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                principled.inputs['Base Color'].default_value = (
                    float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0
                )
                principled.inputs['Roughness'].default_value = float(roughness)
                principled.inputs['Metallic'].default_value = float(metallic)
                obj.data.materials.clear()
                obj.data.materials.append(mat)

    def get_num_materials(self):
        """Return the number of unique materials among all objects in the container."""
        material_set = set()
        for obj in self.objs:
            if hasattr(obj.data, "materials"):
                for mat in obj.data.materials:
                    if mat is not None:
                        material_set.add(mat)
        return len(material_set)

    def clear_objects(self):
        try:
            for obj in self.objs:
                # Remove all children recursively
                def remove_with_children(o):
                    for child in o.children:
                        remove_with_children(child)
                    if o.name in bpy.data.objects:
                        bpy.data.objects.remove(o)
                remove_with_children(obj)
            if self.empty is not None:
                bpy.data.objects.remove(self.empty)
        except Exception as e:
            print(f'---> Error: {e}, skipping deletion')

    def setup_realtime_update(self, frame_count, translation=None, rotation=None, scale=None):
        """
        Setup realtime update for the object, make all updates to the self.empty.
        Creates keyframe animation for translation, rotation, and scale over time.
        
        Args:
            translation: List of translation vectors for each frame [(x, y, z), ...]
            rotation: List of rotation values for each frame (Z-axis rotation in radians)
            scale: List of scale values for each frame (optional, defaults to 1.0)
        """
        if self.empty is None:
            raise ValueError("Cannot setup realtime update: no empty object found. Use with_empty=True when creating ObjContainer.")
        
        scene = bpy.context.scene

        # Set animation range
        scene.frame_start = 1
        scene.frame_end = frame_count
        
        # Create keyframes for each frame
        for frame in range(1, frame_count + 1):
            # Set current frame
            scene.frame_set(frame)
            
            # Update empty object transform
            if translation is not None:
                self.empty.location = Vector(translation[frame-1])
                self.empty.keyframe_insert(data_path="location", frame=frame)
            if rotation is not None:
                self.empty.rotation_euler[2] = rotation[frame-1]  # Z-axis rotation
                self.empty.keyframe_insert(data_path="rotation_euler", frame=frame)
            if scale is not None:
                self.empty.scale = Vector((scale[frame-1], scale[frame-1], scale[frame-1]))
                self.empty.keyframe_insert(data_path="scale", frame=frame)
            
        # Update AABB after setting up animation
        self.aabb = self.get_aabb()

def clear_scene():
    """Clear all mesh objects from the scene, including hidden objects"""
    # Clear frame handlers to avoid stale callbacks
    try:
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_post.clear()
    except Exception:
        pass

    # Free physics caches and remove rigid body world if present
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass
    try:
        scene = bpy.context.scene
        if getattr(scene, 'rigidbody_world', None):
            bpy.ops.rigidbody.world_remove()
    except Exception:
        pass

    # Unhide all objects so they can be selected and deleted
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_select = False

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    

    # Clear orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    

def add_object_file(object_path, with_empty=True, recenter=True, rescale=True):
    """Load GLB file into Blender"""
    object_path = os.path.abspath(object_path)
    if object_path.endswith('.glb') or object_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif object_path.endswith('.obj'):
        bpy.ops.wm.obj_import(
            filepath=object_path, 
            directory=os.path.dirname(object_path), 
            files=[{"name": os.path.basename(object_path)}]
        )
    appended_objs = bpy.context.selected_objects

    return ObjContainer(appended_objs, with_empty, recenter, file_name=object_path, rescale=rescale)

def add_blender_object(blendfile, obj_name, with_empty=True, recenter=True, rescale=True):
    # Use the appropriate path separator for the OS
    section = os.path.join("Object", "")
    directory = os.path.join(blendfile, section)
    # Append the object
    bpy.ops.wm.append(
        filepath=os.path.join(directory, obj_name),
        directory=directory,
        filename=obj_name
    )
    
    appended_objs = bpy.context.selected_objects
    return ObjContainer(appended_objs, with_empty, recenter, file_name=obj_name, rescale=rescale)

def apply_object_transform(obj, location, rotation=0):
    """Apply the given location and rotation to the object"""
    obj.location = location
    # rotation only around Z-axis in radians
    obj.rotation_euler = (0, 0, rotation)


def setup_camera_pose(cam_pose):
    """Setup camera pose from a 4x4 matrix"""
    # cam_pose: [N, 7] (x, y, z, qw, qx, qy, qz)
    for i, pose in enumerate(cam_pose):
        # Create a new camera object
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = f"Camera_{i:02d}"

        # Set camera location
        camera.location = Vector(pose[:3])

        # Set camera rotation from quaternion
        camera.rotation_mode = 'QUATERNION'
        quat = Quaternion(pose[3:])
        camera.rotation_quaternion = quat

def setup_realtime_camera_update(cam_pose, cam_mode='QUATERNION', fov_sequence=None):
    """
    Setup camera updates that respond to timeline scrubbing.
    
    Args:
        cam_pose: List of camera poses (position + rotation)
        cam_mode: 'QUATERNION' or 'MATRIX' 
        fov_sequence: Optional list of FOV values for each frame (for dolly zoom effect)
    """
    # Remove existing handlers to avoid duplicates
    bpy.app.handlers.frame_change_pre.clear()

    # Get or create camera
    if 'Camera' in bpy.data.objects:
        camera = bpy.data.objects['Camera']
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = 'Camera'

    # Set this camera as the active camera for the scene
    scene = bpy.context.scene
    scene.camera = camera

    # Create camera animation with provided poses
    frame_count = len(cam_pose)

    # Set animation range
    scene.frame_start = 1
    scene.frame_end = frame_count

    # Create keyframes for each pose
    for frame, pose in enumerate(cam_pose, 1):
        # Set current frame
        scene.frame_set(frame)
        if cam_mode == 'QUATERNION':
            # Update camera position and rotation from quaternion pose
            camera.location = Vector(pose[:3])
            camera.rotation_mode = 'QUATERNION'
            camera.rotation_quaternion = Quaternion(pose[3:])
        elif cam_mode == 'MATRIX':
            # Assume pose is a 4x4 camera matrix (row-major, numpy or list)
            # Convert to Blender location and rotation
            # Blender expects a 4x4 Matrix, so convert if needed
            if not isinstance(pose, Matrix):
                cam_mat = Matrix([pose[0], pose[1], pose[2], pose[3]])
            else:
                cam_mat = pose
            # Decompose matrix to location and rotation
            loc, rot, scale = cam_mat.decompose()
            camera.location = loc
            camera.rotation_mode = 'QUATERNION'
            camera.rotation_quaternion = rot

        # Insert keyframes for position and rotation
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        
        # Insert FOV keyframes if fov_sequence is provided
        if fov_sequence is not None and frame <= len(fov_sequence):
            fov_radians = fov_sequence[frame-1]
            sensor_width = camera.data.sensor_width
            focal_length = sensor_width / (2 * math.tan(fov_radians / 2))
            camera.data.lens = focal_length
            camera.data.keyframe_insert(data_path="lens", frame=frame)

    # Force LINEAR interpolation on all camera keyframes to prevent Bezier overshoot
    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'
    if camera.data.animation_data and camera.data.animation_data.action:
        for fcurve in camera.data.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'

    # Create frame change handler to update camera during timeline scrubbing
    def update_camera(cur_scene):
        frame = cur_scene.frame_current
        if 1 <= frame <= frame_count:
            # Use the pose data directly from the animation
            pass  # Animation keyframes will handle the updates

    # Register the handler
    bpy.app.handlers.frame_change_pre.append(update_camera)

def setup_camera_settings(resolution_x=1920, resolution_y=1080, fov_rad=1.047, cam_type='PERSP'):
    """Setup camera resolution and field of view"""
    # cam_type: 'PERSP', 'PANO'
    scene = bpy.context.scene

    # Ensure we have a camera set for rendering
    if scene.camera is None:
        logger.info("Warning: No camera set for rendering. Attempting to find and set a camera.")
        if 'Camera' in bpy.data.objects:
            scene.camera = bpy.data.objects['Camera']
            logger.info("Set 'Camera' as the active camera for rendering.")

    # Set resolution
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100

    camera_obj = bpy.data.objects['Camera']
    camera_data = camera_obj.data
    camera_data.clip_start = 0.02

    if cam_type == 'PERSP':
        # Set field of view
        camera_data.type = 'PERSP'
        camera_data.lens_unit = 'FOV'
        camera_data.angle = fov_rad
        logger.info(f"Camera settings: {resolution_x}x{resolution_y}, FOV: {fov_rad}")
    elif cam_type == 'PANO':
        camera_data.type = 'PANO'
        camera_data.panorama_type = 'EQUIRECTANGULAR'
        logger.info(f"Camera settings: {resolution_x}x{resolution_y}, PANO: EQUIRECTANGULAR")


def setup_cycles_rendering(samples=128, use_denoise=None, transparent_bg=False):
    """Setup Cycles rendering engine with specified settings"""
    scene = bpy.context.scene

    # Set render engine to Cycles
    scene.render.engine = 'CYCLES'

    # Set samples
    scene.cycles.samples = samples
    scene.render.film_transparent = transparent_bg

    # Enable GPU rendering if available
    scene.cycles.device = 'GPU'
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons['cycles'].preferences

    # Try to enable OptiX if available
    # Check if GPU is available
    # cycles_preferences.compute_device_type = 'OPTIX'
    gpu_available = False

    compute_device_type = None
    for device_type in cycles_preferences.get_device_types(bpy.context):
        cycles_preferences.get_devices_for_type(device_type[0])

    for gpu_type in ['OPTIX', 'CUDA']:#, 'METAL']:
        for device in cycles_preferences.devices:
            if device.type == gpu_type and (compute_device_type is None or compute_device_type == gpu_type):
                cycles_preferences.compute_device_type = gpu_type
                device.use = True
                logger.info('Device {} of type {} found and used.'.format(device.name, device.type))
                gpu_available = True
                break
        if gpu_available:
            break

    # # This line is critical! It forces Blender to re-scan for devices.
    # # Without it, the devices list might be empty.
    # cycles_preferences.get_devices()

    # for device in cycles_preferences.devices:
    #     if device.type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL']:
    #         gpu_available = True
    #         device.use = True
    #         logger.info(f"Enabled GPU device: {device.name}")

    # Set rendering device based on availability
    if gpu_available:
        # scene.cycles.device = 'GPU'
        if use_denoise:
            # Try to use OptiX denoising if available
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = use_denoise
            logger.info(f"Using {use_denoise} denoiser")
    else:
        # CPU rendering with OpenImageDenoise
        scene.cycles.device = 'CPU'
        if use_denoise:
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = use_denoise
            logger.info(f"Using {use_denoise} denoiser")

    logger.info(f"Cycles setup: {samples} samples, denoising enabled")

def setup_render_passes(ext_passes=None):
    """Setup render passes for RGB, Normal, Depth, and Diffuse Color"""
    if ext_passes is None:
        ext_passes = ['normal', 'depth', 'diffcol', 'object', 'material']

    scene = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]

    # Enable required passes
    if 'normal' in ext_passes:
        view_layer.use_pass_normal = True
    if 'depth' in ext_passes:
        view_layer.use_pass_z = True
    if 'diffcol' in ext_passes:
        view_layer.use_pass_diffuse_color = True
    if 'object' in ext_passes:
        view_layer.use_pass_cryptomatte_object = True
    if 'material' in ext_passes:
        view_layer.use_pass_cryptomatte_material = True

    logger.info("Render passes enabled: RGB, Normal, Depth, Diffuse Color")


def setup_compositor_nodes(output_dir, passes=None, suffix=''):
    """Setup compositor nodes for saving different passes"""
    if passes is None:
        passes = ['rgb', 'normal', 'depth', 'diffcol']
    scene = bpy.context.scene

    # Enable compositor
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    # Clear existing nodes
    nodes.clear()

    # Create render layers node
    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)

    # Create file output nodes for each pass
    pass_map = {
        'rgb': ('Image', 'OPEN_EXR'),      # RGB - linear EXR
        'normal': ('Normal', 'OPEN_EXR'),  # Normal - linear EXR
        'depth': ('Depth', 'OPEN_EXR'),    # Depth - linear EXR
        'diffcol': ('DiffCol', 'OPEN_EXR'), # Diffuse Color - linear EXR
        'rgb_ldr': ('Image', 'JPEG'),      # RGB - tone-mapped LDR
        'alpha': ('Alpha', 'PNG'), # Alpha - PNG
    }

    y_offset = 0
    for p in passes:
        if p not in pass_map:
            continue
        pass_name, file_format = pass_map[p]
        # Create file output node
        file_output = nodes.new(type='CompositorNodeOutputFile')
        file_output.location = (400, y_offset)
        file_output.base_path = output_dir
        file_output.format.file_format = file_format
        if file_format == 'OPEN_EXR':
            file_output.format.color_depth = '16'
            file_output.format.exr_codec = 'ZIP'
        elif file_format == 'JPEG':
            file_output.format.quality = 95

        # Set filename
        filename = f"{p}{suffix}." if not isinstance(passes, dict) else f'{passes[p]}{suffix}.'
        file_output.file_slots[0].path = filename

        # Connect appropriate pass
        output_socket = render_layers.outputs.get(pass_name)
        if output_socket is None:
            logger.info(f"Pass '{pass_name}' not found in render layers node.")
        else:
            links.new(output_socket, file_output.inputs['Image'])

        y_offset -= 200

    logger.info(f"Compositor setup complete. Output directory: {output_dir}")

def get_colormap(num_items, cmap='turbo'):
    """Get a colormap for the given number of items"""
    try:
        import matplotlib.pyplot as plt
        # Create a colormap with the desired number of colors
        cmap = plt.get_cmap(cmap, num_items)
        # Convert the colormap to an array
        cmap = np.array([cmap(i)[:3] for i in range(cmap.N)])
    except ImportError:
        # If matplotlib is not available, use a simple grayscale colormap
        cmap = np.linspace(1, 0, num_items, endpoint=False)[:, None]
        cmap = np.concatenate((cmap, cmap, cmap), axis=1)
    return cmap

def blender_color(color):
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if len(color) == 3:
        color = color + [1.0]
    return color

def setup_colored_mask_nodes(item_list, output_dir, mask_mode='object', mask_name=None, mask_cmap=None, y_offset=0):
    # Assume `setup_compositor_nodes` has already been called
    scene = bpy.context.scene
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Get the render layers node
    render_layers_node = nodes.get('Render Layers')
    mask_layer_name = 'ViewLayer.CryptoMaterial' if mask_mode == 'material' else 'ViewLayer.CryptoObject'
    if mask_cmap is None:
        mask_cmap = get_colormap(len(item_list))

    y_offset = 0
    colored_mask_nodes = []
    for i, item in enumerate(item_list):
        cryptomatte_node = nodes.new(type='CompositorNodeCryptomatteV2')
        cryptomatte_node.location = (600, y_offset)
        cryptomatte_node.scene = scene
        cryptomatte_node.layer_name = mask_layer_name
        links.new(render_layers_node.outputs['Image'], cryptomatte_node.inputs['Image'])
        cryptomatte_node.matte_id = item
        colorize_node = nodes.new(type='CompositorNodeMixRGB')
        colorize_node.location = (800, y_offset)
        colorize_node.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0) # Black
        colorize_node.inputs[2].default_value = blender_color(mask_cmap[i]) # Color from colormap
        links.new(cryptomatte_node.outputs['Matte'], colorize_node.inputs['Fac'])
        colored_mask_nodes.append(colorize_node)
        y_offset -= 300

    # Connect the final combined output to the Composite and Viewer nodes
    file_output = nodes.new(type='CompositorNodeOutputFile')
    file_output.location = (900 + (len(colored_mask_nodes)-1)*300, 0)
    file_output.base_path = output_dir
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '16'
    file_output.format.exr_codec = 'ZIP'
    file_output.file_slots[0].path = f'mask_{mask_mode}.' if mask_name is None else f'{mask_name}.'
    # Now, chain the colored masks together
    if len(colored_mask_nodes) > 1:

        # Start with the first two colored masks
        current_combined_node = nodes.new(type='CompositorNodeMixRGB')
        current_combined_node.location = (900, 0)
        current_combined_node.blend_type = 'ADD'

        links.new(colored_mask_nodes[0].outputs['Image'], current_combined_node.inputs[1])
        links.new(colored_mask_nodes[1].outputs['Image'], current_combined_node.inputs[2])

        # Chain the rest of the masks
        for i in range(2, len(colored_mask_nodes)):
            new_combined_node = nodes.new(type='CompositorNodeMixRGB')
            new_combined_node.location = (900 + (i-1)*300, 0)
            new_combined_node.blend_type = 'ADD'

            links.new(current_combined_node.outputs['Image'], new_combined_node.inputs[1])
            links.new(colored_mask_nodes[i].outputs['Image'], new_combined_node.inputs[2])

            current_combined_node = new_combined_node


        links.new(current_combined_node.outputs['Image'], file_output.inputs['Image'])

    else:
        # If there's only one mask, just connect it directly to the output
        links.new(colored_mask_nodes[0].outputs['Image'], file_output.inputs['Image'])


def _compositor_new_separate_color(tree):
    """Blender 4.x: CompositorNodeSeparateColor; older: SeparateRGBA / SepRGBA."""
    for typ in (
        "CompositorNodeSeparateColor",
        "CompositorNodeSeparateRGBA",
        "CompositorNodeSepRGBA",
    ):
        if hasattr(bpy.types, typ):
            return tree.nodes.new(type=typ), typ
    return None, None


def _link_to_separate_color_input(links, from_socket, sep_node):
    for key in ("Image", "Color"):
        if key in sep_node.inputs:
            links.new(from_socket, sep_node.inputs[key])
            return
    links.new(from_socket, sep_node.inputs[0])


def _separate_node_red_socket(sep_node):
    for key in ("Red", "R"):
        if key in sep_node.outputs:
            return sep_node.outputs[key]
    return sep_node.outputs[0]


def setup_object_mask_output(output_dir, object_names, suffix=''):
    """Compositor JPEG masks for post_process_rendering: glob mask.{pidx}.*.jpg (union of objects).

    Must run after setup_compositor_nodes (keeps existing Render Layers). Cryptomatte object pass
    must be enabled on the view layer.
    """
    if not object_names:
        logger.warning("setup_object_mask_output: empty object_names, skipping")
        return

    pidx = suffix.strip(".") if suffix else "0"
    if not pidx.isdigit():
        pidx = "0"

    scene = bpy.context.scene
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    render_layers_node = nodes.get("Render Layers")
    if render_layers_node is None:
        logger.error("setup_object_mask_output: compositor has no 'Render Layers' node")
        return

    def matte_to_white_rgb(crypto_node, loc_x, loc_y):
        """Cryptomatte -> MixRGB: black background, white where matte > 0."""
        sep, _sep_typ = _compositor_new_separate_color(tree)
        if sep is None:
            logger.error("setup_object_mask_output: no Separate Color / RGBA compositor node available")
            mix = nodes.new(type="CompositorNodeMixRGB")
            mix.location = (loc_x + 180, loc_y)
            mix.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
            mix.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)
            try:
                links.new(crypto_node.outputs["Matte"], mix.inputs["Fac"])
            except Exception as e:
                logger.error(f"setup_object_mask_output: could not wire Matte to Mix Fac: {e}")
            return mix
        sep.location = (loc_x, loc_y)
        _link_to_separate_color_input(links, crypto_node.outputs["Matte"], sep)
        mix = nodes.new(type="CompositorNodeMixRGB")
        mix.location = (loc_x + 180, loc_y)
        mix.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
        mix.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(_separate_node_red_socket(sep), mix.inputs["Fac"])
        return mix

    mask_layer_name = "ViewLayer.CryptoObject"
    base_x, base_y = 500, -400
    white_masks = []

    for i, obj_name in enumerate(object_names):
        cy = base_y - i * 220
        crypto = nodes.new(type="CompositorNodeCryptomatteV2")
        crypto.location = (base_x, cy)
        crypto.scene = scene
        crypto.layer_name = mask_layer_name
        links.new(render_layers_node.outputs["Image"], crypto.inputs["Image"])
        crypto.matte_id = obj_name
        wm = matte_to_white_rgb(crypto, base_x + 220, cy)
        white_masks.append(wm)

    if len(white_masks) == 1:
        combined_socket = white_masks[0].outputs["Image"]
    else:
        # Union of mattes: per-channel max (white on black) without overshoot
        cur = white_masks[0]
        for i in range(1, len(white_masks)):
            mx = nodes.new(type="CompositorNodeMixRGB")
            mx.blend_type = "LIGHTEN"
            mx.location = (base_x + 500 + (i - 1) * 220, base_y)
            links.new(cur.outputs["Image"], mx.inputs[1])
            links.new(white_masks[i].outputs["Image"], mx.inputs[2])
            cur = mx
        combined_socket = cur.outputs["Image"]

    fout = nodes.new(type="CompositorNodeOutputFile")
    fout.location = (base_x + 900, base_y - 100)
    fout.base_path = output_dir
    fout.format.file_format = "JPEG"
    fout.format.quality = 95
    fout.file_slots[0].path = f"mask.{pidx}."
    links.new(combined_socket, fout.inputs["Image"])
    logger.info(
        f"setup_object_mask_output: mask.{pidx}.####.jpg -> {output_dir} ({len(object_names)} object(s))"
    )


def setup_instance_mask_outputs(output_dir, object_names, suffix=''):
    """Create a separate grayscale PNG mask file output per object using Cryptomatte.

    Produces files like instance_000.XXXX.png, instance_001.XXXX.png, etc.
    Each mask is white (255) where the object is visible and black (0) elsewhere.
    Must run after setup_compositor_nodes. Cryptomatte object pass must be enabled.
    """
    if not object_names:
        logger.warning("setup_instance_mask_outputs: empty object_names, skipping")
        return

    pidx = suffix.strip(".") if suffix else "0"
    if not pidx.isdigit():
        pidx = "0"

    scene = bpy.context.scene
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    render_layers_node = nodes.get("Render Layers")
    if render_layers_node is None:
        logger.error("setup_instance_mask_outputs: no 'Render Layers' node")
        return

    mask_layer_name = "ViewLayer.CryptoObject"
    base_x, base_y = 500, -900

    for i, obj_name in enumerate(object_names):
        cy = base_y - i * 300

        crypto = nodes.new(type="CompositorNodeCryptomatteV2")
        crypto.location = (base_x, cy)
        crypto.scene = scene
        crypto.layer_name = mask_layer_name
        links.new(render_layers_node.outputs["Image"], crypto.inputs["Image"])
        crypto.matte_id = obj_name

        sep, _sep_typ = _compositor_new_separate_color(tree)
        if sep is not None:
            sep.location = (base_x + 220, cy)
            _link_to_separate_color_input(links, crypto.outputs["Matte"], sep)
            red_socket = _separate_node_red_socket(sep)
        else:
            red_socket = crypto.outputs["Matte"]

        mix = nodes.new(type="CompositorNodeMixRGB")
        mix.location = (base_x + 440, cy)
        mix.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
        mix.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(red_socket, mix.inputs["Fac"])

        fout = nodes.new(type="CompositorNodeOutputFile")
        fout.location = (base_x + 660, cy)
        fout.base_path = output_dir
        fout.format.file_format = "PNG"
        fout.format.color_mode = "BW"
        fout.format.color_depth = "8"
        fout.file_slots[0].path = f"instance_{i:03d}.{pidx}."
        links.new(mix.outputs["Image"], fout.inputs["Image"])

    logger.info(
        f"setup_instance_mask_outputs: {len(object_names)} per-object masks -> {output_dir}"
    )


def render_all_frames(output_dir, num_frames, suffix='render', capture_transforms_callback=None):
    """Render all frames and save them to the output directory.

    If ``capture_transforms_callback`` is set, it is called as ``callback(frame_number)``
    after each successful render; return values are collected and returned as a list
    (index ``i`` corresponds to Blender frame ``i + 1``). Otherwise returns an empty list.
    """
    scene = bpy.context.scene
    captured = []
    for frame_number in range(1, num_frames + 1):
        scene.frame_set(frame_number)
        scene.render.filepath = f"{output_dir}/{suffix}.{frame_number:04d}"
        bpy.ops.render.render(write_still=True)
        logger.info(f"Rendered frame {frame_number} to {output_dir}")
        if capture_transforms_callback is not None:
            captured.append(capture_transforms_callback(frame_number))
    return captured


def hide_object(obj):
    """Hierarchically hide the object and its children"""
    if isinstance(obj, str):
        obj = bpy.data.objects.get(obj)
    obj.hide_render = True
    obj.hide_viewport = True
    for child in obj.children:
        hide_object(child)

def set_envmap_texture(envmap_path, rotation=0., strength=1.0, flip=False, rot_offset=-PI_2):
    """Set the environment map texture for the scene"""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create new nodes
    env_texture = nodes.new(type='ShaderNodeTexEnvironment')
    env_texture.image = bpy.data.images.load(envmap_path)
    env_texture.location = (0, 0)

    background = nodes.new(type='ShaderNodeBackground')
    background.location = (200, 0)
    background.inputs['Strength'].default_value = strength

    output = nodes.new(type='ShaderNodeOutputWorld')
    output.location = (400, 0)

    # Link nodes
    links.new(env_texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    # Apply rotation
    # Create a texture coordinate node and mapping node for proper flipping
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-400, 0)
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-200, 0)
    # Connect the nodes
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])

    mapping.inputs['Rotation'].default_value[2] = rotation + rot_offset  # Convert degrees to radians
    links.new(mapping.outputs['Vector'], env_texture.inputs['Vector'])

    # Flip the environment map if needed
    if flip:
        mapping.inputs['Scale'].default_value[0] = -1.0  # Flip horizontally

def setup_realtime_envmap_update(envmap_rot_list, rot_offset=-PI_2):
    """
    Setup environment map updates that respond to timeline scrubbing.
    Creates keyframe animation for environment map rotation over time.
    
    Args:
        envmap_rot_list: List of rotation values for each frame (in degrees)
    """
    # Get the world shader nodes
    world = bpy.context.scene.world
    if not world:
        logger.warning("No world shader found. Cannot setup environment map animation.")
        return
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Find the mapping node for environment map rotation
    mapping_node = None
    for node in nodes:
        if node.type == 'MAPPING':
            mapping_node = node
            break
    
    if mapping_node is None:
        logger.warning("No mapping node found in world shader. Cannot setup environment map rotation animation.")
        return

    scene = bpy.context.scene
    frame_count = len(envmap_rot_list)

    # Set animation range
    scene.frame_start = 1
    scene.frame_end = frame_count

    # Create keyframes for each frame
    for frame in range(1, frame_count + 1):
        # Set current frame
        scene.frame_set(frame)
        
        # Update environment map rotation
        # Convert degrees to radians for Blender
        mapping_node.inputs['Rotation'].default_value[2] = envmap_rot_list[frame-1] + rot_offset
        
        # Insert keyframe for rotation
        mapping_node.inputs['Rotation'].keyframe_insert(data_path="default_value", frame=frame)
    
    logger.info(f"Environment map rotation animation setup complete for {frame_count} frames")

def is_trans_mat(material):
    if not material or not material.use_nodes:
        return False

    nodes = material.node_tree.nodes

    # Check for specific transparency shader nodes
    for node in nodes:
        if node.type in ('BSDF_GLASS', 'BSDF_TRANSPARENT'):
            return True
        # Check Principled BSDF for alpha or transmission
        if node.type == 'BSDF_PRINCIPLED':
            # Check Alpha input
            alpha_input = node.inputs.get('Alpha')
            if alpha_input and alpha_input.default_value < 1.0 and alpha_input.default_value > 0:
                return True
            # Check Transmission input
            transmission_input = node.inputs.get('Transmission Weight')
            if transmission_input and transmission_input.default_value > 0.5:
                return True

    # # Check Material Properties settings for transparency
    # if material.blend_method != 'OPAQUE':
    #     return True

    return False

def get_trans_mat_ids(obj_list):
    trans_mat_ids_map = {}
    for obj_name in obj_list:
        obj = bpy.data.objects.get(obj_name)
        for slot in obj.material_slots:
            if slot.material and is_trans_mat(slot.material):
                trans_mat_ids_map[obj_name] = slot.material.name

    return trans_mat_ids_map

def get_scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)) or isinstance(obj.data, (bpy.types.Curve)):
            yield obj

def clean_up_bsdf(basd_node, material):
    if "Specular" in basd_node.inputs:
        basd_node.inputs["Specular"].default_value = 0
        if len(basd_node.inputs["Specular"].links) > 0:
            material.node_tree.links.remove(basd_node.inputs["Specular"].links[0])
    if "Transmission" in basd_node.inputs:
        if basd_node.inputs["Transmission"].default_value > 0:
            basd_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
        basd_node.inputs["Transmission"].default_value = 0
        if len(basd_node.inputs["Transmission"].links) > 0:
            material.node_tree.links.remove(basd_node.inputs["Transmission"].links[0])
    if "Transmission Weight" in basd_node.inputs:
        if basd_node.inputs["Transmission Weight"].default_value > 0:
            basd_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
        basd_node.inputs["Transmission Weight"].default_value = 0
        if len(basd_node.inputs["Transmission Weight"].links) > 0:
            material.node_tree.links.remove(basd_node.inputs["Transmission Weight"].links[0])
    if "Sheen" in basd_node.inputs:
        basd_node.inputs["Sheen"].default_value = 0
    if "Subsurface" in basd_node.inputs:
        basd_node.inputs["Subsurface"].default_value = 0
    if "Clearcoat" in basd_node.inputs:
        basd_node.inputs["Clearcoat"].default_value = 0
    if "Emission" in basd_node.inputs:
        basd_node.inputs["Emission"].default_value = (0, 0, 0, 1)

def render_albedo_and_material(output_dir, passes=['orm'], clean_materials=False, suffix=''):
    roughness, metalness = 0.0, 0.0
    render_layers = bpy.context.scene.node_tree.nodes.get("Render Layers")
    if render_layers is None:
        render_layers = bpy.context.scene.node_tree.nodes.new("CompositorNodeRLayers")

    # if len(bpy.data.objects) > 100 or len(bpy.data.materials) > 100:
    #     return None, None, None, None, None

    # Clear existing AOVs to avoid conflicts
    view_layer = bpy.context.view_layer
    while len(view_layer.aovs) > 0:
        view_layer.aovs.remove(view_layer.aovs[0])

    pass_map = {
        'orm': ('ORM', 'OPEN_EXR'),
        'albedo': ('Albedo', 'OPEN_EXR')
    }

    # Add AOVs once at the view layer level
    for i, p in enumerate(passes):
        bpy.ops.scene.view_layer_add_aov()
        view_layer.aovs[i].name = pass_map[p][0]

    # Process materials
    for obj in get_scene_meshes():
        if len(obj.material_slots) == 0:
            mat = bpy.data.materials.new("Material")
            obj.data.materials.append(mat)
        for slot_id, slot in enumerate(obj.material_slots):
            material = slot.material
            if not material:
                material = bpy.data.materials.new("Material")
                obj.material_slots[slot_id].material = material
            if not material.node_tree:
                material.use_nodes = True

            bsdf_node = None
            use_diff_col = False

            # Find or create Principled BSDF node
            for n in material.node_tree.nodes:
                if n.bl_idname == "ShaderNodeBsdfPrincipled":
                    bsdf_node = n
                # if n.bl_idname == "ShaderNodeEmission":
                #     n.inputs["Strength"].default_value = 0

            if not bsdf_node:
                bsdf_node = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
                bsdf_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                orig_node = None
                for n in material.node_tree.nodes:
                    if n.bl_idname == "ShaderNodeBsdfGlossy":
                        orig_node = n
                if not orig_node:
                    for n in material.node_tree.nodes:
                        if n.bl_idname == "ShaderNodeBsdfDiffuse":
                            orig_node = n
                    if not orig_node:
                        for n in material.node_tree.nodes:
                            if n.bl_idname == "ShaderNodeOutputMaterial" and clean_materials:
                                material.node_tree.links.new(bsdf_node.outputs["BSDF"], n.inputs[0])
                        bsdf_node.inputs["Roughness"].default_value = 0.25
                    else:
                        bsdf_node.inputs["Roughness"].default_value = 0.25
                        if len(orig_node.inputs["Color"].links) > 0:
                            color_output = orig_node.inputs["Color"].links[0].from_socket
                            material.node_tree.links.new(color_output, bsdf_node.inputs["Base Color"])
                        else:
                            bsdf_node.inputs["Base Color"].default_value = orig_node.inputs["Color"].default_value
                else:
                    bsdf_node.inputs["Roughness"].default_value = orig_node.inputs["Roughness"].default_value
                    bsdf_node.inputs["Metallic"].default_value = 1 - orig_node.inputs["Roughness"].default_value
                    if len(orig_node.inputs["Color"].links) > 0:
                        color_output = orig_node.inputs["Color"].links[0].from_socket
                        material.node_tree.links.new(color_output, bsdf_node.inputs["Base Color"])
                    else:
                        bsdf_node.inputs["Base Color"].default_value = orig_node.inputs["Color"].default_value


            # Clean up material properties
            if clean_materials:
                clean_up_bsdf(bsdf_node, material)
                # Clean up all Principled BSDF nodes
                for n in material.node_tree.nodes:
                    if n.bl_idname == "ShaderNodeBsdfPrincipled":
                        clean_up_bsdf(n, material)

            # Create AOV nodes for this material
            aov_map = {}
            for i, p in enumerate(passes):
                pass_name = pass_map[p][0]
                aov_node = material.node_tree.nodes.new(type="ShaderNodeOutputAOV")
                aov_node.name = f"{pass_name}_AOV"
                aov_node.aov_name = pass_name  # This is the key property!
                aov_node.location = (800, -200 * (i+1))
                aov_map[p] = aov_node

            combine_node = material.node_tree.nodes.new(type="ShaderNodeCombineColor")

            if 'orm' in passes:
                # Connect metallic to blue channel
                try:
                    metallic_connected_output = bsdf_node.inputs["Metallic"].links[0].from_socket
                    material.node_tree.links.new(metallic_connected_output, combine_node.inputs["Blue"])
                except IndexError:
                    metalness = bsdf_node.inputs["Metallic"].default_value
                    combine_node.inputs["Blue"].default_value = metalness

                # Connect roughness to green channel
                try:
                    roughness_connected_output = bsdf_node.inputs["Roughness"].links[0].from_socket
                    material.node_tree.links.new(roughness_connected_output, combine_node.inputs["Green"])
                except IndexError:
                    roughness = bsdf_node.inputs["Roughness"].default_value
                    combine_node.inputs["Green"].default_value = roughness

                # Connect ORM
                material.node_tree.links.new(combine_node.outputs["Color"], aov_map['orm'].inputs["Color"])

            if 'albedo' in passes:
                # Connect albedo
                try:
                    albedo_connected_output = bsdf_node.inputs["Base Color"].links[0].from_socket
                    material.node_tree.links.new(albedo_connected_output, aov_map['albedo'].inputs["Color"])
                except IndexError:
                    aov_map['albedo'].inputs["Color"].default_value = bsdf_node.inputs["Base Color"].default_value


    # Setup compositor nodes (outside the material loop)
    for i, p in enumerate(passes):
        pass_name, file_format = pass_map[p]
        file_node = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        file_node.location = (1200, -200 * (i+1))  # Set location for file_node
        alpha_set = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        alpha_set.location = (1000, -200 * (i+1))  # Set location for alpha_set
        file_node.base_path = output_dir
        file_node.format.file_format = file_format
        if file_format == 'OPEN_EXR':
            file_node.format.color_depth = '16'
            file_node.format.exr_codec = 'ZIP'
        elif file_format == 'JPEG':
            file_node.format.quality = 95
        file_node.file_slots[0].path = f"{p}{suffix}."

        # Connect AOV outputs in compositor
        use_diff_col = False
        bpy.context.scene.node_tree.links.new(render_layers.outputs[pass_name], alpha_set.inputs["Image"])
        if p == 'albedo' and use_diff_col:
            bpy.context.scene.node_tree.links.new(render_layers.outputs["DiffCol"], alpha_set.inputs["Image"])
        bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set.inputs["Alpha"])
        bpy.context.scene.node_tree.links.new(alpha_set.outputs["Image"], file_node.inputs[0])


def get_look_at_matrix(pos, look_at, up):
    forward = look_at - pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    return np.array([[right[0], up[0], -forward[0], pos[0]],
                     [right[1], up[1], -forward[1], pos[1]],
                     [right[2], up[2], -forward[2], pos[2]],
                     [0, 0, 0, 1]])


def get_cam_matrix(azimuth, elevation, t, cam_radius):
    # return camera matrix + translation
    z = np.sin(elevation)
    r = np.cos(elevation)
    x = r * np.cos(azimuth)
    y = r * np.sin(azimuth)
    pos = np.array([x, y, z]) * cam_radius
    look_at = np.array([0., 0., 0.])
    if t is not None:
        look_at += t
    up = np.array([0, 0, 1])
    return get_look_at_matrix(pos, look_at, up)
