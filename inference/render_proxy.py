import cv2
import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
    TexturesAtlas,
)


class ColoredCube:
    """A unit cube mesh with distinct solid colors on each face for pose tracking.

    Face color mapping (matches COLORS in track_proxy.py):
        front  (+z) → white
        back   (-z) → yellow
        left   (-x) → cyan
        right  (+x) → green
        bottom (-y) → blue
        top    (+y) → red
    """

    def __init__(self, size=1.0, device="cuda"):
        self.size = float(size)
        self.device = device
        self.face_colors = {
            "front":  torch.tensor((255, 255, 255), device=device) / 255.,  # white
            "back":   torch.tensor((0, 255, 255), device=device) / 255.,    # yellow
            "left":   torch.tensor((230, 230, 0), device=device) / 255.,    # cyan
            "right":  torch.tensor((0, 240, 75), device=device) / 255.,     # green
            "bottom": torch.tensor((255, 30, 30), device=device) / 255.,    # blue
            "top":    torch.tensor((10, 10, 255), device=device) / 255.,    # red
        }
        self.mesh = self._create_mesh()

    def _create_mesh(self) -> Meshes:
        s = self.size / 2.0
        vertices = torch.tensor([
            [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
            [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
        ], dtype=torch.float32, device=self.device)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],   # front  (-z)
            [5, 4, 7], [5, 7, 6],   # back   (+z)
            [4, 0, 3], [4, 3, 7],   # left   (-x)
            [1, 5, 6], [1, 6, 2],   # right  (+x)
            [4, 5, 1], [4, 1, 0],   # bottom (-y)
            [3, 2, 6], [3, 6, 7],   # top    (+y)
        ], dtype=torch.int64, device=self.device)

        atlas_size = 1
        atlas = torch.zeros((1, 12, atlas_size, atlas_size, 3), device=self.device)
        face_colors_list = (
            [self.face_colors["front"]] * 2 +
            [self.face_colors["back"]] * 2 +
            [self.face_colors["left"]] * 2 +
            [self.face_colors["right"]] * 2 +
            [self.face_colors["bottom"]] * 2 +
            [self.face_colors["top"]] * 2
        )
        for tri_idx in range(12):
            atlas[0, tri_idx, :, :, :] = face_colors_list[tri_idx]

        textures = TexturesAtlas(atlas=atlas)
        return Meshes(verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures)

    def contours(self):
        """Return per-face contour metadata (vertices + neighbor colors).

        Each entry in the returned list is a dict with:
          - 'color':           RGB list of the face's own color.
          - 'vertices':        List of 4 [x, y, z] vertex positions.
          - 'neighbor_colors': List of 4 pairs of neighbor-face RGB colors.
        """
        s = self.size / 2.0
        vertices_list = [
            [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
            [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
        ]
        face_vertices = {
            "front":  [0, 1, 2, 3],
            "back":   [4, 5, 6, 7],
            "left":   [4, 0, 3, 7],
            "right":  [5, 1, 2, 6],
            "bottom": [4, 5, 1, 0],
            "top":    [7, 6, 2, 3],
        }
        vertex_faces = {
            0: ["front", "left",  "bottom"],
            1: ["front", "right", "bottom"],
            2: ["front", "right", "top"],
            3: ["front", "left",  "top"],
            4: ["back",  "left",  "bottom"],
            5: ["back",  "right", "bottom"],
            6: ["back",  "right", "top"],
            7: ["back",  "left",  "top"],
        }

        face_colors_dict = {k: v.tolist() for k, v in self.face_colors.items()}

        contours = []
        for face, verts in face_vertices.items():
            contour = []
            neighbor_colors = []
            own_color = face_colors_dict[face]
            for v_idx in verts:
                v_pos = vertices_list[v_idx]
                other_faces = [f for f in vertex_faces[v_idx] if f != face]
                neigh_colors = [face_colors_dict[f] for f in other_faces]
                contour.append(v_pos)
                neighbor_colors.append(neigh_colors)

            contours.append({
                "color": own_color,
                "vertices": contour,
                "neighbor_colors": neighbor_colors,
            })

        return contours


class CubeTrajectoryRenderer:
    """PyTorch3D renderer for the colored proxy cube using explicit camera intrinsics.

    Args:
        scale:       Edge length of the cube mesh.
        image_size:  Output image resolution (square).
        device:      Torch device string.
        supersample: Supersampling factor for anti-aliasing (default 2×).
    """

    def __init__(self, scale, image_size=512, device="cuda", supersample=2):
        self.image_size = int(image_size)
        self.device = device
        self.supersample = max(1, int(supersample))
        self.render_size = self.image_size * self.supersample

        self.cube = ColoredCube(size=scale, device=device)
        self.renderer = self._setup_renderer()

    def _setup_renderer(self):
        raster_settings = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            cull_backfaces=False,
        )
        lights = AmbientLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),))
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=self.device, lights=lights, blend_params=blend_params),
        )

    @torch.no_grad()
    def render_frame_rt(self, cam_rt, object_rt, intrinsics):
        """Render the proxy cube given explicit camera and object transforms.

        Args:
            cam_rt:     (4, 4) camera-to-world transform [R|T; 0|1].
            object_rt:  (4, 4) object-to-world transform [R|T; 0|1].
            intrinsics: (3, 3) camera intrinsic matrix K [[fx,0,cx],[0,fy,cy],[0,0,1]].

        Returns:
            bgr_np:   (H, W, 3) uint8 BGR rendered image.
            alpha_np: (H, W, 1) uint8 alpha mask.
        """
        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.float().to(self.device)
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        cam_rt   = _t(cam_rt)
        object_rt = _t(object_rt)
        K        = _t(intrinsics)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        focal_length    = torch.stack([fx, fy]).unsqueeze(0)
        principal_point = torch.stack([cx, cy]).unsqueeze(0)

        cameras = PerspectiveCameras(
            R=cam_rt[:3, :3].unsqueeze(0),
            T=cam_rt[:3, 3].unsqueeze(0),
            focal_length=focal_length * self.supersample,
            principal_point=principal_point * self.supersample,
            image_size=((self.render_size, self.render_size),),
            in_ndc=False,
            device=self.device,
        )

        original_verts = self.cube.mesh.verts_packed()
        transformed_verts = (original_verts @ object_rt[:3, :3].T) + object_rt[:3, 3]
        transformed_mesh = self.cube.mesh.update_padded(transformed_verts.unsqueeze(0))

        images = self.renderer(transformed_mesh, cameras=cameras)
        rgb   = images[0, ..., :3].clamp(0, 1)
        alpha = images[0, ..., 3:4].clamp(0, 1)

        rgb_np   = (rgb.cpu().numpy()   * 255).astype(np.uint8)
        alpha_np = (alpha.cpu().numpy() * 255).astype(np.uint8)

        if self.supersample > 1:
            rgb_np   = cv2.resize(rgb_np,   (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            alpha_np = cv2.resize(alpha_np, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            if alpha_np.ndim == 2:
                alpha_np = alpha_np[..., None]

        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        return bgr_np, alpha_np


def camera_space_point_to_proxy(intrinsics, image_size, pixel_xy, depth, proxy_scale=0.1):
    """Convert a 2D pixel prompt + depth to a 3D proxy object transform.

    Places a proxy cube at the 3D point corresponding to the given pixel,
    oriented so its corner points toward the camera (for easy color visibility).

    Args:
        intrinsics:  (3, 3) camera intrinsic matrix K.
        image_size:  Scalar image width/height (assumes square image).
        pixel_xy:    (u, v) pixel coordinate (column, row).
        depth:       Depth of the point in camera space (meters or arbitrary units).
        proxy_scale: Fraction of the image half-width that the cube should subtend.

    Returns:
        T_proxy: (4, 4) float32 object-to-camera transform.
        scale:   Scalar edge length of the proxy cube in the same units as depth.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    focal_length = (fx + fy) / 2

    x_ndc = -(pixel_xy[0] - cx) / fx
    y_ndc = -(pixel_xy[1] - cy) / fy

    z_cam = depth
    x_cam = x_ndc * z_cam
    y_cam = y_ndc * z_cam
    t_obj = np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    scale = z_cam * proxy_scale * (image_size / focal_length)

    z_dir = t_obj / np.linalg.norm(t_obj)
    y_dir = np.array([0, 1, 0], dtype=np.float32)
    y_dir = y_dir - np.dot(y_dir, z_dir) * z_dir
    y_dir = y_dir / np.linalg.norm(y_dir)
    x_dir = np.cross(y_dir, z_dir)
    R_proxy = np.stack([x_dir, y_dir, z_dir], axis=1)

    # Rotate 45° around Y then tilt around X so the cube corner faces the camera
    angle_y = np.radians(45)
    rot_y = np.array([
        [ np.cos(angle_y), 0, np.sin(angle_y)],
        [             0,   1,             0  ],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ], dtype=np.float32)
    angle_x = -np.arctan(1 / np.sqrt(2))
    rot_x = np.array([
        [1,              0,               0],
        [0,  np.cos(angle_x), -np.sin(angle_x)],
        [0,  np.sin(angle_x),  np.cos(angle_x)],
    ], dtype=np.float32)

    R_proxy = R_proxy @ rot_x @ rot_y

    T_proxy = np.eye(4, dtype=np.float32)
    T_proxy[:3, :3] = R_proxy
    T_proxy[:3, 3]  = t_obj

    return T_proxy, scale
