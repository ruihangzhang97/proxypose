import numpy as np
import torch
import torch.nn.functional as F

NVDIFFRAST_RENDER = True

try:    
    import nvdiffrast.torch as dr
except:
    NVDIFFRAST_RENDER = False

def convert_cam_mat_blender_to_dr(cam_mat):
    """
    Convert Blender C2W (Z-up) matrix to DR C2W (Y-up) matrix.
    Since C2W maps camera local axes to world, only the world basis changes:
    M_dr = S_world @ M_blender, with S_world = R_x(-90deg).
    Accepts numpy or torch 4x4.
    """
    BLENDER_TO_DR = np.array([
        [1, 0,  0, 0],
        [0, 0,  1, 0],
        [0, -1, 0, 0],
        [0, 0,  0, 1],
    ], dtype=np.float32)
    if isinstance(cam_mat, np.ndarray):
        return BLENDER_TO_DR @ cam_mat
    elif torch.is_tensor(cam_mat):
        S = torch.tensor(BLENDER_TO_DR, dtype=cam_mat.dtype, device=cam_mat.device)
        return S @ cam_mat
    else:
        raise TypeError("cam_mat must be numpy.ndarray or torch.Tensor")

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def cam_intrinsics(fov, width, height, device=None):
    """
    fov is along the height axis
    """
    focal = 0.5 * height / np.tan(0.5 * fov)
    intrinsics = torch.tensor([
        [focal, 0, 0.5 * width],
        [0, focal, 0.5 * height],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    return intrinsics

def uv_mesh(width, height, device=None):
    uv = torch.stack(
        torch.meshgrid(torch.arange(width) + 0.5, torch.arange(height) + 0.5, indexing='xy'), dim=-1
    ).float().to(device)
    uv = torch.cat([uv, torch.ones((height, width, 1), device=device)], dim=-1) # [H, W, 3]
    return uv

def reinhard(x, max_point=16):
    # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    # lumi = lumi[..., None]
    # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
    # y_rein = x / (1+x)
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein

def rgb_to_srgb(f):
    """
    Convert linear RGB to sRGB.
    Supports both torch.Tensor and numpy.ndarray.
    """
    if isinstance(f, torch.Tensor):
        return torch.where(
            f <= 0.0031308,
            f * 12.92,
            torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055
        )
    elif isinstance(f, np.ndarray):
        out = np.where(
            f <= 0.0031308,
            f * 12.92,
            np.power(np.clip(f, 0.0031308, None), 1.0 / 2.4) * 1.055 - 0.055
        )
        return out
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

def srgb_to_rgb(f):
    """
    Convert sRGB to linear RGB.
    Supports both torch.Tensor and numpy.ndarray.
    """
    if isinstance(f, torch.Tensor):
        return torch.where(
            f <= 0.04045,
            f / 12.92,
            torch.pow((torch.clamp(f, min=0.04045) + 0.055) / 1.055, 2.4)
        )
    elif isinstance(f, np.ndarray):
        import numpy as np
        out = np.where(
            f <= 0.04045,
            f / 12.92,
            np.power((np.clip(f, 0.04045, None) + 0.055) / 1.055, 2.4)
        )
        return out
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_vec(res, device=None):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    # return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
    return reflvec

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    # normal_map = normal_map.numpy()
    # mask = mask.numpy()
    return normal_map, mask

def get_ref_vector(normal: np.array, incoming_vector: np.array):
    """
    BLENDER CONVENSION
    normal: the normal vector of the point
    incoming_vector: the vector from the point to the camera
    """
    #R = 2(N ⋅ I)N - I
    R = 2 * (normal * incoming_vector).sum(-1, keepdims=True) * normal - incoming_vector
    return R

def latlong_to_cubemap_torch(latlong_map, res): # no dr dependency
    ndim = latlong_map.ndim
    batch_size = 1 if ndim == 3 else latlong_map.shape[0]
    if ndim == 3:
        latlong_map = latlong_map.unsqueeze(0)
    device = latlong_map.device
    cubemap = torch.zeros(batch_size, 6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
            indexing='ij'
        )
        v = cube_to_dir(s, gx, gy)  # Shape: (res[0], res[1], 3)
        v = F.normalize(v, dim=-1)
        
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5  # Shape: (res[0], res[1], 1)
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi  # Shape: (res[0], res[1], 1)
        
        tu = tu * 2 - 1  # Shape: (res[0], res[1], 1)
        tv = tv * 2 - 1  # Shape: (res[0], res[1], 1)

        grid = torch.cat((tu, tv), dim=-1)  # Shape: (res[0], res[1], 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, res[0], res[1], 2)
        texture = latlong_map.permute(0, 3, 1, 2)  # Shape: (1, channels, H_latlong, W_latlong)
        
        # Perform sampling
        sampled = F.grid_sample(texture, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Result shape: (1, channels, res[0], res[1])
        
        # Reshape and store in cubemap
        # cubemap[s] = sampled.squeeze(0).permute(1, 2, 0)  # Shape: (res[0], res[1], channels)
        cubemap[:, s] = sampled.permute(0, 2, 3, 1)  # Shape: (batch_size, res[0], res[1], channels)
    
    if ndim == 3:
        cubemap = cubemap.squeeze(0)

    return cubemap

def cubemap_sample_torch(cubemap, dirs):
    """Sample from cubemap with smoother transitions across face edges."""
    device = cubemap.device
    dirs_shape = dirs.shape
    dirs = dirs.reshape(-1, 3)
    N = dirs.shape[0]  # Number of directions (e.g., H * W)
    C = cubemap.shape[-1]  # Number of channels (e.g., 3 for RGB)
    h, w = cubemap.shape[-3:-1]  # Cubemap face resolution
    ndim = cubemap.ndim
    batch_size = cubemap.shape[0] if ndim == 5 else 1  # Handle batch size
    if ndim == 4:
        cubemap = cubemap.unsqueeze(0)

    # Normalize directions
    dirs = F.normalize(dirs, dim=-1)  # Shape: (N, 3)
    
    # Compute absolute directions and max axis
    abs_dirs = torch.abs(dirs)  # Shape: (N, 3)
    max_axis = torch.argmax(abs_dirs, dim=-1)  # Shape: (N,)
    max_vals = abs_dirs[torch.arange(N, device=device), max_axis]  # Shape: (N,)
    
    # Get signs for face selection
    sign = torch.sign(dirs[torch.arange(N, device=device), max_axis])  # Shape: (N,)
    
    # Preallocate result tensor
    result = torch.zeros(batch_size, N, C, device=device)  # Shape: (N, channels)
    
    # Process each face iteratively
    for s in range(6):
        # Determine face index based on max_axis and sign
        face_idx = torch.where(max_axis == 0, torch.where(sign > 0, 0, 1),  # +X, -X
                               torch.where(max_axis == 1, torch.where(sign > 0, 2, 3),  # +Y, -Y
                                           torch.where(sign > 0, 4, 5)))  # +Z, -Z
        mask = (face_idx == s)  # Shape: (N,)
        
        if not mask.any():
            continue  # Skip if no rays hit this face
        
        # Select directions for this face
        dirs_s = dirs[mask]  # Shape: (N_s, 3)
        max_axis_s = max_axis[mask]  # Shape: (N_s,)
        max_vals_s = max_vals[mask]  # Shape: (N_s,)
        
        # Compute UV coordinates based on face
        # We need to map the two non-max axes to U and V
        u = torch.zeros_like(max_vals_s)
        v = torch.zeros_like(max_vals_s)
        
        # Face-specific UV mapping
        if s == 0:  # +X
            u = -dirs_s[..., 2] / max_vals_s  # -Z
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 1:  # -X
            u = dirs_s[..., 2] / max_vals_s   # +Z
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 2:  # +Y
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = dirs_s[..., 2] / max_vals_s   # +Z
        elif s == 3:  # -Y
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = -dirs_s[..., 2] / max_vals_s  # -Z
        elif s == 4:  # +Z
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 5:  # -Z
            u = -dirs_s[..., 0] / max_vals_s  # -X
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        
        # Normalize UV to [0, 1]
        u = (u + 1) * 0.5  # Shape: (N_s,)
        v = (v + 1) * 0.5  # Shape: (N_s,)
        
        # Convert UV to grid_sample format: [0, 1] -> [-1, 1]
        u = u * 2 - 1
        v = v * 2 - 1
        grid = torch.stack((u, v), dim=-1)  # Shape: (N_s, 2)
        
        # Reshape for grid_sample: expects (batch, H, W, 2)
        grid = grid.view(-1, 1, 1, 2)  # Shape: (N_s, 1, 1, 2)
        
        # Prepare texture: grid_sample expects (batch, channels, height, width)
        texture = cubemap[:, s].permute(0, 3, 1, 2).flatten(0,1).unsqueeze(0)  # Shape: (batch_size, channels, h, w)
        texture = texture.expand(grid.shape[0], -1, -1, -1)  # Shape: (N_s, channels, h, w)

        # Perform sampling
        sampled = F.grid_sample(texture, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Result shape: (batch_size * N_s, channels, 1, 1)

        # Reshape and store in result
        result[:, mask] = sampled.view(-1, batch_size, C).permute(1, 0, 2)  # Shape: (batch_size, N_s, channels)
    
    if ndim == 4:
        result = result.squeeze(0) # Shape: (N, channels)
    
    result = result.reshape(*dirs_shape[:-1], C)

    return result


def latlong_to_cubemap_dr(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    latlong_map = latlong_map.to(device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                                indexing='ij')
        v = cube_to_dir(s, gx, gy)
        v = F.normalize(v, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_sample_dr(cubemap, dirs):
    result = dr.texture(
        cubemap.unsqueeze(0), dirs.unsqueeze(0).contiguous(), filter_mode='linear', boundary_mode='cube'
    )[0]
    return result