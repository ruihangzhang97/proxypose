import os
import numpy as np

import OpenEXR
import Imath
import imageio
import imageio.v3 as iio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def read_img(img_file):
    if img_file.endswith('.exr'):
        img = iio.imread(img_file, flags=cv2.IMREAD_UNCHANGED, plugin='opencv')
    else:
        img = iio.imread(img_file)
    if img.ndim == 2:
        img = img[..., None]
    return img

def save_image(fn, x : np.ndarray) -> np.ndarray:
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
        elif os.path.splitext(fn)[1] == ".jpg":
            if x.ndim == 3 and x.shape[2] == 4:
                x = x[..., :3]
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), quality=98)
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"WARNING: FAILED to save image {fn}: {e}")

def read_normal_exr(exr_file):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(exr_file)
    header = exr_file.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Specify the pixel type (FLOAT for 32-bit)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the X, Y, and Z channels
    x_channel = np.frombuffer(exr_file.channel('X', FLOAT), dtype=np.float32).reshape(height, width)
    y_channel = np.frombuffer(exr_file.channel('Y', FLOAT), dtype=np.float32).reshape(height, width)
    z_channel = np.frombuffer(exr_file.channel('Z', FLOAT), dtype=np.float32).reshape(height, width)

    # You can combine these into a single 3-channel array if you need
    normals_or_positions = np.stack([x_channel, y_channel, z_channel], axis=-1)

    # Read the alpha channel
    # alpha_channel = np.frombuffer(exr_file.channel('A', FLOAT), dtype=np.float32).reshape(height, width)

    return normals_or_positions

def read_depth_exr(exr_file):
    exr_data = OpenEXR.InputFile(exr_file)
    header = exr_data.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if 'V' in header['channels']:
        depth = np.frombuffer(exr_data.channel('V', FLOAT), dtype=np.float32).reshape(height, width)
    else:
        depth = np.frombuffer(exr_data.channel('R', FLOAT), dtype=np.float32).reshape(height, width)
    return depth[..., None] # (H, W, 1)