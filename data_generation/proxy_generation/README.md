# Proxy Video Generation (Ground Truth)

Uses PyTorch3D to render the ground-truth colored-cube "proxy" video paired with one
data_generation clip's RGB video — the training target ProxyPose needs — directly from the known
poses in `0000.meta.json`. A separate, ordinary environment from Blender's `bpy`.

**Runs on CPU by default (recommended):** it's a 12-triangle cube with flat shading, no model
involved, and CPU skips CUDA-version matching entirely. Pass `--device cuda` only if you already
have a matching CUDA PyTorch3D build.

## Installation

Needs `torch`, `pytorch3d`, `opencv-python`, `numpy`, `tqdm`, and a system `ffmpeg`. Any env that
already has PyTorch3D works fine; for a new one:

```bash
pip install torch numpy opencv-python tqdm
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Usage

```bash
python generate_proxy_video.py --input-dir /path/to/output/drop_phy/drop_phy_s000042
```

