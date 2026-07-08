# 🎯 ProxyPose

**ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation**

[![arXiv](https://img.shields.io/badge/arXiv-2607.06555-b31b1b.svg)](https://arxiv.org/abs/2607.06555)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://ruihangzhang97.github.io/proxypose/)

[Ruihang Zhang](https://ruihangzhang97.github.io/)\*<sup>1</sup>, [Felix Taubner](https://felixtaubner.github.io/)\*<sup>1,2</sup>, [Pooja Ravi](https://01pooja10.github.io/)<sup>1</sup>, [Kiriakos N. Kutulakos](https://www.cs.toronto.edu/~kyros/)<sup>1,2</sup>, [David B. Lindell](https://davidlindell.com/)<sup>1,2</sup>

<sup>1</sup>University of Toronto &nbsp; <sup>2</sup>Vector Institute &nbsp; \*Equal contribution

---

**TL;DR:** One query pixel in, a full 6‑DoF pose trajectory out.

---

## ⚡️ Quick start

### 🛠️ 1. Install

```bash
# Clone the repository
git clone https://github.com/ruihangzhang97/proxypose.git
cd proxypose

# Create and activate a conda environment
conda create -n proxypose python=3.10 -y
conda activate proxypose

# Install PyTorch 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch3D 
export FORCE_CUDA=1
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable

# Install ProxyPose and all remaining dependencies
pip install -e .
```

### 📦 2. Download model weights

The weights are **downloaded automatically on first run** — no manual steps needed.

| Weight | HuggingFace | Size |
|--------|-------------|------|
| Wan2.1-T2V-14B (base) | [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~30 GB |
| ProxyPose LoRA | [`ruihangzhang79/proxypose`](https://huggingface.co/ruihangzhang79/proxypose) | ~600 MB |


### 🖱️ 3. Pick your prompt point

```bash
proxypose-annotate --input-video video/my_video.mp4
```

Opens `http://localhost:7860` in your browser. Click on a query point in the first frame, press **Save**. Coordinates are written to `video/my_video.points.json`.

### 🔭 4. (Optional) Estimate focal length with Depth Anything 3

By default, we assume a 45° horizontal field of view. For improved accuracy, consider using [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

First, install Depth Anything 3 inside your ProxyPose environment:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

pip install --no-build-isolation -e .
```

Then, from the ProxyPose repository, run:

```bash
# Requires:  pip install hatchling editables
python -m inference.annotation.depth_anything video/my_video.mp4
```

### ✅ 5. Run inference

Pass the query JSON saved by the annotator directly to `--prompt`:

```bash
proxypose-infer \
    --video_path   video/my_video.mp4 \
    --output_path  output/result.mp4 \
    --prompt       video/my_video.points.json \
    --depth_anything_path video/my_video.da3.npz   # optional, omit to use fixed 45° FOV
```
---

## 📖 Citation

```bibtex
@article{zhang2026proxypose,
  title={ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation},
  author={Ruihang Zhang and Felix Taubner and Pooja Ravi and Kiriakos N. Kutulakos and David B. Lindell},
  journal={arXiv preprint arXiv:2607.06555},
  year={2026}
}
```