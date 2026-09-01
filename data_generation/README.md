# ProxyPose — Synthetic Data Generation

Generates ProxyPose's synthetic training data: RGB videos of objects moved around a 3D scene
(scripted motion or physics-based dropping), rendered in Blender Cycles alongside ground-truth
6-DoF object pose, depth, instance masks, and other auxiliary buffers. A Blender-based
re-implementation of the [Diffusion Renderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/)
data pipeline — some buffers may differ from that original.

Self-contained, with its own dependencies (Blender/`bpy`), separate from the root `inference/` package.

## Installation

Blender 4.2 installs as a plain Python package, which avoids most dependency/server-setup pain:

```bash
cd data_generation
conda create -n proxypose-datagen python=3.11.0
conda activate proxypose-datagen
pip install -r requirements.txt
```

## Quick start

Example configs run out of the box against the small demo assets bundled under `data/`:

```bash
python blender_datagen_compose.py --config configs/render_orbit_cam.yaml out_dir=output/blender_compose
python blender_datagen_compose.py --config configs/render_drop_phy.yaml out_dir=output/drop_phy
```

For real generation, point `base_path` at your own object library (e.g. an
[Objaverse](https://objaverse.allenai.org/) subset) instead of the demo objects — `data/objects/`
is an empty placeholder for that; see the comment at the top of `configs/render_drop_phy.yaml` or
`configs/render_fly_physics.yaml`.

## Configuration

Configs are YAML, loaded with OmegaConf; override any key with `key=value` dotlist args after
`--config` (e.g. `num_frames=8 resolution=[512,512]`). All keys are set in `configs/*.yaml` — the
most commonly changed ones:

- `video_mode`: `orbit_cam`, `oscil_cam`, `orbit_lgt`, `rotat_obj`, `vtran_obj`, `dolly_cam`, `drop_phy`, `arbitrary_motion`
- `base_path` / `envlight` / `placement_plane_textures`: your object / HDRI / ground-texture asset paths
- `num_frames`, `num_rendering`, `num_lighting`, `resolution`, `spp`, `use_denoise`
- `glbs_per_scene`, `glbs_scale_range`, `glbs_placement_bbox`: how many objects and where they go
- `drop_phy`'s physics behavior (gravity, restitution, spawn region, ...) lives under the `physics:`/`spawn:`/`initial_motion:` blocks — see `configs/render_drop_phy.yaml` for a fully-commented example

## Output

```text
output/drop_phy/drop_phy_s000042/
  0000.rgb.mp4  0000.masked.mp4  0000.meta.json
  depth/            # per-frame depth EXRs
  instance_maps/    # per-object masks
  meshes/           # canonical object meshes (.glb/.ply)
```

`0000.meta.json` has per-frame camera/environment info and per-object 6-DoF pose (`object_transforms`).

## Next step

`0000.rgb.mp4` alone isn't a training pair yet — see [`proxy_generation/`](proxy_generation/README.md)
for rendering the matching ground-truth proxy video from `0000.meta.json`.

## Acknowledgments

Envmap rendering inspired by [nvdiffrec](https://github.com/nvlabs/nvdiffrec); albedo/material
rendering adapted from [MaterialFusion](https://github.com/yehonathanlitman/MaterialFusion);
Blender-as-python-package approach from [InfiniGen](https://github.com/princeton-vl/infinigen).

## License

Licensed under [Apache License 2.0](./LICENSE) (may differ from the repo root's license).
Bundled `data/` assets are CC0/public domain — see `data/README.md`. Assets you supply yourself
(e.g. Objaverse) keep their own licenses. For citation, see the [root README](../README.md).
