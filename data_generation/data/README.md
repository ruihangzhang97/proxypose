# Bundled demo assets

This directory ships a small set of demo assets so the example configs in `configs/*.yaml` run
out of the box. For real data generation, point `base_path` / `envlight` /
`placement_plane_textures` at your own, larger asset library instead (see the top-level
[README](../README.md#data-and-assets)). `data/objects/` is an intentionally empty placeholder
for that purpose.

## Sources and licenses

All bundled assets below are CC0 / public domain.

- `envmaps/*.hdr`, `textures/*/` — HDRIs and PBR material sets from [Poly Haven](https://polyhaven.com/), licensed [CC0](https://polyhaven.com/license).
- `toy_objects/spot/` — the "Spot" model by Keenan Crane, from the [CMU Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/index.html#spot), released into the public domain.
- `toy_objects/bob/` — model from the [CMU Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/), dedicated under CC0 1.0 Universal.
- `basicshapes/`, `plane_basic/`, `plane_box.glb` — simple primitives authored for this project.

If you populate `data/objects/` (or point `base_path` elsewhere) with your own asset library
(e.g. Objaverse), those assets retain their own licenses — check compliance before publishing
any dataset generated with them.
