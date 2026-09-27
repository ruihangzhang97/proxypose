import cv2
import json
import shutil
import pickle
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import pdist


HO3D_DEPTH_SCALE = 0.00012498664727900177


def decode_ho3d_depth(depth_3ch):
    """3-channel uint8 PNG to meters as float32 (H, W)."""

    if depth_3ch.ndim != 3:
        raise ValueError(f"expected 3-channel depth, got shape {depth_3ch.shape}")
    R = depth_3ch[..., 2].astype(np.float32)
    G = depth_3ch[..., 1].astype(np.float32)
    return (R + G * 256.0) * HO3D_DEPTH_SCALE


def bbox_from_mask(binary):
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def mesh_diameter_cached(mesh_path, cache):
    if mesh_path in cache:
        return cache[mesh_path]
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    v = mesh.vertices
    if len(v) > 2000:
        idx = np.random.default_rng(0).choice(len(v), 2000, replace=False)
        v = v[idx]
    d = float(pdist(v).max())
    cache[mesh_path] = d
    return d


def load_models_info(models_info_path):
    with open(models_info_path) as f:
        return json.load(f)


def reshape_symmetries(entry):
    out = {"discrete": [], "continuous": []}
    for flat in entry.get("symmetries_discrete", []):
        M = np.array(flat, dtype=np.float64).reshape(4, 4)
        M[:3, 3] = M[:3, 3] / 1000.0
        out["discrete"].append(M.tolist())
    for sym in entry.get("symmetries_continuous", []):
        out["continuous"].append(
            {
                "axis": list(sym["axis"]),
                "offset": [v / 1000.0 for v in sym["offset"]],
            }
        )
    return out


def convert_sequence(
    src_seq_dir,
    dst_root,
    ycbv_meshes_rel,
    models_info,
    mesh_diameter_cache,
    apply_pose_flip=True,
    scene_id_offset=0,
):
    seq_name = src_seq_dir.name
    new_scene_id = scene_id_offset
    dst_scene = dst_root / f"ho3d_{new_scene_id:05d}"

    for sub in ("color", "depth", "mask", "meta"):
        (dst_scene / sub).mkdir(parents=True, exist_ok=True)

    rgb_dir = src_seq_dir / "rgb"
    depth_dir = src_seq_dir / "depth"
    seg_dir = src_seq_dir / "seg"
    meta_dir = src_seq_dir / "meta"

    frame_stems = sorted(p.stem for p in rgb_dir.glob("*.jpg"))
    if not frame_stems:
        raise ValueError(f"no rgb frames in {rgb_dir}")

    # read first frame meta, get object info, K, img shape
    with open(meta_dir / f"{frame_stems[0]}.pkl", "rb") as f:
        first_meta = pickle.load(f)

    obj_id = int(first_meta["objLabel"])
    obj_name = first_meta["objName"]
    K_first = np.asarray(first_meta["camMat"], dtype=np.float64)

    first_rgb = cv2.imread(str(rgb_dir / f"{frame_stems[0]}.jpg"))
    imh, imw = first_rgb.shape[:2]

    mesh_path_rel = f"{ycbv_meshes_rel}/obj_{obj_id:06d}.ply"
    mesh_path_abs = (dst_scene / mesh_path_rel).resolve()

    diameter = mesh_diameter_cached(str(mesh_path_abs), mesh_diameter_cache)
    symmetries = reshape_symmetries(models_info.get(str(obj_id), {}))

    flip = np.diag([1.0, -1.0, -1.0]) if apply_pose_flip else np.eye(3)

    K_changed = False

    # per-frame conversion
    new_idx = 0
    source_frame_map = {}
    for stem in tqdm(frame_stems, desc=seq_name, leave=False):
        with open(meta_dir / f"{stem}.pkl", "rb") as f:
            fmeta = pickle.load(f)

        if fmeta.get("objRot") is None or fmeta.get("objTrans") is None:
            continue

        K_frame = np.asarray(fmeta["camMat"], dtype=np.float64)
        if not np.allclose(K_frame, K_first, atol=1e-3) and not K_changed:
            print(f"[warn] {seq_name}: K varies, using first frame K")
            K_changed = True

        rvec = np.asarray(fmeta["objRot"], dtype=np.float64).reshape(3, 1)
        R_raw, _ = cv2.Rodrigues(rvec)
        t_raw = np.asarray(fmeta["objTrans"], dtype=np.float64).flatten()

        R_cv = flip @ R_raw
        t_cv = flip @ t_raw

        new_stem = f"{new_idx:04d}"
        source_frame_map[new_stem] = stem

        # color
        bgr = cv2.imread(str(rgb_dir / f"{stem}.jpg"))
        cv2.imwrite(str(dst_scene / "color" / f"{new_stem}_color.png"), bgr)

        # depth, 3-chn uint8 to meters to uint16 PNG at 0.1mm
        depth_3ch = cv2.imread(str(depth_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        depth_m = decode_ho3d_depth(depth_3ch)
        depth_u16 = np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(dst_scene / "depth" / f"{new_stem}_depth.png"), depth_u16)

        # mask, binary to uint8 with obj_id
        seg = cv2.imread(str(seg_dir / f"{int(stem):05d}.png"), cv2.IMREAD_UNCHANGED)
        if seg.ndim == 3:
            seg = seg[..., 0]
        binary = seg > 0
        label = np.zeros(seg.shape, dtype=np.uint8)
        label[binary] = obj_id
        cv2.imwrite(str(dst_scene / "mask" / f"{new_stem}_mask.png"), label)

        bbox = bbox_from_mask(binary)

        frame_meta = {
            "frame_id": new_idx,
            "source_frame_id": stem,
            "objects": {
                str(obj_id): {
                    "R": R_cv.tolist(),
                    "t": t_cv.tolist(),
                    "visib_fract": 1.0,
                    "bbox_visib": bbox,
                }
            },
        }
        with open(dst_scene / "meta" / f"{new_stem}_meta.json", "w") as f:
            json.dump(frame_meta, f, indent=2)

        new_idx += 1

    valid_frame_count = new_idx

    scene_meta = {
        "dataset": "ho3d",
        "source_scene_id": seq_name,
        "scene_name": f"ho3d_{new_scene_id:05d}",
        "n_frames": valid_frame_count,
        "image_width": int(imw),
        "image_height": int(imh),
        "K": K_first.tolist(),
        "depth_scale_to_meters": 0.0001,
        "source_frame_map": source_frame_map,
        "objects": {
            str(obj_id): {
                "class_name": obj_name,
                "mesh_path": mesh_path_rel,
                "diameter_m": round(diameter, 6),
                "symmetries": symmetries,
            }
        },
    }
    with open(dst_scene / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)

    print(
        f"[ok] {seq_name} -> {dst_scene.name}: {valid_frame_count}/{len(frame_stems)} valid frames, obj={obj_name}"
    )


def main():
    ap = argparse.ArgumentParser(description="Convert the HO3D v3 evaluation split to the uniform format.")
    ap.add_argument("--src", type=Path, required=True, help="HO3D v3 'evaluation' directory")
    ap.add_argument("--dst", type=Path, required=True, help="Output root in uniform format")
    ap.add_argument(
        "--ycbv_meshes_rel",
        default="../meshes/ycbv",
        help="YCB-V mesh directory (containing obj_XXXXXX.ply), relative to each scene directory",
    )
    ap.add_argument(
        "--models_info",
        type=Path,
        required=True,
        help="YCB-V (BOP) models_info.json, used for object symmetries",
    )
    ap.add_argument("--scene_id_start", type=int, default=0, help="Index of the first converted scene")
    ap.add_argument("--sequences", nargs="+", default=None, help="Only convert these sequences")
    ap.add_argument("--no_pose_flip", action="store_true", help="Skip the OpenGL to OpenCV pose flip")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(args.models_info)
    mesh_diameter_cache = {}

    if args.sequences:
        seq_dirs = [args.src / s for s in args.sequences]
    else:
        seq_dirs = sorted(d for d in args.src.iterdir() if d.is_dir())

    print(f"converting {len(seq_dirs)} HO3D sequences")
    for i, seq_dir in enumerate(seq_dirs):
        if not seq_dir.is_dir():
            print(f"[skip] missing: {seq_dir}")
            continue
        try:
            convert_sequence(
                seq_dir,
                args.dst,
                args.ycbv_meshes_rel,
                models_info,
                mesh_diameter_cache,
                apply_pose_flip=not args.no_pose_flip,
                scene_id_offset=args.scene_id_start + i,
            )
        except Exception as e:
            print(f"[fail] {seq_dir.name}: {type(e).__name__}: {e}")

    print(f"done: {args.dst}")


if __name__ == "__main__":
    main()
