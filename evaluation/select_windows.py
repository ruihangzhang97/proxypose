import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1.T @ R2
    cos = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    return float(np.linalg.norm(t1 - t2))


def mask_area(scene_dir, fid, obj_id):
    m = cv2.imread(
        str(scene_dir / "mask" / f"{fid:04d}_mask.png"), cv2.IMREAD_UNCHANGED
    )
    return int((m == obj_id).sum())


def load_scene_poses(scene_dir: Path):
    with open(scene_dir / "scene_meta.json") as f:
        scene_meta = json.load(f)
    n_frames = scene_meta["n_frames"]

    poses = {}
    for fid in range(n_frames):
        with open(scene_dir / "meta" / f"{fid:04d}_meta.json") as f:
            fm = json.load(f)
        frame = {}
        for oid_str, entry in fm["objects"].items():
            frame[int(oid_str)] = {
                "R": np.array(entry["R"]),
                "t": np.array(entry["t"]).flatten(),
                "visib_fract": float(entry.get("visib_fract", 1.0)),
            }
        poses[fid] = frame

    return scene_meta, poses, n_frames


def pick_reference_object(poses: dict, n_frames: int) -> int:
    stats = defaultdict(lambda: {"count": 0, "visib_sum": 0.0})
    for frame in poses.values():
        for oid, p in frame.items():
            stats[oid]["count"] += 1
            stats[oid]["visib_sum"] += p["visib_fract"]

    eligible = [
        (oid, s["visib_sum"] / s["count"])
        for oid, s in stats.items()
        if s["count"] >= 0.95 * n_frames
    ]
    if not eligible:
        return max(stats.items(), key=lambda kv: kv[1]["count"])[0]
    return max(eligible, key=lambda x: x[1])[0]


def analyze_scene(
    scene_dir,
    poses: dict,
    n_frames: int,
    window_size: int,
    max_step_deg: float,
    min_anchor_visib: float,
    stride: int,
):

    span = stride * (window_size - 1) + 1
    if n_frames < span:
        return {"n_frames": n_frames, "windows": [], "ref_obj_id": None}

    ref_obj = pick_reference_object(poses, n_frames)
    windows = []

    if min_anchor_visib > 0:
        areas = {}
        for fid in range(n_frames):
            m = cv2.imread(
                str(scene_dir / "mask" / f"{fid:04d}_mask.png"), cv2.IMREAD_UNCHANGED
            )
            areas[fid] = int((m == ref_obj).sum())
        max_area = max(areas.values()) if areas else 0
        visib = {f: (a / max_area if max_area > 0 else 0.0) for f, a in areas.items()}
    else:
        visib = defaultdict(lambda: 1.0)

    for start in range(n_frames - span + 1):
        fids = list(range(start, start + span, stride))
        if any(ref_obj not in poses[f] for f in fids):
            continue

        R_a = poses[fids[0]][ref_obj]["R"]
        R_e = poses[fids[-1]][ref_obj]["R"]
        t_a = poses[fids[0]][ref_obj]["t"]
        t_e = poses[fids[-1]][ref_obj]["t"]

        rot_delta = rotation_angle_deg(R_a, R_e)
        trans_delta = translation_distance(t_a, t_e)

        steps = [
            rotation_angle_deg(
                poses[fids[i]][ref_obj]["R"], poses[fids[i + 1]][ref_obj]["R"]
            )
            for i in range(len(fids) - 1)
        ]
        mean_step = float(np.mean(steps))
        max_step = float(np.max(steps))

        # checking smoothness
        if max_step > max_step_deg:
            continue

        # checking visibility
        if visib[fids[0]] < min_anchor_visib:
            continue

        shared = set(poses[fids[0]].keys())
        for f in fids[1:]:
            shared &= set(poses[f].keys())

        windows.append(
            {
                "anchor_frame": fids[0],
                "end_frame": fids[-1],
                "rot_delta_deg": round(rot_delta, 3),
                "trans_delta_m": round(trans_delta, 4),
                "mean_keyframe_step_deg": round(mean_step, 3),
                "max_keyframe_step_deg": round(max_step, 3),
                "n_shared_objs": len(shared),
                "shared_obj_ids": sorted(shared),
            }
        )

    return {"n_frames": n_frames, "windows": windows, "ref_obj_id": int(ref_obj)}


def top_windows(
    analysis: dict, top_n: int, used_anchors: list = None, min_gap: int = 49
):
    if not analysis["windows"]:
        return []
    ranked = sorted(
        analysis["windows"],
        key=lambda w: (-w["rot_delta_deg"], w["mean_keyframe_step_deg"]),
    )
    used_anchors = used_anchors or []
    out = []
    for w in ranked:
        if all(abs(w["anchor_frame"] - u) >= min_gap for u in used_anchors):
            out.append(w)
            used_anchors = used_anchors + [w["anchor_frame"]]
            if len(out) == top_n:
                break
    return out


def main():
    p = argparse.ArgumentParser(
        description="Select the evaluation window (anchor frame + 49 frames) for each scene."
    )
    p.add_argument("--uniform_root", required=True, help="Root of the benchmark in uniform format")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. ho3d, ycbineoat, proxy")
    p.add_argument("--scene_prefix", default=None, help="Scene directory prefix (default: '<dataset>_')")
    p.add_argument("--window_size", type=int, default=49)
    p.add_argument("--top_n", type=int, default=1)
    p.add_argument("--max_step_deg", type=float, default=10.0)
    p.add_argument("--min_anchor_visib", type=float, default=0.0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    args.scene_prefix = args.scene_prefix or f"{args.dataset}_"

    root = Path(args.uniform_root)
    scene_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith(args.scene_prefix)
    )
    print(f"found {len(scene_dirs)} scenes")

    all_results = []
    seen_windows = {}
    span = args.stride * (args.window_size - 1) + 1
    for sdir in tqdm(scene_dirs):
        scene_meta, poses, n_frames = load_scene_poses(sdir)
        analysis = analyze_scene(
            sdir,
            poses,
            n_frames,
            args.window_size,
            args.max_step_deg,
            args.min_anchor_visib,
            args.stride,
        )
        ref = analysis["ref_obj_id"]
        used = seen_windows.get(ref, [])
        selected = top_windows(analysis, args.top_n, used_anchors=used, min_gap=span)
        if selected:
            seen_windows.setdefault(ref, []).extend(w["anchor_frame"] for w in selected)

        # check if windows were chosen
        if not selected:
            print(f"{sdir.name}: no valid windows, searching again")
            analysis = analyze_scene(
                sdir,
                poses,
                n_frames,
                args.window_size,
                args.max_step_deg,
                0.0,
                args.stride,
            )
            selected = top_windows(
                analysis,
                args.top_n,
                used_anchors=seen_windows.get(analysis["ref_obj_id"], []),
                min_gap=span,
            )
            if not selected:
                print("still no valid windows")
                continue
            else:
                print("found window")

        scene_id_int = int(sdir.name.split("_")[-1])
        all_results.append(
            {
                "scene_id": scene_id_int,
                "n_keyframes": n_frames,
                "reference_obj_id": analysis["ref_obj_id"],
                "top_windows": selected,
            }
        )

    if args.max_scenes:
        all_results.sort(
            key=lambda s: s["top_windows"][0]["rot_delta_deg"],
            reverse=True,
        )
        all_results = all_results[: args.max_scenes]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"w{args.top_n}_f{args.window_size}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "window_size": args.window_size,
                "stride": args.stride,
                "top_n_per_scene": args.top_n,
                "n_scenes": len(all_results),
                "scenes": all_results,
            },
            f,
            indent=2,
        )

    print(f"saved {len(all_results)} scenes to {out_path}")


if __name__ == "__main__":
    main()
