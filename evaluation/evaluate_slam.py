import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from evaluation.dataset import UniformDataset
from evaluation.slam_metrics import compute_slam_metrics


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_anchor_lookup(filter_json):
    filt = load_json(filter_json)
    lookup = {}
    for s in filt["scenes"]:
        name = f"{filt['dataset']}_{s['scene_id']:05d}"
        lookup[name] = s["top_windows"][0]["anchor_frame"]
    return lookup, filt


def gather_poses(results_json, filter_json, uniform_root):
    data = load_json(results_json)
    anchor_lookup, filt = build_anchor_lookup(filter_json)
    n_frames = filt["window_size"]
    stride = filt.get("stride", 1)

    ds_cache = {}
    groups = defaultdict(lambda: {"pred": {}, "gt": {}})

    if "predictions" not in data:
        data = {"predictions": data}

    for p in data["predictions"]:
        scene = p["scene_name"]
        oid = p.get("obj_id") or p.get("object_id")
        fid = p["frame_idx"]
        anchor = anchor_lookup.get(scene, 0)

        if not filt["dataset"] in scene:
            continue

        cache_key = (scene, oid, anchor)
        if cache_key not in ds_cache:
            scene_dir = Path(uniform_root) / scene
            if not scene_dir.exists():
                continue
            try:
                ds = UniformDataset(str(scene_dir), oid, anchor, n_frames, stride)
            except (KeyError, ValueError):
                continue
            ds_cache[cache_key] = ds
        ds = ds_cache[cache_key]

        key = (scene, oid)
        groups[key]["pred"][fid] = np.array(p["pose_4x4"])
        groups[key]["gt"][fid] = ds.get_gt_pose(fid)

    return (
        data.get("model", "unknown"),
        data.get("dataset", "unknown"),
        groups,
        ds_cache,
    )


def evaluate(results_json, filter_json, uniform_root):
    model, dataset, groups, ds_cache = gather_poses(
        results_json, filter_json, uniform_root
    )

    all_summaries = []
    per_scene = {}

    for (scene, oid), poses in sorted(groups.items()):
        fids = sorted(poses["pred"].keys())
        if len(fids) < 3:
            continue

        pred_list = [poses["pred"][f] for f in fids]
        gt_list = [poses["gt"][f] for f in fids]

        p0 = None
        matching_key = [k for k in ds_cache if k[0] == scene and k[1] == oid]

        # try proxy points
        proxy_pts_path = Path(uniform_root) / scene / "prompt_meta.json"
        if proxy_pts_path.exists():
            pp = load_json(str(proxy_pts_path))
            obj_coords = np.array(pp["object_coordinates"][0])
            # p0 = obj_coords.mean(axis=0)
            p0 = obj_coords
        elif matching_key and ds_cache[matching_key[0]].has_mesh():
            p0 = np.array(ds_cache[matching_key[0]].get_gt_mesh().vertices).mean(axis=0)

        K = ds_cache[matching_key[0]].K if matching_key else None
        result = compute_slam_metrics(pred_list, gt_list, p0=p0, K=K)

        all_summaries.append(result["summary"])
        per_scene[f"{scene}_obj{oid}"] = result["summary"]

    # success counting
    thresholds_mm = [10, 20, 50]
    success = {t: 0 for t in thresholds_mm}
    for s in all_summaries:
        for t in thresholds_mm:
            if s["ate_mean_mm"] < t:
                success[t] += 1

    n_total = len(all_summaries)
    if n_total == 0:
        return None

    success_rates = {}
    for t in thresholds_mm:
        success_rates[f"success_{t}mm"] = f"{success[t]}/{n_total}"
        success_rates[f"success_{t}mm_pct"] = round(success[t] / n_total * 100, 1)

    avg = {}
    for k in all_summaries[0]:
        avg[k] = round(float(np.mean([s[k] for s in all_summaries])), 4)

    return {
        "model": model,
        "dataset": dataset,
        "n_sequences": len(all_summaries),
        "mean": avg,
        "success": success_rates,
        "per_scene": per_scene,
    }


def run_one(results_json, filter_json, uniform_root, label, output):
    if not Path(results_json).exists():
        print(f"skip {label}, {results_json} not found")
        return

    result = evaluate(results_json, filter_json, uniform_root)
    if result is None:
        print(f"skip {label}, not enough frames")
        return

    result["model"] = label

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)

    print(f"{label}: {result['n_sequences']} seqs")


def main():
    ap = argparse.ArgumentParser(
        description="Trajectory metrics (ATE, ARE, RPE, drift, 2D error) reported in the paper."
    )
    ap.add_argument("--results_json", required=True, help="Predictions JSON (list of per-frame poses)")
    ap.add_argument("--filter_json", required=True, help="Benchmark window file, e.g. evaluation/benchmarks/ho3d/w1_f49.json")
    ap.add_argument("--uniform_root", required=True, help="Root of the benchmark in uniform format")
    ap.add_argument("--output", default=None, help="Where to write the metrics JSON")
    ap.add_argument("--label", default="method", help="Method name stored in the output")
    args = ap.parse_args()

    run_one(args.results_json, args.filter_json, args.uniform_root, args.label, args.output)


if __name__ == "__main__":
    main()
