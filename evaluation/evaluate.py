import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from evaluation.dataset import UniformDataset
from evaluation.metrics import (
    proxy_point_error,
    find_local_ADD,
    find_local_ADD_S,
    find_ADD,
    find_ADD_S,
    find_MSSD,
    find_MSPD,
    find_3d_iou,
    recall_at_threshold,
    auc_at_threshold,
    rotation_error,
    translation_error,
    pose_jitter,
    relative_pose_error,
)

from evaluation.temp_metrics import compute_proxy_metrics, summarize_proxy_metrics


def load_json(file_path):
    with open(file_path) as fl:
        return json.load(fl)


def build_anchor_lookup(filter_json):
    filt = load_json(filter_json)
    lookup = {}
    for s in filt["scenes"]:
        scene_name = f"{filt['dataset']}_{s['scene_id']:05d}"
        lookup[scene_name] = s["top_windows"][0]["anchor_frame"]
    return lookup, filt


def evaluate_predictions(results_json, filter_json, uniform_root):

    data = load_json(results_json)

    anchor_lookup, filt = build_anchor_lookup(filter_json)
    n_frames = filt["window_size"]
    stride = filt.get("stride", 1)

    per_frame = []
    ds_cache = {}

    if "predictions" not in data:
        data = {"predictions": data, "model": "unknown", "dataset": "unknown"}

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

        pred_pose = np.array(p["pose_4x4"])
        R_pred = pred_pose[:3, :3]
        t_pred = pred_pose[:3, 3]

        gt_pose = ds.get_gt_pose(fid)
        R_gt = gt_pose[:3, :3]
        t_gt = gt_pose[:3, 3]

        entry = {
            "scene": scene,
            "obj_id": oid,
            "frame_idx": fid,
            "obj_class": p.get("object_class", ds.get_object_class()),
            "rot_err_deg": rotation_error(R_pred, R_gt),
            "trans_err_m": translation_error(t_pred, t_gt),
            "pred_pose": pred_pose,
            "gt_pose": gt_pose,
        }

        if ds.has_mesh():
            mesh = ds.get_gt_mesh()
            verts = np.array(mesh.vertices)
            proxy_point = verts.mean(axis=0)

            entry["ppe"] = proxy_point_error(R_pred, t_pred, R_gt, t_gt, proxy_point)
            entry["local_add"] = find_local_ADD(
                R_pred, t_pred, R_gt, t_gt, verts, proxy_point
            )
            entry["local_add_s"] = find_local_ADD_S(
                R_pred, t_pred, R_gt, t_gt, verts, proxy_point
            )

            sym = ds.get_object_symmetries()
            diameter = ds.get_gt_mesh_diameter()
            K = ds.K

            # use full mesh vertex range
            entry["mssd"] = find_MSSD(R_pred, t_pred, R_gt, t_gt, verts, sym)
            entry["mspd"] = find_MSPD(R_pred, t_pred, R_gt, t_gt, verts, K, sym)

            entry["add"] = find_ADD(R_pred, t_pred, R_gt, t_gt, verts)
            entry["add_s"] = find_ADD_S(R_pred, t_pred, R_gt, t_gt, verts)
            entry["diameter"] = diameter

        bbox = ds.get_bbox_side_len()
        if bbox is not None:
            entry["iou_3d"] = find_3d_iou(R_pred, t_pred, R_gt, t_gt, bbox)

        per_frame.append(entry)

    temporal = compute_temporal_metrics(per_frame, ds_cache)
    return data["model"], data["dataset"], per_frame, temporal


def compute_temporal_metrics(per_frame, ds_cache):
    groups = defaultdict(list)
    for e in per_frame:
        groups[(e["scene"], e["obj_id"])].append(e)

    jitter, rpe, proxy = [], [], []

    for key, entries in groups.items():
        entries.sort(key=lambda x: x["frame_idx"])
        if len(entries) < 3:
            continue
        pred_poses = [e["pred_pose"] for e in entries]
        gt_poses = [e["gt_pose"] for e in entries]

        # pose jitter, relative pose
        jitter.append(pose_jitter(pred_poses, gt_poses))
        rpe.append(relative_pose_error(pred_poses, gt_poses))

        # SLAM proxy metrics
        scene, oid = key
        ds_key = [k for k in ds_cache if k[0] == scene and k[1] == oid]
        if ds_key and ds_cache[ds_key[0]].has_mesh():
            mesh = ds_cache[ds_key[0]].get_gt_mesh()
            p0 = np.array(mesh.vertices).mean(axis=0)
            raw = compute_proxy_metrics(pred_poses, gt_poses, p0, fps=30.0)
            proxy.append(summarize_proxy_metrics(raw))

    return {"jitter": jitter, "rpe": rpe, "proxy": proxy}


def aggregate_metrics(per_frame, has_mesh, has_bbox, temporal=None):
    rot_errs = [e["rot_err_deg"] for e in per_frame]
    trans_errs = [e["trans_err_m"] for e in per_frame]

    summary = {
        "n_predictions": len(per_frame),
        "rot_err_deg_mean": round(np.mean(rot_errs), 2),
        "rot_err_deg_median": round(np.median(rot_errs), 2),
        "trans_err_cm_mean": round(np.mean(trans_errs) * 100, 2),
        "trans_err_cm_median": round(np.median(trans_errs) * 100, 2),
    }

    rot_np = np.array(rot_errs)
    trans_cm = np.array(trans_errs) * 100
    for r, t in [(5, 2), (5, 5), (10, 2), (10, 5)]:
        acc = float(((rot_np < r) & (trans_cm < t)).mean())
        summary[f"VUS@{r}deg{t}cm"] = round(acc * 100, 1)

    if has_mesh:
        adds = [e["add"] for e in per_frame if "add" in e]
        adds_s = [e["add_s"] for e in per_frame if "add_s" in e]
        mssds = [e["mssd"] for e in per_frame if "mssd" in e]
        mspds = [e["mspd"] for e in per_frame if "mspd" in e]
        diameters = [e["diameter"] for e in per_frame if "diameter" in e]

        if adds and diameters:
            mean_d = np.mean(diameters)
            for frac in [0.02, 0.05, 0.10]:
                thresh = mean_d * frac
                summary[f"ADD<{int(frac*100)}%d"] = round(
                    recall_at_threshold(adds, thresh), 4
                )
                summary[f"ADD-S<{int(frac*100)}%d"] = round(
                    recall_at_threshold(adds_s, thresh), 4
                )
            summary["ADD_AUC@10%d"] = round(auc_at_threshold(adds, mean_d * 0.1), 4)
            summary["ADD-S_AUC@10%d"] = round(auc_at_threshold(adds_s, mean_d * 0.1), 4)
            summary["ADD-S_AUC"] = round(auc_at_threshold(adds_s, 0.1) * 100, 1)
            summary["ADD_AUC"] = round(auc_at_threshold(adds, 0.1) * 100, 1)

        ppes = [e["ppe"] for e in per_frame if "ppe" in e]
        local_adds = [e["local_add"] for e in per_frame if "local_add" in e]
        local_adds_s = [e["local_add_s"] for e in per_frame if "local_add_s" in e]
        if ppes:
            summary["PPE_mean_mm"] = round(np.mean(ppes) * 1000, 2)
            summary["PPE_median_mm"] = round(np.median(ppes) * 1000, 2)
            summary["LocalADD_AUC"] = round(auc_at_threshold(local_adds, 0.1) * 100, 1)
            summary["LocalADD-S_AUC"] = round(
                auc_at_threshold(local_adds_s, 0.1) * 100, 1
            )

        if mssds:
            mssd_recalls = [
                recall_at_threshold(mssds, t / 1000) for t in range(5, 51, 5)
            ]
            summary["MSSD_AR"] = round(np.mean(mssd_recalls) * 100, 1)

        if mspds:
            mspd_recalls = [recall_at_threshold(mspds, t) for t in range(5, 51, 5)]
            summary["MSPD_AR"] = round(np.mean(mspd_recalls) * 100, 1)

    if has_bbox:
        ious = [e["iou_3d"] for e in per_frame if "iou_3d" in e]
        if ious:
            summary["IoU_mean"] = round(np.mean(ious), 4)

    if temporal and temporal["rpe"]:
        rpe_rot = [t["rpe_rot_median"] for t in temporal["rpe"]]
        rpe_trans = [t["rpe_trans_cm_median"] for t in temporal["rpe"]]
        jitter_rot = [t["rot_jitter_excess"] for t in temporal["jitter"]]
        jitter_trans = [t.get("trans_jitter_excess_cm", 0) for t in temporal["jitter"]]
        summary["RPE_rot_median"] = round(np.mean(rpe_rot), 2)
        summary["RPE_trans_cm_median"] = round(np.mean(rpe_trans), 2)
        summary["Jitter_rot_excess"] = round(np.mean(jitter_rot), 3)
        summary["Jitter_trans_excess_cm"] = round(np.mean(jitter_trans), 3)

    if temporal and temporal.get("proxy"):
        proxy = temporal["proxy"]
        summary["ATE_mean_mm"] = round(np.mean([p["ate_mean_mm"] for p in proxy]), 2)
        summary["ATE_median_mm"] = round(
            np.mean([p["ate_median_mm"] for p in proxy]), 2
        )
        summary["ARE_mean_deg"] = round(np.mean([p["are_mean_deg"] for p in proxy]), 2)
        summary["TVE_mean_mm"] = round(np.mean([p["tve_mean_mm"] for p in proxy]), 2)
        summary["RVE_mean_deg"] = round(np.mean([p["rve_mean_deg"] for p in proxy]), 2)
        summary["trans_drift"] = round(np.mean([p["trans_drift"] for p in proxy]), 2)
        summary["rot_drift"] = round(np.mean([p["rot_drift"] for p in proxy]), 2)

    by_class = defaultdict(list)
    for e in per_frame:
        by_class[e["obj_class"]].append(e)

    class_summary = {}
    for cls, entries in sorted(by_class.items()):
        cs = {
            "n": len(entries),
            "rot_err_mean": round(np.mean([e["rot_err_deg"] for e in entries]), 2),
            "trans_err_cm_mean": round(
                np.mean([e["trans_err_m"] for e in entries]) * 100, 2
            ),
        }
        if has_mesh and any("add" in e for e in entries):
            cs["add_mean_mm"] = round(
                np.mean([e["add"] for e in entries if "add" in e]) * 1000, 2
            )
            cs["add_s_mean_mm"] = round(
                np.mean([e["add_s"] for e in entries if "add_s" in e]) * 1000, 2
            )
        if any("ppe" in e for e in entries):
            cs["ppe_mean_mm"] = round(
                np.mean([e["ppe"] for e in entries if "ppe" in e]) * 1000, 2
            )
        class_summary[cls] = cs

    summary["per_class"] = class_summary
    return summary


def run_single(results_json, filter_json, uniform_root, label, output):
    if not Path(results_json).exists():
        print(f"Skipped {label}, {results_json} not found")
        return None

    model, dataset, per_frame, temporal = evaluate_predictions(
        results_json, filter_json, uniform_root
    )
    if not per_frame:
        print(f"Skipped {label}, no predictions matched the benchmark windows")
        return None
    has_mesh = any("add" in e for e in per_frame)
    has_bbox = any("iou_3d" in e for e in per_frame)
    summary = aggregate_metrics(per_frame, has_mesh, has_bbox, temporal)
    summary["model"] = label
    summary["dataset"] = dataset

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Per-frame pose metrics (rotation/translation error, ADD(-S), MSSD, MSPD, VUS)."
    )
    ap.add_argument("--results_json", required=True, help="Predictions JSON (list of per-frame poses)")
    ap.add_argument("--filter_json", required=True, help="Benchmark window file, e.g. evaluation/benchmarks/ho3d/w1_f49.json")
    ap.add_argument("--uniform_root", required=True, help="Root of the benchmark in uniform format")
    ap.add_argument("--output", required=True, help="Where to write the metrics JSON")
    ap.add_argument("--label", default="method", help="Method name stored in the output")
    args = ap.parse_args()

    run_single(args.results_json, args.filter_json, args.uniform_root, args.label, args.output)


if __name__ == "__main__":
    main()
