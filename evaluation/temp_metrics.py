import numpy as np
from scipy.spatial.transform import Rotation


def project_point(pose, p0):
    return pose[:3, :3] @ p0 + pose[:3, 3]


def rotation_error_deg(R_pred, R_gt):
    R_rel = R_pred.T @ R_gt
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def relative_rotation(R_curr, R_prev):
    return R_curr @ R_prev.T


def rotation_to_angle(R):
    trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def compute_proxy_metrics(pred_poses, gt_poses, p0, fps=30.0):
    # p0, tracked point in object frame (mesh centroid)

    n = len(pred_poses)
    assert n == len(gt_poses), f"pred {n} vs gt {len(gt_poses)}"

    # per-frame projected points
    p_pred = np.array([project_point(pred_poses[t], p0) for t in range(n)])
    p_gt = np.array([project_point(gt_poses[t], p0) for t in range(n)])

    # absolute translation error per frame
    ate = np.linalg.norm(p_pred - p_gt, axis=1)

    # absolute rotation error per frame
    are = np.array(
        [
            rotation_error_deg(pred_poses[t][:3, :3], gt_poses[t][:3, :3])
            for t in range(n)
        ]
    )

    # translational velocity error (frame-to-frame)
    vel_pred = np.diff(p_pred, axis=0)
    vel_gt = np.diff(p_gt, axis=0)
    tve = np.linalg.norm(vel_pred - vel_gt, axis=1)

    # rotational velocity error (frame-to-frame)
    rve = []
    for t in range(1, n):
        R_rel_pred = relative_rotation(pred_poses[t][:3, :3], pred_poses[t - 1][:3, :3])
        R_rel_gt = relative_rotation(gt_poses[t][:3, :3], gt_poses[t - 1][:3, :3])
        R_diff = R_rel_pred.T @ R_rel_gt
        rve.append(rotation_to_angle(R_diff))
    rve = np.array(rve)

    # translational drift (cumulative error / time)
    # drift normalized by distance traveled (SLAM convention)
    gt_distances = np.cumsum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))
    total_dist = gt_distances[-1] if gt_distances[-1] > 1e-6 else 1e-6
    trans_drift_pct = (ate[-1] / total_dist) * 100
    rot_drift_deg_per_m = are[-1] / total_dist

    return {
        "ate": ate,
        "are": are,
        "tve": tve,
        "rve": rve,
        "trans_drift": float(trans_drift_pct),
        "rot_drift": float(rot_drift_deg_per_m),
    }


def summarize_proxy_metrics(metrics):
    ate = metrics["ate"]
    are = metrics["are"]
    tve = metrics["tve"]
    rve = metrics["rve"]
    trans_drift = metrics["trans_drift"]
    rot_drift = metrics["rot_drift"]

    return {
        "ate_mean_mm": float(np.mean(ate) * 1000),
        "ate_median_mm": float(np.median(ate) * 1000),
        "ate_max_mm": float(np.max(ate) * 1000),
        "are_mean_deg": float(np.mean(are)),
        "are_median_deg": float(np.median(are)),
        "are_max_deg": float(np.max(are)),
        "tve_mean_mm": float(np.mean(tve) * 1000),
        "tve_median_mm": float(np.median(tve) * 1000),
        "rve_mean_deg": float(np.mean(rve)),
        "rve_median_deg": float(np.median(rve)),
        "trans_drift": float(trans_drift * 1000),
        "rot_drift": float(rot_drift),
    }


def evaluate_sequence(pred_poses, gt_poses, mesh_vertices, fps=30.0):
    # using mesh centroid as proxy point
    p0 = mesh_vertices.mean(axis=0)
    metrics = compute_proxy_metrics(pred_poses, gt_poses, p0, fps)
    return summarize_proxy_metrics(metrics), metrics
