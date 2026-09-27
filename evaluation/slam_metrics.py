import numpy as np


def project_point(pose, p0):
    return pose[:3, :3] @ p0 + pose[:3, 3]


def angle_from_rotation(R):
    trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.arccos(trace)


def pose_error(P_pred, P_gt):
    return np.linalg.inv(P_pred) @ P_gt


def relative_pose_error(P_pred_i, P_pred_j, P_gt_i, P_gt_j):
    delta_gt = np.linalg.inv(P_gt_i) @ P_gt_j
    delta_est = np.linalg.inv(P_pred_i) @ P_pred_j
    return np.linalg.inv(delta_gt) @ delta_est


# absolute translation error (ATE)
def compute_ate(pred_poses, gt_poses, p0=None):
    n = len(pred_poses)
    errors = np.zeros(n)
    for i in range(n):
        if p0 is not None:
            p_pred = project_point(pred_poses[i], p0)
            p_gt = project_point(gt_poses[i], p0)
            errors[i] = np.linalg.norm(p_pred - p_gt)
        else:
            E = pose_error(pred_poses[i], gt_poses[i])
            errors[i] = np.linalg.norm(E[:3, 3])
    return errors


# absolute rotation error (ARE)
def compute_are(pred_poses, gt_poses):
    """Per-frame absolute rotation errors (degrees)"""
    n = len(pred_poses)
    errors = np.zeros(n)
    for i in range(n):
        E = pose_error(pred_poses[i], gt_poses[i])
        errors[i] = np.degrees(angle_from_rotation(E[:3, :3]))
    return errors


def compute_rpe_trans(pred_poses, gt_poses, p0=None):
    n = len(pred_poses)
    errors = np.zeros(n - 1)
    for i in range(n - 1):
        if p0 is not None:
            vel_pred = project_point(pred_poses[i + 1], p0) - project_point(
                pred_poses[i], p0
            )
            vel_gt = project_point(gt_poses[i + 1], p0) - project_point(gt_poses[i], p0)
            errors[i] = np.linalg.norm(vel_pred - vel_gt)
        else:
            E = relative_pose_error(
                pred_poses[i], pred_poses[i + 1], gt_poses[i], gt_poses[i + 1]
            )
            errors[i] = np.linalg.norm(E[:3, 3])
    return errors


def compute_rpe_rot(pred_poses, gt_poses):
    """Frame-to-frame relative rotation errors (degrees)"""
    n = len(pred_poses)
    errors = np.zeros(n - 1)
    for i in range(n - 1):
        E = relative_pose_error(
            pred_poses[i], pred_poses[i + 1], gt_poses[i], gt_poses[i + 1]
        )
        errors[i] = np.degrees(angle_from_rotation(E[:3, :3]))
    return errors


def compute_rpe_combined(pred_poses, gt_poses):
    n = len(pred_poses)
    errors = []
    for i in range(n - 1):
        E = relative_pose_error(
            pred_poses[i], pred_poses[i + 1], gt_poses[i], gt_poses[i + 1]
        )
        t_err = np.linalg.norm(E[:3, 3])
        r_err = np.degrees(angle_from_rotation(E[:3, :3]))
        errors.append({"trans_m": t_err, "rot_deg": r_err})
    return errors


def compute_trans_drift(pred_poses, gt_poses, p0=None):
    """
    Translational drift (mm/mm %)
    """
    ate = compute_ate(pred_poses, gt_poses, p0)

    if p0 is not None:
        gt_positions = np.array([project_point(P, p0) for P in gt_poses])
    else:
        gt_positions = np.array([P[:3, 3] for P in gt_poses])

    path_segments = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    total_path = np.sum(path_segments)

    trans_drift_pct = (ate[-1] / max(total_path, 1e-6)) * 100

    return {
        "trans_drift_pct": float(trans_drift_pct),
    }


def compute_rot_drift(pred_poses, gt_poses):
    """
    Rotational drift
    Returns (deg/deg %)
    """
    are = compute_are(pred_poses, gt_poses)
    n = len(gt_poses)

    total_rot = 0.0
    for i in range(1, n):
        R_rel = gt_poses[i][:3, :3] @ gt_poses[i - 1][:3, :3].T
        total_rot += np.degrees(angle_from_rotation(R_rel))
    rot_drift_ratio = are[-1] / max(total_rot, 1e-6)
    rot_drift_pct = rot_drift_ratio * 100
    return {
        "rot_drift_pct": float(rot_drift_pct),
    }

def compute_accumulated_trans_drift(pred_poses, gt_poses, p0=None):
    ate = compute_ate(pred_poses, gt_poses, p0)

    if p0 is not None:
        gt_positions = np.array([project_point(P, p0) for P in gt_poses])
    else:
        gt_positions = np.array([P[:3, 3] for P in gt_poses])

    path_segments = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    cumulative_path = np.cumsum(path_segments)

    # skip frame 0 (no path traveled yet)
    ratios = []
    for i in range(1, len(ate)):
        if cumulative_path[i - 1] > 1e-6:
            ratios.append(ate[i] / cumulative_path[i - 1])

    return {
        "acc_trans_drift_pct": float(np.mean(ratios) * 100) if ratios else 0.0,
    }


def compute_accumulated_rot_drift(pred_poses, gt_poses):
    are = compute_are(pred_poses, gt_poses)
    n = len(gt_poses)

    # cumulative GT rotation at each frame
    cumulative_rot = np.zeros(n)
    for i in range(1, n):
        R_rel = gt_poses[i][:3, :3] @ gt_poses[i - 1][:3, :3].T
        cumulative_rot[i] = cumulative_rot[i - 1] + np.degrees(angle_from_rotation(R_rel))

    ratios = []
    for i in range(1, n):
        if cumulative_rot[i] > 1e-6:
            ratios.append(are[i] / cumulative_rot[i])

    return {
        "acc_rot_drift_pct": float(np.mean(ratios) * 100) if ratios else 0.0,
    }

def compute_2d_dist(pred_poses, gt_poses, K, p0=None):
    n = len(pred_poses)

    errors = np.zeros(n)
    for i in range(n):
        if p0 is not None:
            p_pred = project_point(pred_poses[i], p0)
            p_gt = project_point(gt_poses[i], p0)
        else:
            p_pred = pred_poses[i][:3, 3]
            p_gt = gt_poses[i][:3, 3]

        uv_pred = K @ p_pred
        uv_pred = uv_pred[:2] / uv_pred[2]

        uv_gt = K @ p_gt
        uv_gt = uv_gt[:2] / uv_gt[2]

        errors[i] = np.linalg.norm(uv_pred - uv_gt)

    return errors


def compute_slam_metrics(pred_poses, gt_poses, p0=None, K=None):
    n = len(pred_poses)
    assert n == len(gt_poses), f"length mismatch: {n} vs {len(gt_poses)}"
    assert n >= 2, "need at least 2 frames"

    ate = compute_ate(pred_poses, gt_poses, p0)
    are = compute_are(pred_poses, gt_poses)
    rpe_t = compute_rpe_trans(pred_poses, gt_poses, p0)
    rpe_r = compute_rpe_rot(pred_poses, gt_poses)
    rpe_combined = compute_rpe_combined(pred_poses, gt_poses)
    t_drift = compute_trans_drift(pred_poses, gt_poses, p0)
    r_drift = compute_rot_drift(pred_poses, gt_poses)
    ate_2d = compute_2d_dist(pred_poses, gt_poses, K, p0) if K is not None else None
    acc_t_drift = compute_accumulated_trans_drift(pred_poses, gt_poses, p0)
    acc_r_drift = compute_accumulated_rot_drift(pred_poses, gt_poses)

    return {
        "per_frame": {
            "ate": ate,
            "are": are,
            "rpe_trans": rpe_t,
            "rpe_rot": rpe_r,
        },
        "summary": {
            "ate_rmse_m": float(np.sqrt(np.mean(ate**2))),
            "ate_mean_m": float(np.mean(ate)),
            "ate_mean_mm": float(np.mean(ate) * 1000),
            "are_rmse_deg": float(np.sqrt(np.mean(are**2))),
            "are_mean_deg": float(np.mean(are)),
            "rpe_trans_rmse_m": float(np.sqrt(np.mean(rpe_t**2))),
            "rpe_trans_mean_mm": float(np.mean(rpe_t) * 1000),
            "rpe_rot_rmse_deg": float(np.sqrt(np.mean(rpe_r**2))),
            "rpe_rot_mean_deg": float(np.mean(rpe_r)),
            "rpe_mean_mm": float(np.mean([e["trans_m"] for e in rpe_combined]) * 1000),
            "rpe_mean_deg": float(np.mean([e["rot_deg"] for e in rpe_combined])),
            "trans_drift_pct": t_drift["trans_drift_pct"],
            "rot_drift_pct": r_drift["rot_drift_pct"],
            "2d_mean_px": float(np.mean(ate_2d)) if ate_2d is not None else None,
            "2d_rmse_px": (
                float(np.sqrt(np.mean(ate_2d**2))) if ate_2d is not None else None
            ),
            "acc_trans_drift_pct": acc_t_drift["acc_trans_drift_pct"],
            "acc_rot_drift_pct": acc_r_drift["acc_rot_drift_pct"],
        },
    }


def summarize_sequences(all_metrics):
    keys = all_metrics[0]["summary"].keys()
    avg = {}
    for k in keys:
        vals = [m["summary"][k] for m in all_metrics]
        avg[k] = float(np.mean(vals))
    return avg
