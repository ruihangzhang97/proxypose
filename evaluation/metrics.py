import numpy as np
from scipy.spatial import cKDTree


def rotation_error(R_pred, R_gt):
    cos = np.clip((np.trace(R_pred.T @ R_gt) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def translation_error(t_pred, t_gt):
    return float(np.linalg.norm(t_pred - t_gt))


def proxy_point_error(R_pred, t_pred, R_gt, t_gt, proxy_point):
    pred_pos = R_pred @ proxy_point + t_pred
    gt_pos = R_gt @ proxy_point + t_gt
    return float(np.linalg.norm(pred_pos - gt_pos))


def find_local_ADD(R_pred, t_pred, R_gt, t_gt, verts, proxy_point, radius=0.02):
    dists_to_proxy = np.linalg.norm(verts - proxy_point, axis=1)
    local_verts = verts[dists_to_proxy < radius]
    if len(local_verts) == 0:
        local_verts = verts[np.argsort(dists_to_proxy)[:50]]
    pred = (R_pred @ local_verts.T).T + t_pred
    gt = (R_gt @ local_verts.T).T + t_gt
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def find_ADD(R_pred, t_pred, R_gt, t_gt, vertices):
    pred = (R_pred @ vertices.T).T + t_pred
    gt = (R_gt @ vertices.T).T + t_gt
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def find_local_ADD_S(R_pred, t_pred, R_gt, t_gt, verts, proxy_point, radius=0.02):
    dists_to_proxy = np.linalg.norm(verts - proxy_point, axis=1)
    local_verts = verts[dists_to_proxy < radius]
    if len(local_verts) == 0:
        local_verts = verts[np.argsort(dists_to_proxy)[:50]]
    pred = (R_pred @ local_verts.T).T + t_pred
    gt = (R_gt @ local_verts.T).T + t_gt
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    return float(dists.mean())


def find_ADD_S(R_pred, t_pred, R_gt, t_gt, vertices):
    pred = (R_pred @ vertices.T).T + t_pred
    gt = (R_gt @ vertices.T).T + t_gt
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    return float(dists.mean())


def pose_jitter(pred_poses, gt_poses):
    pred_rot_deltas = []
    pred_trans_deltas = []
    gt_rot_deltas = []
    gt_trans_deltas = []

    for i in range(1, len(pred_poses)):
        # predicted frame-to-frame rotation change
        R_delta_pred = pred_poses[i][:3, :3] @ pred_poses[i - 1][:3, :3].T
        cos_p = np.clip((np.trace(R_delta_pred) - 1) / 2, -1, 1)
        pred_rot_deltas.append(np.degrees(np.arccos(cos_p)))
        pred_trans_deltas.append(
            np.linalg.norm(pred_poses[i][:3, 3] - pred_poses[i - 1][:3, 3])
        )

        # GT frame-to-frame rotation change
        R_delta_gt = gt_poses[i][:3, :3] @ gt_poses[i - 1][:3, :3].T
        cos_g = np.clip((np.trace(R_delta_gt) - 1) / 2, -1, 1)
        gt_rot_deltas.append(np.degrees(np.arccos(cos_g)))
        gt_trans_deltas.append(
            np.linalg.norm(gt_poses[i][:3, 3] - gt_poses[i - 1][:3, 3])
        )

    return {
        "rot_jitter_pred": round(np.std(pred_rot_deltas), 3),
        "rot_jitter_gt": round(np.std(gt_rot_deltas), 3),
        "rot_jitter_excess": round(np.std(pred_rot_deltas) - np.std(gt_rot_deltas), 3),
        "trans_jitter_pred_cm": round(np.std(pred_trans_deltas) * 100, 3),
        "trans_jitter_gt_cm": round(np.std(gt_trans_deltas) * 100, 3),
        "trans_jitter_excess_cm": round(
            np.std(pred_trans_deltas) * 100 - np.std(gt_trans_deltas) * 100, 3
        ),
    }


def relative_pose_error(pred_poses, gt_poses):
    rel_rot_errors = []
    rel_trans_errors = []

    for i in range(1, len(pred_poses)):
        gt_rel = gt_poses[i] @ np.linalg.inv(gt_poses[i - 1])
        pred_rel = pred_poses[i] @ np.linalg.inv(pred_poses[i - 1])

        R_err = pred_rel[:3, :3] @ gt_rel[:3, :3].T
        cos = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rel_rot_errors.append(np.degrees(np.arccos(cos)))
        rel_trans_errors.append(np.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3]))

    return {
        "rpe_rot_mean": round(np.mean(rel_rot_errors), 2),
        "rpe_rot_median": round(np.median(rel_rot_errors), 2),
        "rpe_trans_cm_mean": round(np.mean(rel_trans_errors) * 100, 2),
        "rpe_trans_cm_median": round(np.median(rel_trans_errors) * 100, 2),
    }


def find_MSSD(R_pred, t_pred, R_gt, t_gt, vertices, symmetries):
    sym_transforms = [np.eye(4)]
    for symm in symmetries.get("discrete", []):
        sym_transforms.append(np.array(symm))

    best = np.inf
    for symm in sym_transforms:
        R_s = np.array(symm[:3, :3]) if np.array(symm).shape == (4, 4) else np.eye(3)
        t_s = np.array(symm[:3, 3]) if np.array(symm).shape == (4, 4) else np.zeros(3)
        gt_sym = (R_gt @ (R_s @ vertices.T + t_s[:, None])).T + t_gt
        pred = (R_pred @ vertices.T).T + t_pred
        d = np.linalg.norm(pred - gt_sym, axis=1).max()
        best = min(best, d)
    return float(best)


def find_MSPD(R_pred, t_pred, R_gt, t_gt, vertices, K, symmetries):
    sym_transforms = [np.eye(4)]
    for symm in symmetries.get("discrete", []):
        sym_transforms.append(np.array(symm))

    best = np.inf
    for symm in sym_transforms:
        R_s = np.array(symm)[:3, :3] if np.array(symm).shape == (4, 4) else np.eye(3)
        t_s = np.array(symm)[:3, 3] if np.array(symm).shape == (4, 4) else np.zeros(3)

        gt_sym = (R_gt @ (R_s @ vertices.T + t_s[:, None])).T + t_gt
        pred = (R_pred @ vertices.T).T + t_pred
        kpred = (K @ pred.T).T
        proj_pred = kpred[:, :2] / kpred[:, 2:]
        kgt = (K @ gt_sym.T).T
        proj_gt = kgt[:, :2] / kgt[:, 2:]
        dist = np.linalg.norm(proj_pred - proj_gt, axis=1).max()
        best = min(best, dist)

    return float(best)


def find_3d_iou(R_pred, t_pred, R_gt, t_gt, bbox_side_len):
    s = np.array(bbox_side_len)
    vol_single = s[0] * s[1] * s[2]
    diff = np.abs(t_pred - t_gt)
    overlap = np.maximum(s - diff, 0)
    vol_inter = overlap[0] * overlap[1] * overlap[2]
    vol_union = 2 * vol_single - vol_inter
    if vol_union < 1e-12:
        return 0.0
    return float(vol_inter / vol_union)


def recall_at_threshold(values, threshold):
    return float((np.array(values) < threshold).mean())


def auc_at_threshold(values, max_val):
    values = np.array(values)
    thresholds = np.linspace(0, max_val, 100)
    recalls = [(values < t).mean() for t in thresholds]
    return float(np.trapz(recalls, thresholds) / max_val)
