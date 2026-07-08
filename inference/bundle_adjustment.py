import cv2
import numpy as np
from scipy.optimize import least_squares

def _rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _matrix_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def relative_pose(rvec1, tvec1, rvec2, tvec2):
    """Rigid offset of pose 2 expressed in the local frame of pose 1.

    Args:
        rvec1, tvec1: Reference pose (Rodrigues + translation).
        rvec2, tvec2: Target pose (Rodrigues + translation).

    Returns:
        (rvec_rel, tvec_rel): Relative rotation and translation.
    """
    R1 = _rodrigues_to_matrix(rvec1)
    R2 = _rodrigues_to_matrix(rvec2)
    R_rel = R1.T @ R2
    t_rel = R1.T @ (np.asarray(tvec2).flatten() - np.asarray(tvec1).flatten())
    return _matrix_to_rodrigues(R_rel), t_rel


def apply_relative_pose(rvec, tvec, rvec_rel, tvec_rel):
    """Apply a local-frame rigid offset to a primary pose.

    Args:
        rvec, tvec:         Primary pose (Rodrigues + translation).
        rvec_rel, tvec_rel: Local-frame offset (from :func:`relative_pose`).

    Returns:
        (rvec_new, tvec_new): Resulting pose.
    """
    R = _rodrigues_to_matrix(rvec)
    R_rel = _rodrigues_to_matrix(rvec_rel)
    R_new = R @ R_rel
    t_new = R @ np.asarray(tvec_rel).flatten() + np.asarray(tvec).flatten()
    return _matrix_to_rodrigues(R_new), t_new


def bundle_adjust_proxies(
    rvecs,
    tvecs,
    points_2d_list,
    points_3d_list,
    intrinsics,
    max_iterations=2000,
    w_translational_reg=200.0,
    w_rotational_reg=40.0,
):
    """Refine multi-proxy pose trajectories via bundle adjustment.

    Args:
        rvecs:              List of per-proxy rotation-vector trajectories.
                            ``rvecs[i][f]`` is the rvec for proxy *i* at frame *f*
                            (or ``None`` if tracking failed that frame).
        tvecs:              Matching list of translation-vector trajectories.
        points_2d_list:     ``points_2d_list[i][f]`` — (N, 2) float64 observed
                            2-D corners for proxy *i* at frame *f* (or ``None``).
        points_3d_list:     ``points_3d_list[i][f]`` — (N, 3) float64 matching
                            3-D object-space points.
        intrinsics:         (3, 3) camera intrinsic matrix.
        max_iterations:     Maximum LM iterations.
        w_translational_reg: Weight on frame-to-frame translation smoothness.
        w_rotational_reg:   Weight on frame-to-frame rotation smoothness.

    Returns:
        out_rvecs:            Refined rotation trajectories (same shape as input).
        out_tvecs:            Refined translation trajectories (same shape as input).
        n_observations_per_frame: Number of 2-D observations in each frame for proxy 0.
    """
    assert len(rvecs) == len(tvecs) == len(points_2d_list) == len(points_3d_list)

    n_proxies = len(rvecs)
    n_frames = len(rvecs[0])
    n_observations_per_frame = [
        len(pts) if pts is not None else 0 for pts in points_2d_list[0]
    ]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    valid_frames = [f for f in range(n_frames) if rvecs[0][f] is not None]
    if not valid_frames:
        return rvecs, tvecs, n_observations_per_frame

    base_rvec = np.asarray(rvecs[0][valid_frames[0]]).flatten()
    base_tvec = np.asarray(tvecs[0][valid_frames[0]]).flatten()

    # Fixed anchor pose for each secondary proxy (first valid frame)
    fixed_rvecs, fixed_tvecs, init_depths = [], [], []
    for i in range(1, n_proxies):
        f0 = next((f for f in range(n_frames) if rvecs[i][f] is not None), valid_frames[0])
        fixed_rvecs.append(np.asarray(rvecs[i][f0]).flatten())
        fixed_tvecs.append(np.asarray(tvecs[i][f0]).flatten())
        init_depths.append(1.0)

    n_valid = len(valid_frames)
    rel_offset = n_valid * 6
    x0 = np.empty(rel_offset + (n_proxies - 1))
    for j, f in enumerate(valid_frames):
        x0[j * 6:j * 6 + 3] = np.asarray(rvecs[0][f]).flatten()
        x0[j * 6 + 3:j * 6 + 6] = np.asarray(tvecs[0][f]).flatten()
    for i, d in enumerate(init_depths):
        x0[rel_offset + i] = d

    def residuals(x):
        res = []
        depths = x[rel_offset:rel_offset + (n_proxies - 1)]

        # Relative offsets from primary to each secondary proxy
        rel_rvs, rel_tvs = [], []
        for i in range(1, n_proxies):
            rv_rel, tv_rel = relative_pose(
                base_rvec, base_tvec,
                fixed_rvecs[i - 1], depths[i - 1] * fixed_tvecs[i - 1],
            )
            rel_rvs.append(rv_rel)
            rel_tvs.append(tv_rel)

        for j, f in enumerate(valid_frames):
            rv0 = x[j * 6:j * 6 + 3]
            tv0 = x[j * 6 + 3:j * 6 + 6]

            proxy_rvs = [rv0]
            proxy_tvs = [tv0]
            for i in range(n_proxies - 1):
                rv, tv = apply_relative_pose(rv0, tv0, rel_rvs[i], rel_tvs[i])
                proxy_rvs.append(rv)
                proxy_tvs.append(tv)

            # Reprojection residuals
            for i in range(n_proxies):
                pts2d_raw = points_2d_list[i][f]
                pts3d_raw = points_3d_list[i][f]
                if pts2d_raw is None or len(pts2d_raw) == 0:
                    continue
                pts2d = np.asarray(pts2d_raw, dtype=np.float64)
                pts3d = np.asarray(pts3d_raw, dtype=np.float64)
                if i > 0:
                    pts3d = pts3d * depths[i - 1]
                projected, _ = cv2.projectPoints(
                    pts3d,
                    proxy_rvs[i].reshape(3, 1),
                    proxy_tvs[i].reshape(3, 1),
                    intrinsics, None,
                )
                res.append((projected[:, 0, :] - pts2d).flatten())

            # Temporal smoothness regularisation
            if j > 0:
                prev_rv0 = x[(j - 1) * 6:(j - 1) * 6 + 3]
                prev_tv0 = x[(j - 1) * 6 + 3:(j - 1) * 6 + 6]

                prev_proxy_rvs = [prev_rv0]
                prev_proxy_tvs = [prev_tv0]
                for i in range(n_proxies - 1):
                    prev_rv, prev_tv = apply_relative_pose(prev_rv0, prev_tv0, rel_rvs[i], rel_tvs[i])
                    prev_proxy_rvs.append(prev_rv)
                    prev_proxy_tvs.append(prev_tv)

                for i in range(n_proxies):
                    if w_rotational_reg > 0:
                        dR = (_rodrigues_to_matrix(proxy_rvs[i])
                              @ _rodrigues_to_matrix(prev_proxy_rvs[i]).T)
                        res.append(w_rotational_reg * _matrix_to_rodrigues(dR))
                    if w_translational_reg > 0:
                        res.append(w_translational_reg * (proxy_tvs[i] - prev_proxy_tvs[i]))

        return np.concatenate(res) if res else np.zeros(1)

    result = least_squares(residuals, x0, method="lm", max_nfev=max_iterations)
    x_opt = result.x
    opt_depths = x_opt[rel_offset:rel_offset + (n_proxies - 1)]

    rel_rvs, rel_tvs = [], []
    for i in range(1, n_proxies):
        rv_rel, tv_rel = relative_pose(
            base_rvec, base_tvec,
            fixed_rvecs[i - 1], opt_depths[i - 1] * fixed_tvecs[i - 1],
        )
        rel_rvs.append(rv_rel)
        rel_tvs.append(tv_rel)

    out_rvecs = [[None] * n_frames for _ in range(n_proxies)]
    out_tvecs = [[None] * n_frames for _ in range(n_proxies)]
    for j, f in enumerate(valid_frames):
        rv0 = x_opt[j * 6:j * 6 + 3]
        tv0 = x_opt[j * 6 + 3:j * 6 + 6]
        out_rvecs[0][f] = rv0.reshape(3, 1)
        out_tvecs[0][f] = tv0.reshape(3, 1)
        for i in range(n_proxies - 1):
            rv, tv = apply_relative_pose(rv0, tv0, rel_rvs[i], rel_tvs[i])
            out_rvecs[i + 1][f] = rv.reshape(3, 1)
            out_tvecs[i + 1][f] = tv.reshape(3, 1)

    print(f"Bundle adjustment complete. Optimised depths: {opt_depths}")
    return out_rvecs, out_tvecs, n_observations_per_frame
