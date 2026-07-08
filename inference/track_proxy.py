import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from inference.utils import load_video, save_video, load_config


COLOR_THRESHOLD = 70      # BGR color distance threshold for color masking
CONTOUR_AREA_THRESHOLD = 500  # Minimum contour area in pixels² for detection

# BGR colors matching each face of the proxy cube (see render_proxy.py ColoredCube)
COLORS = {
    "white":  np.array((255, 255, 255)),
    "yellow": np.array((0, 255, 255)),
    "cyan":   np.array((230, 230, 0)),
    "green":  np.array((0, 240, 75)),
    "blue":   np.array((255, 30, 30)),
    "red":    np.array((10, 10, 255)),
}

# 3D vertices of each cube face in object space (CCW winding viewed from outside)
PROXY_CONTOURS = {
    "+z": np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * 0.5,
    "-z": np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]) * 0.5,
    "-x": np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]) * 0.5,
    "+x": np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]) * 0.5,
    "-y": np.array([[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]]) * 0.5,
    "+y": np.array([[1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1]]) * 0.5,
}


def color_distance_threshold(bgr_img, target_color, threshold=COLOR_THRESHOLD):
    """Create a binary mask of pixels within `threshold` BGR distance of `target_color`."""
    distance = np.linalg.norm(bgr_img.astype(np.float32) - target_color[None, None], axis=-1)
    mask = (distance < threshold).astype(np.uint8) * 255
    return mask


def detect_contours_by_color(hsv_frame, color="red", contour_threshold=None, use_sub_pix_accuracy=True, border_threshold=5):
    """Detect quadrilateral contours matching a named proxy cube face color.

    Args:
        hsv_frame:            HSV image (H, W, 3).
        color:                One of the keys in COLORS.
        contour_threshold:    Minimum contour area; defaults to CONTOUR_AREA_THRESHOLD.
        use_sub_pix_accuracy: Refine corners to sub-pixel accuracy.
        border_threshold:     Ignore contours touching the image border within this margin.

    Returns:
        contours: List of detected quadrilateral contours (each (4, 1, 2) or (4, 2) float).
        mask:     Binary mask (H, W) uint8.
    """
    if color not in COLORS:
        raise ValueError(f"Color '{color}' not supported. Choose from: {', '.join(COLORS.keys())}")

    if contour_threshold is None:
        contour_threshold = CONTOUR_AREA_THRESHOLD

    bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    mask = color_distance_threshold(bgr_frame, COLORS[color], threshold=COLOR_THRESHOLD)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > contour_threshold]
    contours = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in contours]
    contours = [c for c in contours if len(c) == 4]

    h, w = hsv_frame.shape[:2]
    contours = [
        c for c in contours
        if np.all(c[:, 0, 0] >= border_threshold) and np.all(c[:, 0, 0] <= w - border_threshold - 1) and
           np.all(c[:, 0, 1] >= border_threshold) and np.all(c[:, 0, 1] <= h - border_threshold - 1)
    ]

    if use_sub_pix_accuracy:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        contours = [
            cv2.cornerSubPix(gray, c.astype(np.float32), (5, 5), (-1, -1), criteria)
            for c in contours
        ]

    return contours, mask


def match_contour(rvec, tvec, intrinsics, contour_2d, scale=0.5, valids=None):
    """Find the best-matching 3D cube face for a detected 2D contour.

    Args:
        rvec:        Rotation vector (3, 1).
        tvec:        Translation vector (3, 1).
        intrinsics:  (3, 3) camera intrinsic matrix.
        contour_2d:  (4, 2) array of detected 2D corners.
        scale:       Half-size of the cube in meters.
        valids:      Set of face names to consider; None means all.

    Returns:
        best_match: Name of the best-matching face key in PROXY_CONTOURS.
        best_perm:  Index permutation aligning the projected face to contour_2d.
    """
    best_match = None
    best_perm = None
    best_dist = float("inf")

    for contour_name, contour_3d in PROXY_CONTOURS.items():
        if valids is not None and contour_name not in valids:
            continue

        projected_2d, _ = cv2.projectPoints(contour_3d * scale, rvec, tvec, intrinsics, None)
        projected_2d = projected_2d[:, 0, :]

        for permute_id in range(4):
            permutation = np.roll(np.arange(4), permute_id)
            dist = np.max(np.linalg.norm(projected_2d - contour_2d[permutation], axis=-1))
            if dist < best_dist:
                best_dist = dist
                best_perm = permutation
                best_match = contour_name

    return best_match, best_perm


def visualize_contours(frame, contours, color=np.array([0, 255, 0])):
    """Draw contours and their areas onto a copy of `frame`."""
    contours = [c.astype(int) for c in contours]
    cv2.drawContours(frame, contours, -1, tuple(color.tolist()), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for contour in contours:
        area = cv2.contourArea(contour.astype(np.float32))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"{area:.1f}", (x + 5, y + 20), font, 0.4, tuple(color.tolist()), 1)

    return frame


def visualize_transform(frame, rvec, tvec, intrinsics, s=0.3):
    """Overlay a 3D coordinate frame axes on `frame` at the given pose."""
    origin = np.array([[0, 0, 0]], dtype=np.float64)
    axes = np.array([[s, 0, 0], [0, s, 0], [0, 0, s]], dtype=np.float64)

    pts, _ = cv2.projectPoints(np.vstack([origin, axes]), rvec, tvec, intrinsics, None)
    pts = pts[:, 0, :].astype(int)

    o = tuple(pts[0])
    cv2.line(frame, o, tuple(pts[1]), (0, 0, 255), 2)   # X: red
    cv2.line(frame, o, tuple(pts[2]), (0, 255, 0), 2)   # Y: green
    cv2.line(frame, o, tuple(pts[3]), (255, 0, 0), 2)   # Z: blue
    return frame


def locate_cube(frame, intrinsics, config, rvec_prop=None, tvec_prop=None, matches_prop=None):
    """Detect the colored proxy cube in a single frame and solve for its 6-DoF pose.

    Args:
        frame:        BGR image (H, W, 3).
        intrinsics:   (3, 3) camera intrinsic matrix.
        config:       Tracking config dict (cube_scale, use_sub_pix_accuracy).
        rvec_prop:    Proposed rotation vector from the previous frame (or None).
        tvec_prop:    Proposed translation vector from the previous frame (or None).
        matches_prop: Previous-frame color-to-face assignment for temporal consistency.

    Returns:
        (rvec, tvec, contour_matches, vis_frame) on success, or None if no cube detected.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cube_contours = []
    contour_colors = []

    for color in COLORS.keys():
        contours, _ = detect_contours_by_color(
            hsv, color=color, use_sub_pix_accuracy=config["use_sub_pix_accuracy"])
        if len(contours) > 0:
            cube_contours.append(contours[0])
            contour_colors.append(color)

    if len(cube_contours) == 0:
        return None

    cube_contours, contour_colors = zip(
        *sorted(zip(cube_contours, contour_colors), key=lambda x: cv2.contourArea(x[0]), reverse=True)
    )

    s = config["cube_scale"]
    contour_matches = []

    if rvec_prop is None:
        main_face_2d = cube_contours[0][:, 0, :]
        main_face_3d = PROXY_CONTOURS["+z"]
        _, rvec, tvec = cv2.solvePnP(main_face_3d * s, main_face_2d.astype(np.float64), intrinsics, None)
        contour_matches.append((contour_colors[0], "+z", main_face_2d))
    else:
        rvec, tvec = rvec_prop.copy(), tvec_prop.copy()

    contour_valids = []
    for color in contour_colors:
        if matches_prop is None:
            contour_valids.append(list(PROXY_CONTOURS.keys()))
        else:
            valid_faces = set(PROXY_CONTOURS.keys())
            for key, value, _ in matches_prop:
                if key == color:
                    valid_faces = {value}
                else:
                    valid_faces.discard(value)
            contour_valids.append(valid_faces)

    for i in range(len(contour_matches), len(cube_contours)):
        other_contour = cube_contours[i][:, 0, :]
        other_color = contour_colors[i]

        matched_cube_face_name, matched_perm = match_contour(
            rvec, tvec, intrinsics, other_contour, scale=s, valids=contour_valids[i],
        )
        contour_matches.append((other_color, matched_cube_face_name, other_contour[matched_perm]))

    points_2d = np.vstack([c for _, _, c in contour_matches]).astype(np.float64)
    points_3d = np.vstack([PROXY_CONTOURS[f] * s for _, f, _ in contour_matches])

    _, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d, intrinsics, None, rvec.copy(), tvec.copy(), useExtrinsicGuess=True
    )

    vis_frame = frame.copy()
    for color, _, contour in contour_matches:
        vis_frame = visualize_contours(vis_frame, [contour], color=255 - COLORS[color])
    vis_frame = visualize_transform(vis_frame, rvec, tvec, intrinsics)

    return rvec, tvec, contour_matches, vis_frame


def track_video(video, config, intrinsics=None):
    """Track the colored proxy cube across all frames of a video.

    Args:
        video:      List of BGR frames (H, W, 3) uint8.
        config:     Tracking config dict (cube_scale, use_sub_pix_accuracy, default_focal_deg).
        intrinsics: Optional (3, 3) numpy intrinsic matrix. Estimated from FOV if None.

    Returns:
        rvecs:          List of rotation vectors (one per frame; None if tracking failed).
        tvecs:          List of translation vectors (one per frame; None if tracking failed).
        vis_frames:     List of BGR visualization frames with axes overlaid.
        points_2d_list: List of (N, 2) float64 arrays of observed 2-D corner pixels per frame
                        (None for frames where tracking failed).
        points_3d_list: List of (N, 3) float64 arrays of corresponding 3-D object-space points
                        (None for frames where tracking failed).
    """
    if intrinsics is None:
        focal_deg = config["default_focal_deg"]
        focal_length_px = 0.5 * video[0].shape[1] / np.tan(np.radians(focal_deg / 2))
        intrinsics = np.array([
            [focal_length_px, 0, video[0].shape[1] / 2],
            [0, focal_length_px, video[0].shape[0] / 2],
            [0, 0, 1]
        ])

    s = config["cube_scale"]
    vis_frames = []
    rvecs, tvecs = [], []
    points_2d_list, points_3d_list = [], []

    prev_rvec, prev_tvec, prev_matches = None, None, None
    r_velocity = np.zeros((3, 1))
    t_velocity = np.zeros((3, 1))

    for frame in video:
        if prev_rvec is not None:
            rvec_proposal = prev_rvec + r_velocity
            tvec_proposal = prev_tvec + t_velocity
        else:
            rvec_proposal = None
            tvec_proposal = None

        result = locate_cube(
            frame, intrinsics, config,
            rvec_prop=rvec_proposal, tvec_prop=tvec_proposal, matches_prop=prev_matches
        )

        if prev_rvec is None:
            assert result is not None, (
                "Failed to detect cube in the first frame. "
                "Please ensure the first frame of the generated video shows the proxy cube clearly."
            )

        if result is None:
            vis_frame = frame.copy()
            if prev_rvec is not None:
                vis_frame = visualize_transform(vis_frame, rvec_proposal, tvec_proposal, intrinsics)
            vis_frames.append(vis_frame)

            prev_rvec = rvec_proposal
            prev_tvec = tvec_proposal
            prev_matches = None

            rvecs.append(prev_rvec)
            tvecs.append(prev_tvec)
            points_2d_list.append(None)
            points_3d_list.append(None)
            continue

        new_rvec, new_tvec, new_matches, vis_frame = result

        if prev_rvec is None:
            prev_rvec, prev_tvec, prev_matches = new_rvec, new_tvec, new_matches
        else:
            r_velocity = new_rvec - prev_rvec
            t_velocity = new_tvec - prev_tvec
            prev_rvec = new_rvec
            prev_tvec = new_tvec
            prev_matches = new_matches

        rvecs.append(prev_rvec)
        tvecs.append(prev_tvec)
        vis_frames.append(vis_frame)
        points_2d_list.append(
            np.vstack([c for _, _, c in new_matches]).astype(np.float64))
        points_3d_list.append(
            np.vstack([PROXY_CONTOURS[f] * s for _, f, _ in new_matches]))

    return rvecs, tvecs, vis_frames, points_2d_list, points_3d_list


def main():
    parser = argparse.ArgumentParser(description="Track colored proxy cube in a video")
    parser.add_argument("--config_path", type=str, required=True, help="Path to tracking config YAML")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output visualization video")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame index to start tracking")
    args = parser.parse_args()

    config = load_config(args.config_path)
    video = load_video(args.video_path)[args.start_frame:]

    print(f"Tracking {len(video)} frames ...")
    _, _, vis_frames = track_video(video, config)

    save_video(vis_frames, args.output_path)
    print(f"Saved tracking visualization to {args.output_path}")


if __name__ == "__main__":
    main()
