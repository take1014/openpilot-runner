"""Draw SuperCombo model outputs onto an OpenCV image."""
import cv2
import numpy as np

from .constants import MEDMODEL_K, MEDMODEL_HEIGHT
from .runner.constants import X_IDXS

# Colour palette (BGR)
_LANE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 255, 0), (255, 0, 0)]
_EDGE_COLORS = [(200, 0, 180), (200, 0, 180)]
_PATH_COLOR  = (0, 128, 255)
_LEAD_COLOR  = (0, 0, 255)

LEGEND = [
    ('Lane lines', _LANE_COLORS[1]),
    ('Road edges', _EDGE_COLORS[0]),
    ('Path',       _PATH_COLOR),
    ('Lead car',   _LEAD_COLOR),
]


def road_to_img(x_fwd: np.ndarray, y_right: np.ndarray, z_down: np.ndarray,
                K: np.ndarray = MEDMODEL_K) -> np.ndarray:
    """Project road/device-frame 3-D points to model image pixel coordinates.

    Coordinate convention matching openpilot's calib_frame_to_full_frame:
        x = forward, y = right (positive = right), z = down from camera
        (device frame: road surface ≈ z=1.22 for a 1.22-m-mounted camera)

    Projection:
        u = fx * y / x + cx
        v = fy * z / x + cy

    Returns (N,2) float32 array; rows with x_fwd ≤ 0.5 are set to NaN.
    """
    pts = np.full((len(x_fwd), 2), np.nan, dtype=np.float32)
    valid = x_fwd > 0.5
    if not valid.any():
        return pts
    xv, yv, zv = x_fwd[valid], y_right[valid], z_down[valid]
    pts[valid, 0] = (yv / xv) * K[0, 0] + K[0, 2]
    pts[valid, 1] = (zv / xv) * K[1, 1] + K[1, 2]
    return pts


def _polyline(img: np.ndarray, pts: np.ndarray,
              color: tuple, thickness: int) -> None:
    """Draw a polyline from (N,2) float pixel coords, skipping NaN segments."""
    h, w = img.shape[:2]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        if np.isnan(p0).any() or np.isnan(p1).any():
            continue
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        if (-w < x0 < 2 * w) and (-h < y0 < 2 * h):
            cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def draw_overlays(img: np.ndarray, outputs: dict,
                  scale: float, K: np.ndarray = MEDMODEL_K,
                  *, min_thickness: int = 1) -> None:
    """Draw lane lines, road edges, predicted path, and lead-car circle on *img* in-place.

    img must be in model image space (512×256 or a scaled version thereof).
    scale     = pixel multiplier from model pixels to img pixels (e.g. 2.0 for 1024×512).
    min_thickness = minimum line width in pixels (override for back-projected overlays).
    Expects outputs keys: lane_lines, road_edges, plan, lead (from ModelRunner.run).
    """
    tk = max(min_thickness, round(scale))

    lanes = outputs.get('lane_lines')
    if lanes is not None:
        for i in range(min(4, lanes.shape[1])):
            pts = road_to_img(X_IDXS, lanes[0, i, :, 0], lanes[0, i, :, 1], K) * scale
            _polyline(img, pts, _LANE_COLORS[i], tk + 1)

    edges = outputs.get('road_edges')
    if edges is not None:
        for i in range(min(2, edges.shape[1])):
            pts = road_to_img(X_IDXS, edges[0, i, :, 0], edges[0, i, :, 1], K) * scale
            _polyline(img, pts, _EDGE_COLORS[i], tk)

    plan = outputs.get('plan')
    if plan is not None:
        path = plan[0, 0]  # (IDX_N, PLAN_WIDTH), columns 0-2 = (x_fwd, y_right_dev, z_road_rel)
        # plan z is road-surface-relative (z=0 at road surface); add MEDMODEL_HEIGHT
        # to convert to device frame (z_down from camera)
        z_plan = path[:, 2] + MEDMODEL_HEIGHT
        pts = road_to_img(path[:, 0], path[:, 1], z_plan, K) * scale
        _polyline(img, pts, _PATH_COLOR, tk + 2)

    lead = outputs.get('lead')
    if lead is not None and lead.shape[1] > 0:
        lpt = lead[0, 0, 0]  # closest sample of best hypothesis
        lx, ly = float(lpt[0]), float(lpt[1])
        if lx > 1.0:
            # lead z: use plan_position.z at nearest x-index + MEDMODEL_HEIGHT,
            # matching openpilot's update_leads: z = model_position.getZ()[idx] + 1.22
            if plan is not None:
                idx = min(int(np.searchsorted(X_IDXS, lx)), len(X_IDXS) - 1)
                z_lead = float(plan[0, 0, idx, 2]) + MEDMODEL_HEIGHT
            else:
                z_lead = MEDMODEL_HEIGHT
            px = road_to_img(np.array([lx]), np.array([ly]), np.array([z_lead]), K) * scale
            if not np.isnan(px).any():
                cx = int(round(float(px[0, 0])))
                cy = int(round(float(px[0, 1])))
                cv2.circle(img, (cx, cy), max(8, int(14 * scale)),
                           _LEAD_COLOR, -1, cv2.LINE_AA)


def put_legend(img: np.ndarray, y0: int = 46, lh: int = 18) -> None:
    """Draw a small colour legend in the top-left corner of *img*."""
    lx = 8
    for label, color in LEGEND:
        cv2.rectangle(img, (lx, y0 - 12), (lx + 14, y0 + 2), color, -1)
        cv2.putText(img, label, (lx + 18, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y0 += lh
