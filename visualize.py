"""Draw SuperCombo model outputs onto an OpenCV image, matching openpilot v0.8.10 UI style.

Key differences from a plain polyline overlay:
  - Lane lines   : white filled bands; alpha = lane_line_prob  (fade when uncertain)
  - Road edges   : red  filled bands; alpha = clamp(1 − std, 0, 1)
  - Path / plan  : white filled band ±0.5 m; vertical gradient (opaque at bottom → transparent)
                   truncated at lead-car distance when lead prob > 0.5
  - Lead car     : yellow-glow / red-body chevron (▲); size scales with distance;
                   red fill alpha increases when close / approaching; only shown if prob > 0.5
"""
import cv2
import numpy as np

from .constants import MEDMODEL_K, MEDMODEL_HEIGHT
from .runner.constants import X_IDXS

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
_WHITE      = (255, 255, 255)
_RED_BGR    = ( 49,  34, 201)   # openpilot red   RGB(201, 34, 49)
_YELLOW_BGR = ( 37, 202, 218)   # openpilot yellow RGB(218, 202, 37)

# ── Bird's-eye view geometry ──────────────────────────────────────────────────
BEV_SCALE     = 5.0   # pixels per metre
BEV_MAX_X     = 80.0  # metres forward visible
BEV_MAX_Y     = 20.0  # metres lateral visible (±)
BEV_MARGIN_PX = 20    # extra pixels below origin (car icon space)

BEV_W = int(BEV_MAX_Y * 2 * BEV_SCALE)              # 200 px
BEV_H = int(BEV_MAX_X * BEV_SCALE) + BEV_MARGIN_PX  # 420 px

_GRID_COLOR = ( 50,  50,  50)   # dark grey grid lines
_GRID_LABEL = (110, 110, 110)   # distance labels
_CAR_COLOR  = (200, 200, 200)   # own vehicle

LEGEND = [
    ('Lane lines', _WHITE),
    ('Road edges', _RED_BGR),
    ('Path',       _WHITE),
    ('Lead car',   _YELLOW_BGR),
]


# ── Projection ────────────────────────────────────────────────────────────────

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


# ── Band polygon helpers ───────────────────────────────────────────────────────

def _band_polygon(x_arr: np.ndarray, y_center: np.ndarray, z_arr: np.ndarray,
                  y_half: float, K: np.ndarray, scale: float) -> np.ndarray | None:
    """Return a closed polygon (N,2) int32 that outlines a band of half-width y_half metres.

    The polygon goes forward along the left edge (y_center − y_half) then
    backward along the right edge (y_center + y_half), matching the vertex
    winding used in openpilot's update_line_data().
    Returns None when no valid projected points exist.
    """
    left_pts  = road_to_img(x_arr, y_center - y_half, z_arr, K) * scale
    right_pts = road_to_img(x_arr, y_center + y_half, z_arr, K) * scale
    valid = ~np.isnan(left_pts).any(axis=1) & ~np.isnan(right_pts).any(axis=1)
    if not valid.sum():
        return None
    lp = left_pts[valid].astype(np.int32)
    rp = right_pts[valid].astype(np.int32)
    return np.vstack([lp, rp[::-1]])


def _fill_poly_alpha(img: np.ndarray, poly: np.ndarray | None,
                     color_bgr: tuple, alpha: float) -> None:
    """Fill a polygon on *img* in-place with semi-transparent colour (alpha in [0, 1])."""
    if poly is None or len(poly) < 3 or alpha <= 0.0:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _fill_path_gradient(img: np.ndarray, poly: np.ndarray | None,
                        grad_top: float = 0.4) -> None:
    """Fill path polygon with a white→transparent vertical gradient.

    grad_top : fraction of image height below which the gradient starts.
               Derived from K[1,2]/img_h so the fade begins near the horizon.
               Fully opaque at bottom, fully transparent above grad_top.
    """
    if poly is None or len(poly) < 3:
        return
    h, w = img.shape[:2]

    # Rasterise polygon to an alpha mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if not mask.any():
        return

    # α(y) = clamp((y − grad_top*h) / (h − grad_top*h), 0, 1)
    y_transp = h * grad_top
    span = max(1.0, h - y_transp)
    ys = np.clip((np.arange(h, dtype=np.float32) - y_transp) / span, 0.0, 1.0)
    alpha_map = (mask.astype(np.float32) / 255.0) * ys[:, np.newaxis]  # (h, w)

    # Composite white × alpha_map onto img
    img_f = img.astype(np.float32)
    for c in range(3):
        img[:, :, c] = np.clip(
            img_f[:, :, c] * (1.0 - alpha_map) + 255.0 * alpha_map, 0, 255
        ).astype(np.uint8)


# ── Lead-car chevron ──────────────────────────────────────────────────────────

def _draw_chevron(img: np.ndarray, cx: int, cy: int,
                  sz: float, fill_alpha: float) -> None:
    """Draw openpilot-style lead-car chevron (▲) centred on *(cx, cy)*.

    Matches draw_chevron() in paint.cc:
        Yellow glow (slightly oversized) + red body whose alpha encodes proximity.
    sz         : half-height of the chevron in pixels
    fill_alpha : red body opacity in [0, 1]
    """
    sz = max(6, int(sz))
    gxo, gyo = sz // 5, sz // 10

    # Glow – yellow, fixed opacity
    glow = np.array([
        [cx + int(sz * 1.35) + gxo, cy + sz + gyo],
        [cx,                         cy - gxo      ],
        [cx - int(sz * 1.35) - gxo, cy + sz + gyo],
    ], dtype=np.int32)
    _fill_poly_alpha(img, glow, _YELLOW_BGR, 0.85)

    # Body – red, distance / speed dependent alpha
    body = np.array([
        [cx + int(sz * 1.25), cy + sz],
        [cx,                  cy     ],
        [cx - int(sz * 1.25), cy + sz],
    ], dtype=np.int32)
    _fill_poly_alpha(img, body, _RED_BGR, float(np.clip(fill_alpha, 0.0, 1.0)))


# ── Main overlay entry point ──────────────────────────────────────────────────

def draw_overlays(img: np.ndarray, outputs: dict,
                  scale: float, K: np.ndarray = MEDMODEL_K,
                  *, min_thickness: int = 1) -> None:
    """Draw lane lines, road edges, path, and lead car on *img* in-place (openpilot style).

    img must be in model image space (512×256 or a scaled version thereof).
    scale = pixel multiplier from model pixels to img pixels (e.g. 2.0 → 1024×512).
    Expects outputs keys produced by runner.parser.parse_outputs():
        lane_lines, lane_line_probs, road_edges, road_edge_stds,
        plan, lead, lead_prob.
    """
    plan      = outputs.get('plan')
    lead      = outputs.get('lead')
    lead_prob = float(outputs.get('lead_prob', 0.0))

    # ── Determine how far to draw the path (truncate at lead car) ─────────────
    max_path_idx = len(X_IDXS) - 1
    if lead is not None and lead_prob > 0.5:
        d_rel    = float(lead[0, 0, 0, 0])
        lead_d   = d_rel * 2.0
        max_dist = max(0.0, lead_d - min(lead_d * 0.35, 10.0))
        idx = int(np.searchsorted(X_IDXS, max_dist))
        max_path_idx = max(0, min(idx, len(X_IDXS) - 1))

    # ── Path — white filled band ±0.5 m with gradient fade ────────────────────
    # Drawn first so lane lines and edges appear on top.
    if plan is not None:
        path   = plan[0, 0, :max_path_idx + 1]   # (K, 15)
        z_plan = path[:, 2] + MEDMODEL_HEIGHT      # road-surface-relative → device frame
        poly   = _band_polygon(path[:, 0], path[:, 1], z_plan, 0.5, K, scale)
        # Derive gradient start from principal point (horizon position in image)
        grad_top = float(np.clip(K[1, 2] / img.shape[0], 0.05, 0.55))
        _fill_path_gradient(img, poly, grad_top)

    # ── Road edges — red filled bands; alpha = clamp(1 − std, 0, 1) ───────────
    # Matches paint.cc: nvgRGBAf(1.0, 0.0, 0.0, clamp(1 − road_edge_stds[i], 0, 1))
    edges     = outputs.get('road_edges')
    edge_stds = outputs.get('road_edge_stds')
    if edges is not None:
        for i in range(min(2, edges.shape[1])):
            std   = float(edge_stds[i]) if edge_stds is not None else 1.0
            alpha = float(np.clip(1.0 - std, 0.0, 1.0))
            if alpha < 0.05:
                continue
            poly = _band_polygon(X_IDXS, edges[0, i, :, 0], edges[0, i, :, 1],
                                  0.025, K, scale)
            _fill_poly_alpha(img, poly, _RED_BGR, alpha)

    # ── Lane lines — white filled bands; alpha = lane_line_prob ───────────────
    # Band half-width = 0.025 × prob metres (thinner when uncertain).
    # Matches paint.cc: nvgRGBAf(1, 1, 1, lane_line_probs[i])
    lanes      = outputs.get('lane_lines')
    lane_probs = outputs.get('lane_line_probs')
    if lanes is not None:
        for i in range(min(4, lanes.shape[1])):
            prob   = float(lane_probs[i]) if lane_probs is not None else 0.5
            prob   = float(np.clip(prob, 0.0, 1.0))
            if prob < 0.05:
                continue
            y_half = 0.025 * prob
            poly   = _band_polygon(X_IDXS, lanes[0, i, :, 0], lanes[0, i, :, 1],
                                    y_half, K, scale)
            _fill_poly_alpha(img, poly, _WHITE, prob)

    # ── Lead car — chevron, only when lead_prob > 0.5 ─────────────────────────
    # Matches paint.cc draw_lead(): chevron size ∝ 1/distance,
    # red fill alpha increases when close or approaching.
    if lead is not None and lead_prob > 0.5:
        lpt    = lead[0, 0, 0]                    # closest sample [x, y, v, a]
        lx, ly = float(lpt[0]), float(lpt[1])
        v_rel  = float(lpt[2])
        if lx > 1.0:
            if plan is not None:
                idx    = min(int(np.searchsorted(X_IDXS, lx)), len(X_IDXS) - 1)
                z_lead = float(plan[0, 0, idx, 2]) + MEDMODEL_HEIGHT
            else:
                z_lead = MEDMODEL_HEIGHT

            px = road_to_img(np.array([lx]), np.array([ly]),
                             np.array([z_lead]), K) * scale
            if not np.isnan(px).any():
                cx = int(round(float(px[0, 0])))
                cy = int(round(float(px[0, 1])))

                # Chevron size: clamp((25×30)/(d/3+30), 15, 30) × scale × 0.5
                # (0.5 factor down-scales from openpilot's ~2560-wide canvas)
                sz = float(np.clip((25 * 30) / (lx / 3 + 30), 15.0, 30.0)) * scale * 0.5

                # Red fill alpha — brighter when close or approaching
                speed_buff, lead_buff = 10.0, 40.0
                fill_alpha = 0.0
                if lx < lead_buff:
                    fill_alpha = 1.0 - lx / lead_buff
                    if v_rel < 0:              # approaching
                        fill_alpha += -v_rel / speed_buff
                    fill_alpha = min(fill_alpha, 1.0)

                _draw_chevron(img, cx, cy, sz, fill_alpha)

                # Distance label — e.g. "12.34m"
                dist_label = f'{lx:.2f}m'
                font_scale = max(0.40, scale * 0.30)
                thickness  = max(1, int(scale * 0.7))
                (tw, th), baseline = cv2.getTextSize(
                    dist_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = max(0, cx - tw // 2)
                ty = cy + int(sz) + th + 4
                # Shadow for legibility
                cv2.putText(img, dist_label, (tx + 1, ty + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(img, dist_label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            _YELLOW_BGR, thickness, cv2.LINE_AA)


def put_legend(img: np.ndarray, y0: int = 46, lh: int = 18) -> None:
    """Draw a small colour legend in the top-left corner of *img*."""
    lx = 8
    for label, color in LEGEND:
        cv2.rectangle(img, (lx, y0 - 12), (lx + 14, y0 + 2), color, -1)
        cv2.putText(img, label, (lx + 18, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y0 += lh


# ── Bird's-eye view ────────────────────────────────────────────────────────────

_LANE_LABELS = ['LLF', 'LLN', 'RLN', 'RLF']   # left-far, left-near, right-near, right-far


def _fit_lane_quadratic(xs: np.ndarray, ys: np.ndarray
                        ) -> tuple[float, float, float] | None:
    """Fit y = a*x^2 + b*x + c to lane (xs, ys) points.

    a: curvature, b: yaw angle, c: lateral position.
    Returns None when fewer than 3 valid points exist.
    """
    valid = xs > 0.5
    xv, yv = xs[valid], ys[valid]
    if len(xv) < 3:
        return None
    coeffs = np.polyfit(xv, yv, 2)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

def _bev_pt(x_fwd: float, y_right: float) -> tuple[int, int]:
    """Device-frame (x_fwd metres forward, y_right metres right) → BEV pixel (px, py).

    Orientation: forward = upward on screen, right = rightward.
    Origin (car) sits BEV_MARGIN_PX above the bottom edge.
    """
    px = int(round(BEV_W / 2.0 + y_right * BEV_SCALE))
    py = int(round(BEV_H - BEV_MARGIN_PX - x_fwd * BEV_SCALE))
    return px, py


def draw_birdseye(outputs: dict) -> np.ndarray:
    """Return a (BEV_H × BEV_W) BGR image with a top-down scene view.

    Layout:
      - Black background
      - Dark-grey grid: lines every 10 m forward, every 5 m lateral; centre-line slightly brighter
      - Distance labels along the left edge (10 m, 20 m, …)
      - Own vehicle: grey rectangle at origin
      - Path: semi-transparent white ±0.5 m band
      - Lane lines: white polylines, alpha = lane_line_prob  (fade when uncertain)
      - Road edges: red polylines, alpha = clamp(1 − std, 0, 1)
      - Lead car: yellow rectangle (only when lead_prob > 0.5)
    """
    img = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    # ── grid ──────────────────────────────────────────────────────────────────
    # Lateral lines every 5 m
    for y_m in np.arange(-BEV_MAX_Y, BEV_MAX_Y + 0.01, 5.0):
        px, _ = _bev_pt(0.0, y_m)
        if 0 <= px < BEV_W:
            color = (90, 90, 90) if abs(y_m) < 0.01 else _GRID_COLOR
            cv2.line(img, (px, 0), (px, BEV_H), color, 1)
    # Forward lines every 10 m
    for x_m in np.arange(10.0, BEV_MAX_X + 0.01, 10.0):
        _, py = _bev_pt(x_m, 0.0)
        if 0 <= py < BEV_H:
            cv2.line(img, (0, py), (BEV_W, py), _GRID_COLOR, 1)
            cv2.putText(img, f'{int(x_m)}m', (3, py - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, _GRID_LABEL, 1, cv2.LINE_AA)

    plan      = outputs.get('plan')
    lead      = outputs.get('lead')
    lead_prob = float(outputs.get('lead_prob', 0.0))

    # ── path: ±0.5 m filled band ──────────────────────────────────────────────
    if plan is not None:
        path  = plan[0, 0]          # (IDX_N, 15)
        xs, ys = path[:, 0], path[:, 1]
        valid = xs > 0.5
        if valid.sum() >= 2:
            xv, yv = xs[valid], ys[valid]
            l_pts = np.array([_bev_pt(float(x), float(y) - 0.5)
                               for x, y in zip(xv, yv)], dtype=np.int32)
            r_pts = np.array([_bev_pt(float(x), float(y) + 0.5)
                               for x, y in zip(xv, yv)], dtype=np.int32)
            poly = np.vstack([l_pts, r_pts[::-1]])
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], _WHITE)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # ── road edges ──────────────────────────────────────────────────────────
    edges     = outputs.get('road_edges')
    edge_stds = outputs.get('road_edge_stds')
    if edges is not None:
        for i in range(min(2, edges.shape[1])):
            xs_e = X_IDXS
            ys_e = edges[0, i, :, 0]
            valid = xs_e > 0.5
            pts = np.array([_bev_pt(float(x), float(y))
                             for x, y in zip(xs_e[valid], ys_e[valid])], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, _RED_BGR, 1, cv2.LINE_AA)

    # ── lane lines ────────────────────────────────────────────────────────────
    lanes      = outputs.get('lane_lines')
    lane_probs = outputs.get('lane_line_probs')
    if lanes is not None:
        for i in range(min(4, lanes.shape[1])):
            xs_l = X_IDXS
            ys_l = lanes[0, i, :, 0]
            valid = xs_l > 0.5
            pts = np.array([_bev_pt(float(x), float(y))
                             for x, y in zip(xs_l[valid], ys_l[valid])], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, _WHITE, 1, cv2.LINE_AA)

    # ── lead car: yellow rectangle ~ 1.8 m × 4 m ────────────────────────────
    if lead is not None and lead_prob > 0.5:
        lpt = lead[0, 0, 0]
        lx, ly = float(lpt[0]), float(lpt[1])
        if lx > 0.5:
            cx, cy = _bev_pt(lx, ly)
            hw = max(1, int(round(0.5 * BEV_SCALE)))   # half-width  ≈ 0.5 m
            hh = max(1, int(round(1.5 * BEV_SCALE)))   # half-length ≈ 1.5 m
            cv2.rectangle(img, (cx - hw, cy - hh), (cx + hw, cy + hh),
                           _YELLOW_BGR, -1)

    # ── own vehicle: grey rectangle ~ 1.8 m × 4.4 m ──────────────────────────
    ox, oy = _bev_pt(0.0, 0.0)
    cw = max(1, int(round(0.9 * BEV_SCALE)))   # half-width  ≈ 0.9 m
    ch = max(1, int(round(2.2 * BEV_SCALE)))   # half-length ≈ 2.2 m
    cv2.rectangle(img, (ox - cw, oy - ch), (ox + cw, oy + ch), _CAR_COLOR, -1)

    # ── quadratic fit labels (top-right) ──────────────────────────────────────
    # y = a*x^2 + b*x + c  (a:曲率, b:ヨー角, c:横位置)
    if lanes is not None:
        tx0 = BEV_W // 2 + 28
        ty  = 14
        fs  = 0.26
        for i in range(min(4, lanes.shape[1])):
            coeffs = _fit_lane_quadratic(X_IDXS, lanes[0, i, :, 0])
            if coeffs is None:
                continue
            a, b, c = coeffs
            lbl = _LANE_LABELS[i]
            cv2.putText(img, f'{lbl} a={a:.2e}', (tx0, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, f'  b={b:+.3f}', (tx0, ty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, f'  c={c:+.3f}', (tx0, ty + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
            ty += 34

    # ── prob / confidence values (bottom-left) ────────────────────────────────
    # Path prob (softmax), lane_line_probs (sigmoid), road_edge conf (1-std)
    fs2 = 0.26
    py_prob = BEV_H - 8   # start from bottom, go upward
    line_h  = 11

    # road edges: conf = clamp(1 - std, 0, 1)
    if edge_stds is not None:
        re_conf = [float(np.clip(1.0 - float(edge_stds[i]), 0.0, 1.0))
                   for i in range(min(2, len(edge_stds)))]
        re_str = '  '.join(f'{"LR"[i]}={re_conf[i]:.2f}' for i in range(len(re_conf)))
        cv2.putText(img, f'RE {re_str}', (3, py_prob),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2, _RED_BGR, 1, cv2.LINE_AA)
        py_prob -= line_h

    # lane line probs
    if lane_probs is not None:
        lp = [float(np.clip(float(lane_probs[i]), 0.0, 1.0))
              for i in range(min(4, len(lane_probs)))]
        labels_lr = ['LF', 'LN', 'RN', 'RF']
        ll_str = '  '.join(f'{labels_lr[i]}={lp[i]:.2f}' for i in range(len(lp)))
        cv2.putText(img, f'LL {ll_str}', (3, py_prob),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2, _WHITE, 1, cv2.LINE_AA)
        py_prob -= line_h

    # path prob
    plan_prob = outputs.get('plan_prob')
    if plan_prob is not None:
        cv2.putText(img, f'Path p={float(plan_prob):.2f}', (3, py_prob),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2, (200, 200, 200), 1, cv2.LINE_AA)

    # ── label ─────────────────────────────────────────────────────────────────
    cv2.putText(img, 'BEV', (3, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
    return img
