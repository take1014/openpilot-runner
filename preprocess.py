"""Webcam image → model input buffer preprocessing.

Mirrors the OpenCL kernels in selfdrive/modeld/transforms/:
  - warpPerspective  (transform.cl)
  - loadys / loaduv  (loadyuv.cl)
"""
import cv2
import numpy as np

from .runner.constants import MODEL_W, MODEL_H
from .constants import MEDMODEL_K, SBIGMODEL_K


def build_warp_matrix(cam_w: int, cam_h: int, focal_length: float,
                      big: bool = False, flip: bool = False,
                      pitch_deg: float = 0.0) -> np.ndarray:
    """Return 3×3 homography M for cv2.warpPerspective(..., WARP_INVERSE_MAP).

    For each destination (model) pixel (dx, dy) the source (webcam) pixel is
    M @ [dx, dy, 1]^T — identical to the warpPerspective OpenCL kernel in
    selfdrive/modeld/transforms/transform.cl.

    With a level, forward-facing camera and zero calibration:
        M = webcam_K @ inv(model_K)

    If flip=True, a 180° rotation is baked into M so that cv2.flip() is not
    needed separately — saves ~0.7 ms per frame on a 1280×720 input.

    If pitch_deg != 0, a pitch-rotation homography is pre-multiplied so the
    warped image looks as if the camera were tilted nose-down by that angle.
    The SuperCombo model was trained with ~5.1° nose-down pitch (derived from
    MEDMODEL_CY=47.6, FL=910, image height 256 px).  Passing pitch_deg=5.1
    compensates for a horizontally-mounted webcam.
    """
    webcam_K = np.array([
        [focal_length, 0.,           cam_w / 2],
        [0.,           focal_length, cam_h / 2],
        [0.,           0.,           1.        ]], dtype=np.float64)
    model_K = SBIGMODEL_K if big else MEDMODEL_K
    M = webcam_K @ np.linalg.inv(model_K)
    if pitch_deg != 0.0:
        # Apply nose-down pitch rotation in webcam image space:
        #   H = webcam_K @ R_x(θ) @ inv(webcam_K)
        # where R_x(θ) rotates the camera around its x-axis (pitch down = +θ).
        theta = np.deg2rad(pitch_deg)
        c, s = np.cos(theta), np.sin(theta)
        R_x = np.array([[1., 0.,  0.],
                        [0., c,  -s],
                        [0., s,   c]], dtype=np.float64)
        H_pitch = webcam_K @ R_x @ np.linalg.inv(webcam_K)
        M = H_pitch @ M
    if flip:
        # 180° rotation of the source: (x,y) → (cam_w-1-x, cam_h-1-y)
        flip_mat = np.array([
            [-1.,  0., cam_w - 1],
            [ 0., -1., cam_h - 1],
            [ 0.,  0., 1.       ]], dtype=np.float64)
        M = flip_mat @ M
    return M


def warp_to_model_space(bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp a BGR frame into 512×256 model space (BGR output)."""
    return cv2.warpPerspective(
        bgr, M, (MODEL_W, MODEL_H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)


def bgr_to_yuv_planes(bgr_512x256: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 512×256 BGR to full-res Y (256,512) and half-res U/V (128,256)."""
    yuv_full = cv2.cvtColor(bgr_512x256, cv2.COLOR_BGR2YUV)  # (256,512,3) uint8
    Y = yuv_full[:, :, 0]                                     # (256,512)
    U = cv2.resize(yuv_full[:, :, 1], (MODEL_W // 2, MODEL_H // 2),
                   interpolation=cv2.INTER_AREA)              # (128,256)
    V = cv2.resize(yuv_full[:, :, 2], (MODEL_W // 2, MODEL_H // 2),
                   interpolation=cv2.INTER_AREA)              # (128,256)
    return Y, U, V


def pack_loadyuv(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Pack Y/U/V planes into the loadyuv kernel output format (196 608 bytes).

    Mirror of selfdrive/modeld/transforms/loadyuv.cl.
    Output layout: [y0 | y1 | y2 | y3 | U | V]
      y0 = even rows, even cols  (32 768 bytes each)
      y1 = odd  rows, even cols
      y2 = even rows, odd  cols
      y3 = odd  rows, odd  cols
    """
    y0 = Y[0::2, 0::2]
    y1 = Y[1::2, 0::2]
    y2 = Y[0::2, 1::2]
    y3 = Y[1::2, 1::2]
    return np.concatenate(
        [y0.ravel(), y1.ravel(), y2.ravel(), y3.ravel(),
         U.ravel(), V.ravel()]).astype(np.uint8)
