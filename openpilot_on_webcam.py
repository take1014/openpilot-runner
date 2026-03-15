#!/usr/bin/env python3
"""
Real-time webcam → SuperCombo AI model → lane/road-edge/path overlay display.

This is the main entry point.  All heavy logic lives in dedicated modules:
  constants.py   — shared numeric constants and model file paths
  preprocess.py  — webcam frame → loadyuv-packed model input buffer
  visualize.py   — road-to-image projection and OpenCV overlay drawing
  runner.py      — ModelRunner: onnxruntime supercombo.onnx execution

Usage:
  python -m openpilot_runner.openpilot_on_webcam --camera 0
  python -m openpilot_runner.openpilot_on_webcam --rhd          # Japan/UK (right-hand drive)

Options:
  --camera INT          webcam device index (default 0)
  --focal-length FLOAT  webcam focal length in pixels (default 908;
                        try 820 for a 78° HFOV webcam at 1280×720)
  --display-scale FLOAT upscale factor for the 512×256 model display (default 2)
  --fps-cap INT         maximum loop rate in Hz (default 20)
  --rhd                 right-hand drive traffic convention (Japan/UK)
  --pitch FLOAT         nose-down pitch correction in degrees (default 0;
                        use ~5.1 for a horizontally-mounted webcam)

Model file (openpilot v0.8.12, placed under models/):
  supercombo.onnx
If absent the script runs in preview-only mode.

Press 'q' or ESC to quit.
"""
import argparse
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from .runner.constants import (
    MODEL_W, MODEL_H, MODEL_FRAME_SIZE, TEMPORAL_SKIP,
    SUPERCOMBO_ONNX_PATH,
)
from .preprocess import (
    build_warp_matrix, warp_to_model_space, bgr_to_yuv_planes, pack_loadyuv,
)
from .visualize import draw_overlays, put_legend, draw_birdseye, BEV_W, BEV_H
from .runner import ModelRunner
from .camera import CameraThread, AsyncVideoWriter


# ─────────────────────────────────────────────────────────────────────────────
# MAIN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Real-time webcam → SuperCombo AI model → overlay display')
    p.add_argument('--camera', type=int, default=0,
                   help='webcam device index (default 0)')
    p.add_argument('--width', type=int, default=1280,
                   help='capture width in pixels (default 1280)')
    p.add_argument('--height', type=int, default=720,
                   help='capture height in pixels (default 720)')
    p.add_argument('--cam-fps', type=int, default=20,
                   help='requested capture frame rate (default 20)')
    p.add_argument('--focal-length', type=float, default=908.0,
                   help='webcam focal length in pixels (default 908; '
                        'try 820 for 78° HFOV webcam at 1280×720)')
    p.add_argument('--display-scale', type=float, default=2.0,
                   help='upscale factor for the 512×256 model display (default 2)')
    p.add_argument('--fps-cap', type=int, default=20,
                   help='maximum frames per second (default 20)')
    p.add_argument('--pitch', type=float, default=0.0,
                   help='nose-down pitch correction in degrees (default 0). '
                        'Use ~5.1 for a horizontally-mounted webcam to match '
                        'the ~5.1° downward pitch assumed by the SuperCombo model.')
    p.add_argument('--rhd', action='store_true',
                   help='right-hand drive traffic (Japan/UK: cars drive on left)')
    p.add_argument('--no-flip', dest='flip', action='store_false',
                   help='disable vertical flip (default: flip is ON for upside-down mounting)')
    p.add_argument('--save-video', metavar='FILE', nargs='?', const='',
                   help='save display canvas to an MP4 file; '
                        'omit FILE to auto-generate a timestamped filename')
    p.set_defaults(flip=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── try to load model ────────────────────────────────────────
    model: ModelRunner | None = None
    if SUPERCOMBO_ONNX_PATH.exists():
        print('[INFO] Loading supercombo.onnx …')
        try:
            model = ModelRunner(rhd=args.rhd)
            print('[INFO] Model loaded. Running inference.')
        except Exception as exc:
            print(f'[WARN] Model load failed: {exc}')
    else:
        print('[WARN] supercombo.onnx not found → preview-only mode.')
        print(f'      Expected: {SUPERCOMBO_ONNX_PATH}')

    # ── open webcam ──────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    # Set resolution and FPS before reading (matching openpilot front_mount_helper)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.cam_fps)
    # Keep internal buffer at 1 frame to minimise capture latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {args.camera}', file=sys.stderr)
        sys.exit(2)
    ret, first = cap.read()
    if not ret:
        print('[ERROR] Cannot read from camera', file=sys.stderr)
        sys.exit(3)
    CAM_H, CAM_W = first.shape[:2]
    print(f'[INFO] Webcam: {CAM_W}×{CAM_H},  focal_length={args.focal_length}')

    # ── perspective warp matrix (medmodel only — v0.8.12 has one camera input)
    # flip=args.flip bakes the 180° rotation into M, eliminating a separate cv2.flip() call
    M_main = build_warp_matrix(CAM_W, CAM_H, args.focal_length,
                               big=False, flip=args.flip, pitch_deg=args.pitch)
    if args.pitch != 0.0:
        print(f'[INFO] Pitch correction: {args.pitch:.1f}° nose-down')
    # M_main maps model pixel → camera pixel (used both for warpPerspective and back-projection)

    # ── camera intrinsic matrix (for direct overlay projection onto camera image) ──
    webcam_K = np.array([
        [args.focal_length, 0.,                CAM_W / 2],
        [0.,               args.focal_length,  CAM_H / 2],
        [0.,               0.,                1.        ]], dtype=np.float64)

    # ── start background capture thread ─────────────────────────
    cam = CameraThread(cap)
    print('[INFO] Camera thread started.')

    # ── display window ───────────────────────────────────────────
    DISP_W = int(MODEL_W * args.display_scale)
    DISP_H = int(MODEL_H * args.display_scale)
    CANVAS_H = DISP_H * 2                         # top + bottom panel stacked
    bev_disp_w = int(BEV_W * CANVAS_H / BEV_H)   # BEV scaled to full canvas height
    cv2.namedWindow('SuperCombo Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SuperCombo Webcam', DISP_W + bev_disp_w, CANVAS_H)

    # ── video writer (optional) ───────────────────────────────────
    writer: AsyncVideoWriter | None = None
    if args.save_video is not None:
        if args.save_video == '':
            args.save_video = datetime.now().strftime('openpilot_%Y%m%d_%H%M%S.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        _vw = cv2.VideoWriter(
            args.save_video, fourcc, float(args.fps_cap), (DISP_W + bev_disp_w, CANVAS_H))
        if not _vw.isOpened():
            print(f'[WARN] Cannot open video writer for {args.save_video}', file=sys.stderr)
        else:
            writer = AsyncVideoWriter(_vw)
            print(f'[INFO] Recording to {args.save_video} (async)')

    # ── temporal frame ring buffer ───────────────────────────────
    # v0.8.10: single camera; 2-frame temporal input [oldest | current]
    ring_main: deque[np.ndarray] = deque(maxlen=TEMPORAL_SKIP + 1)
    zero_frame = np.zeros(MODEL_FRAME_SIZE, dtype=np.uint8)

    outputs: dict = {}
    frame_count = 0
    t_start = time.monotonic()
    interval = 1.0 / args.fps_cap

    print("[INFO] Press 'q' or ESC to quit.")

    while True:
        t_loop = time.monotonic()
        ret, bgr = cam.read()  # non-blocking: returns latest captured frame
        if not ret:
            break
        # flip is now baked into M_main — no separate cv2.flip() needed

        # ── preprocess: warp → YUV → loadyuv pack ────────────────
        warped_main = warp_to_model_space(bgr, M_main)
        Y, U, V  = bgr_to_yuv_planes(warped_main)
        cur_main = pack_loadyuv(Y, U, V)

        ring_main.append(cur_main)

        # Temporal input: [oldest_frame | current_frame]
        oldest_main = ring_main[0] if len(ring_main) == TEMPORAL_SKIP + 1 else zero_frame

        # ── model inference ───────────────────────────────────────
        if model is not None:
            frame_buf = np.concatenate([oldest_main, cur_main])
            try:
                outputs = model.run(frame_buf)
            except Exception as exc:
                print(f'[WARN] Inference error: {exc}')
                outputs = {}

        # ── build display ─────────────────────────────────────────
        # Bottom panel: model input (warped) with overlay in model space
        bottom_panel = cv2.resize(warped_main, (DISP_W, DISP_H),
                                  interpolation=cv2.INTER_LINEAR)
        if outputs:
            draw_overlays(bottom_panel, outputs, scale=args.display_scale)

        # Top panel: project overlay directly onto camera image in camera space
        # For display, show the image as the user sees it (flip back if needed)
        display_bgr = cv2.flip(bgr, -1) if args.flip else bgr
        if outputs:
            cam_tk = max(2, round(CAM_W / MODEL_W))
            cam_with_overlay = display_bgr.copy()
            draw_overlays(cam_with_overlay, outputs, scale=1.0,
                          K=webcam_K, min_thickness=cam_tk)
            top_panel = cv2.resize(cam_with_overlay, (DISP_W, DISP_H),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            top_panel = cv2.resize(display_bgr, (DISP_W, DISP_H),
                                   interpolation=cv2.INTER_LINEAR)

        # HUD text
        elapsed = time.monotonic() - t_start
        fps = (frame_count + 1) / elapsed if elapsed > 0 else 0.0
        frame_count += 1

        cv2.putText(top_panel,
                    f'Webcam  {CAM_W}x{CAM_H}  |  fl={args.focal_length:.0f}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 100), 1, cv2.LINE_AA)

        mode = 'SuperCombo v0.8.10 inference' if model else 'Preview only (no supercombo.onnx)'
        color = (0, 220, 100) if model else (80, 80, 255)
        cv2.putText(bottom_panel, mode, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(bottom_panel, f'{fps:.1f} fps',
                    (8, DISP_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (49, 34, 201), 1, cv2.LINE_AA)
        put_legend(bottom_panel)

        bev_panel = cv2.resize(draw_birdseye(outputs),
                                (bev_disp_w, CANVAS_H), interpolation=cv2.INTER_NEAREST)
        canvas = np.hstack([np.vstack([top_panel, bottom_panel]), bev_panel])
        cv2.imshow('SuperCombo Webcam', canvas)
        if writer is not None:
            writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # FPS cap
        sleep_t = interval - (time.monotonic() - t_loop)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if writer is not None:
        writer.release()
        print(f'[INFO] Video saved to {args.save_video}')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
