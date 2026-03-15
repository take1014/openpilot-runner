#!/usr/bin/env python3
"""Run SuperCombo inference on a folder of images and save annotated results.

Usage:
    python -m openpilot_runner.test.test_images --input ~/fun/openpilot-runner/c920
    python -m openpilot_runner.test.test_images --input ~/fun/openpilot-runner/c920 \\
        --output /tmp/c920_out --focal-length 820 --rhd

Options:
    --input  DIR      folder of input images (sorted by filename)  [required]
    --output DIR      folder to write annotated images (default: <input>_out)
    --focal-length F  webcam focal length in pixels (default 908.0)
    --display-scale S upscale factor for the 512×256 model view (default 2.0)
    --rhd             right-hand drive traffic (Japan/UK)
    --no-flip         disable 180° vertical flip (default: flip OFF)
    --ext EXT         input image extension glob, e.g. png or jpg (default png)
    --limit N         process at most N frames (default: all)
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..runner.constants import (
    MODEL_W, MODEL_H, MODEL_FRAME_SIZE, TEMPORAL_SKIP,
    SUPERCOMBO_ONNX_PATH,
)
from ..preprocess import (
    build_warp_matrix, warp_to_model_space, bgr_to_yuv_planes, pack_loadyuv,
)
from ..visualize import draw_overlays, put_legend
from ..runner import ModelRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Run SuperCombo on a folder of images and save annotated results')
    p.add_argument('--input', required=True,
                   help='folder containing input images')
    p.add_argument('--output', default=None,
                   help='folder to write annotated images (default: <input>_out)')
    p.add_argument('--focal-length', type=float, default=908.0,
                   help='webcam focal length in pixels (default 908.0)')
    p.add_argument('--display-scale', type=float, default=2.0,
                   help='upscale factor for 512×256 model view (default 2.0)')
    p.add_argument('--rhd', action='store_true',
                   help='right-hand drive traffic convention')
    p.add_argument('--flip', dest='flip', action='store_true',
                   help='enable 180° vertical flip (for upside-down mounted cameras)')
    p.add_argument('--ext', default='png',
                   help='input image extension (default: png)')
    p.add_argument('--limit', type=int, default=None,
                   help='process at most N frames')
    p.set_defaults(flip=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        print(f'[ERROR] Input directory not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).expanduser().resolve() if args.output else (
        input_dir.parent / (input_dir.name + '_out'))
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob(f'*.{args.ext}'))
    if not images:
        print(f'[ERROR] No .{args.ext} files found in {input_dir}', file=sys.stderr)
        sys.exit(1)
    if args.limit:
        images = images[:args.limit]

    print(f'[INFO] Input : {input_dir}  ({len(images)} images)')
    print(f'[INFO] Output: {output_dir}')

    # ── load model ────────────────────────────────────────────────
    model: ModelRunner | None = None
    if SUPERCOMBO_ONNX_PATH.exists():
        print('[INFO] Loading supercombo.onnx …')
        try:
            model = ModelRunner(rhd=args.rhd)
            print('[INFO] Model loaded.')
        except Exception as exc:
            print(f'[WARN] Model load failed: {exc}')
    else:
        print(f'[WARN] supercombo.onnx not found at {SUPERCOMBO_ONNX_PATH}')
        print('[WARN] Running in preview-only mode (no inference).')

    # ── read first image to get camera dimensions ─────────────────
    first = cv2.imread(str(images[0]))
    if first is None:
        print(f'[ERROR] Cannot read {images[0]}', file=sys.stderr)
        sys.exit(1)
    CAM_H, CAM_W = first.shape[:2]
    print(f'[INFO] Image size: {CAM_W}×{CAM_H},  focal_length={args.focal_length}')

    # ── warp matrix ───────────────────────────────────────────────
    M = build_warp_matrix(CAM_W, CAM_H, args.focal_length, big=False, flip=args.flip)

    # ── camera intrinsic matrix (for direct overlay projection) ──
    webcam_K = np.array([
        [args.focal_length, 0.,                CAM_W / 2],
        [0.,               args.focal_length,  CAM_H / 2],
        [0.,               0.,                1.        ]], dtype=np.float64)

    # ── display layout ────────────────────────────────────────────
    DISP_W = int(MODEL_W * args.display_scale)
    DISP_H = int(MODEL_H * args.display_scale)

    # ── temporal ring buffer ──────────────────────────────────────
    ring: deque[np.ndarray] = deque(maxlen=TEMPORAL_SKIP + 1)
    zero_frame = np.zeros(MODEL_FRAME_SIZE, dtype=np.uint8)

    # ── process loop ──────────────────────────────────────────────
    for i, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'[WARN] Cannot read {img_path.name}, skipping.')
            continue

        # preprocess
        warped = warp_to_model_space(bgr, M)
        Y, U, V = bgr_to_yuv_planes(warped)
        cur = pack_loadyuv(Y, U, V)
        ring.append(cur)
        oldest = ring[0] if len(ring) == TEMPORAL_SKIP + 1 else zero_frame

        # inference
        outputs: dict = {}
        if model is not None:
            frame_buf = np.concatenate([oldest, cur])
            try:
                outputs = model.run(frame_buf)
            except Exception as exc:
                print(f'[WARN] frame {img_path.name}: inference error: {exc}')

        # ── right panel: model view (512×256) with overlay ────────
        right_panel = cv2.resize(warped, (DISP_W, DISP_H),
                                 interpolation=cv2.INTER_LINEAR)
        if outputs:
            draw_overlays(right_panel, outputs, scale=args.display_scale)
        mode_text = 'SuperCombo v0.8.10' if model else 'Preview (no model)'
        color = (0, 220, 100) if model else (80, 80, 255)
        cv2.putText(right_panel, mode_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(right_panel, img_path.name, (8, DISP_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        put_legend(right_panel)

        # ── left panel: overlay projected directly in camera space ─
        display_bgr = cv2.flip(bgr, -1) if args.flip else bgr
        if outputs:
            cam_tk = max(2, round(CAM_W / MODEL_W))
            cam_with_overlay = display_bgr.copy()
            draw_overlays(cam_with_overlay, outputs, scale=1.0,
                          K=webcam_K, min_thickness=cam_tk)
            left_panel = cv2.resize(cam_with_overlay, (DISP_W, DISP_H),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            left_panel = cv2.resize(display_bgr, (DISP_W, DISP_H),
                                    interpolation=cv2.INTER_LINEAR)

        cv2.putText(left_panel,
                    f'C920  {CAM_W}x{CAM_H}  fl={args.focal_length:.0f}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

        canvas = np.hstack([left_panel, right_panel])
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), canvas)

        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f'[INFO] {i + 1}/{len(images)}  → {out_path.name}')

    print(f'[INFO] Done. Results saved to {output_dir}')


if __name__ == '__main__':
    main()
