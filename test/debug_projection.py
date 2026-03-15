"""Visualize projected points numerically to diagnose misalignment."""
import sys
import numpy as np
import cv2

from ..runner.constants import MODEL_FRAME_SIZE, X_IDXS
from ..constants import MEDMODEL_K, MEDMODEL_HEIGHT
from ..preprocess import build_warp_matrix, warp_to_model_space, bgr_to_yuv_planes, pack_loadyuv
from ..runner import ModelRunner

FL = float(sys.argv[2]) if len(sys.argv) > 2 else 908.0
img_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/take/fun/openpilot-runner/c920/0885.png'

img = cv2.imread(img_path)
H, W = img.shape[:2]
print(f'Image: {W}x{H}  fl={FL}')

webcam_K = np.array([[FL, 0, W/2], [0, FL, H/2], [0, 0, 1]], dtype=np.float64)
M = build_warp_matrix(W, H, FL)

warped = warp_to_model_space(img, M)
Y, U, V = bgr_to_yuv_planes(warped)
cur = pack_loadyuv(Y, U, V)
frame_buf = np.concatenate([np.zeros(MODEL_FRAME_SIZE, dtype=np.uint8), cur])

model = ModelRunner(rhd=False)
out = model.run(frame_buf)

ll = out['lane_lines'][0]  # (4, 33, 2)
print('\n--- Lane line projected positions in MODEL image (512x256) ---')
for i, name in enumerate(['left_far', 'left_near', 'right_near', 'right_far']):
    pts = []
    for j in [5, 8, 10, 12, 15]:  # select a few x-distances
        xf = X_IDXS[j]
        yr = ll[i, j, 0]
        zd = ll[i, j, 1]
        u = MEDMODEL_K[0,0] * yr/xf + MEDMODEL_K[0,2]
        v = MEDMODEL_K[1,1] * zd/xf + MEDMODEL_K[1,2]
        pts.append(f'  x={xf:.1f}m: model=({u:.0f},{v:.0f})')
    print(f'{name}:')
    for p in pts: print(p)

print('\n--- Lane line projected positions in CAMERA image (1280x720) ---')
for i, name in enumerate(['left_far', 'left_near', 'right_near', 'right_far']):
    pts = []
    for j in [5, 8, 10, 12, 15]:
        xf = X_IDXS[j]
        yr = ll[i, j, 0]
        zd = ll[i, j, 1]
        u = webcam_K[0,0] * yr/xf + webcam_K[0,2]
        v = webcam_K[1,1] * zd/xf + webcam_K[1,2]
        pts.append(f'  x={xf:.1f}m: cam=({u:.0f},{v:.0f})')
    print(f'{name}:')
    for p in pts: print(p)

print('\n--- Expected flat-road z=1.22 projection in camera ---')
for xf in [4.69, 9.19, 18.75, 30, 50]:
    v_flat = webcam_K[1,1] * 1.22/xf + webcam_K[1,2]
    print(f'  x={xf:.1f}m: expected v={v_flat:.0f}  (for flat road at camera_height=1.22m)')
