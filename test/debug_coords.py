"""Debug: print model output 3D coordinate ranges to understand coordinate convention."""
import sys
import numpy as np
import cv2

from ..runner.constants import MODEL_FRAME_SIZE
from ..preprocess import build_warp_matrix, warp_to_model_space, bgr_to_yuv_planes, pack_loadyuv
from ..runner import ModelRunner

img_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/take/fun/openpilot-runner/c920/0885.png'
img = cv2.imread(img_path)
H, W = img.shape[:2]

M = build_warp_matrix(W, H, 908.0)
warped = warp_to_model_space(img, M)
Y, U, V = bgr_to_yuv_planes(warped)
cur = pack_loadyuv(Y, U, V)
frame_buf = np.concatenate([np.zeros(MODEL_FRAME_SIZE, dtype=np.uint8), cur])

model = ModelRunner(rhd=False)
out = model.run(frame_buf)

print('=== lane_lines (y_right, z) ===')
ll = out['lane_lines'][0]  # (4, 33, 2)
for i in range(4):
    y = ll[i, :, 0]
    z = ll[i, :, 1]
    print(f'  lane[{i}]: y=[{y[0]:.2f}..{y[-1]:.2f}]  z=[{z[0]:.3f}..{z[-1]:.3f}]  z_mean={z.mean():.3f}')

print('=== plan position (x, y, z) ===')
p = out['plan'][0, 0]  # (33, 15)
print(f'  x = [{p[0,0]:.2f} .. {p[-1,0]:.2f}]')
print(f'  y = [{p[0,1]:.3f} .. {p[-1,1]:.3f}]')
print(f'  z = [{p[0,2]:.3f} .. {p[-1,2]:.3f}]  mean={p[:,2].mean():.3f}')

print('=== lead ===')
lead = out['lead'][0, 0, 0]
print(f'  x={lead[0]:.2f}  y={lead[1]:.2f}  v={lead[2]:.2f}  a={lead[3]:.2f}')
