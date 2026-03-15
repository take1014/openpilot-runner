"""Camera intrinsics and coordinate-frame constants for openpilot_runner.

All model constants (input geometry, output indices, MHP layout, etc.) live
in runner/constants.py.  This file holds only the optics/geometry data
needed by preprocess.py and visualize.py.

Ported from common/transformations/model.py (camera intrinsics).
"""
import numpy as np

from .runner.constants import MODEL_W, MODEL_H  # needed for principal-point column

# ── Camera intrinsics (common/transformations/model.py) ──────────────────────
# medmodel  (standard road camera → 512×256)
MEDMODEL_HEIGHT = 1.22      # assumed camera height above road (metres)
MEDMODEL_FL = 910.0
MEDMODEL_CY = 47.6
MEDMODEL_K  = np.array([
    [MEDMODEL_FL, 0.,          MODEL_W / 2],
    [0.,          MEDMODEL_FL, MEDMODEL_CY],
    [0.,          0.,          1.         ]], dtype=np.float64)

# sbigmodel  (wide road camera → also 512×256)
SBIGMODEL_FL = 455.0
SBIGMODEL_K  = np.array([
    [SBIGMODEL_FL, 0.,           MODEL_W / 2              ],
    [0.,           SBIGMODEL_FL, 0.5 * (256 + MEDMODEL_CY)],
    [0.,           0.,           1.                        ]], dtype=np.float64)

# device frame: x→forward, y→right,  z→down
# view frame:   x→right,   y→down,   z→forward
VIEW_FROM_DEVICE = np.array([
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 0., 0.]], dtype=np.float64)
