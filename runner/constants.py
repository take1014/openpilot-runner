"""SuperCombo v0.8.10 — all model constants.

Single source of truth for every value that describes the model's interface:
  - input geometry & buffer layout  (commonmodel.h)
  - inference timing                (driving.cc)
  - output indices & MHP layout     (driving.h)
  - trajectory sample points        (modeldata.h)
  - model file location
"""
import os as _os
from pathlib import Path

import numpy as np

# ── Model input geometry (commonmodel.h) ─────────────────────────────────────
MODEL_W, MODEL_H = 512, 256
UV_SIZE          = (MODEL_W // 2) * (MODEL_H // 2)      # 32 768 bytes per sub-plane
MODEL_FRAME_SIZE = MODEL_W * MODEL_H * 3 // 2          # 196 608 bytes per packed frame
MODEL_BUF_SIZE   = MODEL_FRAME_SIZE * 2                # 393 216 bytes (2 temporal frames)

MODEL_FREQ    = 20   # Hz
TEMPORAL_SKIP = 1    # frames between temporal inputs: openpilot uses adjacent frames [t-1 | t]

# ── Trajectory points (modeldata.h) ──────────────────────────────────────────
IDX_N  = 33  # TRAJECTORY_SIZE
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(IDX_N)])

# ── Model I/O scalar sizes (driving.h) ───────────────────────────────────────
DESIRE_LEN             = 8
TRAFFIC_CONVENTION_LEN = 2
TEMPORAL_SIZE          = 512   # recurrent hidden state

# ── Trajectory points / plan layout ─────────────────────────────────────────
PLAN_WIDTH = 15   # floats per trajectory sample (pos x,y,z + vel + rot + …)

# Plan MHP (Multiple Hypotheses Prediction)
PLAN_MHP_N          = 5
PLAN_MHP_VALS       = PLAN_WIDTH * IDX_N        # 495
PLAN_MHP_SELECTION  = 1
PLAN_MHP_GROUP_SIZE = 2 * PLAN_MHP_VALS + PLAN_MHP_SELECTION  # 991

# Lead MHP
LEAD_TRAJ_LEN       = 6
LEAD_PRED_DIM       = 4   # x, y, velocity, acceleration
LEAD_MHP_N          = 2
LEAD_MHP_VALS       = LEAD_PRED_DIM * LEAD_TRAJ_LEN  # 24
LEAD_MHP_SELECTION  = 3
LEAD_MHP_GROUP_SIZE = 2 * LEAD_MHP_VALS + LEAD_MHP_SELECTION  # 51

# Meta / pose sizes
OTHER_META_SIZE  = 48
DESIRE_PRED_SIZE = 32
POSE_SIZE        = 12

# ── Flat output offsets (driving.h — verified against static_asserts) ────────
PLAN_IDX         = 0
LL_IDX           = PLAN_IDX      + PLAN_MHP_N * PLAN_MHP_GROUP_SIZE  # 4955
LL_PROB_IDX      = LL_IDX        + 4 * 2 * 2 * IDX_N                # 5483
RE_IDX           = LL_PROB_IDX   + 8                                  # 5491
LEAD_IDX         = RE_IDX        + 2 * 2 * 2 * IDX_N                # 5755
LEAD_PROB_IDX    = LEAD_IDX      + LEAD_MHP_N * LEAD_MHP_GROUP_SIZE  # 5857
DESIRE_STATE_IDX = LEAD_PROB_IDX + LEAD_MHP_SELECTION                # 5860
META_IDX         = DESIRE_STATE_IDX + DESIRE_LEN                     # 5868
POSE_IDX         = META_IDX      + OTHER_META_SIZE + DESIRE_PRED_SIZE # 5948
OUTPUT_SIZE      = POSE_IDX      + POSE_SIZE                          # 5960
NET_OUTPUT_SIZE  = OUTPUT_SIZE   + TEMPORAL_SIZE                      # 6472

# ── Model file path ───────────────────────────────────────────────────────────
# Default: openpilot_runner/models/supercombo.onnx
# Override with SUPERCOMBO_MODEL_DIR env var.
_env_dir  = _os.environ.get('SUPERCOMBO_MODEL_DIR')
_MODEL_DIR = Path(_env_dir) if _env_dir else (
    Path(__file__).resolve().parent.parent / 'models')  # runner/ → openpilot_runner/models/

SUPERCOMBO_ONNX_PATH = _MODEL_DIR / 'supercombo.onnx'
