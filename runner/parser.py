"""SuperCombo v0.8.10 output parser.

Decodes the flat NET_OUTPUT_SIZE float array from onnxruntime into
visualisation-ready numpy arrays.

Mirrors the C++ struct-cast logic in selfdrive/modeld/models/driving.cc
(fill_model / fill_lane_lines / fill_plan / fill_lead).
"""
import numpy as np

from .constants import (
    PLAN_IDX, LL_IDX, LL_PROB_IDX, RE_IDX, LEAD_IDX, LEAD_PROB_IDX,
    PLAN_MHP_N, PLAN_MHP_VALS, PLAN_MHP_GROUP_SIZE,
    LEAD_MHP_N, LEAD_MHP_VALS, LEAD_MHP_GROUP_SIZE,
    LEAD_TRAJ_LEN, LEAD_PRED_DIM,
    IDX_N, PLAN_WIDTH,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def parse_outputs(raw: np.ndarray) -> dict:
    """Decode the flat NET_OUTPUT_SIZE float array into visualisation-ready arrays.

    Output shapes are chosen to be compatible with visualize.draw_overlays:
      lane_lines  (1, 4, IDX_N, 2)  last dim = [y_left, z_up]
      road_edges  (1, 2, IDX_N, 2)  last dim = [y_left, z_up]
      plan        (1, 1, IDX_N, 15) last dim = [x_fwd, y_left, z_up, …]
      lead        (1, 1, LEAD_TRAJ_LEN, 4) = [x, y, velocity, accel]
    """
    out: dict = {}
    N = IDX_N  # 33

    # ── lane_lines ─────────────────────────────────────────────────────────
    # Memory layout at LL_IDX (driving.h ModelDataRawLaneLines):
    #   mean: [left_far | left_near | right_near | right_far] × 33 × {y,z}
    #   std:  same layout
    # Reshape as (mean_std=2, 4_lines, 33_pts, 2_yz), take mean (index 0).
    if len(raw) > LL_PROB_IDX:
        ll = raw[LL_IDX:LL_PROB_IDX].reshape(2, 4, N, 2)
        out['lane_lines'] = ll[np.newaxis, 0, :, :, :]   # (1, 4, N, 2)
        # lane_line_probs: ModelDataRawLinesProb = 4 × {val_deprecated, val}
        # The active probability is 'val' (odd indices); apply sigmoid.
        # Order: left_far[1], left_near[3], right_near[5], right_far[7]
        ll_prob_raw = raw[LL_PROB_IDX:LL_PROB_IDX + 8]
        out['lane_line_probs'] = sigmoid(ll_prob_raw[1::2])  # (4,)

    # ── road_edges ──────────────────────────────────────────────────────────
    # Memory layout (ModelDataRawRoadEdges):
    #   mean: [left | right] × 33 × {y,z}  = 132 floats
    #   std:  same layout                  = 132 floats
    # Reshape as (mean_std=2, 2_edges, 33_pts, 2_yz), take mean (index 0).
    re_end = RE_IDX + 2 * 2 * N * 2  # = LEAD_IDX
    if len(raw) > re_end:
        re = raw[RE_IDX:re_end].reshape(2, 2, N, 2)
        out['road_edges'] = re[np.newaxis, 0, :, :, :]   # (1, 2, N, 2)
        # road_edge_stds: exp(y-std at first point) for each edge — matches
        # framed.setRoadEdgeStds in fill_road_edges (driving.cc)
        out['road_edge_stds'] = np.exp(re[1, :, 0, 0])   # (2,)

    # ── plan (best MHP) ─────────────────────────────────────────────────────
    # 5 plan hypotheses × PLAN_MHP_GROUP_SIZE=991 floats each.
    # Group layout: [mean(495) | std(495) | prob(1)].
    if len(raw) > LL_IDX:
        plans = raw[PLAN_IDX:LL_IDX].reshape(PLAN_MHP_N, PLAN_MHP_GROUP_SIZE)
        probs = plans[:, 2 * PLAN_MHP_VALS]   # one prob per hypothesis
        best  = int(np.argmax(probs))
        plan_mean = plans[best, :PLAN_MHP_VALS].reshape(N, PLAN_WIDTH)
        out['plan'] = plan_mean[np.newaxis, np.newaxis, :, :]  # (1,1,N,15)

    # ── lead (best t=0 hypothesis) ──────────────────────────────────────────
    # 2 lead hypotheses × LEAD_MHP_GROUP_SIZE=51 floats each.
    # Group layout: [mean(24) | std(24) | prob(3)]; mean = 6 pts × 4 (x,y,v,a).
    if len(raw) > LEAD_PROB_IDX:
        lead_preds = raw[LEAD_IDX:LEAD_PROB_IDX].reshape(
            LEAD_MHP_N, LEAD_MHP_GROUP_SIZE)
        t0_probs = lead_preds[:, 2 * LEAD_MHP_VALS]   # first of 3 time horizons
        best     = int(np.argmax(t0_probs))
        lead_mean = lead_preds[best, :LEAD_MHP_VALS].reshape(
            LEAD_TRAJ_LEN, LEAD_PRED_DIM)
        out['lead'] = lead_mean[np.newaxis, np.newaxis, :, :]  # (1,1,6,4)
        # lead_prob: ModelDataRawLeads.prob[0] — overall lead probability at t=0
        # sigmoid(raw[LEAD_PROB_IDX]) matches fill_lead: lead.setProb(sigmoid(leads.prob[t_idx]))
        out['lead_prob'] = float(sigmoid(raw[LEAD_PROB_IDX:LEAD_PROB_IDX + 1])[0])

    return out
