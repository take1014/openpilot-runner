"""SuperCombo v0.8.10 model runner using onnxruntime.

Single unified supercombo.onnx model — replaces the tinygrad vision+policy pair.
Inputs per forward pass:
  1. image buffer  (float32, MODEL_BUF_SIZE values = 2 × loadyuv-packed frames)
  2. desire        (float32, 8)
  3. traffic_conv  (float32, 2)
  4. recurrent     (float32, 512)  ← fed back from previous output
Output: flat float32 array of NET_OUTPUT_SIZE=6472 values.
"""
import numpy as np

from .constants import (
    MODEL_BUF_SIZE, SUPERCOMBO_ONNX_PATH,
    DESIRE_LEN, TRAFFIC_CONVENTION_LEN, TEMPORAL_SIZE,
    OUTPUT_SIZE, NET_OUTPUT_SIZE,
)
from .parser import parse_outputs


class ModelRunner:
    """Load supercombo.onnx and run one frame at a time with onnxruntime.

    Mirrors ModelState / model_init / model_eval_frame in
    selfdrive/modeld/models/driving.cc (v0.8.10).
    """

    def __init__(self, rhd: bool = False):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required.  pip install onnxruntime") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CUDA if available (Jetson Xavier / NVIDIA GPU), fall back to CPU.
        # On macOS, CoreML is available but slower than CPU for this model — skip it.
        providers: list[str] = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self._session = ort.InferenceSession(
            str(SUPERCOMBO_ONNX_PATH), opts, providers=providers)
        print(f'[INFO] onnxruntime providers: {self._session.get_providers()}')

        # Identify inputs by their total element count (batch-dim = 1 or None).
        # Sizes from driving.h: image=MODEL_BUF_SIZE, desire=8, traf=2, rnn=512.
        IMG_ELEMS = MODEL_BUF_SIZE  # 393 216 float32 values (uint8 pixels cast to float)
        _key_for_size = {
            IMG_ELEMS:             'img',
            DESIRE_LEN:            'desire',
            TRAFFIC_CONVENTION_LEN:'traf',
            TEMPORAL_SIZE:         'rnn',
        }
        self._named: dict[str, str] = {}   # role → ort input name
        self._shapes: dict[str, tuple] = {}
        for inp in self._session.get_inputs():
            elems = int(np.prod([d if isinstance(d, int) and d > 0 else 1
                                 for d in inp.shape]))
            role = _key_for_size.get(elems)
            if role:
                self._named[role] = inp.name
                self._shapes[role] = tuple(
                    d if isinstance(d, int) and d > 0 else 1
                    for d in inp.shape)

        # Persistent state (matches ModelState in driving.cc)
        self._recurrent = np.zeros((1, TEMPORAL_SIZE), dtype=np.float32)
        self._desire    = np.zeros((1, DESIRE_LEN),    dtype=np.float32)
        self._traf_conv = np.zeros((1, TRAFFIC_CONVENTION_LEN), dtype=np.float32)
        # traffic_convention[0]=1 → LHD (drive on right, US/Europe)
        # traffic_convention[1]=1 → RHD (drive on left, Japan/UK)
        self._traf_conv[0, int(rhd)] = 1.0

        # Expose the image input shape so callers can prepare the right buffer
        self.img_shape: tuple = self._shapes.get('img', (1, MODEL_BUF_SIZE))

    def run(self, frame_buf: np.ndarray) -> dict:
        """Run one forward pass.

        Parameters
        ----------
        frame_buf : uint8 ndarray of length MODEL_BUF_SIZE (393 216 bytes)
            Packed temporal input = [oldest_frame | current_frame] as produced
            by preprocess.pack_loadyuv().  Oldest frame is zeroed until the
            ring buffer fills up.

        Returns
        -------
        Dict with keys lane_lines, road_edges, plan, lead (numpy arrays),
        or empty dict if onnxruntime is unavailable / model file missing.
        """
        # The ONNX model was trained with uint8 pixel values stored as float32
        img_float = frame_buf.astype(np.float32).reshape(self.img_shape)

        feed: dict[str, np.ndarray] = {}
        if 'img'    in self._named: feed[self._named['img']]    = img_float
        if 'desire' in self._named: feed[self._named['desire']] = self._desire
        if 'traf'   in self._named: feed[self._named['traf']]   = self._traf_conv
        if 'rnn'    in self._named: feed[self._named['rnn']]    = self._recurrent

        raw: np.ndarray = self._session.run(None, feed)[0].flatten()

        # Feed recurrent state back for next frame
        if len(raw) >= NET_OUTPUT_SIZE:
            self._recurrent[0, :] = raw[OUTPUT_SIZE:OUTPUT_SIZE + TEMPORAL_SIZE]

        return parse_outputs(raw)
