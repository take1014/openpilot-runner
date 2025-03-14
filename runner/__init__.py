import os
import sys
import cv2
import numpy as np
import onnxruntime
from .utils import sigmoid

class SuperComboRunner:

    _EON_FOCAL_LENGTH = 910.0   # pixel
    _EON_IMG_SIZE_W   = 1164    # pixel
    _EON_IMG_SIZE_H   = 874     # pixel
    _EON_CAMERA_OFFSET = 0.06   # m

    def __init__(self, onnx_model_path='./models/supercombo_v0_8_10.onnx',
                 excam_para=(_EON_FOCAL_LENGTH, _EON_IMG_SIZE_W, _EON_IMG_SIZE_H, _EON_CAMERA_OFFSET)):
        assert os.path.exists(onnx_model_path), f'Do not exists onnx model path {onnx_model_path}'
        self.sess = onnxruntime.InferenceSession(onnx_model_path, None)
        self.input_keys = [i.name for i in self.sess.get_inputs()]

        # indexs
        # plan
        self.plan_start_idx = 0
        self.plan_end_idx = 4955
        # lanes
        self.lanes_start_idx = self.plan_end_idx
        self.lanes_end_idx   = self.lanes_start_idx + 528
        # lanes prob
        self.lanes_prob_start_idx = self.lanes_end_idx
        self.lanes_prob_end_idx   = self.lanes_prob_start_idx + 8
        # road_edges
        self.road_edges_start_idx = self.lanes_prob_end_idx
        self.road_edges_end_idx = self.road_edges_start_idx + 264
        # recurrent state
        self.recurrent_state_start_idx = 6472 - 512
        self.recurrent_state_data = np.zeros((1, 512), dtype=np.float32)

        # camera's intrinsic param
        # intrinsic = [
        #   [fx,  0, cx],
        #   [ 0, fy, cy],
        #   [ 0,  0,  1]
        # ]
        eon_dcam_intrinsic = np.array([
            [self._EON_FOCAL_LENGTH,                       0., self._EON_IMG_SIZE_W/2.],
            [                     0.,  self._EON_FOCAL_LENGTH, self._EON_IMG_SIZE_H/2.],
            [                     0.,                      0.,                      1.]
        ], dtype=np.float32)

        self.excam_intrinsic = np.array([
            [excam_para[0],             0., excam_para[1]/2.],
            [            0., excam_para[0], excam_para[2]/2.],
            [            0.,            0.,               1.]
        ], dtype=np.float32)
        self._trans_excam_to_eon_front = np.dot(eon_dcam_intrinsic, np.linalg.inv(self.excam_intrinsic))

        self._X_IDXS = np.array([
            0.,  0.1875,  0.75, 1.6875, 3., 4.6875, 6.75, 9.1875, 12., 15.1875, 18.75,
            22.6875, 27., 31.6875, 36.75, 42.1875, 48., 54.1875, 60.75, 67.6875, 75., 82.6875,
            90.75, 99.1875, 108., 117.1875, 126.75, 136.6875, 147., 157.6875, 168.75, 180.1875, 192.
            ], dtype=np.float32)

        self._camera_offset = excam_para[3]

    @property
    def x_idxs(self):
        return self._X_IDXS

    @property
    def intrinsic(self):
        return self.excam_intrinsic

    def _preprocess_image(self, frame):
        H = (frame.shape[0]*2) // 3  # 256
        W = frame.shape[1]           # 512

        y_img = frame[0:H, :]
        u_img = frame[H:H+H//4, :].reshape(128, -1)
        v_img = frame[H+H//4:, :].reshape(128, -1)

        return np.stack([
            y_img[::2, ::2], y_img[::2, 1::2], y_img[1::2, ::2], y_img[1::2, 1::2],
            u_img, v_img
            ], dtype=np.uint8)

    def _create_input_img(self, img):
        img = cv2.warpPerspective(img, self._trans_excam_to_eon_front, (self._EON_IMG_SIZE_W, self._EON_IMG_SIZE_H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        resize_img = cv2.resize(img, (512, 256))
        yuv_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2YUV_I420)
        preprocessed_imgs = self._preprocess_image(yuv_img)
        return preprocessed_imgs

    def run(self, img_curr, img_pre, desire=[1,0,0,0,0,0,0,0], traffic_convention=[1,0]):
        """
            Args:
                img_curr (numpy ndarray) : Current frame image.
                img_pre  (numpy ndarray) : Previous frame image.
                desire   (list)          : One-hot encoded desire information.
                                            [0,0,0,0,0,0,0,0] : none
                                            [1,0,0,0,0,0,0,0] : lane centering (default)
                                            [0,1,0,0,0,0,0,0] : lane change left
                                            [0,0,1,0,0,0,0,0] : lane change right
                traffic_convention (list): One-hot encoded traffic convention information.
                                            [1, 0] : Right-hand
                                            [0, 1] : Left-hand
            Returns:
                outputs (numpy ndarray) : supercombo's outputs. 6742 size vector
        """

        inputs = []
        # input image data
        input_imgs_data = np.concatenate([
            self._create_input_img(img_curr),
            self._create_input_img(img_pre)
            ], axis=0, dtype=np.float32)
        # cv2.imwrite("./input_imgs.png", input_imgs_data.reshape(-1, 256))
        inputs.append(input_imgs_data.reshape(self.sess.get_inputs()[0].shape))
        # desire
        assert len(desire) == 8, f"The length of the desire list must be 8, but {len(desire)}"
        inputs.append(np.array(desire, dtype=np.float32).reshape(self.sess.get_inputs()[1].shape))
        # traffic convention
        assert len(traffic_convention) == 2, f"The length of the traffic_convention list must be 2, but {len(traffic_convention)}"
        inputs.append(np.array(traffic_convention, dtype=np.float32).reshape(self.sess.get_inputs()[2].shape))
        # recurrent state
        inputs.append(self.recurrent_state_data)

        # inference
        result = self.sess.run(None, dict(zip(self.input_keys, inputs)))

        # set recurrent state
        self.recurrent_state_data = result[0][:, self.recurrent_state_start_idx:]

        return result[0].reshape(-1)

    def parse_lanes(self, raw_result):
        # lanes shape: (mean, std) x (ll, el, er, rr) x 33points x (y, z)
        # lane: 4 lane's output (each 2x66) concat axis=2 -> (2x(4x66)). 66(y,z) ->(33,2)
        lane_lines = raw_result[self.lanes_start_idx:self.lanes_end_idx].reshape(2, 4, 33, 2)
        lane_probs = sigmoid(raw_result[self.lanes_prob_start_idx:self.lanes_prob_end_idx])
        return {
            "x": self._X_IDXS,
            "lanes": {
                "LL": {
                    "y": lane_lines[0, 0, :, 0] - self._camera_offset,
                    "z": lane_lines[0, 0, :, 1],
                    "prob": lane_probs[1]
                },
                "EL": {
                    "y": lane_lines[0, 1, :, 0] - self._camera_offset,
                    "z": lane_lines[0, 1, :, 1],
                    "prob": lane_probs[3]
                },
                "ER": {
                    "y": lane_lines[0, 2, :, 0] + self._camera_offset,
                    "z": lane_lines[0, 2, :, 1],
                    "prob": lane_probs[5]
                },
                "RR": {
                    "y": lane_lines[0, 3, :, 0] + self._camera_offset,
                    "z": lane_lines[0, 3, :, 1],
                    "prob": lane_probs[7]
                }
            }
        }

    def parse_road_edges(self, raw_result):
        # lanes shape: (mean, std) x (l, r) x 33points x (y, z)
        # road_edges: 2 road_edges's output (each 2x66) concat axis=2 -> (2x(2x66)). 66(y,z) ->(33,2)
        road_edges = raw_result[self.road_edges_start_idx:self.road_edges_end_idx].reshape(2, 2, 33, 2)
        return {
            "x": self._X_IDXS,
            "road_edges": {
                "L": {
                    "y": road_edges[0, 0, :, 0] - self._camera_offset,
                    "z": road_edges[0, 0, :, 1]
                },
                "R": {
                    "y": road_edges[0, 1, :, 0] + self._camera_offset,
                    "z": road_edges[0, 1, :, 1]
                }
            }
        }

    def parse_plan(self, raw_result):
        return raw_result[self.plan_start_idx:self.plan_end_idx]
