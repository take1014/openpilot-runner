import cv2
import sys
from copy import deepcopy
import numpy as np
from glob import glob
from natsort import natsorted
from runner import SuperComboRunner

color_table = {
    "LL": [0, 0, 255],
    "EL": [0, 255, 0],
    "ER": [0, 255, 0],
    "RR": [255, 0, 0]
}

cam_height = 1.22

def project_to_image(intrinsic, lanes, z):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    cx = 1164//2
    cy = 874//2
    x = lanes["y"]
    y = lanes["z"]
    us = (fx * x) / (z + 1e-5) + cx
    vs = (fy * y) / (z + 1e-5) + cy
    return us, vs

def main(onnx_file_path):
    runner = SuperComboRunner(onnx_file_path)

    image_path_list = natsorted(glob("./test_imgs/*.png"))

    img_pre = None
    for idx, image_path in enumerate(image_path_list):
        print(image_path)
        img_curr = cv2.imread(image_path)
        img_pre = img_curr if idx == 0 else img_pre

        # inference
        raw_result = runner.run(img_curr, img_pre)

        # lanes
        cam_intrinsic = runner.intrinsic

        # img_curr = cv2.resize(img_curr, (256, 128))
        lanes = runner.parse_lanes(raw_result)
        for line_kind in ["LL", "EL", "ER", "RR"]:
            us, vs = project_to_image(cam_intrinsic, lanes["lanes"][line_kind], lanes["x"])
            for u, v in zip(us, vs):
                if 0 <= u < 1164 and 0 <= v < 874:
                    print(line_kind, (int(u), int(v)))
                    cv2.circle(img_curr, (int(u), int(v)-60), 1, color_table[line_kind], thickness=1)

        # road_edges
        road_edges = runner.parse_road_edges(raw_result)
        for line_kind in ["L", "R"]:
            us, vs = project_to_image(cam_intrinsic, road_edges["road_edges"][line_kind], road_edges["x"])
            for u, v in zip(us, vs):
                if 0 <= u < 1164 and 0 <= v < 874:
                    cv2.circle(img_curr, (int(u), int(v) - 60), 1, [255, 255, 255], thickness=1)


        img_pre = deepcopy(img_curr)
        cv2.imshow("preview", img_curr)
        cv2.waitKey(1000)

if __name__ == '__main__':
    onnx_file_path = './models/supercombo_v0_8_10.onnx'
    main(onnx_file_path)