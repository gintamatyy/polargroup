import os
import numpy as np
import json
import cv2
import random
from tqdm import tqdm

dataset_dir = "/mnt/Data2/jiahua/coco/images/val2017"
dataset_save = "/mnt/Data2/jiahua/test_result/coco/val/dekr2"


def show_skeleton(img, kpts, color=(255, 128, 128), thr=0.5):
    kpts = np.array(kpts).reshape(-1, 3)
    skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                 [9, 11], [1, 2], [1, 3], [2, 4], [3, 5]]
    points_num = [num for num in range(1, 18)]
    for sk in skelenton:

        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0] - 1, 2] > thr and kpts[
            sk[1] - 1, 2] > thr:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points - 1, 0]), int(kpts[points - 1, 1]))
        if pos[0] > 0 and pos[1] > 0 and kpts[points - 1, 2] > thr:
            cv2.circle(img, pos, 4, (0, 0, 255), -1)  # 为肢体点画红色实心圆
    return img


with open(
        "/home/jiahua/pytorch-code/dekr/output/coco_kpt/hrnet_dekr/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140/results/keypoints_val2017regression_results.json",
        "r") as load_f:
    # with open("/mnt/Data2/jiahua/test_result/coco/annotation/keypoints_val2017regression_results.json", "r") as load_f:
    load_dict = json.load(load_f)
imgIds_old = 0
image = cv2.imread(os.path.join(dataset_dir, str(397133).zfill(12) + '.jpg'))
skeleton_color = [(154, 194, 182), (123, 151, 138), (0, 208, 244), (8, 131, 229), (18, 87, 220)]  # 选择自己喜欢的颜色
for dict_num in tqdm(load_dict):
    imgIds = dict_num["image_id"]
    if imgIds != imgIds_old:
        cv2.imwrite(os.path.join(dataset_save, str(imgIds_old).zfill(12) + '.jpg'), image)
        image_path = os.path.join(dataset_dir, str(imgIds).zfill(12) + '.jpg')
        image = cv2.imread(image_path)
    color = random.choice(skeleton_color)
    show_skeleton(image, dict_num["keypoints"], color=color)

    imgIds_old = imgIds
