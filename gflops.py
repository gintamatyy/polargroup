from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
# import os
# import sys
# # import stat
# import pprint
from thop import clever_format
from thop import profile
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
from torchstat import stat
import _init_paths
import models

from config import cfg
from config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args
def main():
    args = parse_args()
    update_config(cfg, args)
    # logger, final_output_dir, _ = create_logger(
    #     cfg, args.cfg, 'valid'
    # )
    #
    # logger.info(pprint.pformat(args))
    # logger.info(cfg)

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False)

    # if cfg.TEST.MODEL_FILE:
        # logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    # else:
    #     model_state_file = os.path.join(
    #         final_output_dir, 'model_best.pth.tar')
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(torch.load(model_state_file))
    stat(model, (3, 512, 512))
    # input = torch.randn(1, 3, 512, 512)
    # flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")



if __name__ == '__main__':
    main()
