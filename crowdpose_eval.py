from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

gt_file = '/mnt/Data2/jiahua/crowdpose/json/crowdpose_test.json'
preds = '/home/jiahua/pytorch-code/dekr/output/crowd_pose_kpt/hrnet_anchor9/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300_0701/results/keypoints_testregression_results.json'

cocoGt = COCO(gt_file)
cocoDt = cocoGt.loadRes(preds)
cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()