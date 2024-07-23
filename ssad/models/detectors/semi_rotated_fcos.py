import torch
from mmrotate.models import RotatedFCOS, ROTATED_DETECTORS, RotatedSingleStageDetector
from mmrotate.core import rbbox2result
import mmcv
import numpy as np


@ROTATED_DETECTORS.register_module()
class SemiRotatedFCOS(RotatedFCOS):

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        get_data=False,
        get_pred=False,
    ):

        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        if not get_pred:
            return self.bbox_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, get_data=get_data
            )
        with torch.no_grad():
            self.eval()
            bbox_results = self.simple_test(img, img_metas, rescale=True)
            self.train()
        logits = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, get_data=get_data
        )
        return logits, bbox_results
