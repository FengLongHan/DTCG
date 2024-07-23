from mmrotate import ROTATED_HEADS
from ssad.models.dense_heads.ellipse_fcos_head import EllipseFCOSHead


@ROTATED_HEADS.register_module()
class SemiRotatedFCOSHead(EllipseFCOSHead):
    def __init__(self, num_classes, in_channels, **kwargs):
        super(SemiRotatedFCOSHead, self).__init__(num_classes, in_channels, **kwargs)

    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        get_data=False,
        **kwargs
    ):

        if get_data:
            return self(x)
        return super(SemiRotatedFCOSHead, self).forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs
        )
