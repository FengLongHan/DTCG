import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init, Scale
from mmcv.runner import force_fp32
from mmcv.ops import deform_conv2d
from mmdet.core import distance2bbox, multi_apply, bbox_overlaps, reduce_mean, filter_scores_and_topk, select_single_mlvl, bbox2distance

from mmdet.models.dense_heads.paa_head import levels_to_images 
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from .rotated_fcos_head import RotatedFCOSHead
from ..builder import ROTATED_HEADS, build_loss
from mmrotate.core import rbbox_overlaps
from mmrotate.core.bbox.coder import distance_angle_point_coder
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated

EPS = 1e-12

class CenterPrior(nn.Module):
    def __init__(self,
                 soft_prior=True,
                #  num_classes=80,
                num_classes = 1,
                 strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(num_classes, 2)+0.11, requires_grad=False)
        self.strides = strides
        self.soft_prior = soft_prior

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):

        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,
                                    num_gts), inside_gt_bbox_mask
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            # slvl_points: points from single level in FPN, has shape (h*w, 2)
            # single_level_points has shape (h*w, num_gt, 2)
            single_level_points = slvl_points[:, None, :].expand(
                (slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
            gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            # instance_center has shape (1, num_gt, 2)
            instance_center = self.mean[labels][None]
            # instance_sigma has shape (1, num_gt, 2)
            instance_sigma = self.sigma[labels][None]
            # distance has shape (num_points, num_gt, 2)
            distance = (((single_level_points - gt_center) / float(stride) -
                        instance_center)**2)
            center_prior = torch.exp(-distance /
                                     (2 * instance_sigma**2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        if not self.soft_prior:
            prior_mask = center_prior_weights > 0.3    
            center_prior_weights[prior_mask] = 1
            center_prior_weights[~prior_mask] = 0

        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


@ROTATED_HEADS.register_module()
class DWHead(RotatedAnchorFreeHead):
    def __init__(self,
                 *args,
                 soft_prior=True,
                 reg_refine=True,
                 prior_offset=0.5,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bbox_coder=dict(type='DistanceAnglePointCoder',angle_version = 'le90'),
                 init_cfg=dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01)),
                 **kwargs):
        self.with_reg_refine = reg_refine
        super().__init__(*args, 
                         conv_bias=True,
                         norm_cfg=norm_cfg,
                         init_cfg=init_cfg,
                         **kwargs)
        self.bbox_coder = bbox_coder
        self.center_prior = CenterPrior(
            soft_prior=soft_prior,
            num_classes=self.num_classes,
            strides=self.strides)
        self.prior_generator.offset = prior_offset
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
    def init_weights(self):
        super(DWHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_angle, std=0.01)
        if self.with_reg_refine:
            normal_init(self.reg_offset, std=0.01)
            self.reg_offset.bias.data.zero_()
    
    def _init_layers(self):
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.with_reg_refine:
            self.reg_offset = nn.Conv2d(self.feat_channels, 4 * 2, 3, padding=1)
    
    def deform_sampling(self, feat, offset):
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y
    
    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)
        
    def forward_single(self, x, scale, stride):
        b, c, h, w = x.shape
        # cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        centerness = self.conv_centerness(reg_feat)
        angle_pred = self.conv_angle(reg_feat)
        
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)
        bbox_pred *= stride
        if self.with_reg_refine:
            bbox_dist = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            angle_dist = angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            reg_dist = torch.cat([bbox_dist, angle_dist], dim=-1)
            # bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            # print('reg_dist\n', reg_dist.shape)
            # print('bbox_pred\n', bbox_pred.shape)
            points = self.prior_generator.single_level_grid_priors((h,w), self.strides.index(stride), dtype=x.dtype, device=x.device)
            points = points.repeat(b, 1) 
            
            #!!!  改为angle_bbox_decoder
            # decoded_bbox_preds = distance2bbox(points, reg_dist).reshape(b, h, w, 5).permute(0, 3, 1, 2)
            
            decoded_bbox_preds = self.bbox_coder.decode(points, reg_dist).reshape(b, h, w, 5).permute(0, 3, 1, 2)
            
            decoded_bbox_preds = decoded_bbox_preds[:,:4,:,:]
            reg_offset = self.reg_offset(reg_feat)
            bbox_pred_d  = bbox_pred / stride 
            # 计算偏移
            reg_offset = torch.stack([reg_offset[:,0], reg_offset[:,1] - bbox_pred_d[:, 0],\
                                        reg_offset[:,2] - bbox_pred_d[:, 1], reg_offset[:,3],
                                        reg_offset[:,4], reg_offset[:,5] + bbox_pred_d[:, 2],
                                        reg_offset[:,6] + bbox_pred_d[:, 3], reg_offset[:,7],
                                        # reg_offset[:,8], reg_offset[:,9] + bbox_pred_d[:, 4],
                                        ], 1)
            bbox_pred = self.deform_sampling(decoded_bbox_preds.contiguous(), reg_offset.contiguous())
            decoded_bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            # 编码成距离。归一
            # ！！！ 改为angleb_box_encoder
            # bbox_pred = F.relu(bbox2distance(points, bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)).reshape(b, h, w, 5).permute(0, 3, 1, 2).contiguous())
            decoded_bbox_pred = F.relu(self.bbox_coder.encode(points, decoded_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)).reshape(b, h, w, 5).permute(0, 3, 1, 2).contiguous())
            bbox_pred = decoded_bbox_pred[:,:4,:,:]
            angle_pred = decoded_bbox_pred[:,4:,:,:]
        return cls_score, bbox_pred, angle_pred, centerness
    
    def _loss_single(self, cls_score, objectness, reg_loss, gt_labels,
                            center_prior_weights, ious, inside_gt_bbox_mask):
        num_gts = len(gt_labels)
        joint_conf = (cls_score * objectness)
        #To more precisely estimate the consistency degree between cls and reg heads, we represent IoU score as an expentional function of the reg loss.
        p_loc = torch.exp(-reg_loss*5)
        p_cls = (cls_score * objectness)[:, gt_labels] 
        p_pos = p_cls * p_loc
        
        p_neg_weight = torch.ones_like(joint_conf)
        neg_metrics = torch.zeros_like(ious).fill_(-1)
        alpha = 2
        t = lambda x: 1/(0.5**alpha-1)*x**alpha - 1/(0.5**alpha-1)
        if num_gts > 0:
            def normalize(x): 
                x_ = t(x)
                t1 = x_.min()
                t2 = min(1., x_.max())
                y = (x_ - t1 + EPS ) / (t2 - t1 + EPS )
                y[x<0.5] = 1
                return y
            for instance_idx in range(num_gts):
                idxs = inside_gt_bbox_mask[:, instance_idx]
                if idxs.any():
                    neg_metrics[idxs, instance_idx] = normalize(ious[idxs, instance_idx])
            foreground_idxs = torch.nonzero(neg_metrics != -1, as_tuple=True)
            p_neg_weight[foreground_idxs[0],
                         gt_labels[foreground_idxs[1]]] = neg_metrics[foreground_idxs]
        
        p_neg_weight = p_neg_weight.detach()
        neg_avg_factor = (1 - p_neg_weight).sum()
        p_neg_weight = p_neg_weight * joint_conf ** 2
        neg_loss = p_neg_weight * F.binary_cross_entropy(joint_conf, torch.zeros_like(joint_conf), reduction='none')
        neg_loss = neg_loss.sum() 
        
        p_pos_weight = (torch.exp(5*p_pos) * p_pos * center_prior_weights) / (torch.exp(3*p_pos) * p_pos * center_prior_weights).sum(0, keepdim=True).clamp(min=EPS)
        p_pos_weight = p_pos_weight.detach()
        
        cls_loss = F.binary_cross_entropy(
            p_cls,
            torch.ones_like(p_cls),
            reduction='none') * p_pos_weight 
        loc_loss = F.binary_cross_entropy(
            p_loc,
            torch.ones_like(p_loc),
            reduction='none') * p_pos_weight 
        cls_loss = cls_loss.sum() * 0.25
        loc_loss = loc_loss.sum() * 0.25
        
        return cls_loss, loc_loss, neg_loss, neg_avg_factor
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds','angle_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses) == len(angle_preds)
        all_num_gt = sum([len(gt_bbox) for gt_bbox in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes)

        center_prior_weight_list = []
        temp_inside_gt_bbox_mask_list = []
        for gt_bboxe, gt_label, inside_gt_bbox_mask in zip(gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = self.center_prior(all_level_points, gt_bboxe, gt_label, inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
            temp_inside_gt_bbox_mask_list.append(inside_gt_bbox_mask)
        inside_gt_bbox_mask_list = temp_inside_gt_bbox_mask_list

        mlvl_points = torch.cat(all_level_points, dim=0)
        # [16368,5]
        bbox_preds = levels_to_images(bbox_preds) 
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)
        angle_preds = levels_to_images(angle_preds)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)

        for bbox_pred, angle_pred, objectness,gt_bboxe, inside_gt_bbox_mask in zip(
                bbox_preds, angle_preds,objectnesses, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 5)
            bbox_pred = torch.cat((bbox_pred, angle_pred), dim=-1)
            
            pos_centerness_targets = self.centerness_target(gt_bboxe)
            centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 5).reshape(-1, 5)
            # !!! 改为angle_bbox_decode
            # decoded_bbox_preds = distance2bbox(expand_mlvl_points,
            #                                    expand_bbox_pred)
            decoded_bbox_preds = self.bbox_coder.decode(expand_mlvl_points,
                                               expand_bbox_pred)
            #  !!! 改为angle_bbox_decode
            # decoded_target_preds = distance2bbox(expand_mlvl_points, gt_bboxe)
            decoded_target_preds = self.bbox_coder.decode(expand_mlvl_points, gt_bboxe)
            with torch.no_grad():
                ious = rbbox_overlaps(
                    decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    ious = ious
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            print(loss_bbox)
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))
        
        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        # angle_preds = [item.sigmod() for item in angle_preds]
        cls_loss_list, loc_loss_list, cls_neg_loss_list, neg_avg_factor_list = multi_apply(self._loss_single, cls_scores,
                                    objectnesses, reg_loss_list, gt_labels, center_prior_weight_list, ious_list, inside_gt_bbox_mask_list)

        pos_avg_factor = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        neg_avg_factor = sum(item.data.sum()
                            for item in neg_avg_factor_list).float()
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        cls_loss = sum(cls_loss_list) / pos_avg_factor
        loc_loss = sum(loc_loss_list) / pos_avg_factor
        cls_neg_loss = sum(cls_neg_loss_list) / neg_avg_factor

        loss = dict(
            loss_cls_pos=cls_loss, loss_loc=loc_loss, loss_cls_neg=cls_neg_loss)
        return loss

    def get_targets(self, points, gt_bboxes_list):
        concat_points = torch.cat(points, dim=0)
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        return inside_gt_bbox_mask_list, bbox_targets_list
    
    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_target_single(self, gt_bboxes, points):
        
        # 改为angle_distance 
        # num_points = points.size(0)
        # num_gts = gt_bboxes.size(0)
        # gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        # xs, ys = points[:, 0], points[:, 1]
        # xs = xs[:, None]
        # ys = ys[:, None]
        # left = xs - gt_bboxes[..., 0]
        # right = gt_bboxes[..., 2] - xs
        # top = ys - gt_bboxes[..., 1]
        # bottom = gt_bboxes[..., 3] - ys
        # bbox_targets = torch.stack((left, top, right, bottom), -1)
        
        
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
                
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        angle_targets = gt_angle.squeeze(-1) 
        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        
        bbox_targets = torch.stack((left, top, right, bottom, angle_targets), -1)
        
        

        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets
    
    
    
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):

        assert len(cls_scores) == len(bbox_preds) == len(score_factors) == len(angle_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            angle_pred_list = select_single_mlvl(angle_preds, img_id)
            score_factor_list = select_single_mlvl(score_factors, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              angle_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, 
                                              **kwargs)
            result_list.append(results)
        return result_list
    


    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                        #    scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        # if rescale:
        #     scale_factor = mlvl_bboxes.new_tensor(scale_factor)
        #     mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        
        return det_bboxes, det_labels
        
