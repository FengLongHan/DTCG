# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable
beta  = 0.8


def check_overlap1(ellipse_param1, ellipse_param2, size_im):
    # create x-y coordinate grid and draw a grid on the plane based on the image size
    pixels_x, pixels_y = np.meshgrid(np.arange(size_im[0]) + 1, np.arange(size_im[1]) + 1)
    a1, b1, x1, y1, theta1 = ellipse_param1
    if ((a1 != 0) | (b1 != 0)):
        # calculate the number of pixels inside the standard ellipse

        f1 = ((pixels_x - x1) * np.sin(theta1) - (pixels_y - y1) * np.cos(theta1)) ** 2 / b1 ** 2 + (
                    (pixels_x - x1) * np.cos(theta1) + (pixels_y - y1) * np.sin(theta1)) ** 2 / a1 ** 2 - 1
        pixels_inside_ellipse1 = ~(f1 > 0)
    else:
        return 0
    a2, b2, x2, y2, theta2 = ellipse_param2
    if ((a2 != 0) | (b2 != 0)):
        # calculate the number of pixels inside the test ellipse

        f2 = ((pixels_x - x2) * np.sin(theta2) - (pixels_y - y2) * np.cos(theta2)) ** 2 / b2 ** 2 + (
                    (pixels_x - x2) * np.cos(theta2) + (pixels_y - y2) * np.sin(theta2)) ** 2 / a2 ** 2 - 1
        pixels_inside_ellipse2 = ~(f2 > 0)
    else:
        return 0

    # calculate the overlap ratio based on the number of overlapping pixels
    # a = np.sum((np.logical_xor(pixels_inside_ellipse1,pixels_inside_ellipse2)))
    # b = np.sum(np.sum((np.logical_or(pixels_inside_ellipse1,pixels_inside_ellipse2))))
    # c = np.sum(a/b)
    # overlap_ratio = 1 -c
    overlap_ratio = 1 - np.sum(np.sum((np.logical_xor(pixels_inside_ellipse1, pixels_inside_ellipse2)))) / np.sum(
        np.sum((np.logical_or(pixels_inside_ellipse1, pixels_inside_ellipse2))))
    return overlap_ratio



def tpfp_default(
                 det_bboxes,
                 gt_bboxes,
                 TP,
                 FP,
                 FN,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None,
                 ):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]



    
    gt_ellipses = np.transpose(gt_bboxes)
    sorted_indices = np.argsort(gt_ellipses[2])
    gt_ellipses = gt_ellipses[:, sorted_indices]
    
    det_ellipses = np.transpose(det_bboxes)
    sorted_indices = np.argsort(det_ellipses[2])
    
    if ((len(det_ellipses.shape) > 1)):
            det_ellipses = det_ellipses[:, sorted_indices]

    if ((len(det_ellipses.shape) <= 1) | (len(gt_ellipses.shape) <= 1)):
        TP[0, i] = 0
        FN[0, i] = np.shape(gt_ellipses)[1] - TP[0, i]
        FP[0, i] = 0
    else:
        Overlap = np.zeros((gt_ellipses.shape[1], det_ellipses.shape[1]))

        for ii in range(gt_ellipses.shape[1]):
            for jj in range(det_ellipses.shape[1]):
                max_x = max(gt_ellipses[2, ii] + gt_ellipses[0, ii], det_ellipses[2, jj] + det_ellipses[0, jj])
                max_y = max(gt_ellipses[3, ii] + gt_ellipses[0, ii], det_ellipses[3, jj] + det_ellipses[0, jj])
                Overlap[ii, jj] = check_overlap1(gt_ellipses[:, ii], det_ellipses[:, jj], [max_x + 5, max_y + 5])

        x += np.shape(gt_ellipses)[1]
        TP[0, i] = np.count_nonzero(np.sum(Overlap > beta, axis=1) > 0)
        FN[0, i] = np.shape(gt_ellipses)[1] - TP[0, i]
        FP[0, i] = det_ellipses.shape[1] - np.count_nonzero(np.sum(Overlap > beta, axis=0) > 0)
        
    
    print(TP)
    
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp

    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
  
    
    
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    

    
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    
    
    N = num_imgs
    TP = np.zeros((1, N))
    FP = np.zeros((1, N))
    FN = np.zeros((1, N))
    
    
    for i in range(num_classes):
        
        
        
    #     # 得到 椭圆指标
        
    #     for key,value in annotations[0]:
    #         pass
    #         print(i)
    #         gt_ellipses = np.array(annotations[i])
    #         print(gt_ellipses.shape)
    #         # gt_ellipses = np.transpose(gt_ellipses)
    #         # sorted_indices = np.argsort(gt_ellipses[2])
    #         # gt_ellipses = gt_ellipses[:, sorted_indices]
            
    #         det_ellipses = np.array(det_results[i])
    #         # det_ellipses = np.transpose(det_ellipses)
    #         # sorted_indices = np.argsort(det_ellipses[2])
        
        
        
        
        
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, TP, FP, FN, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    
    pool.close()
    
    TPs = np.sum(TP)
    FPs = np.sum(FP)
    FNs = np.sum(FN)
    
    if (TPs == 0):
        Precision = 0
        Recall = 0
        resultFM = 0
    else:
        Precision = TPs / (TPs + FPs)
        Recall = TPs / (TPs + FNs)
        resultFM = 2 * Precision * Recall / (Precision + Recall)
    print(TPs,FPs,FNs)
    
    
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
