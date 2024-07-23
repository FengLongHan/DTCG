import os
import numpy as np
import math



path_GT = "data/ellipseData/ged_gt/gt_norm/"

path_Det = "work_dir/test_img/10_labeled_ellipse/det/"


path_DetOut = "workdirs/new_ged/all_labeled/test_img/iter_99200_best/img/det/"

beta = 0.8

import numpy as np


def ellipseRegularized(e, inpath):

    if e[4] > np.pi:
        e[4] = e[4] - 2 * np.pi
    if e[4] < -np.pi:
        e[4] = e[4] + 2 * np.pi

    if e[3] > e[2]:
        min_axes = e[2]
        e[2] = e[3]
        e[3] = min_axes

        if e[4] > 0:
            e[4] = e[4] - 2 - np.pi / 2
        if e[4] < 0:
            e[4] = e[4] - 2 + np.pi / 2

    if e[4] > np.pi:
        e[4] = e[4] - 2 * np.pi
    if e[4] < -np.pi:
        e[4] = e[4] + 2 * np.pi

    return e


def normal_det(input_Path, output_Path):
    files = os.listdir(input_Path)
    N = len(files)

    for i in range(N):
        filename = files[i]
        inPath = input_Path + filename
        outPath = output_Path + filename
        with open(inPath, "r") as f:
            data = f.readlines()
            ellipses = np.zeros((int(data[0]), 5))
            for i in range(1, int(data[0]) + 1, 1):
                # dic = data[i].split(' ')

                dic = data[i].split("\t")

                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = float(an[0])
                ellipse = [cx, cy, a, b, theta]

                # ellipse[4] = np.deg2rad(ellipse[4])

                # ellipse = ellipseRegularized(ellipse,inPath)
                ellipses[i - 1, 0:5] = ellipse[0:5]
        with open(outPath, "w") as f:
            f.write(data[0])
            for i in range(int(data[0])):
                line = "\t".join(str(x) for x in ellipses[i])
                f.write(line + "\n")


import numpy as np


def check_overlap1(ellipse_param1, ellipse_param2, size_im):
    # create x-y coordinate grid and draw a grid on the plane based on the image size
    pixels_x, pixels_y = np.meshgrid(
        np.arange(size_im[0]) + 1, np.arange(size_im[1]) + 1
    )
    a1, b1, x1, y1, theta1 = ellipse_param1
    if (a1 != 0) | (b1 != 0):
        # calculate the number of pixels inside the standard ellipse

        f1 = (
            ((pixels_x - x1) * np.sin(theta1) - (pixels_y - y1) * np.cos(theta1)) ** 2
            / b1**2
            + ((pixels_x - x1) * np.cos(theta1) + (pixels_y - y1) * np.sin(theta1)) ** 2
            / a1**2
            - 1
        )
        pixels_inside_ellipse1 = ~(f1 > 0)
    else:
        return 0
    a2, b2, x2, y2, theta2 = ellipse_param2
    if (a2 != 0) | (b2 != 0):
        # calculate the number of pixels inside the test ellipse

        f2 = (
            ((pixels_x - x2) * np.sin(theta2) - (pixels_y - y2) * np.cos(theta2)) ** 2
            / b2**2
            + ((pixels_x - x2) * np.cos(theta2) + (pixels_y - y2) * np.sin(theta2)) ** 2
            / a2**2
            - 1
        )
        pixels_inside_ellipse2 = ~(f2 > 0)
    else:
        return 0

    # calculate the overlap ratio based on the number of overlapping pixels
    # a = np.sum((np.logical_xor(pixels_inside_ellipse1,pixels_inside_ellipse2)))
    # b = np.sum(np.sum((np.logical_or(pixels_inside_ellipse1,pixels_inside_ellipse2))))
    # c = np.sum(a/b)
    # overlap_ratio = 1 -c
    overlap_ratio = 1 - np.sum(
        np.sum((np.logical_xor(pixels_inside_ellipse1, pixels_inside_ellipse2)))
    ) / np.sum(np.sum((np.logical_or(pixels_inside_ellipse1, pixels_inside_ellipse2))))
    return overlap_ratio


def getFValue(tp, fn, fp):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_value = 2 * precision * recall / (precision + recall)
    return f_value, precision, recall


def computePerformanceAllLu():
    x = 0
    files = os.listdir(path_GT)
    count = 0
    N = len(files)
    TP = np.zeros((1, N))
    FP = np.zeros((1, N))
    FN = np.zeros((1, N))
    std_devs = np.zeros((1, 5))
    std_count = 0
    for i in range(N):
        f = files[i]

        file_GT = path_GT + f
        # file_det = path_DetOut + 'det_' + f[3:-8] + '.jpg.txt'
        file_det = path_DetOut + f[3:-8] + ".png.txt"

        if f[3:-8] == "00004":
            print(123)

        with open(file_GT, "r") as f:
            data = f.readlines()
            gt_ellipses = np.zeros((int(data[0]), 5))
            for ind1 in range(1, int(data[0]) + 1, 1):
                dic = data[ind1].split("\t")
                # print(file_GT,dic)
                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = float(an[0])
                gt_ellipses[ind1 - 1, 0:5] = [a, b, cx, cy, theta]

        gt_ellipses = np.transpose(gt_ellipses)
        sorted_indices = np.argsort(gt_ellipses[2])
        gt_ellipses = gt_ellipses[:, sorted_indices]

        count += np.shape(gt_ellipses)[1]

        with open(file_det, "r") as f:
            data = f.readlines()
            det_ellipses = np.zeros((int(data[0]), 5))
            for ind2 in range(1, int(data[0]) + 1, 1):
                dic = data[ind2].split("\t")
                # print(file_det,dic)
                an = dic[4].split("\n")
                cx = float(dic[0])
                cy = float(dic[1])
                a = float(dic[2])
                b = float(dic[3])
                theta = float(an[0])
                det_ellipses[ind2 - 1, 0:5] = [a, b, cx, cy, theta]

        # 按cx坐标升序排列
        det_ellipses = np.transpose(det_ellipses)
        sorted_indices = np.argsort(det_ellipses[2])

        if len(det_ellipses.shape) > 1:
            det_ellipses = det_ellipses[:, sorted_indices]

        if (len(det_ellipses.shape) <= 1) | (len(gt_ellipses.shape) <= 1):
            TP[0, i] = 0
            FN[0, i] = np.shape(gt_ellipses)[1] - TP[0, i]
            FP[0, i] = 0
        else:
            Overlap = np.zeros((gt_ellipses.shape[1], det_ellipses.shape[1]))

            for ii in range(gt_ellipses.shape[1]):
                for jj in range(det_ellipses.shape[1]):
                    max_x = max(
                        gt_ellipses[2, ii] + gt_ellipses[0, ii],
                        det_ellipses[2, jj] + det_ellipses[0, jj],
                    )
                    max_y = max(
                        gt_ellipses[3, ii] + gt_ellipses[0, ii],
                        det_ellipses[3, jj] + det_ellipses[0, jj],
                    )
                    Overlap[ii, jj] = check_overlap1(
                        gt_ellipses[:, ii], det_ellipses[:, jj], [max_x + 5, max_y + 5]
                    )

                    if Overlap[ii, jj] > beta:

                        std_devs[0, 0] += math.pow(
                            gt_ellipses[:, ii][0] - det_ellipses[:, jj][0], 2
                        )
                        std_devs[0, 1] += math.pow(
                            gt_ellipses[:, ii][1] - det_ellipses[:, jj][1], 2
                        )
                        std_devs[0, 2] += math.pow(
                            gt_ellipses[:, ii][2] - det_ellipses[:, jj][2], 2
                        )
                        std_devs[0, 3] += math.pow(
                            gt_ellipses[:, ii][3] - det_ellipses[:, jj][3], 2
                        )
                        std_devs[0, 4] += math.pow(
                            gt_ellipses[:, ii][4] - det_ellipses[:, jj][4], 2
                        )
                        std_count += 1
                        # print(std_count)

            x += np.shape(gt_ellipses)[1]
            TP[0, i] = np.count_nonzero(np.sum(Overlap > beta, axis=1) > 0)
            FN[0, i] = np.shape(gt_ellipses)[1] - TP[0, i]
            FP[0, i] = det_ellipses.shape[1] - np.count_nonzero(
                np.sum(Overlap > beta, axis=0) > 0
            )

    TPs = np.sum(TP)
    FPs = np.sum(FP)
    FNs = np.sum(FN)

    if TPs == 0:
        Precision = 0
        Recall = 0
        resultFM = 0
    else:
        Precision = TPs / (TPs + FPs)
        Recall = TPs / (TPs + FNs)
        resultFM = 2 * Precision * Recall / (Precision + Recall)

    return Precision, Recall, resultFM


if __name__ == "__main__":
    normal_det(path_Det, path_DetOut)
    computePerformanceAllLu()

    def momentum_static_update(self, model, momentum):

        if self.skip_buffer:
            for (
                (src_name, src_parm),
                (tgt_name, tgt_parm),
                (stgt_name, stgt_parm),
            ) in zip(
                model.student.named_parameters(),
                model.teacher.named_parameters(),
                model.static_teacher.named_parameters(),
            ):
                # tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

                stgt_parm.data.copy_(src_parm.data)
        else:
            for src_parm, dst_parm in zip(
                model.student.state_dict().values(),
                #   model.teacher.state_dict().values(),
                model.static_teacher.state_dict().values(),
            ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    # dst_parm.data.mul_(momentum).add_(
                    #     src_parm.data, alpha=1 - momentum)
                    dst_parm.data.copy_(src_parm.data)
