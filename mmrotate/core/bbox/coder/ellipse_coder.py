import math
import torch
import cv2
import numpy as np


def ellipseToxywha(ellipses):
    """
    input:(N,[x,y,u,v,m,a])
    :param ellipses:
    :return:(n,[x,y,w,h,a])
    """
    # 从输入张量中提取各个分量
    x, y, u, v, m, alpha = ellipses.split(1, dim=1)

    # 计算其他中间张量，以便进一步向量化操作
    fl = torch.sqrt(u**2 + v**2)
    # ulv = torch.cat((u / fl, -v / fl), dim=1)
    # usv = torch.cat((ulv[:, 0:1], -ulv[:, 1:2]), dim=1)

    # ulv = torch.where(alpha <= 0.5, torch.stack((u / fl, -v / fl), dim=1), torch.stack((u / fl, v / fl), dim=1))
    ulv_sl = torch.where(alpha <= 0.5,  -v / fl,  v / fl)
    ulv = torch.cat([u/fl, ulv_sl],dim=1)
    usv = torch.cat([ulv[0], -ulv[1]])

    al = (m + fl) / 2
    bl = torch.sqrt(al**2 - (fl / 2)**2)
    av = al * ulv
    bv = bl * usv

    # 计算四个角点 顺时针
    x1 = x - av[:, 0:1] - bv[:, 1:2]
    y1 = y - av[:, 1:2] - bv[:, 0:1]
    x2 = x + av[:, 0:1] - bv[:, 1:2]
    y2 = y + av[:, 1:2] - bv[:, 0:1]
    x3 = x + av[:, 0:1] + bv[:, 1:2]
    y3 = y + av[:, 1:2] + bv[:, 0:1]
    x4 = x - av[:, 0:1] + bv[:, 1:2]
    y4 = y - av[:, 1:2] + bv[:, 0:1]

    # 拼接为输出张量
    polys = torch.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    
    obb_bboxes = []
    for poly in polys:
        obb_bbox = poly2obb_torch_le90(poly)
        # if obb_bbox is None:
        #     obb_bbox = torch.tensor([0,0,0,0,0])
        obb_bboxes.append(obb_bbox)

    obb_bboxes = torch.stack(obb_bboxes, dim=0).reshape(-1, 5)
    return obb_bboxes



# def ellipseToxywha(ellipses):
#     """
#     input:(N,[x,y,u,v,m,a])
#     :param ellipses:
#     :return:(n,[x,y,w,h,a])
#     """
    
#     r_bboxes = []
#     for ellipse in ellipses:
#         if ellipse[2]==0 and ellipse[3]==0:
#             r_bbox = torch.tensor([ellipse[0],ellipse[1],0,0,0])
#             # print(r_bbox)
#             r_bboxes.append(r_bbox)
#             continue
#         poly = ellipseTodota(ellipse)
#         r_bbox = poly2obb_torch_le90(poly)
#         # print(poly)
#         # print(r_bbox)
#         if r_bbox is None:
#             r_bbox = torch.tensor([0,0,0,0,0])
#         r_bboxes.append(r_bbox)
    
#     obb_bbox = torch.stack(r_bboxes, dim=0).reshape(-1, 5)
#     return obb_bbox



# def ellipseTodota(ellipse):
#     # ellipse = ellipse.cpu().numpy()

#     x, y, u, v, m, alpha = ellipse.split([1,1,1,1,1,1])
#     fl = torch.sqrt(u * u + v * v)

#     if alpha <= 0.5:
#         ulv = (u / fl, -v / fl)
#     else:
#         ulv = (u / fl, v / fl)

#     usv = (ulv[0], -ulv[1])

#     al = (m + fl) / 2
#     bl = torch.sqrt(al * al - (fl / 2) * (fl / 2))

#     av = (al * ulv[0], al * ulv[1])
#     bv = (bl * usv[0], bl * usv[1])
#     # 顺时针
#     x1 = x - av[0] - bv[1]
#     y1 = y - av[1] - bv[0]
#     x2 = x + av[0] - bv[1]
#     y2 = y + av[1] - bv[0]
#     x3 = x + av[0] + bv[1]
#     y3 = y + av[1] + bv[0]
#     x4 = x - av[0] + bv[1]
#     y4 = y - av[1] + bv[0]

#     # poly = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4),axis=0)
#     # poly = torch.tensor((x1, y1, x2, y2, x3, y3, x4, y4))
#     # return poly

#     return torch.tensor([x1, y1, x2, y2, x3, y3, x4, y4])



def dotaToellipse(polys):
    polys = polys.cpu().numpy()
    x1, y1, x2, y2, x3, y3, x4, y4 = polys
    x = (x1 + x2 + x3 + x4) / 4
    y = (y1 + y2 + y3 + y4) / 4
    p = x1 - x
    q = y1 - y
    h = x2 - x
    n = y2 - y
    svec = (p - h, q - n)
    lvec = (p + h, q + n)
    sls2 = svec[0] * svec[0] + svec[1] * svec[1]
    sl = math.sqrt(sls2)
    lls2 = lvec[0] * lvec[0] + lvec[1] * lvec[1]
    ll = math.sqrt(lls2)
    if sl > ll:
        vec = svec
        laxis = sl
    else:
        vec = lvec
        laxis = ll
    fl = math.sqrt(abs(sls2 - lls2))
    s1 = 1
    s2 = 1
    if vec[0] < 0:
        vec = (-vec[0], vec[1])
        s1 = -1
    if vec[1] < 0:
        vec = (vec[0], -vec[1])
        s2 = -1
    alpha = 0 if s1 + s2 == 0 else 1
    # alpha = torch.where((s1 + s2) == 0, torch.tensor([0.0]), torch.tensor([1.0]))

    ux = vec[0] / laxis
    uy = vec[1] / laxis
    u = fl * ux
    v = fl * uy
    m = laxis - fl
    return torch.tensor((x, y, u, v, m, alpha))


def xywhaToellipse(rbboxes):
    x = rbboxes[:, 0].reshape(-1, 1)
    y = rbboxes[:, 1].reshape(-1, 1)
    w = rbboxes[:, 2].reshape(-1, 1)
    h = rbboxes[:, 3].reshape(-1, 1)
    a = rbboxes[:, 4].reshape(-1, 1)

    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = torch.cat([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


# rbbox转为dota 验证
    # r_bboxes = []
    # for poly in polys:
    #     r_bbox = poly2obb_torch_le90(poly)
    #     r_bboxes.append(r_bbox)
    # r_bboxes = torch.stack(r_bboxes,dim=0).reshape(-1,5)

    e_bboxes = []
    for poly in polys:
        e_bbox = dotaToellipse(poly)
        e_bboxes.append(e_bbox)
    ellipse_bbox = torch.stack(e_bboxes, dim=0).reshape(-1, 6)
    return ellipse_bbox



def poly2obb_torch_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        poly (Tensor): [x0, y0, x1, y1, x2, y2, x3, y3]

    Returns:
        obb (Tensor): [x_ctr, y_ctr, w, h, angle]
    """
    # 将输入张量转换为NumPy数组
    bboxps = poly.cpu().numpy().reshape((4, 2)).astype(np.float32)
    # bboxps = poly.reshape((4, 2))

    # bbox = bboxps.numpy().astype(np.float32)
    # bboxps = bboxps.astype(np.float32)
    # 使用OpenCV计算旋转矩形
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]


    # if w < 2 or h < 2:
    #     return None

    # 将角度从度数转换为弧度
    a = a / 180 * np.pi

    if w < h:
        w, h = h, w
        a += np.pi / 2

    # 将角度调整为在 -pi/2 到 pi/2 之间
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi

    assert np.pi / 2 > a >= -np.pi / 2
    return torch.tensor([x, y, w, h, a])

def ellipseToObb(ellipses):
    x, y, u, v, m, alpha = ellipses.split(1, dim=1)

    # 计算其他中间张量，以便进一步向量化操作
    # 2*c
    fl = torch.sqrt(u ** 2 + v ** 2).clamp(1e-8)
    # ulv = torch.cat((u / fl, -v / fl), dim=1)
    # usv = torch.cat((ulv[:, 0:1], -ulv[:, 1:2]), dim=1)

    # ulv = torch.where(alpha <= 0.5, torch.stack((u / fl, -v / fl), dim=1), torch.stack((u / fl, v / fl), dim=1))
    ulv_sl = torch.where(alpha <= 0.5, -v / fl, v / fl)
    # ulv = torch.cat([u / fl, ulv_sl], dim=1)
    # usv = torch.cat([ulv[0], -ulv[1]])

    al = (m + fl) / 2
    bl = torch.sqrt(al ** 2 - (fl / 2) ** 2)
    w = 2*al
    h = 2*bl

    a = torch.arcsin(ulv_sl)

    obb_bbox = torch.cat([x,y,w,h,a],dim=1)
    return obb_bbox