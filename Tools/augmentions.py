import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
# from Tools.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from Tools.metrics import bbox_ioa



def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    # HSV色彩空间增强
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    # 使用 im.shape(n,m,3) 和范围 0-255 均衡 BGR 图像“im”上的直方图
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    # 复制标签
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # 在满足跨步约束的同时调整图像大小和填充图像
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):

        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP) : 只缩小，不放大（为了更好的 val mAP）
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5, isTest=False):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            labels[ymin:ymax, xmin:xmax] = [random.randint(0, 0) for _ in range(3)]

            # return unobscured labels
            # if len(labels) and s > 0.03:
            #     box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            #     ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            #     labels = labels[ioa < 0.60]  # remove >60% obscured labels

    if isTest:
        return im, labels

    return labels


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(img1, img2, mask1, mask2, p=0.5):
    # ------------------------------  CutMix  ------------------------------------------
    if random.random() < p:
        img1 = np.array(img1)
        mask1 = np.array(mask1)
        rand_image = np.array(img2)
        rand_mask = np.array(mask2)
        lam1 = np.random.beta(1, 1)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img1.shape, lam1)
        img1[bbx1:bbx2, bby1:bby2, :] = rand_image[bbx1:bbx2, bby1:bby2, :]
        mask1[bbx1:bbx2, bby1:bby2] = rand_mask[bbx1:bbx2, bby1:bby2]
    # img1 = Image.fromarray(img1.astype(np.uint8))
    # mask1 = Image.fromarray(mask1.astype(np.uint8))

    return img1, mask1

def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(2, 5)  # mixup ratio, alpha=beta=32.0
    im2 = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, im2


def cut_bbox(size, r, p=0.5):
    W = size[0]
    H = size[1]
    flag = True

    if random.random() <= p:
        cut_h_start = np.random.randint(H // 2)
        cut_h_end = int(r * H + cut_h_start)
        return cut_h_start, cut_h_end, flag
    else:
        cut_w_start = np.random.randint(W // 2)
        cut_w_end = int(r * W + cut_w_start)
        flag = False
        return cut_w_start, cut_w_end, flag


def CutTwo(img1, mask1, img2, mask2, p=0.5):
    # im = np.array(im)
    # labels = np.array(labels)
    # im2 = np.array(im2)
    # labels2 = np.array(labels2)
    r  = 0.5
    bb1, bb2, bflag = cut_bbox(img1.shape, r)

    if random.random() < p:
        if bflag:
            img2 = img2[:, bb1:bb2, :]
            augment_hsv(img2)
            img1[:, bb1:bb2, :] = img2
            mask1[:, bb1:bb2, :] = mask2[:, bb1:bb2, :]
        else:
            img2 = img2[bb1:bb2, :, :]
            augment_hsv(img2)
            img1[bb1:bb2, :, :] = img2
            mask1[bb1:bb2, :, :] = mask2[bb1:bb2, :, :]

    return img1, mask1

def read(img1, img2, mask1, mask2):
    # img1 = Image.open(img1)
    # img2 = Image.open(img2)
    # mask1 = Image.open(mask1)
    # mask2 = Image.open(mask2)

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    mask1 = cv2.imread(mask1)
    mask2 = cv2.imread(mask2)

    p = 1

    # re_im, re_label = CutTwo(img1, mask1, img2, mask2, p=p)
    # re_im, re_label = cutmix(re_im, img2, re_label, mask2, p=p)
    re_im, re_label = cutout(img1, mask1, p=p, isTest=True)
    # re_im, re_label =  mixup(img1, mask1, img2, mask2)

    # re_im.save('re_im.jpg')
    # re_label.save('re_label.png')
    cv2.imwrite('re_im.jpg', re_im)
    cv2.imwrite('re_label.png', re_label)



if __name__ == '__main__':


    img1 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/JPEGImages/1293.jpg'
    img2 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/JPEGImages/638.jpg'
    mask1 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/mask/1293.png'
    mask2 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/mask/638.png'

    read(img1, img2, mask1, mask2)



